from flask import Flask, request, render_template, redirect, url_for, flash, session
import sqlite3
import numpy as np
import cv2
import base64
import torch
from io import BytesIO
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import random  # For generating OTP 

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Flask app setup
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Temporary storage for OTPs (use a more secure storage like Redis in production)
otp_storage = {}

# Database setup
def init_db():
    with sqlite3.connect('users.db') as conn:
        cursor = conn.cursor()

        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT NOT NULL,
                            email TEXT NOT NULL UNIQUE,
                            mobile TEXT NOT NULL,
                            password TEXT NOT NULL,
                            face_embedding BLOB NOT NULL
                        )''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS linked_accounts (
                            main_user_id INTEGER,
                            linked_user_id INTEGER,
                            FOREIGN KEY (main_user_id) REFERENCES users(id),
                            FOREIGN KEY (linked_user_id) REFERENCES users(id)
                        )''')

        # ✅ New Table for Profile Pics
        cursor.execute('''CREATE TABLE IF NOT EXISTS profile_pics (
                            user_id INTEGER PRIMARY KEY,
                            image_path TEXT,
                            FOREIGN KEY (user_id) REFERENCES users(id)
                        )''')

        conn.commit()

init_db()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        captured_image_data = request.form['captured_image']

        if not captured_image_data:
            flash('Live face authentication required.', 'error')
            return redirect(request.url)

        # Decode base64 image
        header, encoded = captured_image_data.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))
        img = np.array(image)

        # Convert to grayscale and detect face
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            flash('No face detected.', 'error')
            return redirect(request.url)

        # Extract first detected face
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = np.transpose(face, (2, 0, 1)) / 255.0
        face_tensor = np.expand_dims(face, axis=0).astype(np.float32)

        with torch.no_grad():
            uploaded_face_embedding = model(torch.tensor(face_tensor)).numpy()[0]

        # Fetch user details from database
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, password, face_embedding FROM users WHERE email = ?', (email,))
            result = cursor.fetchone()

            if not result:
                flash('User not found.', 'error')
                return redirect(request.url)

            user_id, stored_password, stored_face_embedding = result

            if stored_password != password:
                flash('Invalid email or password.', 'error')
                return redirect(request.url)

            stored_face_embedding = np.frombuffer(stored_face_embedding, dtype=np.float32)
            similarity = 1 - cosine(stored_face_embedding, uploaded_face_embedding)

            if similarity >= 0.4:
                session['user_id'] = user_id
                session['email'] = email

                # Fetch linked accounts (both ways)
                cursor.execute('''
                    SELECT u.id, u.email FROM users u 
                    WHERE u.id IN (
                        SELECT linked_user_id FROM linked_accounts WHERE main_user_id = ?
                        UNION
                        SELECT main_user_id FROM linked_accounts WHERE linked_user_id = ?
                    )
                ''', (user_id, user_id))

                linked_accounts = cursor.fetchall()
                session['linked_accounts'] = linked_accounts

                return redirect(url_for('dashboard'))
            else:
                flash('Face authentication failed.', 'error')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        mobile = request.form['mobile']
        password = request.form['password']
        face_image = request.files['face_image']

        if not face_image:
            flash('Please upload a face image.', 'error')
            return redirect(request.url)

        # Convert image to embedding
        image = Image.open(face_image)
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            flash('No face detected.', 'error')
            return redirect(request.url)

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = np.transpose(face, (2, 0, 1)) / 255.0
        face_tensor = np.expand_dims(face, axis=0).astype(np.float32)

        with torch.no_grad():
            new_face_embedding = model(torch.tensor(face_tensor)).numpy()[0]

        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()

            # Check if the email already exists
            cursor.execute('SELECT id, face_embedding FROM users WHERE email = ?', (email,))
            existing_user = cursor.fetchone()

            if existing_user:
                flash('Email is already registered.', 'error')
                return redirect(request.url)

            # Check if face exists
            cursor.execute('SELECT id, email, face_embedding FROM users')
            users = cursor.fetchall()

            matching_user_id = None
            for user_id, user_email, stored_face_embedding in users:
                stored_face_embedding = np.frombuffer(stored_face_embedding, dtype=np.float32)
                similarity = 1 - cosine(stored_face_embedding, new_face_embedding)

                if similarity >= 0.4:
                    matching_user_id = user_id  # Store user ID with matching face
                    existing_email_for_face = user_email
                    break

            if matching_user_id:
                # Case: Face exists, but email is new → Ask for confirmation
                session['pending_user'] = {
                    'username': username,
                    'email': email,
                    'mobile': mobile,
                    'password': password,
                    'face_embedding': new_face_embedding.tobytes(),
                    'existing_user_id': matching_user_id,
                    'existing_email': existing_email_for_face
                }
                return redirect(url_for('confirm_linking'))  # Redirect to confirmation page

            else:
                # Completely new user
                cursor.execute('''INSERT INTO users (username, email, mobile, password, face_embedding)
                                  VALUES (?, ?, ?, ?, ?)''',
                               (username, email, mobile, password, new_face_embedding.tobytes()))
                conn.commit()
                flash('Signup successful!', 'success')

        return redirect(url_for('signup'))

    return render_template('signup.html')

@app.route('/confirm_linking', methods=['GET', 'POST'])
def confirm_linking():
    if 'pending_user' not in session:
        flash('No pending user data found.', 'error')
        return redirect(url_for('signup'))

    pending_user = session['pending_user']

    if request.method == 'POST':
        choice = request.form.get('choice')

        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()

            if choice == 'yes':
                cursor.execute('''INSERT INTO users (username, email, mobile, password, face_embedding)
                                  VALUES (?, ?, ?, ?, ?)''',
                               (pending_user['username'], pending_user['email'], pending_user['mobile'], 
                                pending_user['password'], pending_user['face_embedding']))
                new_user_id = cursor.lastrowid

                # Link new account with existing face account
                cursor.execute('INSERT INTO linked_accounts (main_user_id, linked_user_id) VALUES (?, ?)',
                               (pending_user['existing_user_id'], new_user_id))

                conn.commit()
                flash(f'Account created and linked with existing account ({pending_user["existing_email"]}).', 'success')

            else:
                flash('Signup canceled.', 'error')

        session.pop('pending_user', None)  # Remove session data
        return redirect(url_for('signup'))

    return render_template('confirm_linking.html', existing_email=pending_user['existing_email'])

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    email = session['email']
    linked_accounts = session.get('linked_accounts', [])

    bio = session.get('bio', '')
    profile_pic = None

    # ✅ Handle form submission
    if request.method == 'POST':
        bio = request.form['bio']
        session['bio'] = bio  # (Optional) store bio in session

        profile_pic_file = request.files.get('profile_pic')
        if profile_pic_file:
            filename = email.replace('@', '_') + "_" + profile_pic_file.filename
            profile_pic_path = f'static/uploads/{filename}'
            profile_pic_file.save(profile_pic_path)

            # ✅ Insert or update into profile_pics table
            with sqlite3.connect('users.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''INSERT OR REPLACE INTO profile_pics (user_id, image_path)
                                  VALUES (?, ?)''', (user_id, profile_pic_path))
                conn.commit()

    # ✅ Retrieve profile picture if exists
    with sqlite3.connect('users.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT image_path FROM profile_pics WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        if result:
            profile_pic = result[0]

    return render_template(
        'dashboard.html',
        user_id=user_id,
        email=email,
        linked_accounts=linked_accounts,
        bio=bio,
        profile_pic=profile_pic,
        random=random
    )

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    email = request.args.get('email')
    if request.method == 'POST':
        entered_otp = request.form['otp']
        stored_otp = otp_storage.get(email)

        if stored_otp and entered_otp == stored_otp:
            del otp_storage[email]  # Clear OTP after successful verification
            return render_template('web.html')
        else:
            flash('Invalid OTP. Please try again.', 'error')
            return redirect(url_for('verify_otp', email=email))

    return render_template('verify_otp.html', email=email)


@app.route('/logout')
def logout():
    session.clear()  # Clear session data
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))  # Redirect to login page


#if __name__ == '__main__':
#    app.run(debug=True)


import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))