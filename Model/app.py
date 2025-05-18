from flask import Flask, request, jsonify, render_template
from keras._tf_keras.keras.models import load_model, Model
from keras._tf_keras.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout
import numpy as np
import pandas as pd
import os
import cv2
import threading  # For running retraining asynchronously
import sqlite3
import csv

app = Flask(__name__)

# Load the trained model
model = load_model("ocular_disease_model.keras")

# Global variables
prediction_counter = 0
upload_lock = False  # Lock to prevent uploads during retraining
metadata_file = "C:/Users/jasar/Desktop/OcularDiseaseRecognition/metadata.csv"
# Database setup
DB_FILE = "predictions.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                normal INTEGER,
                diabetes INTEGER,
                glaucoma INTEGER,
                cataract INTEGER,
                amd INTEGER,
                hypertension INTEGER,
                myopia INTEGER,
                other INTEGER
            )
        """)
        conn.commit()

# Initialize the database
init_db()

# Function to preprocess the uploaded images
def preprocess_images(left_img, right_img):
    left_image = cv2.imread(left_img)
    right_image = cv2.imread(right_img)

    if left_image is None or right_image is None:
        return None

    left_image = cv2.resize(left_image, (224, 224)) / 255.0
    right_image = cv2.resize(right_image, (224, 224)) / 255.0

    # Concatenate left and right images along the channel axis to make the shape (224, 224, 6)
    combined_image = np.concatenate((left_image, right_image), axis=-1)

    return combined_image

# Load a retina validation model
retina_validation_model = load_model("retina_validation_model.keras")

def validate_retina_image(img_path):
    """Check if the image is a retina picture."""
    img = cv2.imread(img_path)
    if img is None:
        return False
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = retina_validation_model.predict(img)
    return prediction[0][0] > 0.5  # Returns True if the image is a retina picture

# Function to retrain the model on new data
def retrain_model():
    global model, upload_lock
    print("Retraining model...")

    upload_lock = True

    # Load and preprocess metadata
    data = pd.read_csv(metadata_file)
    metadata = data[["age", "sex"]]
    metadata["sex"] = metadata["sex"].map({"Male": 0, "Female": 1})
    metadata = metadata.to_numpy()

    labels = data[["N", "D", "G", "C", "A", "H", "M", "O"]].to_numpy()

    # Load and preprocess images
    images = []
    filtered_labels = []
    filtered_metadata = []
    for idx, row in data.iterrows():
        left_image_path = row["left_image_path"]
        right_image_path = row["right_image_path"]

        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)

        if left_image is not None and right_image is not None:
            left_image = cv2.resize(left_image, (224, 224)) / 255.0
            right_image = cv2.resize(right_image, (224, 224)) / 255.0
            combined_image = np.concatenate((left_image, right_image), axis=-1)

            images.append(combined_image)
            filtered_labels.append(labels[idx])
            filtered_metadata.append(metadata[idx])

    images = np.array(images)
    filtered_labels = np.array(filtered_labels)
    filtered_metadata = np.array(filtered_metadata)

    print(f"Images: {len(images)}, Labels: {len(filtered_labels)}, Metadata: {len(filtered_metadata)}")

    # Define the model with metadata input
    image_input = Input(shape=(224, 224, 6))
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    metadata_input = Input(shape=(2,))
    metadata_dense = Dense(32, activation='relu')(metadata_input)

    combined = Concatenate()([x, metadata_dense])
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.5)(x)
    output = Dense(8, activation='softmax')(x)

    new_model = Model(inputs=[image_input, metadata_input], outputs=output)
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    new_model.fit([images, filtered_metadata], filtered_labels, epochs=5, batch_size=32)

    # Save the retrained model
    new_model.save('ocular_disease_model.keras')
    print("Model retrained and saved successfully.")

    # Clear the metadata.csv file after retraining and keep only the header
    with open(metadata_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["age", "sex", "left_image_path", "right_image_path","N", "D", "G", "C", "A", "H", "M", "O"])

    upload_lock = False

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_counter, upload_lock, metadata_file

    if upload_lock:
        return jsonify({"error": "Server je zauzet retreniranjem modela. Pokušajte ponovo kasnije."})

    left_image = request.files['left_eye']
    right_image = request.files['right_eye']
    age = int(request.form['age'])
    sex = 0 if request.form['sex'] == 'Male' else 1

    # Save temporarily for validation
    temp_left_path = os.path.join('uploads', f'temp_left_{prediction_counter}.jpg')
    temp_right_path = os.path.join('uploads', f'temp_right_{prediction_counter}.jpg')
    left_image.save(temp_left_path)
    right_image.save(temp_right_path)

    # Validate if both images are retina pictures
    if not validate_retina_image(temp_left_path) or not validate_retina_image(temp_right_path):
        # Delete temporary files if validation fails
        os.remove(temp_left_path)
        os.remove(temp_right_path)
        return jsonify({"error": "Uploadovane fotografije nisu fotografije retina."}), 400

    # Rename validated files to final paths
    left_path = os.path.join('uploads', f'left_{prediction_counter}.jpg')
    right_path = os.path.join('uploads', f'right_{prediction_counter}.jpg')
    os.rename(temp_left_path, left_path)
    os.rename(temp_right_path, right_path)

    # Prepare prediction input
    combined_image = preprocess_images(left_path, right_path)
    if combined_image is None:
        return jsonify({"error": "Error u procesuiranju fotografija."})

    metadata = np.array([[age, sex]])

    # Make prediction
    prediction = model.predict([np.expand_dims(combined_image, axis=0), metadata])
    print("Prediction raw output:", prediction)

    # Define predicted labels
    predicted_labels = {
        "Normal": int(prediction[0][0] >= 0.9),
        "Diabetes": int(prediction[0][1] >= 0.9),
        "Glaucoma": int(prediction[0][2] >= 0.9),
        "Cataract": int(prediction[0][3] >= 0.9),
        "Age-related Macular Degeneration": int(prediction[0][4] >= 0.9),
        "Hypertension": int(prediction[0][5] >= 0.9),
        "Pathological Myopia": int(prediction[0][6] >= 0.9),
        "Other": int(prediction[0][7] >= 0.9),
    }

    prediction_percentage = {key: f"{value * 100:.2f}%" for key, value in zip(predicted_labels.keys(), prediction[0])}

    # Append metadata and predicted labels to the CSV
    with open(metadata_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([age, 'Male' if sex == 0 else 'Female', left_path, right_path] + list(predicted_labels.values()))

    # Save prediction to the database
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, other)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(predicted_labels.values()))
        conn.commit()

    prediction_counter += 1
    print(f"Prediction count: {prediction_counter}")

    # Trigger retraining after every 10 predictions
    if prediction_counter == 10:
        threading.Thread(target=retrain_model).start()
        prediction_counter = 0

    return jsonify(prediction_percentage)




@app.route('/chart_data', methods=['GET'])
def chart_data():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                SUM(normal), SUM(diabetes), SUM(glaucoma), SUM(cataract),
                SUM(amd), SUM(hypertension), SUM(myopia), SUM(other)
            FROM predictions
        """)
        result = cursor.fetchone()

    return jsonify({
        "Normal": result[0] or 0,
        "Diabetes": result[1] or 0,
        "Glaucoma": result[2] or 0,
        "Cataract": result[3] or 0,
        "Age-related Macular Degeneration": result[4] or 0,
        "Hypertension": result[5] or 0,
        "Pathological Myopia": result[6] or 0,
        "Other": result[7] or 0
    })

if __name__ == '__main__':
    app.run(debug=True)
