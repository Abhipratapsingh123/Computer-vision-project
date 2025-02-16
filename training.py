import cv2
import numpy as np
import os

# Initialize LBPH Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

dataset_path = "C://Users//abhip//Desktop//Minor-Project//dataset"

# Get all images and their respective IDs
faces, ids = [], []

for filename in os.listdir(dataset_path):
    if filename.startswith("user"):
        img_path = os.path.join(dataset_path, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping {img_path}: Unable to load image.")
            continue  # Skip invalid images

        # Resize images to ensure consistency
        image = cv2.resize(image, (100, 100))  # Resize as per requirement
        face_id = int(filename.split(".")[1])  # Extract ID from filename

        faces.append(image)
        ids.append(face_id)

# Train the recognizer
face_recognizer.train(np.asarray(faces, dtype='uint8'), np.array(ids))

# Save the trained model
face_recognizer.save("lbph_classifier.yml")

print("Training complete. Model saved as 'lbph_classifier.yml'.")
