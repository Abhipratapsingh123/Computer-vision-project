import cv2
import dlib
import os

# Load dlib's HOG-based face detector
face_detector = dlib.get_frontal_face_detector()

# Creating a dataset directory
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

# Get user ID for labeling
person_id = input("Enter a unique ID for this person: ")

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

image_count = 0

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using dlib HOG
    faces = face_detector(gray_frame)

    for face in faces:
        x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        
        # Ensure coordinates are within bounds
        x, y = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        # Crop the detected face
        face_image = gray_frame[y:y2, x:x2]

        # Save the image
        image_count += 1
        image_filename = f"{dataset_path}/user.{person_id}.{image_count}.jpg"
        cv2.imwrite(image_filename, face_image)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        # Display progress
        cv2.putText(frame, f"Captured: {image_count}/50", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Capturing Faces", frame)

    # Stop when 50 images are captured
    if image_count >= 50:
        print(f"✅ Captured 50 images for person ID {person_id}.")
        break

    # Quit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⛔ Process manually stopped by user.")
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()

print("✅ Face data collection complete!")
