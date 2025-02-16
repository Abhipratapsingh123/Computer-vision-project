import cv2

# Load the face detector and recognizer
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("lbph_classifier.yml")

# Set dimensions and font
width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Start video capture
camera = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    connected, image = camera.read()
    
    if not connected:
        break
    
    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.5, minSize=(30, 30))

    for (x, y, w, h) in detections:
        # Resize detected face
        image_face = cv2.resize(image_gray[y:y + w, x:x + h], (width, height))
        
        # Draw rectangle around face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Predict face
        id, confidence = face_recognizer.predict(image_face)
        name = ""

        if id == 1:
            name = 'Abhi pratap'
        elif id == 2:
            name = 'singh'
        
        # Display name and confidence score
        cv2.putText(image, name, (x, y + (h + 30)), font, 2, (0, 0, 255))
        cv2.putText(image, str(confidence), (x, y + (h + 50)), font, 1, (0, 0, 255))

    # Show the processed frame
    cv2.imshow("Face Recognition", image)
    
    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
camera.release()
cv2.destroyAllWindows()
