import cv2

# Open the camera (change index if needed)
cap = cv2.VideoCapture(0)  # Use the correct index (e.g., 0, 1, etc.)

if not cap.isOpened():
    print("Camera could not be opened!")
    exit()

print("Camera is working! Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame!")
        break

    # Show the video feed
    cv2.imshow("Camera Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close window
cap.release()
cv2.destroyAllWindows()
