
import cv2

# Open laptop webcam
camera = cv2.VideoCapture(0)

while True:

    # Capture video frame
    success, frame = camera.read()

    if not success:
        print("Camera not detected!")
        break

    # Display webcam video
    cv2.imshow("Touchless Scrolling Project", frame)

    # Press Q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
camera.release()
cv2.destroyAllWindows()
