
import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Start hand detection
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        print("Camera not detected!")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Thumb tip = 4
            thumb = hand_landmarks.landmark[4]

            # Index finger tip = 8
            index = hand_landmarks.landmark[8]

            # Display landmark positions
            print("Thumb:", thumb.x, thumb.y)
            print("Index:", index.x, index.y)

            cv2.putText(
                frame,
                "Hand Detected!",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()

