
import cv2
import mediapipe as mp
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            # Get thumb and index finger
            thumb = hand.landmark[4]
            index = hand.landmark[8]

            # Calculate distance
            distance = math.sqrt(
                (thumb.x - index.x) ** 2 +
                (thumb.y - index.y) ** 2
            )

            # Detect pinch
            if distance < 0.05:

                message = "PINCH DETECTED"

                color = (0, 255, 0)

            else:

                message = "NO PINCH"

                color = (0, 0, 255)

            cv2.putText(
                frame,
                message,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            print(message, distance)

    cv2.imshow("Pinch Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()
