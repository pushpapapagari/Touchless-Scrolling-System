
import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

# Scrolling settings
previous_y = None
scroll_sensitivity = 600
pinch_threshold = 0.05

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            # Get thumb and index finger
            thumb = hand.landmark[4]
            index = hand.landmark[8]

            # Calculate pinch distance
            distance = math.sqrt(
                (thumb.x - index.x) ** 2 +
                (thumb.y - index.y) ** 2
            )

            if distance < pinch_threshold:

                current_y = (thumb.y + index.y) / 2

                if previous_y is not None:

                    movement = current_y - previous_y

                    if abs(movement) > 0.002:

                        scroll_amount = int(
                            movement * scroll_sensitivity
                        )

                        pyautogui.scroll(-scroll_amount)

                previous_y = current_y

                message = "SCROLLING ACTIVE"
                color = (0, 255, 0)

            else:

                previous_y = None

                message = "PINCH TO SCROLL"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                message,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    else:
        previous_y = None

    cv2.imshow("Touchless Scrolling System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
hands.close()
cv2.destroyAllWindows()
