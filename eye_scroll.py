
import cv2
import mediapipe as mp
import pyautogui
import math
import time

mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.21

eye_closed = False
closed_start = 0
blink_count = 0
last_blink_time = 0

double_blink_window = 0.6
min_blink_duration = 0.08
max_blink_duration = 0.4

# Increased scrolling speed
SCROLL_AMOUNT = 70


def calculate_ear(landmarks, points):
    p1 = landmarks[points[0]]
    p2 = landmarks[points[1]]
    p3 = landmarks[points[2]]
    p4 = landmarks[points[3]]
    p5 = landmarks[points[4]]
    p6 = landmarks[points[5]]

    vertical1 = math.dist([p2.x, p2.y], [p6.x, p6.y])
    vertical2 = math.dist([p3.x, p3.y], [p5.x, p5.y])
    horizontal = math.dist([p1.x, p1.y], [p4.x, p4.y])

    return (vertical1 + vertical2) / (2 * horizontal)


while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_time = time.time()

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            landmarks = face.landmark

            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            average_ear = (left_ear + right_ear) / 2

            if average_ear < EAR_THRESHOLD:
                if not eye_closed:
                    eye_closed = True
                    closed_start = current_time

                message = "EYES CLOSED"

            else:
                if eye_closed:
                    duration = current_time - closed_start

                    if min_blink_duration <= duration <= max_blink_duration:
                        blink_count += 1
                        last_blink_time = current_time

                    eye_closed = False

                message = "EYES OPEN"

            if (
                blink_count > 0
                and current_time - last_blink_time > double_blink_window
            ):
                if blink_count == 1:
                    pyautogui.scroll(-SCROLL_AMOUNT)
                    print("Single blink: Scroll down")

                elif blink_count >= 2:
                    pyautogui.scroll(SCROLL_AMOUNT)
                    print("Double blink: Scroll up")

                blink_count = 0

            cv2.putText(
                frame, message, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"EAR: {average_ear:.3f}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )

            cv2.putText(
                frame, f"Blink Count: {blink_count}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )

    else:
        eye_closed = False
        blink_count = 0

    cv2.imshow("Eye Controlled Scrolling", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
face_mesh.close()
cv2.destroyAllWindows()
