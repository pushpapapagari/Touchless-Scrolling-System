
import cv2
import mediapipe as mp
import math

# Initialize Face Mesh
mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

# Eye landmark positions
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Blink settings
EAR_THRESHOLD = 0.21
blink_count = 0
eye_closed = False


# Calculate Eye Aspect Ratio
def calculate_ear(landmarks, eye_points):

    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    vertical_1 = math.dist(
        [p2.x, p2.y], [p6.x, p6.y]
    )

    vertical_2 = math.dist(
        [p3.x, p3.y], [p5.x, p5.y]
    )

    horizontal = math.dist(
        [p1.x, p1.y], [p4.x, p4.y]
    )

    ear = (vertical_1 + vertical_2) / (2 * horizontal)

    return ear


while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            landmarks = face.landmark

            left_ear = calculate_ear(
                landmarks, LEFT_EYE
            )

            right_ear = calculate_ear(
                landmarks, RIGHT_EYE
            )

            average_ear = (left_ear + right_ear) / 2

            # Detect eye closing
            if average_ear < EAR_THRESHOLD:

                if not eye_closed:
                    blink_count += 1
                    eye_closed = True

                message = "EYES CLOSED"
                color = (0, 0, 255)

            else:

                eye_closed = False

                message = "EYES OPEN"
                color = (0, 255, 0)

            # Display information
            cv2.putText(
                frame,
                message,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

            cv2.putText(
                frame,
                f"EAR: {average_ear:.3f}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Blink Count: {blink_count}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    cv2.imshow("Eye Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
face_mesh.close()
cv2.destroyAllWindows()
