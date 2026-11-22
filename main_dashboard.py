
import cv2
import mediapipe as mp
import pyautogui
import math
import time
import tkinter as tk
from tkinter import messagebox
import threading

# =====================================
# INITIALIZE MEDIAPIPE
# =====================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================================
# HAND SETTINGS
# =====================================

previous_y = None
scroll_sensitivity = 600
pinch_threshold = 0.05

# =====================================
# EYE SETTINGS
# =====================================

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

SCROLL_AMOUNT = 70

# =====================================
# THREAD CONTROL
# =====================================

stop_event = threading.Event()
camera_thread = None


# =====================================
# CALCULATE EYE ASPECT RATIO
# =====================================

def calculate_ear(landmarks, points):

    p1 = landmarks[points[0]]
    p2 = landmarks[points[1]]
    p3 = landmarks[points[2]]
    p4 = landmarks[points[3]]
    p5 = landmarks[points[4]]
    p6 = landmarks[points[5]]

    vertical1 = math.dist(
        [p2.x, p2.y],
        [p6.x, p6.y]
    )

    vertical2 = math.dist(
        [p3.x, p3.y],
        [p5.x, p5.y]
    )

    horizontal = math.dist(
        [p1.x, p1.y],
        [p4.x, p4.y]
    )

    return (vertical1 + vertical2) / (2 * horizontal)


# =====================================
# COMBINED HAND + EYE DETECTION
# =====================================

def run_combined_system():

    global previous_y
    global eye_closed
    global closed_start
    global blink_count
    global last_blink_time

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        root.after(
            0,
            lambda: messagebox.showerror(
                "Camera Error",
                "Unable to open webcam."
            )
        )
        root.after(0, stop_system)
        return

    previous_y = None
    eye_closed = False
    closed_start = 0
    blink_count = 0
    last_blink_time = 0

    while not stop_event.is_set():

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        current_time = time.time()

        # =================================
        # HAND DETECTION
        # =================================

        hand_results = hands.process(rgb)

        hand_message = "HAND NOT DETECTED"

        if hand_results.multi_hand_landmarks:

            for hand in hand_results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                thumb = hand.landmark[4]
                index = hand.landmark[8]

                distance = math.sqrt(
                    (thumb.x - index.x) ** 2 +
                    (thumb.y - index.y) ** 2
                )

                if distance < pinch_threshold:

                    current_y = (
                        thumb.y + index.y
                    ) / 2

                    if previous_y is not None:

                        movement = current_y - previous_y

                        if abs(movement) > 0.002:

                            scroll_amount = int(
                                movement * scroll_sensitivity
                            )

                            pyautogui.scroll(-scroll_amount)

                    previous_y = current_y

                    hand_message = "HAND SCROLLING ACTIVE"

                else:

                    previous_y = None
                    hand_message = "PINCH TO SCROLL"

        else:
            previous_y = None

        # =================================
        # EYE DETECTION
        # =================================

        face_results = face_mesh.process(rgb)

        eye_message = "FACE NOT DETECTED"

        if face_results.multi_face_landmarks:

            for face in face_results.multi_face_landmarks:

                landmarks = face.landmark

                left_ear = calculate_ear(
                    landmarks,
                    LEFT_EYE
                )

                right_ear = calculate_ear(
                    landmarks,
                    RIGHT_EYE
                )

                average_ear = (
                    left_ear + right_ear
                ) / 2

                if average_ear < EAR_THRESHOLD:

                    if not eye_closed:

                        eye_closed = True
                        closed_start = current_time

                    eye_message = "EYES CLOSED"

                else:

                    if eye_closed:

                        duration = (
                            current_time - closed_start
                        )

                        if (
                            min_blink_duration
                            <= duration
                            <= max_blink_duration
                        ):

                            blink_count += 1
                            last_blink_time = current_time

                        eye_closed = False

                    eye_message = "EYES OPEN"

                # Process blink action
                if (
                    blink_count > 0
                    and current_time - last_blink_time
                    > double_blink_window
                ):

                    if blink_count == 1:

                        pyautogui.scroll(-SCROLL_AMOUNT)
                        print("Single blink: Scroll down")

                    elif blink_count >= 2:

                        pyautogui.scroll(SCROLL_AMOUNT)
                        print("Double blink: Scroll up")

                    blink_count = 0

                cv2.putText(
                    frame,
                    eye_message,
                    (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"EAR: {average_ear:.3f}",
                    (30, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Blink Count: {blink_count}",
                    (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        # =================================
        # DISPLAY HAND STATUS
        # =================================

        cv2.putText(
            frame,
            hand_message,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "HAND + EYE CONTROL ACTIVE",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Combined Touchless Scrolling System",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()

    hands.close()
    face_mesh.close()

    cv2.destroyAllWindows()

    root.after(0, stop_system)


# =====================================
# START SYSTEM
# =====================================

def start_system():

    global camera_thread

    if camera_thread is not None and camera_thread.is_alive():
        return

    stop_event.clear()

    status_label.config(
        text="STATUS: RUNNING",
        fg="green"
    )

    start_button.config(state="disabled")

    camera_thread = threading.Thread(
        target=run_combined_system,
        daemon=True
    )

    camera_thread.start()


# =====================================
# STOP SYSTEM
# =====================================

def stop_system():

    stop_event.set()

    status_label.config(
        text="STATUS: STOPPED",
        fg="red"
    )

    start_button.config(state="normal")


# =====================================
# DASHBOARD DESIGN
# =====================================

root = tk.Tk()

root.title("Touchless Scrolling System")

root.geometry("650x500")

root.configure(bg="#172033")

root.resizable(False, False)


heading = tk.Label(
    root,
    text="TOUCHLESS SCROLLING SYSTEM",
    font=("Arial", 21, "bold"),
    bg="#172033",
    fg="white"
)

heading.pack(pady=(35, 15))


subtitle = tk.Label(
    root,
    text="Hand Gesture and Eye Blink Based Scrolling",
    font=("Arial", 12),
    bg="#172033",
    fg="#B8C7E0"
)

subtitle.pack(pady=10)


info = tk.Label(
    root,
    text=(
        "Both hand and eye detection work together.\n"
        "Pinch and move your hand to scroll.\n"
        "Blink to control scrolling."
    ),
    font=("Arial", 12),
    bg="#263653",
    fg="white",
    width=55,
    height=5
)

info.pack(pady=25)


start_button = tk.Button(
    root,
    text="START SYSTEM",
    font=("Arial", 14, "bold"),
    bg="#2980B9",
    fg="white",
    width=22,
    height=2,
    command=start_system
)

start_button.pack(pady=12)


stop_button = tk.Button(
    root,
    text="STOP SYSTEM",
    font=("Arial", 12, "bold"),
    bg="#E74C3C",
    fg="white",
    width=22,
    height=2,
    command=stop_system
)

stop_button.pack(pady=10)


status_label = tk.Label(
    root,
    text="STATUS: READY",
    font=("Arial", 13, "bold"),
    bg="#172033",
    fg="#27AE60"
)

status_label.pack(pady=15)


root.mainloop()
