import cv2
import mediapipe as mp
import time
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================
# MEDIAPIPE HAND LANDMARKER SETUP
# ============================================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# ============================================
# CAMERA SETUP
# ============================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera didn't open")
    exit()

print("✅ Camera opened")
print()
print("☝️ 1 finger  → NEXT SLIDE")
print("✌️ 2 fingers → PREVIOUS SLIDE")
print("✋ 4 fingers → PAUSE / RESUME")
print("Press Q to quit")


# ============================================
# HAND CONNECTIONS
# ============================================

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# ============================================
# SETTINGS
# ============================================

COOLDOWN = 1.0
REQUIRED_FRAMES = 5

last_action_time = 0
last_gesture = None

gesture_candidate = None
gesture_frames = 0

controller_paused = False


# ============================================
# UI SETTINGS
# ============================================

WINDOW_NAME = "Gesture PowerPoint Controller"

TITLE_HEIGHT = 70
STATUS_HEIGHT = 55


# ============================================
# FINGER DETECTION
# ============================================

def get_finger_states(hand):

    index_up = hand[8].y < hand[6].y
    middle_up = hand[12].y < hand[10].y
    ring_up = hand[16].y < hand[14].y
    pinky_up = hand[20].y < hand[18].y

    return index_up, middle_up, ring_up, pinky_up


# ============================================
# MAIN LOOP
# ============================================

frame_timestamp = 0

while True:

    success, frame = cap.read()

    if not success:
        print("❌ Frame grab failed")
        break

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )

    frame_timestamp += 1


    # ========================================
    # UI HEADER
    # ========================================

    cv2.rectangle(
        frame,
        (0, 0),
        (w, TITLE_HEIGHT),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        frame,
        "GESTURE POWERPOINT CONTROLLER",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    # ========================================
    # DEFAULT VALUES
    # ========================================

    gesture = None
    finger_count = 0


    # ========================================
    # HAND DETECTED
    # ========================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        points = []


        # ====================================
        # DRAW LANDMARKS
        # ====================================

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # ====================================
        # DRAW SKELETON
        # ====================================

        for start, end in connections:

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )


        # ====================================
        # FINGER STATES
        # ====================================

        (
            index_up,
            middle_up,
            ring_up,
            pinky_up
        ) = get_finger_states(hand)


        finger_count = sum([
            index_up,
            middle_up,
            ring_up,
            pinky_up
        ])


        # ====================================
        # GESTURE DETECTION
        # ====================================

        # ☝️ NEXT
        if (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        ):

            gesture = "NEXT"


        # ✌️ PREVIOUS
        elif (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
        ):

            gesture = "PREVIOUS"


        # ✋ PAUSE / RESUME
        elif (
            index_up
            and middle_up
            and ring_up
            and pinky_up
        ):

            gesture = "PAUSE"


    # ========================================
    # ACTION LOGIC
    # ========================================

    current_time = time.time()

    if gesture is not None:

        if gesture == gesture_candidate:

            gesture_frames += 1

        else:

            gesture_candidate = gesture
            gesture_frames = 1


        if (
            gesture_frames >= REQUIRED_FRAMES
            and gesture != last_gesture
            and current_time - last_action_time > COOLDOWN
        ):

            if gesture == "PAUSE":

                controller_paused = not controller_paused

                if controller_paused:

                    print("⏸ CONTROLLER PAUSED")

                else:

                    print("▶️ CONTROLLER RESUMED")


            elif not controller_paused:

                if gesture == "NEXT":

                    print("☝️ NEXT SLIDE")

                    pyautogui.press("right")


                elif gesture == "PREVIOUS":

                    print("✌️ PREVIOUS SLIDE")

                    pyautogui.press("left")


            last_action_time = current_time
            last_gesture = gesture


    else:

        gesture_candidate = None
        gesture_frames = 0
        last_gesture = None


    # ========================================
    # STATUS BAR
    # ========================================

    status_y = h - STATUS_HEIGHT

    cv2.rectangle(
        frame,
        (0, status_y),
        (w, h),
        (35, 35, 35),
        -1
    )


    if controller_paused:

        status_text = "PAUSED"
        status_color = (0, 0, 255)

    else:

        status_text = "ACTIVE"
        status_color = (0, 255, 0)


    cv2.putText(
        frame,
        f"STATUS: {status_text}",
        (20, status_y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )


    # ========================================
    # FINGER COUNT
    # ========================================

    cv2.putText(
        frame,
        f"Fingers: {finger_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )


    # ========================================
    # CURRENT GESTURE
    # ========================================

    if gesture == "NEXT":

        gesture_text = "NEXT SLIDE"

    elif gesture == "PREVIOUS":

        gesture_text = "PREVIOUS SLIDE"

    elif gesture == "PAUSE":

        gesture_text = "PAUSE / RESUME"

    else:

        gesture_text = "NO GESTURE"


    cv2.putText(
        frame,
        gesture_text,
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================
    # INSTRUCTIONS
    # ========================================

    cv2.putText(
        frame,
        "1 Finger = Next",
        (w - 300, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "2 Fingers = Previous",
        (w - 300, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "4 Fingers = Pause",
        (w - 300, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================
    # SHOW CAMERA
    # ========================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================
    # QUIT
    # ========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================
# CLEANUP
# ============================================

cap.release()
detector.close()
cv2.destroyAllWindows()