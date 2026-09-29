import cv2
import mediapipe as mp
import math
import time
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# MediaPipe setup
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera didn't open")
    exit()

print("✅ Camera opened")
print("🤏 Pinch + move left/right")
print("Press Q to quit")


frame_timestamp = 0


# -----------------------------
# Hand connections
# -----------------------------

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# -----------------------------
# Swipe variables
# -----------------------------

pinch_active = False
start_x = None

SWIPE_DISTANCE = 0.10
COOLDOWN = 0.8

last_swipe_time = 0


# -----------------------------
# Main loop
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("❌ Frame grab failed")
        break

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


    # -------------------------
    # Detect hand
    # -------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        h, w, _ = frame.shape

        points = []

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


        # Draw skeleton

        for start, end in connections:

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )


        # -------------------------
        # Pinch detection
        # -------------------------

        thumb = hand[4]
        index = hand[8]

        distance = math.sqrt(
            (thumb.x - index.x) ** 2 +
            (thumb.y - index.y) ** 2
        )

        is_pinching = distance < 0.05


        # -------------------------
        # Pinch started
        # -------------------------

        if is_pinching and not pinch_active:

            pinch_active = True
            start_x = (thumb.x + index.x) / 2

            print("🤏 Pinch started")


        # -------------------------
        # Pinch released
        # -------------------------

        elif not is_pinching and pinch_active:

            pinch_active = False
            start_x = None

            print("Pinch released")


        # -------------------------
        # Track movement while pinching
        # -------------------------

        if pinch_active:

            current_x = (thumb.x + index.x) / 2

            movement = current_x - start_x

            cv2.putText(
                frame,
                "PINCH ACTIVE",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3
            )

            # ---------------------
            # Swipe right
            # ---------------------

            if movement > SWIPE_DISTANCE:

                current_time = time.time()

                if current_time - last_swipe_time > COOLDOWN:

                    print("👉 SWIPE RIGHT")
                    pyautogui.press("right")

                    last_swipe_time = current_time
                    pinch_active = False
                    start_x = None


            # ---------------------
            # Swipe left
            # ---------------------

            elif movement < -SWIPE_DISTANCE:

                current_time = time.time()

                if current_time - last_swipe_time > COOLDOWN:

                    print("👈 SWIPE LEFT")
                    pyautogui.press("left")

                    last_swipe_time = current_time
                    pinch_active = False
                    start_x = None


    else:

        pinch_active = False
        start_x = None


    cv2.imshow(
        "Gesture Controller - Swipe Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
detector.close()
cv2.destroyAllWindows()