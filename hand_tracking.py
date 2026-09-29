import cv2
import mediapipe as mp
import time
import pyautogui
import os
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera didn't open")
    exit()

# Try to keep a stable display size.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

print("✅ Camera opened")
print()
print("☝️ 1 finger  → NEXT SLIDE")
print("✌️ 2 fingers → PREVIOUS SLIDE")
print("✋ 4 fingers → PAUSE / RESUME")
print("Press Q to quit")


# ============================================================
# HAND CONNECTIONS
# ============================================================

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# ============================================================
# GESTURE SETTINGS
# ============================================================

COOLDOWN = 1.0
REQUIRED_FRAMES = 5

last_action_time = 0
last_gesture = None

gesture_candidate = None
gesture_frames = 0

controller_paused = False


# ============================================================
# COLORS
# ============================================================

TEAL = (55, 230, 210)
WHITE = (220, 235, 235)
GREEN = (80, 255, 150)
RED = (255, 80, 90)


# ============================================================
# FONT SETUP
# ============================================================

FONT_REGULAR = r"C:\Windows\Fonts\consola.ttf"
FONT_BOLD = r"C:\Windows\Fonts\consolab.ttf"

if not os.path.exists(FONT_REGULAR):
    FONT_REGULAR = r"C:\Windows\Fonts\arial.ttf"

if not os.path.exists(FONT_BOLD):
    FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


def get_font(size, bold=False):
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size)

# ============================================================
# BACKGROUND WITH DEPTH (soft vignette instead of flat color)
# ============================================================

_bg_cache = {}

def build_background(w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.5, h * 0.42
    max_dist = np.sqrt((w * 0.6) ** 2 + (h * 0.6) ** 2)
    dist = np.clip(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_dist, 0, 1)

    brightness = (30 * (1 - dist) + 5).astype(np.uint8)

    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[..., 0] = brightness
    bg[..., 1] = np.clip(brightness.astype(int) * 1.4, 0, 255).astype(np.uint8)
    bg[..., 2] = np.clip(brightness.astype(int) * 1.3, 0, 255).astype(np.uint8)

    return Image.fromarray(bg, mode="RGB").convert("RGBA")


def get_background(w, h):
    key = (w, h)
    if key not in _bg_cache:
        _bg_cache[key] = build_background(w, h)
    return _bg_cache[key].copy()

# ============================================================
# FINGER DETECTION
# ============================================================

def get_finger_states(hand):
    index_up = hand[8].y < hand[6].y
    middle_up = hand[12].y < hand[10].y
    ring_up = hand[16].y < hand[14].y
    pinky_up = hand[20].y < hand[18].y

    return index_up, middle_up, ring_up, pinky_up


# ============================================================
# GLOW TEXT
# ============================================================

def draw_glow_text(
    base_image,
    position,
    text,
    font,
    fill=TEAL,
    glow_radius=6
):
    glow_layer = Image.new(
        "RGBA",
        base_image.size,
        (0, 0, 0, 0)
    )

    glow_draw = ImageDraw.Draw(glow_layer)

    glow_draw.text(
        position,
        text,
        font=font,
        fill=(*fill, 180)
    )

    glow_layer = glow_layer.filter(
        ImageFilter.GaussianBlur(glow_radius)
    )

    base_image.alpha_composite(glow_layer)

    draw = ImageDraw.Draw(base_image)

    draw.text(
        position,
        text,
        font=font,
        fill=(*fill, 255)
    )


# ============================================================
# SCALE HAND SKELETON FOR HUD
# ============================================================

def transform_hand_points(points, width, height):
    """
    The webcam is only a sensor.
    We don't display the camera image.

    Instead, the detected hand is resized and centered into
    a clean HUD area, similar to the reference mockup.
    """

    if not points:
        return []

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    hand_width = max(max_x - min_x, 1)
    hand_height = max(max_y - min_y, 1)

    # Central area reserved for the hand.
    target_center_x = width // 2
    target_center_y = int(height * 0.55)

    max_target_width = int(width * 0.34)
    max_target_height = int(height * 0.48)

    scale_x = max_target_width / hand_width
    scale_y = max_target_height / hand_height

    scale = min(scale_x, scale_y)

    # Don't allow the skeleton to become absurdly large.
    scale = min(scale, 2.0)

    scaled_width = hand_width * scale
    scaled_height = hand_height * scale

    start_x = target_center_x - scaled_width / 2
    start_y = target_center_y - scaled_height / 2

    transformed = []

    for x, y in points:
        new_x = int(start_x + (x - min_x) * scale)
        new_y = int(start_y + (y - min_y) * scale)
        transformed.append((new_x, new_y))

    return transformed

def draw_hud(frame, finger_count, gesture, points):

    h, w, _ = frame.shape

    hud = get_background(w, h)
    draw = ImageDraw.Draw(hud, "RGBA")

    # subtle outer frame border, ties the whole HUD together
    draw.rectangle((0, 0, w - 1, h - 1), outline=(*TEAL, 60), width=1)

    for y in range(90, h - 55, 8):
        draw.line((18, y, w - 18, y), fill=(35, 120, 115, 14), width=1)

    header_height = 62
    draw.rectangle((0, 0, w, header_height), fill=(5, 15, 14, 255))
    draw.line((0, header_height, w, header_height), fill=(*TEAL, 120), width=1)

    draw_glow_text(hud, (28, 17), "GESTURE POWERPOINT CONTROLLER",
                    get_font(23, bold=True), fill=TEAL, glow_radius=5)

    bracket = 20
    corner_color = (*TEAL, 150)
    draw.line((8, 8, 8 + bracket, 8), fill=corner_color, width=2)
    draw.line((8, 8, 8, 8 + bracket), fill=corner_color, width=2)
    draw.line((w - 8, 8, w - 8 - bracket, 8), fill=corner_color, width=2)
    draw.line((w - 8, 8, w - 8, 8 + bracket), fill=corner_color, width=2)
    draw.line((8, h - 8, 8 + bracket, h - 8), fill=corner_color, width=2)
    draw.line((8, h - 8, 8, h - 8 - bracket), fill=corner_color, width=2)
    draw.line((w - 8, h - 8, w - 8 - bracket, h - 8), fill=corner_color, width=2)
    draw.line((w - 8, h - 8, w - 8, h - 8 - bracket), fill=corner_color, width=2)

    # ---------------- LEFT INFO (now boxed) ----------------
    left_x, left_y = 28, 95
    label_font = get_font(13)
    value_font = get_font(24, bold=True)
    gesture_font = get_font(17, bold=True)

    draw.rounded_rectangle(
        (left_x - 14, left_y - 14, left_x + 220, left_y + 130),
        radius=8, fill=(8, 20, 19, 140), outline=(*TEAL, 130), width=1
    )

    draw.text((left_x, left_y), "FINGERS", font=label_font, fill=(*WHITE, 190))
    draw_glow_text(hud, (left_x, left_y + 18), str(finger_count), value_font, fill=TEAL, glow_radius=5)
    draw.text((left_x, left_y + 70), "GESTURE", font=label_font, fill=(*WHITE, 190))

    gesture_text = {"NEXT": "NEXT SLIDE", "PREVIOUS": "PREVIOUS SLIDE",
                     "PAUSE": "PAUSE / RESUME"}.get(gesture, "WAITING...")
    draw_glow_text(hud, (left_x, left_y + 91), gesture_text, gesture_font, fill=TEAL, glow_radius=4)

    # ---------------- RIGHT CONTROLS (now boxed) ----------------
    right_x, right_y = w - 250, 95

    draw.rounded_rectangle(
        (right_x - 14, right_y - 14, right_x + 235, right_y + 118),
        radius=8, fill=(8, 20, 19, 140), outline=(*TEAL, 130), width=1
    )

    draw.text((right_x, right_y), "CONTROLS", font=label_font, fill=(*WHITE, 190))
    controls = ["1 finger  →  Next", "2 fingers →  Previous", "4 fingers →  Pause"]
    for i, control in enumerate(controls):
        draw.text((right_x, right_y + 30 + i * 28), control, font=get_font(13), fill=(*TEAL, 235))

    # ---------------- CENTRAL HAND SKELETON (unchanged) ----------------
    display_points = transform_hand_points(points, w, h)

    if display_points:
        glow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer, "RGBA")

        for start, end in connections:
            x1, y1 = display_points[start]
            x2, y2 = display_points[end]
            glow_draw.line((x1, y1, x2, y2), fill=(*TEAL, 220), width=7)

        for x, y in display_points:
            glow_draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(*TEAL, 180))

        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(8))
        hud.alpha_composite(glow_layer)

        draw = ImageDraw.Draw(hud, "RGBA")
        for start, end in connections:
            x1, y1 = display_points[start]
            x2, y2 = display_points[end]
            draw.line((x1, y1, x2, y2), fill=(*TEAL, 235), width=3)

        for x, y in display_points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(*TEAL, 235))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(130, 255, 240, 255))

    # ---------------- STATUS BAR (unchanged) ----------------
    status_height = 50
    status_top = h - status_height
    draw.rectangle((0, status_top, w, h), fill=(5, 15, 14, 255))
    draw.line((0, status_top, w, status_top), fill=(*TEAL, 120), width=1)

    if controller_paused:
        status_text, dot_color = "STATUS: PAUSED", RED
    else:
        status_text, dot_color = "STATUS: ACTIVE", GREEN

    draw.ellipse((28, status_top + 18, 40, status_top + 30), fill=(*dot_color, 255))
    draw_glow_text(hud, (52, status_top + 11), status_text, get_font(15, bold=True), fill=TEAL, glow_radius=4)

    rgb = np.array(hud.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)



    # ========================================================
    # RETURN HUD AS OPENCV IMAGE
    # ========================================================

    rgb = np.array(hud.convert("RGB"))

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


# ============================================================
# MAIN LOOP
# ============================================================

frame_timestamp = 0

while True:

    success, frame = cap.read()

    if not success:
        print("❌ Frame grab failed")
        break

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DEFAULT UI VALUES
    # --------------------------------------------------------

    points = []
    finger_count = 0
    gesture = None

    # --------------------------------------------------------
    # HAND DETECTED
    # --------------------------------------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        h, w, _ = frame.shape

        # Get landmark coordinates.
        # These coordinates are used only for the HUD skeleton.
        points = []

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append((x, y))

        # ----------------------------------------------------
        # FINGER COUNT
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # GESTURE RECOGNITION
        # ----------------------------------------------------

        if (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        ):
            gesture = "NEXT"

        elif (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
        ):
            gesture = "PREVIOUS"

        elif (
            index_up
            and middle_up
            and ring_up
            and pinky_up
        ):
            gesture = "PAUSE"

        # ----------------------------------------------------
        # GESTURE STABILITY
        # ----------------------------------------------------

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

    else:

        gesture_candidate = None
        gesture_frames = 0
        last_gesture = None

    # ========================================================
    # DRAW PURE HUD
    # ========================================================

    # IMPORTANT:
    # We pass the camera frame only to get its dimensions.
    # draw_hud() does NOT display the camera image.
    display = draw_hud(
        frame,
        finger_count,
        gesture,
        points
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Gesture PowerPoint Controller",
        display
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
detector.close()
cv2.destroyAllWindows()
