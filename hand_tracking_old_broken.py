import cv2
import mediapipe as mp
import time
import pyautogui
import os

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
DARK_TEAL = (20, 110, 105)
WHITE = (235, 245, 245)
BLACK = (5, 10, 10)
RED = (80, 80, 255)
GREEN = (80, 255, 150)


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

def draw_glow_text(base_image, position, text, font,
                   fill=(55, 230, 210),
                   glow_radius=6):

    glow_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    x, y = position

    # Soft glow
    glow_draw.text(
        (x, y),
        text,
        font=font,
        fill=(*fill, 180)
    )

    glow_layer = glow_layer.filter(
        ImageFilter.GaussianBlur(glow_radius)
    )

    base_image.alpha_composite(glow_layer)

    # Main text
    draw = ImageDraw.Draw(base_image)

    draw.text(
        (x, y),
        text,
        font=font,
        fill=(*fill, 255)
    )


# ============================================================
# DRAW HUD
# ============================================================

    def draw_hud(frame, finger_count, gesture, points):

        h, w, _ = frame.shape

# ========================================================
# CREATE PURE DARK HUD BACKGROUND
# ========================================================

hud = Image.new(
    "RGBA",
    (w, h),
    (5, 10, 10, 255)
)

draw = ImageDraw.Draw(hud, "RGBA")

    # ========================================================
    # COLORS
    # ========================================================

    TEAL = (55, 230, 210)
    SOFT_TEAL = (55, 230, 210, 80)
    WHITE = (220, 235, 235)
    GREEN = (80, 255, 150)
    RED = (255, 80, 90)

    # ========================================================
    # HEADER
    # ========================================================

    header_height = 62

    draw.rectangle(
        (0, 0, w, header_height),
        fill=(5, 15, 14, 255)
    )

    draw.line(
        (0, header_height, w, header_height),
        fill=(*TEAL, 100),
        width=1
    )

    title_font = get_font(23, bold=True)

    draw_glow_text(
        hud,
        (28, 17),
        "GESTURE POWERPOINT CONTROLLER",
        title_font,
        fill=TEAL,
        glow_radius=5
    )

    # ========================================================
    # TOP CORNER BRACKETS
    # ========================================================

    bracket = 18

    # Left
    draw.line(
        (8, 8, 8 + bracket, 8),
        fill=(*TEAL, 130),
        width=2
    )

    draw.line(
        (8, 8, 8, 8 + bracket),
        fill=(*TEAL, 130),
        width=2
    )

    # Right
    draw.line(
        (w - 8, 8, w - 8 - bracket, 8),
        fill=(*TEAL, 130),
        width=2
    )

    draw.line(
        (w - 8, 8, w - 8, 8 + bracket),
        fill=(*TEAL, 130),
        width=2
    )

    # ========================================================
    # LEFT INFORMATION
    # ========================================================

    left_x = 28
    left_y = 95

    label_font = get_font(13)
    value_font = get_font(23, bold=True)

    draw.text(
        (left_x, left_y),
        "FINGERS",
        font=label_font,
        fill=(*WHITE, 180)
    )

    draw_glow_text(
        hud,
        (left_x, left_y + 18),
        str(finger_count),
        value_font,
        fill=TEAL
    )

    draw.text(
        (left_x, left_y + 70),
        "GESTURE",
        font=label_font,
        fill=(*WHITE, 180)
    )

    if gesture == "NEXT":
        gesture_text = "NEXT SLIDE"

    elif gesture == "PREVIOUS":
        gesture_text = "PREVIOUS SLIDE"

    elif gesture == "PAUSE":
        gesture_text = "PAUSE / RESUME"

    else:
        gesture_text = "WAITING..."

    draw_glow_text(
        hud,
        (left_x, left_y + 90),
        gesture_text,
        get_font(18, bold=True),
        fill=TEAL,
        glow_radius=4
    )

    # ========================================================
    # RIGHT CONTROLS
    # ========================================================

    right_x = w - 245
    right_y = 95

    draw.text(
        (right_x, right_y),
        "CONTROLS",
        font=label_font,
        fill=(*WHITE, 180)
    )

    controls = [
        "1 finger  →  Next",
        "2 fingers →  Previous",
        "4 fingers →  Pause"
    ]

    for i, control in enumerate(controls):

        draw.text(
            (
                right_x,
                right_y + 30 + i * 28
            ),
            control,
            font=get_font(13),
            fill=(*TEAL, 235)
        )

    # ========================================================
    # SUBTLE HUD GRID
    # ========================================================

    # Very faint horizontal scan lines
    for y in range(170, h - 70, 9):

        draw.line(
            (20, y, w - 20, y),
            fill=(40, 120, 115, 13),
            width=1
        )

    # ========================================================
    # CENTRAL HAND SKELETON
    # ========================================================

    if points:

        # First create glow layer
        glow_layer = Image.new(
            "RGBA",
            (w, h),
            (0, 0, 0, 0)
        )

        glow_draw = ImageDraw.Draw(
            glow_layer,
            "RGBA"
        )

        # Hand connections
        for start, end in connections:

            x1, y1 = points[start]
            x2, y2 = points[end]

            glow_draw.line(
                (x1, y1, x2, y2),
                fill=(*TEAL, 220),
                width=5
            )

        # Glow
        glow_layer = glow_layer.filter(
            ImageFilter.GaussianBlur(7)
        )

        hud.alpha_composite(glow_layer)

        # Main skeleton
        draw = ImageDraw.Draw(hud, "RGBA")

        for start, end in connections:

            x1, y1 = points[start]
            x2, y2 = points[end]

            draw.line(
                (x1, y1, x2, y2),
                fill=(*TEAL, 230),
                width=3
            )

        # Joints
        for x, y in points:

            draw.ellipse(
                (
                    x - 6,
                    y - 6,
                    x + 6,
                    y + 6
                ),
                fill=(*TEAL, 220)
            )

            draw.ellipse(
                (
                    x - 3,
                    y - 3,
                    x + 3,
                    y + 3
                ),
                fill=(120, 255, 240, 255)
            )

    # ========================================================
    # BOTTOM STATUS BAR
    # ========================================================

    status_height = 48
    status_top = h - status_height

    draw.rectangle(
        (0, status_top, w, h),
        fill=(5, 15, 14, 255)
    )

    draw.line(
        (0, status_top, w, status_top),
        fill=(*TEAL, 110),
        width=1
    )

    if controller_paused:

        status_text = "STATUS: PAUSED"
        dot_color = RED

    else:

        status_text = "STATUS: ACTIVE"
        dot_color = GREEN

    # Status dot
    draw.ellipse(
        (
            28,
            status_top + 18,
            40,
            status_top + 30
        ),
        fill=(*dot_color, 255)
    )

    draw_glow_text(
        hud,
        (52, status_top + 11),
        status_text,
        get_font(15, bold=True),
        fill=TEAL,
        glow_radius=4
    )

    # ========================================================
    # BOTTOM CORNER BRACKETS
    # ========================================================

    draw.line(
        (8, h - 8, 26, h - 8),
        fill=(*TEAL, 130),
        width=2
    )

    draw.line(
        (8, h - 8, 8, h - 26),
        fill=(*TEAL, 130),
        width=2
    )

    draw.line(
        (w - 8, h - 8, w - 26, h - 8),
        fill=(*TEAL, 130),
        width=2
    )

    draw.line(
        (w - 8, h - 8, w - 8, h - 26),
        fill=(*TEAL, 130),
        width=2
    )

    # ========================================================
    # RETURN OPENCV IMAGE
    # ========================================================

    return np_from_pil(hud)


    # --------------------------------------------------------
    # MAIN HUD PANEL
    # --------------------------------------------------------

    margin = 35

    panel_left = margin
    panel_top = 30
    panel_right = w - margin
    panel_bottom = h - 30

    # Transparent dark panel
    draw.rounded_rectangle(
        (panel_left, panel_top, panel_right, panel_bottom),
        radius=8,
        fill=(5, 12, 12, 145),
        outline=(*TEAL, 130),
        width=2
    )


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    header_bottom = 95

    draw.line(
        (panel_left, header_bottom,
         panel_right, header_bottom),
        fill=(*TEAL, 120),
        width=1
    )

    title_font = get_font(25, bold=True)

    draw_glow_text(
        image,
        (panel_left + 25, panel_top + 20),
        "GESTURE POWERPOINT CONTROLLER",
        title_font
    )


    # --------------------------------------------------------
    # CORNER BRACKETS
    # --------------------------------------------------------

    bracket = 20

    corners = [
        (panel_left + 10, panel_top + 10, 1, 1),
        (panel_right - 10, panel_top + 10, -1, 1),
        (panel_left + 10, panel_bottom - 10, 1, -1),
        (panel_right - 10, panel_bottom - 10, -1, -1)
    ]

    for x, y, dx, dy in corners:

        draw.line(
            (x, y, x + dx * bracket, y),
            fill=(*TEAL, 90),
            width=2
        )

        draw.line(
            (x, y, x, y + dy * bracket),
            fill=(*TEAL, 90),
            width=2
        )


    # --------------------------------------------------------
    # LEFT INFORMATION
    # --------------------------------------------------------

    left_x = panel_left + 25
    info_y = header_bottom + 25

    small_font = get_font(14)
    medium_font = get_font(22, bold=True)

    draw.text(
        (left_x, info_y),
        "FINGERS",
        font=small_font,
        fill=(*WHITE, 190)
    )

    draw_glow_text(
        image,
        (left_x, info_y + 22),
        str(finger_count),
        get_font(28, bold=True)
    )


    # Gesture label

    gesture_y = info_y + 75

    draw.text(
        (left_x, gesture_y),
        "GESTURE",
        font=small_font,
        fill=(*WHITE, 190)
    )

    if gesture == "NEXT":
        gesture_text = "NEXT SLIDE"

    elif gesture == "PREVIOUS":
        gesture_text = "PREVIOUS SLIDE"

    elif gesture == "PAUSE":
        gesture_text = "PAUSE / RESUME"

    else:
        gesture_text = "WAITING..."

    draw_glow_text(
        image,
        (left_x, gesture_y + 22),
        gesture_text,
        medium_font
    )


    # --------------------------------------------------------
    # RIGHT INSTRUCTIONS
    # --------------------------------------------------------

    right_x = panel_right - 260
    instruction_y = header_bottom + 25

    draw.text(
        (right_x, instruction_y),
        "CONTROLS",
        font=small_font,
        fill=(*WHITE, 190)
    )

    instructions = [
        "1 finger  →  Next",
        "2 fingers →  Previous",
        "4 fingers →  Pause"
    ]

    for i, instruction in enumerate(instructions):

        draw.text(
            (right_x, instruction_y + 30 + i * 28),
            instruction,
            font=small_font,
            fill=(*TEAL, 230)
        )


    # --------------------------------------------------------
    # SUBTLE HUD LINES
    # --------------------------------------------------------

    middle_y = int(h * 0.78)

    draw.line(
        (panel_left + 20, middle_y,
         panel_right - 20, middle_y),
        fill=(*TEAL, 35),
        width=1
    )

    # Small horizontal scan lines

    for i in range(5):

        y = middle_y + 10 + i * 8

        draw.line(
            (panel_left + 30, y,
             panel_right - 30, y),
            fill=(*TEAL, 12),
            width=1
        )


    # --------------------------------------------------------
    # STATUS BAR
    # --------------------------------------------------------

    status_top = panel_bottom - 55

    draw.line(
        (panel_left, status_top,
         panel_right, status_top),
        fill=(*TEAL, 120),
        width=1
    )

    status_font = get_font(16, bold=True)

    # Status dot
    dot_color = RED if controller_paused else GREEN

    draw.ellipse(
        (
            panel_left + 25,
            status_top + 18,
            panel_left + 37,
            status_top + 30
        ),
        fill=(*dot_color, 255)
    )

    if controller_paused:
        status_text = "STATUS: PAUSED"
    else:
        status_text = "STATUS: ACTIVE"

    draw_glow_text(
        image,
        (panel_left + 50, status_top + 12),
        status_text,
        status_font,
        fill=TEAL
    )


    # --------------------------------------------------------
    # CONVERT BACK TO OPENCV
    # --------------------------------------------------------

    result = np_from_pil(image)

    return result


# ============================================================
# PIL → OPENCV
# ============================================================

def np_from_pil(image):
    import numpy as np

    rgb = np.array(image.convert("RGB"))

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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


    finger_count = 0
    gesture = None


    # --------------------------------------------------------
    # HAND DETECTED
    # --------------------------------------------------------
    points = []
    finger_count = 0
    gesture = None
    if result.hand_landmarks:

    hand = result.hand_landmarks[0]

    h, w, _ = frame.shape

    points = []

    for landmark in hand:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        points.append((x, y))


        # Glow-like hand joints
        for x, y in points:

            cv2.circle(
                frame,
                (x, y),
                9,
                (60, 210, 195),
                -1
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                (170, 255, 240),
                -1
            )


        # Hand connections

        for start, end in connections:

            cv2.line(
                frame,
                points[start],
                points[end],
                (60, 210, 195),
                3
            )


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
    # HUD
    # ========================================================

        frame = draw_hud(
        frame,
        finger_count,
        gesture,
        points
    )

    cv2.imshow(
        "Gesture PowerPoint Controller",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

detector.close()

cv2.destroyAllWindows()