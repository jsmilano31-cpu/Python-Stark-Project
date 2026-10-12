
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
import math
from datetime import datetime

# =====================================
# 1. LOAD THE HAND TRACKING MODEL
# =====================================

MODEL = "hand_landmarker.task"

URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

if not os.path.exists(MODEL):
    print("Downloading hand model...")
    urllib.request.urlretrieve(URL, MODEL)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# =====================================
# 2. DEFINE OUR 3D CUBE
# =====================================

vertices = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1]
], dtype=float)

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

faces = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (1, 2, 6, 5),
    (0, 3, 7, 4)
]


# =====================================
# 3. HELPER FUNCTIONS
# =====================================

def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def get_hand_points(hand, width, height):
    return [
        (
            int(p.x * width),
            int(p.y * height)
        )
        for p in hand
    ]


def is_pinching(points):
    thumb = points[4]
    index = points[8]

    palm_size = distance(
        points[0], points[9]
    )

    threshold = max(18, palm_size * 0.45)

    return distance(thumb, index) < threshold


def rotation_matrix(rx, ry):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)

    rotate_x = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    rotate_y = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    return rotate_y @ rotate_x


def draw_cube(frame, center, size, rx, ry):
    rotation = rotation_matrix(rx, ry)

    # Rotate and scale the 3D points
    transformed = (vertices @ rotation.T) * size

    # Perspective projection
    focal = 600
    camera_distance = 600

    projected = []

    for x, y, z in transformed:
        depth = max(50, camera_distance + z)

        screen_x = int(
            center[0] + x * focal / depth
        )
        screen_y = int(
            center[1] + y * focal / depth
        )

        projected.append((screen_x, screen_y))

    # Translucent hologram faces
    overlay = frame.copy()

    face_order = sorted(
        faces,
        key=lambda face: np.mean(
            transformed[list(face), 2]
        ),
        reverse=True
    )

    for face in face_order:
        polygon = np.array(
            [projected[i] for i in face],
            dtype=np.int32
        )

        cv2.fillConvexPoly(
            overlay,
            polygon,
            (140, 65, 10)
        )

    cv2.addWeighted(
        overlay, 0.20,
        frame, 0.80,
        0, frame
    )

    # Glowing wireframe
    glow = np.zeros_like(frame)

    for a, b in edges:
        cv2.line(
            glow,
            projected[a],
            projected[b],
            (255, 180, 0),
            8,
            cv2.LINE_AA
        )

    glow = cv2.GaussianBlur(
        glow, (0, 0), 9
    )

    cv2.addWeighted(
        frame, 1.0,
        glow, 0.8,
        0, frame
    )

    for a, b in edges:
        cv2.line(
            frame,
            projected[a],
            projected[b],
            (255, 255, 60),
            2,
            cv2.LINE_AA
        )

    for point in projected:
        cv2.circle(
            frame, point, 3,
            (255, 255, 180), -1
        )


# =====================================
# 4. WEBCAM AND OBJECT STATE
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot access webcam")

created = False
cube_center = (320, 240)
cube_size = 70

rotation_x = 0.35
rotation_y = 0.55

hold_start = None
previous_drag = None

start_time = time.monotonic()
last_timestamp = -1

HOLD_SECONDS = 0.8

os.makedirs("screenshots", exist_ok=True)


# =====================================
# 5. MAIN TRACKING LOOP
# =====================================

with HandLandmarker.create_from_options(options) as detector:

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = max(
            last_timestamp + 1,
            int((time.monotonic() - start_time) * 1000)
        )
        last_timestamp = timestamp

        result = detector.detect_for_video(
            mp_image, timestamp
        )

        hands = []

        for hand in result.hand_landmarks:
            points = get_hand_points(
                hand, width, height
            )
            hands.append(points)

        status = "SHOW BOTH HANDS"

        # Display tracked fingertips
        for points in hands:
            cv2.circle(
                frame, points[8], 9,
                (255, 255, 0), 2
            )

            cv2.circle(
                frame, points[4], 7,
                (255, 0, 255), 2
            )

        # =================================
        # 6. CREATE OBJECT WITH TWO HANDS
        # =================================

        if not created:

            previous_drag = None

            if len(hands) == 2:
                left_point = hands[0][8]
                right_point = hands[1][8]

                center = (
                    (left_point[0] + right_point[0]) // 2,
                    (left_point[1] + right_point[1]) // 2
                )

                spacing = distance(
                    left_point, right_point
                )

                cube_center = center

                cube_size = int(
                    np.clip(spacing * 0.35, 35, 125)
                )

                both_pinching = (
                    is_pinching(hands[0])
                    and is_pinching(hands[1])
                )

                if both_pinching:
                    status = "HOLD TO CREATE"

                    if hold_start is None:
                        hold_start = time.monotonic()

                    elapsed = (
                        time.monotonic() - hold_start
                    )

                    progress = min(
                        1.0, elapsed / HOLD_SECONDS
                    )

                    cv2.rectangle(
                        frame, (20, 110),
                        (int(20 + progress * 200), 125),
                        (255, 255, 0), -1
                    )

                    if elapsed >= HOLD_SECONDS:
                        created = True
                        status = "OBJECT CREATED"
                        hold_start = None

                else:
                    hold_start = None
                    status = "SIZE YOUR HOLOGRAM"

                draw_cube(
                    frame,
                    cube_center,
                    cube_size,
                    rotation_x,
                    rotation_y
                )

            else:
                hold_start = None

        # =================================
        # 7. ROTATE THE CREATED OBJECT
        # =================================

        else:
            status = "HOLOGRAM LOCKED"

            pinching_hands = [
                points for points in hands
                if is_pinching(points)
            ]

            if len(hands) == 1 and len(pinching_hands) == 1:
                status = "ROTATING OBJECT"

                current_drag = pinching_hands[0][8]

                if previous_drag is not None:
                    dx = current_drag[0] - previous_drag[0]
                    dy = current_drag[1] - previous_drag[1]

                    rotation_y += dx * 0.012
                    rotation_x += dy * 0.012

                previous_drag = current_drag

            else:
                previous_drag = None

            # Object persists without hands
            draw_cube(
                frame,
                cube_center,
                cube_size,
                rotation_x,
                rotation_y
            )

        # =================================
        # 8. HOLOGRAPHIC INTERFACE
        # =================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, 92),
            (15, 25, 35),
            -1
        )

        cv2.addWeighted(
            overlay, 0.75,
            frame, 0.25,
            0, frame
        )

        cv2.putText(
            frame,
            "STARK HOLOGRAPHIC INTERFACE",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            status,
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "R: RESET | S: SAVE | Q: QUIT",
            (15, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190, 190, 190),
            1
        )

        cv2.imshow("STARK 3D LAB", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            created = False
            hold_start = None
            previous_drag = None
            rotation_x = 0.35
            rotation_y = 0.55

        elif key == ord("s"):
            filename = datetime.now().strftime(
                "screenshots/stark_%Y%m%d_%H%M%S.png"
            )
            cv2.imwrite(filename, frame)
            print("Screenshot saved:", filename)


cap.release()
cv2.destroyAllWindows()
