
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import urllib.request
from datetime import datetime

# ======================================
# 1. MEDIAPIPE SETUP
# ======================================

MODEL = "hand_landmarker.task"
URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

if not os.path.exists(MODEL):
    urllib.request.urlretrieve(URL, MODEL)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)


# ======================================
# 2. GEOMETRY
# ======================================

def make_cube():
    v = np.array([
        [-1, -1, -1], [1, -1, -1],
        [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1],
        [1, 1, 1], [-1, 1, 1]
    ], dtype=float)

    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7]
    ]

    return v, faces


def make_extrusion(outline):
    # Convert a 2D outline into a 3D solid.
    p = np.array(outline, dtype=float)

    minimum = p.min(axis=0)
    maximum = p.max(axis=0)

    center = (minimum + maximum) / 2
    dimension = max(np.max(maximum - minimum), 1)

    p = (p - center) / dimension * 2

    count = len(p)
    depth = 0.3

    front = np.column_stack([
        p, np.full(count, -depth)
    ])

    back = np.column_stack([
        p, np.full(count, depth)
    ])

    vertices = np.vstack([front, back])

    faces = [
        list(range(count)),
        list(range(count, 2 * count))
    ]

    for i in range(count):
        j = (i + 1) % count

        faces.append([
            i, j,
            j + count,
            i + count
        ])

    return vertices, faces


# ======================================
# 3. MATHEMATICS
# ======================================

def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def hand_points(hand, w, h):
    return np.array([
        [p.x * w, p.y * h]
        for p in hand
    ], dtype=float)


def pinch(points):
    thumb = points[4]
    index = points[8]

    palm = distance(points[0], points[9])

    return distance(thumb, index) < max(
        18, palm * 0.45
    )


def pointing(points):
    # Best when fingers point generally upward.
    index_up = points[8][1] < points[6][1]
    middle_down = points[12][1] > points[10][1]

    return index_up and middle_down


def rotation_matrix(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    mx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    my = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cz * 0 + cy]
    ])

    # Correct Y rotation matrix
    my[2, 2] = cy

    mz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])

    return mz @ my @ mx


# ======================================
# 4. HOLOGRAM RENDERER
# ======================================

def draw_model(frame, model, center, scale, angles):
    vertices, faces = model

    matrix = rotation_matrix(*angles)

    rotated = vertices @ matrix.T
    rotated = rotated * scale

    focal = 650.0
    camera_z = 800.0

    projected = []

    for x, y, z in rotated:
        factor = focal / max(100, camera_z + z)

        projected.append([
            center[0] + x * factor,
            center[1] + y * factor
        ])

    projected = np.array(projected, dtype=np.int32)

    # Paint farther faces first.
    ordered_faces = sorted(
        faces,
        key=lambda face: rotated[face, 2].mean(),
        reverse=True
    )

    overlay = frame.copy()

    for face in ordered_faces:
        polygon = projected[face]

        cv2.fillPoly(
            overlay,
            [polygon],
            (180, 90, 15)
        )

    cv2.addWeighted(
        overlay, 0.23,
        frame, 0.77,
        0, frame
    )

    # Draw all unique mesh edges.
    edges = set()

    for face in faces:
        for i in range(len(face)):
            a = face[i]
            b = face[(i + 1) % len(face)]

            edges.add(tuple(sorted((a, b))))

    glow = np.zeros_like(frame)

    for a, b in edges:
        pa = tuple(projected[a])
        pb = tuple(projected[b])

        cv2.line(
            glow, pa, pb,
            (255, 180, 0), 7,
            cv2.LINE_AA
        )

    glow = cv2.GaussianBlur(glow, (0, 0), 8)

    cv2.addWeighted(
        frame, 1,
        glow, 0.65,
        0, frame
    )

    for a, b in edges:
        cv2.line(
            frame,
            tuple(projected[a]),
            tuple(projected[b]),
            (255, 255, 70),
            2,
            cv2.LINE_AA
        )


# ======================================
# 5. APPLICATION STATE
# ======================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam could not open")

model = None
mode = "EMPTY"

center = np.array([320.0, 240.0])
scale = 110.0

angles = np.array([0.35, 0.5, 0.0])

left_path = []
right_path = []
recording = False
waiting_for_release = False

last_single = None
two_grab = None

start_time = time.monotonic()
last_timestamp = -1

os.makedirs("screenshots", exist_ok=True)


# ======================================
# 6. MAIN LOOP
# ======================================

with mp.tasks.vision.HandLandmarker.create_from_options(
    options
) as detector:

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = max(
            last_timestamp + 1,
            int((time.monotonic() - start_time) * 1000)
        )

        last_timestamp = timestamp

        result = detector.detect_for_video(
            image, timestamp
        )

        hands = [
            hand_points(hand, w, h)
            for hand in result.hand_landmarks
        ]

        # Sort by screen position, left to right.
        hands.sort(key=lambda p: p[0][0])

        # Visual fingertip markers
        for hand in hands:
            for index in [4, 8]:
                pt = tuple(hand[index].astype(int))

                cv2.circle(
                    frame, pt, 7,
                    (255, 255, 0), 2
                )

        status = mode

        # ==================================
        # 7. CUSTOM SHAPE TRACING
        # ==================================

      
        # ==================================
        # 7. CUSTOM SHAPE TRACING
        # ==================================

        if mode == "TRACE":
            last_single = None
            two_grab = None

            # Check pinch status for each hand
            if len(hands) == 2:
                left_pinch = pinch(hands[0])
                right_pinch = pinch(hands[1])

            else:
                left_pinch = False
                right_pinch = False

            both_pinching = (
                len(hands) == 2
                and left_pinch
                and right_pinch
            )

            both_released = (
                len(hands) == 2
                and not left_pinch
                and not right_pinch
            )

            # START / CONTINUE DRAWING
            if both_pinching:
                recording = True
                waiting_for_release = False
                status = "PINCH DRAWING ACTIVE"

                a = hands[0][8]
                b = hands[1][8]

                if (
                    not left_path
                    or distance(a, left_path[-1]) > 6
                    or distance(b, right_path[-1]) > 6
                ):
                    left_path.append(a.copy())
                    right_path.append(b.copy())

                # Prevent excessive points
                if len(left_path) > 120:
                    left_path = left_path[::2]
                    right_path = right_path[::2]

            # WAIT WHEN ONLY ONE HAND RELEASES
            elif recording and not both_released:
                waiting_for_release = True
                status = "PAUSED - RELEASE BOTH"

            # FINISH WHEN BOTH HANDS RELEASE
            elif recording and both_released:
                recording = False
                waiting_for_release = False

                if len(left_path) >= 8:
                    outline = (
                        left_path
                        + list(reversed(right_path))
                    )

                    extent = np.ptp(
                        np.array(outline), axis=0
                    )

                    if max(extent) > 45:
                        model = make_extrusion(outline)

                        center = np.mean(
                            np.array(outline), axis=0
                        )

                        scale = max(
                            40,
                            min(155, max(extent) * 0.55)
                        )

                        angles[:] = [0.25, 0.4, 0]
                        mode = "OBJECT"
                        status = "CUSTOM MODEL CREATED"

                    else:
                        status = "TRACE TOO SMALL"

                else:
                    status = "TRACE TOO SHORT"

                left_path.clear()
                right_path.clear()

            else:
                status = "PINCH BOTH HANDS TO DRAW"

            # VISUALIZE TRACED LINES
            for path in [left_path, right_path]:
                if len(path) > 1:
                    line = np.array(
                        path, dtype=np.int32
                    )

                    cv2.polylines(
                        frame,
                        [line],
                        False,
                        (255, 255, 0),
                        3,
                        cv2.LINE_AA
                    )

            # CONNECT BOTH FINGERTIPS
            if (
                left_path
                and right_path
                and len(hands) == 2
            ):
                cv2.line(
                    frame,
                    tuple(left_path[-1].astype(int)),
                    tuple(right_path[-1].astype(int)),
                    (255, 140, 0),
                    2
                )

        # ==================================
        # 8. OBJECT MANIPULATION
        # ==================================

        if mode == "OBJECT" and model is not None:

            pinching = [
                hand for hand in hands
                if pinch(hand)
            ]

            # TWO HANDS: move, scale and twist
            if len(pinching) == 2:
                status = "TWO-HAND GRAB"

                p1 = pinching[0][8]
                p2 = pinching[1][8]

                midpoint = (p1 + p2) / 2
                spacing = max(distance(p1, p2), 1)

                angle = math.atan2(
                    p2[1] - p1[1],
                    p2[0] - p1[0]
                )

                if two_grab is None:
                    two_grab = {
                        "mid": midpoint.copy(),
                        "spacing": spacing,
                        "angle": angle,
                        "center": center.copy(),
                        "scale": scale,
                        "rz": angles[2]
                    }

                # Position follows midpoint movement.
                center = (
                    two_grab["center"]
                    + midpoint
                    - two_grab["mid"]
                )

                # Scale follows distance ratio.
                ratio = (
                    spacing / two_grab["spacing"]
                )

                scale = float(np.clip(
                    two_grab["scale"] * ratio,
                    25, 250
                ))

                # Twist around the Z axis.
                angles[2] = (
                    two_grab["rz"]
                    + angle
                    - two_grab["angle"]
                )

                last_single = None

            # ONE HAND: rotate in 3D
            elif len(pinching) == 1 and len(hands) == 1:
                status = "SINGLE-HAND ROTATION"
                two_grab = None

                current = pinching[0][8]

                if last_single is not None:
                    dx = current[0] - last_single[0]
                    dy = current[1] - last_single[1]

                    angles[1] += dx * 0.012
                    angles[0] += dy * 0.012

                last_single = current.copy()

            else:
                status = "HOLOGRAM LOCKED"
                two_grab = None
                last_single = None

            # Draw it every frame, even with no hands.
            draw_model(
                frame,
                model,
                center,
                scale,
                angles
            )

        # ==================================
        # 9. USER INTERFACE
        # ==================================

        cv2.rectangle(
            frame,
            (0, 0), (w, 104),
            (20, 30, 40), -1
        )

        cv2.putText(
            frame,
            "STARK LAB // V2",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "STATUS: " + status,
            (15, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "C: CUBE  T: TRACE  R: RESET  S: SAVE  Q: QUIT",
            (15, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (190, 220, 220),
            1
        )

        cv2.imshow("STARK LAB V2", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            model = make_cube()
            center = np.array([w / 2, h / 2])
            scale = 110.0
            angles[:] = [0.35, 0.5, 0]
            mode = "OBJECT"
            two_grab = None
            last_single = None
            recording = False
            left_path.clear()
            right_path.clear()

        elif key == ord("t"):
            mode = "TRACE"
            model = None
            recording = False
            left_path.clear()
            right_path.clear()
            two_grab = None
            last_single = None

        elif key == ord("r"):
            model = None
            mode = "EMPTY"
            recording = False
            left_path.clear()
            right_path.clear()
            two_grab = None
            last_single = None

        elif key == ord("s"):
            filename = datetime.now().strftime(
                "screenshots/stark_%Y%m%d_%H%M%S.png"
            )

            cv2.imwrite(filename, frame)
            print("Saved:", filename)

cap.release()
cv2.destroyAllWindows()
