
import cv2
import mediapipe as mp
import time
import os
import urllib.request

# STEP 1: Download model
model_path = "hand_landmarker.task"

model_url = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

if not os.path.exists(model_path):
    print("Downloading hand model...")
    urllib.request.urlretrieve(model_url, model_path)


# STEP 2: MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)


# STEP 3: Finger connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# STEP 4: Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

start_time = time.monotonic()
last_timestamp = -1


# STEP 5: Start tracking
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

        hand_count = len(result.hand_landmarks)


        # STEP 6: Loop through both hands
        for i, hand in enumerate(result.hand_landmarks):

            label = result.handedness[i][0].category_name

            if label == "Left":
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

            points = []

            # STEP 7: Draw landmarks
            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 0, 255),
                    -1
                )

            # STEP 8: Draw skeleton
            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    color,
                    2
                )

            # STEP 9: Track index fingertip
            index_x, index_y = points[8]

            cv2.circle(
                frame,
                (index_x, index_y),
                12,
                color,
                -1
            )

            # STEP 10: Display hand label
            wrist_x, wrist_y = points[0]

            cv2.putText(
                frame,
                label + " Hand",
                (wrist_x, wrist_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )


        # STEP 11: Display hand counter
        cv2.putText(
            frame,
            f"Hands Detected: {hand_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Two Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
