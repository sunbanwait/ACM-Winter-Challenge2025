import cv2
import pyautogui
import time
import platform
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ================== CONFIG ==================
SCREEN_W, SCREEN_H = pyautogui.size()

GAIN_X = 6.5
GAIN_Y = 5.5

SCROLL_AMOUNT = 120
SCROLL_TILT = 0.035

MOUTH_OPEN_THRESHOLD = 0.045
CLICK_COOLDOWN = 1.0

ESCAPE_MARGIN = 20
ESCAPE_TIME = 1.0

SHOW_PREVIEW = True
PREVIEW_W, PREVIEW_H = 420, 260

# ================== STATE ==================
anchor_x = None
anchor_y = None
last_click_time = 0
escape_start = None

# ================== CAMERA ==================
def open_camera():
    system = platform.system()
    backend = (
        cv2.CAP_AVFOUNDATION if system == "Darwin"
        else cv2.CAP_DSHOW if system == "Windows"
        else cv2.CAP_V4L2
    )

    print(f"[INFO] Scanning cameras on {system}...")

    for i in range(8):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            continue

        valid = 0
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                valid += 1
            time.sleep(0.05)

        if valid >= 3:
            print(f"[INFO] Using camera index {i}")
            return cap

        cap.release()

    raise RuntimeError("No usable camera found")

# ================== MEDIAPIPE (TASKS API) ==================
base_options = BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ================== MAIN ==================
cap = open_camera()

print("[INFO] Head control ACTIVE")
print("[INFO] Mouth open = CLICK")
print("[INFO] Head up/down = SCROLL")
print("[INFO] Mouse top-left = EXIT")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(mp_image)

    overlay = frame.copy()

    if SHOW_PREVIEW:
        cv2.putText(overlay, "HEAD CONTROL", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    if not result.face_landmarks:
        if SHOW_PREVIEW:
            cv2.imshow("Head Control", overlay)
            cv2.waitKey(1)
        continue

    lm = result.face_landmarks[0]

    # Nose tip (stable reference)
    nx, ny = lm[1].x, lm[1].y

    if anchor_x is None:
        anchor_x, anchor_y = nx, ny
        if SHOW_PREVIEW:
            cv2.imshow("Head Control", overlay)
            cv2.waitKey(1)
        continue

    # -------- MOVE CURSOR --------
    dx = (nx - anchor_x) * GAIN_X * SCREEN_W
    dy = (ny - anchor_y) * GAIN_Y * SCREEN_H

    screen_x = int(np.clip(SCREEN_W // 2 + dx, 0, SCREEN_W))
    screen_y = int(np.clip(SCREEN_H // 2 + dy, 0, SCREEN_H))

    pyautogui.moveTo(screen_x, screen_y)

    # -------- SCROLL --------
    if ny < anchor_y - SCROLL_TILT:
        pyautogui.scroll(SCROLL_AMOUNT)
    elif ny > anchor_y + SCROLL_TILT:
        pyautogui.scroll(-SCROLL_AMOUNT)

    # -------- CLICK (MOUTH OPEN) --------
    upper_lip = lm[13].y
    lower_lip = lm[14].y
    mouth_open = lower_lip - upper_lip

    if mouth_open > MOUTH_OPEN_THRESHOLD:
        if time.time() - last_click_time > CLICK_COOLDOWN:
            pyautogui.click()
            last_click_time = time.time()

    # -------- EXIT SAFETY --------
    if screen_x < ESCAPE_MARGIN and screen_y < ESCAPE_MARGIN:
        if escape_start is None:
            escape_start = time.time()
        elif time.time() - escape_start > ESCAPE_TIME:
            print("[INFO] Exiting safely")
            break
    else:
        escape_start = None

    # -------- PREVIEW --------
    if SHOW_PREVIEW:
        cx = int(nx * frame.shape[1])
        cy = int(ny * frame.shape[0])
        cv2.circle(overlay, (cx, cy), 5, (0,0,255), -1)

        preview = cv2.resize(overlay, (PREVIEW_W, PREVIEW_H))
        cv2.imshow("Head Control", preview)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
