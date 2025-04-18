import cv2
import numpy as np
import mediapipe as mp
import csv
import time
import os
import matplotlib.pyplot as plt
from collections import deque
from utils.preprocess import preprocess_frame, draw_face_mask
from utils.signal_analysis import estimate_heart_rate_chrom, estimate_heart_rate_pos, compute_hrv

# --- 설정 ---
FRAME_DEPTH = 90
SAVE_LOG = True
DISPLAY_GRAPH = True
LOG_FILE = "hr_log.csv"
DURATION = 20  # seconds

# --- 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)
print("[INFO] 실시간 rPPG 시작 (자동 종료까지 20초)")

frame_window = deque(maxlen=FRAME_DEPTH)
history = deque(maxlen=150)
plot_history = []
hrv_values = []
start_time = time.time()

# --- 로그 초기화 및 저장 ---
def init_log():
    if SAVE_LOG and not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'hr_avg', 'hrv'])

def log_data(hr_avg, hrv):
    if SAVE_LOG:
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), hr_avg, hrv])

def update_plot(ax):
    ax.clear()
    ax.set_title("Average HR")
    ax.set_ylim(40, 120)
    if plot_history:
        ax.plot(plot_history, label="HR")
        ax.legend()

def detect_expression(landmarks, image_shape):
    h, w = image_shape[:2]
    def get_point(idx):
        lm = landmarks[idx]
        return np.array([int(lm.x * w), int(lm.y * h)])

    left_mouth = get_point(61)
    right_mouth = get_point(291)
    top_lip = get_point(13)
    bottom_lip = get_point(14)

    mouth_width = np.linalg.norm(left_mouth - right_mouth)
    mouth_height = np.linalg.norm(top_lip - bottom_lip)
    ratio = mouth_height / mouth_width if mouth_width != 0 else 0

    if ratio > 0.35:
        return "Surprised"
    elif mouth_width > 1.8 * mouth_height:
        return "Smiling"
    else:
        return "Neutral"

def compute_lr_ratio(frame, landmarks):
    h, w = frame.shape[:2]
    left_ids = [234, 93, 132, 58, 172, 136]
    right_ids = [454, 323, 361, 288, 397, 365]

    def region_mean(indices):
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
        if pts.shape[0] > 0:
            cv2.fillConvexPoly(mask, pts, 255)
            region = cv2.bitwise_and(frame, frame, mask=mask)
            return np.mean(region[:, :, 1][mask == 255])
        return 0.0

    left_mean = region_mean(left_ids)
    right_mean = region_mean(right_ids)
    if right_mean == 0:
        return 1.0
    return left_mean / right_mean

init_log()
if DISPLAY_GRAPH:
    plt.ion()
    fig, ax = plt.subplots()

# --- 메인 루프 ---
while True:
    elapsed = time.time() - start_time
    if elapsed >= DURATION:
        break

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0]
        draw_face_mask(frame, mesh.landmark)
        roi = preprocess_frame(frame, mesh.landmark)
        if roi is not None and roi.size > 0:
            roi_resized = cv2.resize(roi, (36, 36))
            frame_window.append(roi_resized)

            if len(frame_window) == FRAME_DEPTH:
                clip = np.stack(frame_window, axis=0)
                hr1 = estimate_heart_rate_chrom(clip)
                hr2 = estimate_heart_rate_pos(clip)

                if 40 <= hr1 <= 180 and 40 <= hr2 <= 180:
                    avg_hr = (hr1 + hr2) / 2
                else:
                    avg_hr = history[-1] if history else 75

                history.append(avg_hr)
                smoothed_hr = np.mean(history)
                hrv = compute_hrv(list(history)[-10:])

                plot_history.append(smoothed_hr)
                hrv_values.append(hrv)
                log_data(smoothed_hr, hrv)

                expression = detect_expression(mesh.landmark, frame.shape)
                lr_ratio = compute_lr_ratio(frame, mesh.landmark)

                cv2.putText(frame, f"HR: {smoothed_hr:.2f} bpm", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"HRV: {hrv:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Expression: {expression}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                cv2.putText(frame, f"LF/RF Ratio: {lr_ratio:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

                remaining = max(0, int(DURATION - elapsed))
                cv2.putText(frame, f"Remaining: {remaining}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

                if DISPLAY_GRAPH:
                    update_plot(ax)
                    plt.pause(0.001)

    cv2.imshow("rPPG Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if DISPLAY_GRAPH:
    plt.ioff()
    plt.close()

print("\n--- 측정 결과 요약 ---")
print("평균 HR:", np.mean(plot_history))
print("HRV 평균:", np.mean(hrv_values))