import cv2
import numpy as np
import mediapipe as mp
import csv
import time
import os
import matplotlib.pyplot as plt
from collections import deque
from threading import Thread
from torchvision import transforms
from utils.preprocess import preprocess_frame, draw_face_mask
from utils.signal_analysis import estimate_heart_rate

# --- 설정 ---
FRAME_DEPTH = 90  # 더 안정적인 분석을 위해 길이 증가
SAVE_LOG = True
DISPLAY_GRAPH = True
LOG_FILE = "hr_log.csv"

# --- 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)
print("[INFO] 실시간 rPPG 시작 (q 키로 종료)")

frame_window = []
history = deque(maxlen=150)
plot_history = []  # 평균 HR 그래프용

# --- 로그 초기화 및 저장 ---
def init_log():
    if SAVE_LOG and not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'heart_rate'])

def log_heart_rate(hr):
    if SAVE_LOG:
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), hr])

# --- HR smoothing ---
def smooth_heart_rate(hr, window=5):
    history.append(hr)
    if len(history) < window:
        return hr
    return np.mean(list(history)[-window:])

# --- 실시간 플롯 ---
def update_plot(ax):
    ax.clear()
    ax.set_title("Average HR (smoothed)")
    ax.set_ylim(40, 120)
    if plot_history:
        ax.plot(plot_history)

# --- 로그 파일 초기화 ---
init_log()

# --- 실시간 그래프 준비 (메인 쓰레드에서 실행) ---
if DISPLAY_GRAPH:
    plt.ion()
    fig, ax = plt.subplots()

# --- 메인 루프 ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0]
        draw_face_mask(frame, mesh.landmark)  # 얼굴 마스킹 시각화
        roi = preprocess_frame(frame, mesh.landmark)
        if roi is not None and roi.size > 0:
            green_channel = cv2.resize(roi, (36, 36))[:, :, 1]
            frame_window.append(np.mean(green_channel))

            if len(frame_window) >= FRAME_DEPTH:
                signal = np.array(frame_window[-FRAME_DEPTH:])
                hr = estimate_heart_rate(signal)

                # 이상치 필터링
                if hr < 40 or hr > 180:
                    hr = history[-1] if history else 75

                smoothed_hr = smooth_heart_rate(hr)
                plot_history.append(smoothed_hr)
                log_heart_rate(smoothed_hr)
                cv2.putText(frame, f"HR: {smoothed_hr:.2f} bpm", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                if DISPLAY_GRAPH:
                    update_plot(ax)
                    plt.pause(0.01)

    cv2.imshow("rPPG Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if DISPLAY_GRAPH:
    plt.ioff()
    plt.close()