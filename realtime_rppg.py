import cv2
import numpy as np
import mediapipe as mp
from torchvision import transforms
from utils.preprocess import preprocess_frame
from utils.signal_analysis import estimate_heart_rate

# --- MediaPipe 얼굴 메쉬 설정 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# --- 슬라이딩 윈도우 준비 ---
frame_window = []
FRAME_DEPTH = 30  # padding 오류 방지용 충분한 길이 확보

cap = cv2.VideoCapture(0)
print("[INFO] 실시간 rPPG 시작 (q 키로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0]
        roi = preprocess_frame(frame, mesh.landmark)
        if roi is not None and roi.size > 0:
            green_channel = cv2.resize(roi, (36, 36))[:, :, 1]  # G 채널만 추출
            frame_window.append(np.mean(green_channel))

            if len(frame_window) >= FRAME_DEPTH:
                signal = np.array(frame_window[-FRAME_DEPTH:])
                hr = estimate_heart_rate(signal)
                cv2.putText(frame, f"HR: {hr:.2f} bpm", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("rPPG Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()