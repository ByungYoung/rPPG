import cv2
import numpy as np

face_outline_indices = list(range(0, 468))

def preprocess_frame(frame, landmarks):
    h, w = frame.shape[:2]
    pts = []
    for idx in face_outline_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(pts, dtype=np.int32), 255)
    roi = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, rw, rh = cv2.boundingRect(np.array(pts))
    return roi[y:y+rh, x:x+rw]

def draw_face_mask(frame, landmarks):
    h, w = frame.shape[:2]
    pts = []
    for idx in face_outline_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))
    if pts:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=1)
