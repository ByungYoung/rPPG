import numpy as np
from scipy.signal import butter, filtfilt, periodogram

def _bandpass(signal, fps):
    nyquist = fps / 2
    low = 0.75 / nyquist
    high = min(4.0 / nyquist, 0.99)
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)

def estimate_heart_rate_chrom(clip, fps=30):
    T, H, W, C = clip.shape
    rgb_means = clip.reshape(T, -1, 3).mean(axis=1)
    rgb_norm = rgb_means / np.linalg.norm(rgb_means, axis=1, keepdims=True)
    Xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
    Ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
    S = Xs / Ys - np.mean(Xs / Ys)
    filtered = _bandpass(S, fps)
    f, pxx = periodogram(filtered, fs=fps)
    valid = (f >= 0.75) & (f <= 4.0)
    return f[valid][np.argmax(pxx[valid])] * 60 if np.any(valid) else 0.0

def estimate_heart_rate_pos(clip, fps=30):
    T, H, W, C = clip.shape
    rgb_means = clip.reshape(T, -1, 3).mean(axis=1).T
    projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
    S = projection_matrix @ rgb_means
    S = S[0] / S[1] - np.mean(S[0] / S[1])
    filtered = _bandpass(S, fps)
    f, pxx = periodogram(filtered, fs=fps)
    valid = (f >= 0.75) & (f <= 4.0)
    return f[valid][np.argmax(pxx[valid])] * 60 if np.any(valid) else 0.0

def compute_hrv(hr_sequence):
    hr_sequence = np.array(hr_sequence)
    if len(hr_sequence) < 2:
        return 0.0
    rr_intervals = 60.0 / hr_sequence
    diff = np.diff(rr_intervals)
    return np.std(diff) * 1000  # HRV in ms
