import numpy as np
from scipy.signal import butter, filtfilt, periodogram

def estimate_heart_rate(signal, fps=30):
    if len(signal) < 21:
        return 0.0
    signal = signal - np.mean(signal)
    b, a = butter(3, [0.75 / (fps / 2), 4.0 / (fps / 2)], btype='band')
    filtered = filtfilt(b, a, signal)
    f, pxx = periodogram(filtered, fs=fps)
    valid = (f >= 0.75) & (f <= 4.0)
    if np.any(valid):
        peak_freq = f[valid][np.argmax(pxx[valid])]
        return peak_freq * 60  # BPM
    else:
        return 0.0