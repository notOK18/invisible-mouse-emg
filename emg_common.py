"""
Shared code for the EMG click pipeline (record -> train -> live).

The SAME FeatureStream and ClickLogic are used for training and for live
inference, so what the model sees live is computed exactly like in training.
"""

from collections import deque

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi

# ==================================================
# CONFIG
# ==================================================

COM_PORT = "COM9"          # the OUTGOING BITalino port
ADC_COLUMN = 5             # column of A2 when acquiring channel [1]
ACQ_CHANNELS = [1]         # A2
FS = 1000                  # Hz

WIN = 200                  # window length, samples (200 ms)
HOP = 25                   # hop, samples (25 ms)
WARMUP_WINDOWS = 120       # 3 s: learn the background level, no output
BASELINE_WINDOWS = 200     # 5 s of history for the background level
MIN_BASELINE_RMS = 0.5     # ADC units, avoids dividing by ~0
MAINS_HZ = 50.0            # Lebanon = 50 Hz

LABEL_NAMES = {0: "rest", 1: "index", 2: "fist"}
CLICK_NAMES = {1: "LEFT", 2: "RIGHT"}

FEATURE_NAMES = ["rms", "mav", "wl", "zc", "ssc", "mnf", "mdf"]

# Click logic (units: windows of HOP = 25 ms)
CAPTURE_WINDOWS = 8        # after onset, track the peak for 8 windows (200 ms)
RELEASE_FRAC = 0.8         # strength must fall below this x onset threshold ...
REST_WINDOWS = 4           # ... for this many windows to re-arm
COOLDOWN_WINDOWS = 12      # minimum gap between clicks (0.3 s)


# ==================================================
# BITALINO
# ==================================================

def connect(port=COM_PORT):
    from bitalino import BITalino

    try:
        device = BITalino(port, timeout=5)
    except TypeError:               # older bitalino.py without timeout arg
        device = BITalino(port)

    print("Connected:", device.version())
    return device


# ==================================================
# STREAMING FILTER + FEATURES
# ==================================================

def make_filter_sos():
    bandpass = butter(4, [20, 450], btype="band", fs=FS, output="sos")
    b, a = iirnotch(MAINS_HZ, 30.0, fs=FS)
    return np.vstack([bandpass, tf2sos(b, a)])


class FeatureStream:
    """
    Feed raw ADC samples in chunks of any size with push().
    Returns a list of (end_sample_index, features) every HOP samples.
    features is None during the warm-up period.
    """

    def __init__(self):
        self.sos = make_filter_sos()
        self.zi = None
        self.buf = np.zeros(0)
        self.buf_start = 0          # absolute index of buf[0]
        self.next_end = WIN         # absolute end index of next window
        self.rms_hist = deque(maxlen=BASELINE_WINDOWS)
        self.n_windows = 0

    def push(self, raw):
        raw = np.asarray(raw, dtype=float)

        if raw.size == 0:
            return []

        if self.zi is None:
            self.zi = sosfilt_zi(self.sos) * raw[0]

        filtered, self.zi = sosfilt(self.sos, raw, zi=self.zi)
        self.buf = np.concatenate([self.buf, filtered])

        out = []
        total = self.buf_start + len(self.buf)

        while self.next_end <= total:
            lo = self.next_end - WIN - self.buf_start
            out.append((self.next_end, self._features(self.buf[lo:lo + WIN])))
            self.next_end += HOP

        drop = self.next_end - WIN - self.buf_start
        if drop > 0:
            self.buf = self.buf[drop:]
            self.buf_start += drop

        return out

    def _features(self, w):
        rms = float(np.sqrt(np.mean(w * w)))
        self.rms_hist.append(rms)
        self.n_windows += 1

        if self.n_windows <= WARMUP_WINDOWS:
            return None

        # Background level = 20th percentile of the last 5 s.
        # Amplitude features are divided by it, so a different electrode
        # contact or a tense arm does not look like a press.
        base = max(float(np.percentile(self.rms_hist, 20)), MIN_BASELINE_RMS)

        d = np.diff(w)
        thr = 0.1 * base

        mav = np.mean(np.abs(w))
        wl = np.mean(np.abs(d))
        zc = np.sum((w[:-1] * w[1:] < 0) & (np.abs(d) > thr)) / WIN
        ssc = np.sum(
            (d[:-1] * d[1:] < 0)
            & ((np.abs(d[:-1]) > thr) | (np.abs(d[1:]) > thr))
        ) / WIN

        power = np.abs(np.fft.rfft(w * np.hanning(WIN))) ** 2
        freqs = np.fft.rfftfreq(WIN, 1.0 / FS)
        band = (freqs >= 20) & (freqs <= 450)
        p, f = power[band], freqs[band]
        total = p.sum() + 1e-12
        mnf = (f * p).sum() / total
        mdf = f[min(np.searchsorted(np.cumsum(p), total / 2), len(f) - 1)]

        return np.array([rms / base, mav / base, wl / base, zc, ssc, mnf, mdf])


# ==================================================
# THRESHOLD CLICK LOGIC
# ==================================================

class ThresholdClicker:
    """
    Fed one strength value per window (rms / background level).

      strength >= x   -> a press started; track the PEAK for CAPTURE_WINDOWS
      peak >= y       -> FIST  (fires immediately, no need to wait)
      otherwise       -> INDEX (fires when the capture period ends)

    After a click, strength must drop below RELEASE_FRAC * x for REST_WINDOWS
    windows (plus a cooldown) before it can fire again, so holding a press
    gives one click.

    push() returns 1 (left / index), 2 (right / fist) or None.
    """

    def __init__(
        self, x, y,
        capture=CAPTURE_WINDOWS,
        release_frac=RELEASE_FRAC,
        rest_windows=REST_WINDOWS,
        cooldown_windows=COOLDOWN_WINDOWS,
    ):
        self.x, self.y = x, y
        self.capture = capture
        self.release = release_frac * x
        self.rest_windows = rest_windows
        self.cooldown_windows = cooldown_windows
        self.state = "wait_rest"
        self.rest_run = 0
        self.cool = 0
        self.peak = 0.0
        self.left = 0

    def _fire(self, cls):
        self.state = "wait_rest"
        self.rest_run = 0
        self.cool = self.cooldown_windows
        return cls

    def push(self, strength):
        if self.cool > 0:
            self.cool -= 1

        if self.state == "wait_rest":
            self.rest_run = self.rest_run + 1 if strength < self.release else 0
            if self.rest_run >= self.rest_windows and self.cool == 0:
                self.state = "armed"
            return None

        if self.state == "armed":
            if strength < self.x:
                return None
            self.state = "capture"
            self.peak = strength
            self.left = self.capture - 1
        else:                                   # capture
            self.peak = max(self.peak, strength)
            self.left -= 1

        if self.peak >= self.y:
            return self._fire(2)
        if self.left <= 0:
            return self._fire(1)
        return None
