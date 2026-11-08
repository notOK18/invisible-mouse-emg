"""
Shared EMG processing for the invisible mouse click detector.

Everything here is used by BOTH training (train_fingers.py) and live inference,
so the model sees exactly the same signal processing in both places.
All filtering is causal (no look-ahead) so it can run in real time.
"""
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

FS = 1000               # Hz (BITalino)
EMG_COLUMN = "A2"
WIN = 200               # samples per analysis window (200 ms)
HOP = 25                # samples between consecutive windows (25 ms)
MAINS_HZ = 50           # power-line frequency
BASELINE_S = 5.0        # history used to track the background (tonic) EMG level
BASELINE_PCT = 20       # percentile of recent log-RMS taken as the background level
WARMUP_S = 3.0          # no clicks until this much history exists

CLASSES = ["rest", "index", "middle", "other", "tap"]
# How recorded fingers map to model classes, per task:
#   fingers: index -> left click, middle -> right click, ring/pinky -> ignored
#   tap:     any finger tap -> one "tap" event (finger-agnostic detector)
TASKS = {
    "fingers": {"index": "index", "middle": "middle", "ring": "other", "pinky": "other"},
    "tap": {"index": "tap", "middle": "tap", "ring": "tap", "pinky": "tap"},
}
CLICK_FOR_CLASS = {"index": "left", "middle": "right", "tap": "tap"}


# ----------------------------------------------------------------------------- loading
def load_opensignals(txt_path):
    """Return (raw EMG as float array, sampling rate) from an OpenSignals .txt file."""
    txt_path = Path(txt_path)
    metadata = None
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# {"):
                metadata = json.loads(line[2:].strip())
            elif line.startswith("# EndOfHeader"):
                break
    if metadata is None:
        raise ValueError(f"Could not read OpenSignals metadata in {txt_path}")
    device = next(iter(metadata.values()))
    df = pd.read_csv(txt_path, sep=r"\s+", comment="#", header=None,
                     names=device["column"], engine="python")
    return df[EMG_COLUMN].to_numpy(float), device["sampling rate"]


def load_recording(folder, name):
    """Load <name>.txt (EMG) and <name>.csv (ground-truth tap times)."""
    folder = Path(folder)
    emg, fs = load_opensignals(folder / f"{name}.txt")
    gt = pd.read_csv(folder / f"{name}.csv")
    return emg, fs, gt["elapsed_seconds"].to_numpy(float), gt["label"].iloc[0]


# ----------------------------------------------------------------------------- filtering
def make_filter(fs=FS):
    band = butter(4, [20, 450], btype="band", fs=fs, output="sos")
    b, a = iirnotch(MAINS_HZ, Q=30, fs=fs)
    return np.vstack([band, tf2sos(b, a)])


class CausalFilter:
    """Stateful band-pass + notch filter. Feed it chunks of any size."""

    def __init__(self, fs=FS):
        self.sos = make_filter(fs)
        self.zi = None
        self.offset = None

    def __call__(self, x):
        x = np.asarray(x, float)
        if self.offset is None:
            # remove the ADC's DC level up front to avoid a start-up transient
            self.offset = x.mean()
            self.zi = np.zeros_like(sosfilt_zi(self.sos))
        y, self.zi = sosfilt(self.sos, x - self.offset, zi=self.zi)
        return y


def filter_offline(x, fs=FS):
    """Same causal filter applied to a whole recording (used for training)."""
    return CausalFilter(fs)(x)


# ----------------------------------------------------------------------------- features
_freqs = np.fft.rfftfreq(WIN, 1 / FS)
_bands = [(20, 50), (50, 100), (100, 200), (200, 450)]
_band_masks = [(_freqs >= lo) & (_freqs < hi) for lo, hi in _bands]
_hann = np.hanning(WIN)

FEATURE_NAMES = (
    ["log_rms", "log_mav", "log_wl", "zc", "ssc", "log_var",
     "mnf", "mdf", "hjorth_mob", "hjorth_cmp"]
    + [f"band_{lo}_{hi}" for lo, hi in _bands]
    + [f"log_rms_q{i}" for i in range(4)]
    + ["rms_slope"]
)


def window_features(w):
    """Classic time + frequency EMG features for one filtered window."""
    eps = 1e-6
    d1 = np.diff(w)
    d2 = np.diff(d1)
    rms = np.sqrt(np.mean(w ** 2))
    mav = np.mean(np.abs(w))
    wl = np.sum(np.abs(d1))
    thr = 0.01 * (np.max(np.abs(w)) + eps)
    zc = np.sum((w[:-1] * w[1:] < 0) & (np.abs(d1) > thr)) / WIN
    ssc = np.sum((d1[:-1] * d1[1:] < 0) & ((np.abs(d1[:-1]) > thr) | (np.abs(d1[1:]) > thr))) / WIN
    var = np.var(w)

    p = np.abs(np.fft.rfft(w * _hann)) ** 2
    ptot = p.sum() + eps
    mnf = np.sum(_freqs * p) / ptot
    mdf = _freqs[np.searchsorted(np.cumsum(p), ptot / 2)]
    bands = [p[m].sum() / ptot for m in _band_masks]

    mob = np.sqrt(np.var(d1) / (var + eps))
    cmp_ = np.sqrt(np.var(d2) / (np.var(d1) + eps)) / (mob + eps)

    q = np.array([np.sqrt(np.mean(c ** 2)) for c in np.array_split(w, 4)])
    lq = np.log(q + eps)

    return np.array(
        [np.log(rms + eps), np.log(mav + eps), np.log(wl + eps), zc, ssc, np.log(var + eps),
         mnf, mdf, mob, cmp_, *bands, *lq, lq[-1] - lq[0]]
    )


# Amplitude features are expressed relative to the recent background EMG level
# (a low percentile of the last few seconds), so electrode noise or the tonic
# tension of holding the hand up doesn't look like a tap.
_LOG_AMP = [FEATURE_NAMES.index(n) for n in
            ["log_rms", "log_mav", "log_wl"] + [f"log_rms_q{i}" for i in range(4)]]
_LOG_VAR = FEATURE_NAMES.index("log_var")
_LOG_RMS = FEATURE_NAMES.index("log_rms")


BASELINE_N = int(BASELINE_S * FS / HOP)


def rolling_baseline(X):
    """Causal background level for each consecutive window (same as live)."""
    lr = X[:, _LOG_RMS]
    return np.array([np.percentile(lr[max(0, i - BASELINE_N + 1):i + 1], BASELINE_PCT)
                     for i in range(len(lr))])


def normalize(X, baseline):
    X = np.array(X, float, copy=True)
    baseline = np.asarray(baseline, float)[..., None]
    X[..., _LOG_AMP] -= baseline
    X[..., _LOG_VAR] -= 2 * baseline[..., 0]
    return X


def sliding_features(y):
    """Features for every window ending at sample index e (e = WIN, WIN+HOP, ...)."""
    ends = np.arange(WIN, len(y) + 1, HOP)
    X = np.array([window_features(y[e - WIN:e]) for e in ends])
    return X, ends


# ----------------------------------------------------------------------------- labels
# Relative to a tap time t0 (seconds), a window ending at time t is:
#   positive (finger)  if  POS_LO <= t - t0 <= POS_HI   (window covers the burst peak)
#   rest               if  the window [t - WIN, t] is far from every tap
#   ignored            otherwise (partial bursts: ambiguous, don't train on them)
POS_LO, POS_HI = 0.00, 0.20
REST_GAP_BEFORE, REST_GAP_AFTER = 0.35, 0.50


def label_windows(ends, fs, taps, finger, class_map=TASKS["fingers"]):
    t_end = ends / fs
    t_start = (ends - WIN) / fs
    labels = np.full(len(ends), None, dtype=object)

    near = np.zeros(len(ends), bool)
    pos = np.zeros(len(ends), bool)
    for t0 in taps:
        near |= (t_end > t0 - REST_GAP_BEFORE) & (t_start < t0 + REST_GAP_AFTER)
        pos |= (t_end - t0 >= POS_LO) & (t_end - t0 <= POS_HI)

    # only trust "rest" inside the labelled part of the recording
    in_session = (t_start > taps.min() - 1.5) & (t_end < taps.max() + 1.5)
    labels[in_session & ~near] = "rest"
    labels[pos] = class_map[finger]
    return labels


# ----------------------------------------------------------------------------- live detector
class ClickDetector:
    """
    Real-time click detector.

        det = ClickDetector.load("models/emg_click_model.joblib")
        for chunk in stream:                 # raw ADC samples from A2, any chunk size
            for click in det.push(chunk):    # 'left' or 'right'
                mouse.click(click)

    No clicks are emitted during the first WARMUP_S seconds while the background
    EMG level is being learned.

    A click fires when `votes_needed` of the last `vote_window` predictions agree
    on a clickable class (index/middle, or tap) with probability >= `min_prob`. After a click
    the detector re-arms only once `refractory_s` has passed AND it has seen `rearm_rest`
    consecutive rest predictions, so one tap (press + release) = one click.
    """

    def __init__(self, model, fs=FS, min_prob=0.6, vote_window=3, votes_needed=2,
                 refractory_s=0.35, rearm_rest=0):
        self.model = model
        self.fs = fs
        self.filter = CausalFilter(fs)
        self.buf = np.zeros(0)
        self.since_hop = 0
        self.min_prob = min_prob
        self.votes = deque(maxlen=vote_window)
        self.votes_needed = votes_needed
        self.refractory = int(refractory_s * fs / HOP)
        self.cooldown = 0
        self.rearm_rest = rearm_rest
        self.rest_run = rearm_rest
        self.armed = True
        self.history = deque(maxlen=BASELINE_N)
        self.feat_hist = deque(maxlen=16)        # recent normalised window features
        self.warmup = int((WARMUP_S * fs - WIN) / HOP) + 1
        self.classes = list(model.classes_)
        self.targets = [c for c in self.classes if c in CLICK_FOR_CLASS]

    @classmethod
    def load(cls, path, **kw):
        import joblib
        bundle = joblib.load(path)
        params = {**bundle.get("detector_params", {}), **kw}
        return cls(bundle["model"], **params)

    def _decide(self, proba):
        k = int(np.argmax(proba))
        cls_ = self.classes[k]
        vote = cls_ if proba[k] >= self.min_prob else "rest"
        self.votes.append(vote)
        self.rest_run = self.rest_run + 1 if vote not in self.targets else 0
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        if self.rest_run < self.rearm_rest and not self.armed:
            return None
        self.armed = True
        for target in self.targets:
            if sum(v == target for v in self.votes) >= self.votes_needed:
                self.cooldown = self.refractory
                self.votes.clear()
                self.armed = False
                self.rest_run = 0
                return CLICK_FOR_CLASS[target]
        return None

    def push(self, raw_chunk):
        y = self.filter(raw_chunk)
        self.buf = np.concatenate([self.buf, y])[-WIN:]
        self.since_hop += len(y)
        clicks = []
        while self.since_hop >= HOP:
            self.since_hop -= HOP
            if len(self.buf) < WIN:
                continue
            f = window_features(self.buf)
            self.history.append(f[_LOG_RMS])
            if self.warmup > 0:
                self.warmup -= 1
                continue
            f = normalize(f[None, :], np.percentile(self.history, BASELINE_PCT))
            self.feat_hist.append(f[0].copy())
            c = self._decide(self.model.predict_proba(f)[0])
            if c:
                clicks.append(c)
        return clicks

    def run_offline(self, raw, chunk=HOP):
        """Replay a whole recording; returns list of (time_s, click)."""
        out = []
        for i in range(0, len(raw) - chunk + 1, chunk):
            for c in self.push(raw[i:i + chunk]):
                out.append(((i + chunk) / self.fs, c))
        return out


# ----------------------------------------------------------------------------- two-stage detector
# Stage 1 finds that a press happened (tap model: rest vs tap). Stage 2 picks the finger from
# the windows around the moment stage 1 fired: FINGER_BACK before it and FINGER_FWD after it
# (about 0.1 s of extra delay). Training uses exactly the same windows.
FINGER_BACK, FINGER_FWD = 4, 4
# log_var is 2 * log_rms, so it adds nothing and makes the linear model unstable
FINGER_COLS = [i for i, n in enumerate(FEATURE_NAMES) if n != "log_var"]


def finger_features(window_feats):
    """Mean of the normalised feature rows around a fire -> the vector the finger model sees."""
    return np.mean(window_feats, axis=0)[FINGER_COLS]


class FingerDetector(ClickDetector):
    """
        det = FingerDetector.load("models/emg_fingers_2stage.joblib")
        for event in det.push(chunk):     # 'left' (index) or 'right' (middle)
            ...
    """

    def __init__(self, tap_model, finger_model, **kw):
        super().__init__(tap_model, **kw)
        self.finger_model = finger_model
        self.pending = None

    @classmethod
    def load(cls, path, **kw):
        import joblib
        bundle = joblib.load(path)
        return cls(bundle["tap_model"], bundle["finger_model"],
                   **{**bundle.get("detector_params", {}), **kw})

    def _decide(self, proba):
        fired = super()._decide(proba)           # 'tap' when the press detector fires
        out = None
        if self.pending is not None:
            self.pending -= 1
            if self.pending == 0:
                self.pending = None
                feats = np.array(list(self.feat_hist)[-(FINGER_BACK + FINGER_FWD + 1):])
                label = self.finger_model.predict(finger_features(feats)[None, :])[0]
                out = CLICK_FOR_CLASS[label]
        if fired:
            self.pending = FINGER_FWD
        return out

