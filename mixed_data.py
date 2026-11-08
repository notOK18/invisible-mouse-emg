"""
Loading for the mixed-finger recordings made with record_cues.py.

Folder layout (names are what you save in OpenSignals):
    s1_mixed.txt + s1_mixed.csv   cued index/middle flexions  (csv from record_cues.py)
    s1_motion.txt                 arm/wrist movement, no taps  -> all "rest"
    s1_rest.txt                   holding still                -> all "rest"
    s2_..., s3_...                further sessions (electrodes re-attached in between)

The session id is the part of the name before the first underscore. It is what the
training script holds out when it tests, so keep it equal to one sitting.

The CSV beat times are on the cue program's clock. They are lined up with the EMG in two
steps: (1) shift by the difference between the CSV wall time and the OpenSignals header
start time, (2) search +-0.4 s for the shift where the EMG bursts best match the beats.
Then every beat is snapped to the nearest real burst, so the label time is the burst, not
your reaction time. Beats with no burst nearby are dropped and counted in the quality report.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from emg_pipeline import (FS, HOP, WARMUP_S, WIN, filter_offline, load_opensignals, normalize,
                          rolling_baseline, sliding_features)

POS_LO, POS_HI = 0.00, 0.20             # windows ending this long after the burst start are positive
REST_GAP_BEFORE, REST_GAP_AFTER = 0.35, 0.50
SNAP_BEFORE, SNAP_AFTER = 0.30, 0.50    # where to look for the burst around an aligned beat (reaction lag varies ~-0.1..+0.4 s)
BURST_TO_TAP = 0.03                     # tap time = envelope peak minus this
MIN_PEAK_RATIO = 2.5                    # burst must exceed this x the median envelope


def header_start(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# {"):
                dev = next(iter(json.loads(line[2:]).values()))
                # some OpenSignals installs write unpadded values, e.g. "2026-11-8" / "9:5:3.1"
                y, mo, d = (int(v) for v in dev["date"].split("-"))
                h, mi, sec = dev["time"].split(":")
                whole = float(sec)
                return datetime(y, mo, d, int(h), int(mi), int(whole),
                                int(round((whole - int(whole)) * 1e6)))
            if line.startswith("# EndOfHeader"):
                break
    return None


def envelope(y, fs, width_s=0.05):
    k = max(1, int(width_s * fs))
    return np.convolve(np.abs(y), np.ones(k) / k, mode="same")


def align_beats(env, fs, beats, guess, search=0.4, step=0.01, prior_sigma=0.3):
    """Shift (s) to add to `beats + guess` so beats land on EMG bursts.

    Only a small search around the clock-based guess: cues are ~1.3 s apart, so a wide
    search can lock onto the neighbouring beat instead (seen on real data)."""
    cs = np.concatenate([[0.0], np.cumsum(env)])
    offs = np.arange(-search, search + 1e-9, step)
    scores = np.full(len(offs), -np.inf)
    for k, o in enumerate(offs):
        a = ((beats + guess + o - 0.05) * fs).astype(int)
        b = ((beats + guess + o + 0.15) * fs).astype(int)
        ok = (a >= 0) & (b < len(env))
        if ok.sum() < max(5, 0.5 * len(beats)):
            continue
        m = (cs[b[ok]] - cs[a[ok]]) / (b[ok] - a[ok])
        scores[k] = np.mean(np.log(m + 1e-3)) - 0.5 * (o / prior_sigma) ** 2 * 0.05
    return offs[int(np.argmax(scores))] + guess


def snap_to_bursts(env, fs, beat_times):
    """Return burst-based tap times and a boolean mask of beats that had a burst."""
    floor = np.median(env) + 1e-6
    taps = np.full(len(beat_times), np.nan)
    for i, t in enumerate(beat_times):
        a, b = int((t - SNAP_BEFORE) * fs), int((t + SNAP_AFTER) * fs)
        if a < 0 or b >= len(env):
            continue
        seg = env[a:b]
        j = int(np.argmax(seg))
        if seg[j] >= MIN_PEAK_RATIO * floor:
            taps[i] = (a + j) / fs - BURST_TO_TAP
    return taps, ~np.isnan(taps)


def label_windows_multi(ends, fs, taps, tap_classes, in_session=None):
    """Per-window class: finger class near a tap, 'rest' far from all taps, None in between."""
    t_end, t_start = ends / fs, (ends - WIN) / fs
    lab = np.full(len(ends), None, dtype=object)
    near = np.zeros(len(ends), bool)
    pos = np.zeros(len(ends), bool)
    for t0, c in zip(taps, tap_classes):
        near |= (t_end > t0 - REST_GAP_BEFORE) & (t_start < t0 + REST_GAP_AFTER)
        sel = (t_end - t0 >= POS_LO) & (t_end - t0 <= POS_HI)
        lab[sel] = c
        pos |= sel
    rest = ~near if in_session is None else (~near & in_session)
    lab[rest] = "rest"
    return lab


def load_session_file(folder, name):
    folder = Path(folder)
    txt = folder / f"{name}.txt"
    csv = folder / f"{name}.csv"
    raw, fs = load_opensignals(txt)
    y = filter_offline(raw, fs)
    X, ends = sliding_features(y)
    X = normalize(X, rolling_baseline(X))
    info = dict(name=name, session=name.split("_")[0], raw=raw, fs=fs)

    if csv.exists():
        gt = pd.read_csv(csv)
        beats = gt["elapsed_seconds"].to_numpy(float)
        labels = gt["label"].astype(str).to_numpy()
        start = header_start(txt)
        wall0 = datetime.fromisoformat(gt["wall_time"].iloc[0]) if "wall_time" in gt else None
        guess = 0.0
        if start is not None and wall0 is not None:
            guess = (wall0 - start).total_seconds() - beats[0]
        env = envelope(y, fs)
        shift = align_beats(env, fs, beats, guess)
        taps, ok = snap_to_bursts(env, fs, beats + shift)
        taps, tap_cls = taps[ok], labels[ok]
        lab = label_windows_multi(ends, fs, taps, tap_cls)
        info.update(kind="mixed", taps=taps, tap_labels=tap_cls, shift=shift, guess=guess,
                    beats_total=len(beats), beats_matched=int(ok.sum()),
                    matched_by_class={c: int(((labels == c) & ok).sum()) for c in sorted(set(labels))},
                    total_by_class={c: int((labels == c).sum()) for c in sorted(set(labels))})
    else:
        lab = np.full(len(ends), "rest", dtype=object)
        info.update(kind=name.split("_", 1)[1] if "_" in name else "rest",
                    taps=np.array([]), tap_labels=np.array([]))

    lab[ends < WARMUP_S * fs] = None            # live detector is silent during warm-up
    keep = lab != None  # noqa: E711
    block = np.minimum((ends / len(raw) * 5).astype(int), 4)
    info.update(X=X[keep], y=lab[keep].astype(str), block=block[keep], ends=ends[keep],
                Xall=X, ends_all=ends)          # every window, for replaying the live detector
    return info


def load_folder(folder):
    folder = Path(folder)
    names = sorted({p.stem for p in folder.glob("*.txt")})
    out = []
    for n in names:
        try:
            out.append(load_session_file(folder, n))
        except Exception as e:                  # a bad file should not hide the others
            print(f"  skipping {n}: {e}")
    return out


def quality_report(sessions):
    print("\nData quality (mixed files):")
    print("  file            beats matched to a burst   shift applied   per finger matched/total")
    for s in sessions:
        if s["kind"] != "mixed":
            continue
        per = ", ".join(f"{c} {s['matched_by_class'][c]}/{s['total_by_class'][c]}" for c in s["total_by_class"])
        frac = s["beats_matched"] / s["beats_total"]
        flag = "" if frac >= 0.9 else "   <-- low: check electrodes / did you tap on every cue?"
        print(f"  {s['name']:14s} {s['beats_matched']:4d}/{s['beats_total']:<4d} ({frac:4.0%})        "
              f"{s['shift'] - s['guess']:+.2f} s        {per}{flag}")
