"""
Fit the two click thresholds from recordings made with record_emg.py.

Usage:
    py train_emg.py                       # uses every data/emg_*.npz
    py train_emg.py data/emg_a.npz data/emg_b.npz
    py train_emg.py --out models/emg_thresholds.json

Strength = 200 ms RMS divided by the background level (see emg_common.py).
    x = onset threshold : above rest noise, below the weakest index press
    y = fist threshold  : between the typical index peak and typical fist peak

Validation (no leakage between overlapping windows):
  - 1 recording  -> 5 contiguous time blocks, thresholds fitted on the rest
  - 2+ recordings -> leave-one-recording-out
Each held-out part is replayed through the SAME click logic used live.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from emg_common import (
    FS, HOP, LABEL_NAMES, WIN, ThresholdClicker, FeatureStream,
)

N_BLOCKS = 5
MIN_RUN = 100                 # samples; shorter label runs are not "presses"
PEAK_SPAN = 250               # samples after press start used for the peak
PAD = WIN + 4 * HOP           # samples kept clear around a held-out block


# ==================================================
# DATA
# ==================================================

def load_recording(path):
    z = np.load(path)
    emg, labels = z["emg"].astype(float), z["labels"]

    stream = FeatureStream()
    ends, strength = [], []

    for i in range(0, len(emg), 500):
        for end, f in stream.push(emg[i:i + 500]):
            if f is not None:
                ends.append(end)
                strength.append(f[0])          # rms / background

    ends = np.array(ends)
    strength = np.array(strength)

    # Windows where every sample is "rest" (used to measure rest noise)
    is_rest = np.array([np.all(labels[e - WIN:e] == 0) for e in ends])

    # Press runs and their peak strength in the first PEAK_SPAN samples
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] in (1, 2) and i - start >= MIN_RUN:
                m = (ends > start) & (ends <= min(i, start + PEAK_SPAN))
                if m.any():
                    runs.append(dict(
                        cls=int(labels[start]), a=start, b=i,
                        peak=float(strength[m].max()),
                    ))
            start = i

    return dict(ends=ends, strength=strength, is_rest=is_rest, runs=runs)


# ==================================================
# FIT
# ==================================================

def fit_thresholds(rest, idx_peaks, fist_peaks):
    rest_hi = float(np.percentile(rest, 99.5))
    idx_lo = float(np.percentile(idx_peaks, 10))

    if idx_lo > rest_hi:
        x = float(np.sqrt(rest_hi * idx_lo))
    else:
        x = (rest_hi + idx_lo) / 2

    y = float(np.sqrt(np.median(idx_peaks) * np.median(fist_peaks)))
    return x, y


# ==================================================
# CLICK-LEVEL SCORING
# ==================================================

def score_clicks(clicks, runs, lo, hi):
    """clicks: [(sample, cls)]. Only runs fully inside [lo, hi] are scored."""
    runs = [r for r in runs if r["a"] >= lo and r["b"] <= hi]
    used = set()
    res = dict(presses=len(runs), correct=0, wrong=0, missed=0, false=0)

    for r in runs:
        hits = [
            (j, c) for j, (s, c) in enumerate(clicks)
            if r["a"] <= s <= r["b"] + 250 and j not in used
        ]
        if not hits:
            res["missed"] += 1
            continue
        used.update(j for j, _ in hits)
        res["correct" if hits[0][1] == r["cls"] else "wrong"] += 1
        res["false"] += len(hits) - 1

    res["false"] += sum(
        1 for j, (s, _) in enumerate(clicks)
        if j not in used and lo <= s <= hi
    )
    return res


def collect(recs, exclude):
    """Rest strengths and per-class press peaks from everything not excluded."""
    rest, idx, fist = [], [], []

    for i, r in enumerate(recs):
        for k in range(len(r["ends"])):
            e = r["ends"][k]
            if r["is_rest"][k] and not exclude(i, e - WIN, e):
                rest.append(r["strength"][k])
        for run in r["runs"]:
            if not exclude(i, run["a"], run["b"]):
                (idx if run["cls"] == 1 else fist).append(run["peak"])

    return np.array(rest), np.array(idx), np.array(fist)


# ==================================================
# VALIDATION
# ==================================================

def run_validation(recs):
    folds = []                                   # (rec, window_start, window_stop)

    if len(recs) > 1:
        folds = [(i, 0, len(r["ends"])) for i, r in enumerate(recs)]
    else:
        edges = np.linspace(0, len(recs[0]["ends"]), N_BLOCKS + 1).astype(int)
        folds = [(0, edges[k], edges[k + 1]) for k in range(N_BLOCKS)]

    totals = dict(presses=0, correct=0, wrong=0, missed=0, false=0)

    for i, w0, w1 in folds:
        r = recs[i]
        lo, hi = r["ends"][w0] - WIN, r["ends"][w1 - 1]

        if len(recs) > 1:
            def exclude(rec, a, b, i=i):
                return rec == i
        else:
            def exclude(rec, a, b, lo=lo, hi=hi):
                return a < hi + PAD and b > lo - PAD

        rest, idx, fist = collect(recs, exclude)

        if len(idx) < 2 or len(fist) < 2 or len(rest) < 20:
            print("  (fold skipped: not enough training presses)")
            continue

        x, y = fit_thresholds(rest, idx, fist)
        clicker = ThresholdClicker(x, y)

        clicks = []
        for end, s in zip(r["ends"][w0:w1], r["strength"][w0:w1]):
            c = clicker.push(s)
            if c:
                clicks.append((end, c))

        res = score_clicks(clicks, r["runs"], lo, hi)
        for k in totals:
            totals[k] += res[k]

    print("\n=== Held-out click test (live click logic) ===")
    print(
        f"presses {totals['presses']} | correct {totals['correct']} | "
        f"wrong button {totals['wrong']} | missed {totals['missed']} | "
        f"false clicks {totals['false']}"
    )
    detected = totals["presses"] - totals["missed"]
    if detected > 0:
        print(f"correct button among detected: {totals['correct'] / detected * 100:.1f}%")


# ==================================================
# MAIN
# ==================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--out", default="models/emg_thresholds.json")
    args = parser.parse_args()

    files = args.files or sorted(glob.glob("data/emg_*.npz"))
    if not files:
        raise SystemExit("No recordings found. Run record_emg.py first.")

    recs = []
    for f in files:
        recs.append(load_recording(f))
        n_i = sum(r["cls"] == 1 for r in recs[-1]["runs"])
        n_f = sum(r["cls"] == 2 for r in recs[-1]["runs"])
        print(f"{f}: {n_i} index presses, {n_f} fist presses")

    rest, idx, fist = collect(recs, lambda *_: False)

    if len(idx) < 3 or len(fist) < 3 or len(rest) < 50:
        raise SystemExit(
            "Not enough labelled data (need a few presses of each class "
            "and plenty of rest). Record again."
        )

    run_validation(recs)

    x, y = fit_thresholds(rest, idx, fist)

    print("\n=== Final thresholds (all data) ===")
    print(f"rest strength   : median {np.median(rest):.2f}, 99.5th pct {np.percentile(rest, 99.5):.2f}, max {rest.max():.2f}")
    print(f"index peaks     : {idx.min():.1f} .. {idx.max():.1f}   (median {np.median(idx):.1f})")
    print(f"fist peaks      : {fist.min():.1f} .. {fist.max():.1f}   (median {np.median(fist):.1f})")
    print(f"x (press onset) = {x:.2f}")
    print(f"y (fist)        = {y:.2f}")

    if idx.max() >= fist.min():
        print(f"WARNING: index and fist peaks overlap ({fist.min():.1f}-{idx.max():.1f}); "
              f"some wrong-button clicks are expected.")
    if rest.max() >= x:
        print("WARNING: some rest windows exceed x; expect occasional false clicks.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"x": x, "y": y, "files": files,
         "index_peaks": idx.tolist(), "fist_peaks": fist.tolist()},
        indent=2,
    ))
    print(f"\nSaved thresholds to {out}")


if __name__ == "__main__":
    main()
