"""
Train and TEST the left/right click model (index = left, middle = right) from mixed recordings.

    python train_fingers.py --data data

Two stages, as in the live detector (emg_pipeline.FingerDetector):
    1. tap model     rest vs "a press happened"        (random forest on 25 ms-hop windows)
    2. finger model  index vs middle for each press    (LDA on the windows around the press)

The test that matters is leave-one-session-out: train on some sessions, replay a whole
session the model never saw (electrodes re-attached) through the live detector, and count
what the mouse would have done: correct click, wrong button, missed, or false click.

Verdict thresholds (correct button among detected presses):
    >= 85%   usable for left/right click with some tuning
    65-85%   partly there; a second channel or better placement will help
    <  65%   not separable on this channel -> use tap timing for right click
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emg_pipeline import (CLICK_FOR_CLASS, FINGER_BACK, FINGER_FWD, FS, WARMUP_S, ClickDetector,
                          FingerDetector, finger_features)
from mixed_data import load_folder, quality_report

# Press-detector settings (votes needed, cooldown after a click, ...), tuned on the held-out sessions.
DETECTOR_PARAMS = dict(min_prob=0.5, vote_window=5, votes_needed=4, refractory_s=0.35, rearm_rest=4)


def models():
    """Name -> factory returning a fresh, unfitted model (used for the press detector)."""
    return {
        "LDA": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "RandomForest": lambda: make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                   class_weight="balanced", n_jobs=1, random_state=0)),
    }


def finger_model():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())


def tap_training_set(sessions):
    X = np.vstack([s["X"] for s in sessions])
    y = np.concatenate([np.where(s["y"] == "rest", "rest", "tap") for s in sessions])
    return X, y


def fire_indices(tap_model, s, params):
    """Window indices where the press detector fires on session s (same logic as live)."""
    det = ClickDetector(tap_model, fs=s["fs"], **params)
    P = tap_model.predict_proba(s["Xall"])
    return [i for i, (p, e) in enumerate(zip(P, s["ends_all"]))
            if e >= WARMUP_S * s["fs"] and det._decide(p)]


def window_around(s, i):
    lo, hi = max(0, i - FINGER_BACK), i + FINGER_FWD + 1
    return finger_features(s["Xall"][lo:hi])


def match_fires(s, fires, tol=(-0.60, 0.45)):
    """Pair each tap with the first unused fire that follows it. Returns [(tap_idx, fire_idx)]."""
    t_fire = np.array([s["ends_all"][i] / s["fs"] for i in fires])
    used, pairs = set(), []
    for k, t0 in enumerate(s["taps"]):
        hit = [j for j in range(len(fires)) if tol[0] <= t_fire[j] - t0 <= tol[1] and j not in used]
        if hit:
            used.add(hit[0])
            pairs.append((k, hit[0]))
    return pairs, used, t_fire


def finger_training_set(tap_model, sessions, params):
    F, y = [], []
    for s in sessions:
        if s["kind"] != "mixed":
            continue
        fires = fire_indices(tap_model, s, params)
        pairs, _, _ = match_fires(s, fires)
        for k, j in pairs:
            F.append(window_around(s, fires[j]))
            y.append(s["tap_labels"][k])
    return np.vstack(F), np.array(y)


def score_session(tap_model, fmodel, s, params):
    """Replay session s like the live detector: returns counts and per-tap results."""
    fires = fire_indices(tap_model, s, params)
    pairs, used, t_fire = match_fires(s, fires)
    ok = wrong = 0
    for k, j in pairs:
        pred = fmodel.predict(window_around(s, fires[j])[None, :])[0]
        if pred == s["tap_labels"][k]:
            ok += 1
        else:
            wrong += 1
    miss = len(s["taps"]) - len(pairs)
    lo, hi = s["taps"].min() - 1.5, s["taps"].max() + 1.5
    fp = sum(1 for j, t in enumerate(t_fire) if lo <= t <= hi and j not in used)
    return len(s["taps"]), ok, wrong, miss, fp


def false_clicks_no_press(tap_model, fmodel, s, params):
    """Clicks fired on a recording that contains no presses (motion or rest)."""
    det = FingerDetector(tap_model, fmodel, fs=s["fs"], **params)
    return len(det.run_offline(s["raw"])), len(s["raw"]) / s["fs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/emg_fingers_2stage.joblib")
    ap.add_argument("--model", default="RandomForest", choices=list(models()),
                    help="model type for the press detector (stage 1)")
    args = ap.parse_args()

    print(f"Loading {args.data}/")
    sessions = load_folder(args.data)
    if not any(s["kind"] == "mixed" for s in sessions):
        raise SystemExit("No <session>_mixed.txt + .csv pairs found.")
    quality_report(sessions)
    params = DETECTOR_PARAMS
    ids = sorted({s["session"] for s in sessions if s["kind"] == "mixed"})
    print(f"\nSessions with mixed taps: {ids}")
    if len(ids) < 2:
        print("Only one session: the held-out test needs at least 2 (3 is better). "
              "Training the final model anyway; do not trust it until you can test it.")

    tot = np.zeros(5, int)
    quiet = []
    if len(ids) >= 2:
        print("\n=== Leave-one-session-out, replayed through the live detector ===")
        print("  held-out   presses  correct  wrong-button  missed  false-clicks   correct among detected")
        for sid in ids:
            tr = [s for s in sessions if s["session"] != sid]
            te_all = [s for s in sessions if s["session"] == sid]
            te = [s for s in te_all if s["kind"] == "mixed"]
            tap = models()[args.model]().fit(*tap_training_set(tr))
            F, y = finger_training_set(tap, tr, params)
            fm = finger_model().fit(F, y)
            for s in te_all:
                if s["kind"] != "mixed":
                    n_fc, dur = false_clicks_no_press(tap, fm, s, params)
                    quiet.append((s["name"], n_fc, dur))
            for s in te:
                n, ok, wr, mi, fp = score_session(tap, fm, s, params)
                tot += [n, ok, wr, mi, fp]
                det = ok / max(1, ok + wr)
                print(f"  {sid:8s} {n:8d} {ok:8d} {wr:13d} {mi:7d} {fp:13d}   {det:.1%}")
        n, ok, wr, mi, fp = tot
        det = ok / max(1, ok + wr)
        print(f"  {'total':8s} {n:8d} {ok:8d} {wr:13d} {mi:7d} {fp:13d}   {det:.1%}")
        verdict = ("usable for left/right click with some tuning" if det >= 0.85 else
                   "partly separable; a second channel or better placement will help" if det >= 0.65 else
                   "NOT separable on this channel; use tap timing for right click")
        for name, n_fc, dur in quiet:
            print(f"  {name}: {n_fc} false click(s) in {dur:.0f} s with no presses (held out) "
                  f"= {n_fc / dur * 60:.1f} per minute")
        print(f"\nVERDICT: correct button on {det:.1%} of detected presses (chance = 50%); "
              f"{mi / n:.0%} of presses missed, {fp} false clicks over {len(ids)} sessions -> {verdict}")

    tap = models()[args.model]().fit(*tap_training_set(sessions))
    F, y = finger_training_set(tap, sessions, params)
    fm = finger_model().fit(F, y)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(tap_model=tap, finger_model=fm, task="fingers-2stage", fs=FS,
                     detector_params=params, trained_on=[s["name"] for s in sessions]), out)
    print(f"saved -> {out}   (load with FingerDetector.load)")


if __name__ == "__main__":
    main()
