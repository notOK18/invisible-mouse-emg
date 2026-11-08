# Invisible mouse: EMG click model

Part of the invisible mouse project: a wrist IMU moves the cursor, and forearm EMG gives the
clicks. This repo holds the **EMG click model**: index finger flexion = left click, middle
finger flexion = right click, from **one** BITalino EMG channel (port A2, 1000 Hz).

## How it works

Two stages, run together in `emg_pipeline.FingerDetector`:

1. **Press detector.** A random forest labels each 200 ms window (25 ms hop) as rest or press.
   Amplitude features are divided by a running background level (20th percentile of the last
   5 s), so a different electrode contact or a tense arm does not look like a press. A vote
   over the last windows, a cooldown, and a "return to rest" rule turn that into one event per press.
2. **Finger classifier.** Linear discriminant analysis on the mean features of the ~0.2 s
   around the moment the detector fires. This adds about 0.1 s of delay.

All filtering is causal (20-450 Hz band-pass plus 50 Hz notch), so the same code runs offline
for training and live on a stream.

## Results

Leave-one-session-out: train on two sessions, replay the third through the live detector
(electrodes re-attached between sessions; each has 60 index + 60 middle cued presses).

| Held-out session | Presses | Correct | Wrong button | Missed | False clicks |
|---|---|---|---|---|---|
| s1 | 109 | 93 | 13 | 3 | 10 |
| s2 | 105 | 82 | 18 | 5 | 12 |
| s3 | 118 | 81 | 34 | 3 | 6 |
| **Total** | 332 | 256 | 65 | 11 | 28 |

- Correct button among detected presses: **79.8%** (chance is 50%).
- Recordings with no presses, held out: 0 false clicks in 37 s of rest, 2 in 210 s of moving the arm.

Limits: one subject, one electrode placement, three sessions; the detector settings were tuned on
the same held-out sessions, so the numbers are a little optimistic. Presses here come every
1.1-1.6 s, faster than normal clicking. The index/middle split is mostly press strength relative
to background, plus some frequency content, so a second EMG channel should help.

## Use

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_fingers.py            # trains, prints the held-out test, saves models/emg_fingers_2stage.joblib
```

```python
from emg_pipeline import FingerDetector
det = FingerDetector.load("models/emg_fingers_2stage.joblib")
for chunk in emg_stream:           # raw A2 ADC samples, any chunk size
    for click in det.push(chunk):  # "left" (index) or "right" (middle)
        ...
```

The detector stays silent for the first 3 s while it learns the background level.

## Recording new data

1. Put the electrodes on, start recording in OpenSignals (EMG on A2, 1000 Hz), save as `sN_mixed`.
2. On the same computer, run `python record_cues.py --out data/sN_mixed.csv` and press SPACE.
   Flex the shown finger when the bar ends. The program refuses to overwrite an existing file.
3. Also record `sN_motion.txt` (arm moving, no presses) and `sN_rest.txt` (hand still).
4. Re-attach the electrodes between sessions. Use a new `N` every time.

## Files

| File | Purpose |
|---|---|
| `emg_pipeline.py` | Filtering, features, `ClickDetector` and the two-stage `FingerDetector` |
| `mixed_data.py` | Loads recordings, lines the cue times up with the EMG, snaps them to real bursts |
| `record_cues.py` | Cue window that shows which finger to press and logs the beats |
| `train_fingers.py` | Training, the leave-one-session-out test, and saving the model |
| `data/` | The recordings the model was trained on |

Data files are OpenSignals text exports (`.txt`) plus the cue log (`.csv`, `.cue.json`).
The session is the part of the name before the first underscore.
