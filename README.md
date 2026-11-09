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

The trained model is included: `models/emg_fingers_2stage.joblib` (trained on every recording in `data/`,
saved with scikit-learn 1.9.1). If your scikit-learn version differs and it fails to load, rebuild it with
`python train_fingers.py`.

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

## Next steps

In rough priority order. Tick them off here as they are done.

**Improve the click model**
- [ ] **Second EMG channel.** Get a second sensor cable (and a reference lead) so one electrode pair sits
  over each finger's muscle. Wrong-button errors (about 1 in 5 now) should drop the most. The code reads
  only column `A2` today: `emg_pipeline.py` needs a list of channels, per-channel features, and the
  ratio between channels as extra features. Record new sessions with both channels.
- [ ] **More sessions** (at least 3 more, electrodes re-attached each time). Session 3 is the weakest (70%).
- [ ] **Realistic presses.** Record a session with presses 2.5-4 s apart, like real clicking, and
  re-check missed presses and false clicks.
- [ ] **Quick calibration at the start of each use:** about 10 index and 10 middle presses to adapt the
  finger classifier to that day's electrode contact.
- [ ] **Nested validation.** The detector settings in `train_fingers.py` were tuned on the same held-out
  sessions; tune them inside the training folds so the reported numbers are not optimistic.

**False clicks from arm movement**
- [ ] Record motion and rest files for every session, including gentle cursor steering and
  **pressing while moving** (the cue program works for this).
- [ ] Use the IMU to suppress clicks for a moment when the hand accelerates. Most remaining false
  clicks come right when a movement suddenly starts.

**Cursor control (IMU, no training needed)**
- [ ] Record a short IMU sample: still, then flicks in each of the four directions, plus finger taps.
- [ ] Build the flick detector: remove gravity, high-pass, threshold with hysteresis, fixed speed,
  a lockout after each flick so the deceleration does not undo the move, and a freeze around clicks.
- [ ] Tune the thresholds and lockout time on the recorded sample.

**Put it together**
- [ ] Live input: stream the BITalino into `FingerDetector.push()` (OpenSignals real-time output or
  the BITalino Python API). Measure the delay from press to click (expected about 0.25 s).
- [ ] Send the cursor moves and clicks to the OS (for example with `pynput` or `pyautogui`).
- [ ] End-to-end test with a small task (move to a target, left click, right click) and count errors.
- [ ] **Fallback if accuracy stays low:** single tap = left click, double tap = right click, or a
  light tap versus a whole-hand squeeze. The press detector already works on one channel.

**Housekeeping**
- [ ] Never reuse a file name when recording; `record_cues.py` now refuses to overwrite.
- [ ] Add a licence once the team decides on one.

## Files

| File | Purpose |
|---|---|
| `emg_pipeline.py` | Filtering, features, `ClickDetector` and the two-stage `FingerDetector` |
| `mixed_data.py` | Loads recordings, lines the cue times up with the EMG, snaps them to real bursts |
| `record_cues.py` | Cue window that shows which finger to press and logs the beats |
| `train_fingers.py` | Training, the leave-one-session-out test, and saving the model |
| `models/emg_fingers_2stage.joblib` | The trained model: press detector + finger classifier |
| `data/` | The recordings the model was trained on |

Data files are OpenSignals text exports (`.txt`) plus the cue log (`.csv`, `.cue.json`).
The session is the part of the name before the first underscore.

---

## Threshold click detector (live pipeline)

The random-forest pipeline above is superseded for live use by a much simpler **threshold detector**. On a first single-session test the forest found every press but picked the wrong button about half the time (10 of 21 correct on held-out blocks, even at 92% window accuracy). Looking at the data showed why: the peak strength of an index press and a fist press barely overlap, and a few thresholds on that peak use it directly. In the first labelled session:

| Gesture | Peak strength (rms / background) |
|---------|----------------------------------|
| Index   | 2.2 to 6.1                       |
| Fist    | 6.8 to 12.2                      |

The peak over the first 250 ms of a press was already enough to tell them apart.

### How it works

1. **Strength.** Raw A2 is band-passed 20-450 Hz with a 50 Hz notch (causal), then the RMS of a 200 ms window (25 ms hop) is divided by a running background level (20th percentile of the last 5 s). This is the same normalisation as the old pipeline, so electrode contact and a tense arm matter less.
2. **Onset.** When strength crosses `x`, a press has started and the peak is tracked for 200 ms.
3. **Button.** If the peak reaches `y`, it is a **fist** (right click, fires immediately). Otherwise it is an **index** press (left click, fires when the 200 ms ends).
4. **One click per press.** After a click, strength must fall below 0.8 x `x` for 100 ms and 0.3 s must pass before the next click, so holding a fist is one click.

The thresholds are fitted automatically: `x` is the geometric mean of the 99.5th percentile of rest strength and the 10th percentile of index peaks; `y` is the geometric mean of the median index peak and the median fist peak. They are saved as readable JSON in `models/emg_thresholds.json`.

### Files

| File             | Purpose                                                                                  |
|------------------|------------------------------------------------------------------------------------------|
| `emg_common.py`  | BITalino connection, streaming filter and features, `ThresholdClicker`                   |
| `record_emg.py`  | Records 60 s from A2; you label it live by holding **A** (index) or **D** (fist)         |
| `train_emg.py`   | Fits `x` and `y`, runs a held-out click test, saves `models/emg_thresholds.json`         |
| `live_emg.py`    | Streams the BITalino and sends left/right clicks with `pynput`                           |

### Use

```
py record_emg.py     # 60 s, stay relaxed for the first 5 s, then alternate holds with ~2 s rest
py train_emg.py      # uses every data/emg_*.npz and prints the held-out click test
py live_emg.py       # run in a second terminal next to imu_mouse.py
```

- **Recording tips:** aim for about 40% rest, keep holds under about 2 s, vary the strength a little, and use a new file each session. A plot at the end lets you check that your labels line up with the bursts.
- **Live keys** (global): **E** toggles clicking, **X** quits. `imu_mouse.py` uses Q, M and C, so these do not clash.
- **Tuning:** `py live_emg.py --monitor` shows live strength and state. Override the saved thresholds without retraining with `--x 2.5 --y 7`.
- **Latency:** about 0.3 s from the start of a press for a left click; right clicks are faster because they fire as soon as strength passes `y`.

### Setup notes (Windows)

- Connect the BITalino through its **outgoing** COM port (`COM9` in `emg_common.py`; check Settings > Bluetooth > More Bluetooth settings > COM Ports). The incoming port hangs, and PyBluez's MAC-address connection crashes on Python 3.10+.
- Extra packages: `numpy`, `scipy`, `keyboard`, `pynput`, `matplotlib`, `pyserial`, plus `bitalino.py`.
- Run the EMG and IMU scripts as two separate processes so neither can stall the other.

### Limits

- One subject and a thin margin between the classes (the highest index peak and lowest fist peak were close). Re-attaching the electrodes shifts the numbers, so record a fresh session and re-run `train_emg.py` when the placement changes.
- Held-out click test on the first session: _fill in from the `train_emg.py` output_.
- Single presses only: holding a fist does not drag.
- Arm movement can still cause false clicks. The IMU-based suppression listed in the next steps is not built yet.