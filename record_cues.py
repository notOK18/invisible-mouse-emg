"""
Cue program for recording mixed index / middle finger flexions.

    .venv/bin/python record_cues.py --out data/s1_mixed.csv
    .venv/bin/python record_cues.py --out /tmp/try.csv --per-finger 3 --rest 2    # quick trial run

1. Start the recording in OpenSignals (red button).
2. Press SPACE in the cue window. You get a rest period, then a long run of cues.
3. Each cue shows WHICH finger to flex next and a bar that empties toward the beat.
   Flex that finger exactly when the bar hits the end and the screen flashes.
4. Stop the OpenSignals recording when the window says DONE, then save it with the same
   name (s1_mixed -> s1_mixed.txt next to s1_mixed.csv).

Output: <name>.csv with columns elapsed_seconds, wall_time, label (one row per beat) and
<name>.cue.json with the settings. The CSV time is the beat on THIS program's clock; the
loader (mixed_data.py) lines it up with the EMG file afterwards, so you do not need to
press the two buttons at the same instant.
"""
import argparse
import csv
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

FINGERS = ["index", "middle"]
COLORS = {"index": "#0a6b60", "middle": "#b4510b"}


def longest_run(seq):
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def make_schedule(per_finger, rest_s, gap, seed, max_run=3):
    """Balanced random finger order, jittered beat times (seconds since SPACE)."""
    rng = random.Random(seed)
    labels = [f for f in FINGERS for _ in range(per_finger)]
    while True:
        rng.shuffle(labels)
        if longest_run(labels) <= max_run:
            break
    beats, t = [], rest_s
    for lab in labels:
        beats.append((t, lab))
        t += rng.uniform(*gap)
    return beats, t


def write_outputs(out, beats_logged, wall0, args):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_seconds", "wall_time", "label"])
        for t, lab in beats_logged:
            wall = (wall0 + timedelta(seconds=t)).isoformat(timespec="milliseconds")
            w.writerow([f"{t:.4f}", wall, lab])
    out.with_suffix(".cue.json").write_text(json.dumps(dict(
        script_start_wall=wall0.isoformat(timespec="milliseconds"),
        per_finger=args.per_finger, rest_s=args.rest, gap=args.gap, seed=args.seed,
        beats_logged=len(beats_logged)), indent=2))
    print(f"wrote {out} ({len(beats_logged)} beats)")


def run_gui(beats, end_t, args):
    import tkinter as tk
    root = tk.Tk()
    root.title("Finger cues")
    root.geometry("900x520")
    canvas = tk.Canvas(root, bg="#12201d", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    state = dict(t0=None, wall0=None, i=0, logged=[], flash_until=0.0, done=False)

    def text(msg, sub="", color="#e4eeeb"):
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        canvas.create_text(w / 2, h * 0.38, text=msg, fill=color, font=("Helvetica", 84, "bold"))
        canvas.create_text(w / 2, h * 0.62, text=sub, fill="#94a8a3", font=("Helvetica", 22))
        return w, h

    def start(_=None):
        if state["t0"] is None:
            state["t0"] = time.perf_counter()
            state["wall0"] = datetime.now()
            tick()

    def finish():
        write_outputs(args.out, state["logged"], state["wall0"] or datetime.now(), args)
        state["done"] = True

    def tick():
        if state["done"]:
            return
        now = time.perf_counter() - state["t0"]
        i = state["i"]
        # a beat is due: log it and flash
        while i < len(beats) and now >= beats[i][0]:
            state["logged"].append((now, beats[i][1]))
            state["flash_until"] = now + 0.15
            root.bell()
            i += 1
        state["i"] = i
        if now >= end_t:
            w, h = text("DONE", "Stop the OpenSignals recording and save it.")
            finish()
            return
        if now < state["flash_until"]:
            text("NOW", f"{i} / {len(beats)}", "#ffffff")
        elif i == 0 and now < beats[0][0] - 1.5:
            text("RELAX", f"first cue in {beats[0][0] - now:.0f} s", "#4fc7b4")
        elif i < len(beats):
            t_next, lab = beats[i]
            prev_t = beats[i - 1][0] if i > 0 else beats[0][0] - 1.5
            w, h = text(lab.upper(), f"{i + 1} / {len(beats)}", COLORS[lab])
            frac = min(1.0, max(0.0, 1.0 - (now - prev_t) / max(1e-3, t_next - prev_t)))
            canvas.create_rectangle(w * 0.15, h * 0.82, w * 0.85, h * 0.88, outline="#2b3d38")
            canvas.create_rectangle(w * 0.15, h * 0.82, w * (0.15 + 0.7 * frac), h * 0.88,
                                    fill=COLORS[lab], outline="")
        root.after(15, tick)

    def on_close(_=None):
        if state["t0"] is not None and not state["done"]:
            finish()
        root.destroy()

    root.bind("<space>", start)
    root.bind("<Escape>", on_close)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.update()
    text("Press RECORD in OpenSignals", "then press SPACE here (ESC to quit)")
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="CSV to write, e.g. data/s1_mixed.csv")
    ap.add_argument("--per-finger", type=int, default=60)
    ap.add_argument("--rest", type=float, default=5.0, help="seconds of rest before the first cue")
    ap.add_argument("--gap", type=float, nargs=2, default=[1.1, 1.6], metavar=("MIN", "MAX"),
                    help="random seconds between beats")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="no window; write a simulated schedule")
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing --out file")
    args = ap.parse_args()
    if Path(args.out).exists() and not args.force:
        raise SystemExit(f"{args.out} already exists, so this run would overwrite its labels. "
                         "Pick a new name (e.g. s2_mixed.csv), or add --force if you really mean it.")
    if args.seed is None:
        args.seed = random.randrange(1 << 30)

    beats, end_t = make_schedule(args.per_finger, args.rest, args.gap, args.seed)
    total = end_t + 2.0
    print(f"{len(beats)} cues, about {total / 60:.1f} min. seed={args.seed}")
    if args.dry_run:
        write_outputs(args.out, beats, datetime.now(), args)
        return
    run_gui(beats, end_t + 2.0, args)


if __name__ == "__main__":
    main()
