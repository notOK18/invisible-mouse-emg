"""
Record a labelled EMG sample from the BITalino (A2, COM9).

While recording, YOU provide the ground truth with the keyboard:
    hold A  = index finger press   (-> left click)
    hold D  = full fist            (-> right click)
    nothing = rest
(A and D together are ignored.)

Usage:
    py record_emg.py                 # 60 s, saved to data/emg_<timestamp>.npz
    py record_emg.py --seconds 90
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import keyboard
import numpy as np

from emg_common import (
    ACQ_CHANNELS, ADC_COLUMN, COM_PORT, FS, LABEL_NAMES, connect,
)

CHUNK = 20   # samples per read (20 ms)


def current_label():
    a = keyboard.is_pressed("a")
    d = keyboard.is_pressed("d")

    if a and d:
        return -1        # ambiguous, ignored in training
    if a:
        return 1
    if d:
        return 2
    return 0


def flush_console_input():
    # The A/D keystrokes also land in the terminal; clear them.
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--port", default=COM_PORT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path("data") / (
        "emg_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".npz"
    )

    if out.exists():
        raise SystemExit(f"{out} already exists, refusing to overwrite.")

    out.parent.mkdir(parents=True, exist_ok=True)

    n_total = int(args.seconds * FS)

    device = connect(args.port)

    print()
    print("Protocol (aim for ~40% rest overall):")
    print("  - Stay RELAXED for the first 5 s (the model learns your rest level).")
    print("  - Then alternate: hold A ~1.5 s, relax ~2 s, hold D ~1.5 s, relax ~2 s ...")
    print("  - Vary the strength a little. Keep holds under ~2 s.")
    print()

    for i in range(5, 0, -1):
        print(f"Starting in {i}...", end="\r")
        time.sleep(1)

    emg = np.zeros(n_total, dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.int8)
    got = 0
    last_print = 0.0

    device.start(FS, ACQ_CHANNELS)
    print("RECORDING" + " " * 40)

    try:
        while got < n_total:
            data = device.read(CHUNK)
            x = data[:, ADC_COLUMN]
            n = min(len(x), n_total - got)

            lab = current_label()

            emg[got:got + n] = x[:n]
            labels[got:got + n] = lab
            got += n

            now = time.perf_counter()
            if now - last_print > 0.2:
                last_print = now
                name = LABEL_NAMES.get(lab, "BOTH KEYS")
                print(
                    f"\r  {got / FS:5.1f} / {args.seconds:.0f} s   [{name.upper():9s}]",
                    end="",
                )

    except KeyboardInterrupt:
        print("\nStopped early.")

    finally:
        try:
            device.stop()
            device.close()
        except Exception:
            pass

    flush_console_input()

    emg, labels = emg[:got], labels[:got]
    np.savez(out, emg=emg, labels=labels, fs=FS)

    print()
    print(f"\nSaved {got} samples ({got / FS:.1f} s) to {out}")
    for k, name in LABEL_NAMES.items():
        print(f"  {name:6s}: {np.mean(labels == k) * 100:5.1f}% of samples")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        t = np.arange(got) / FS
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, emg - np.median(emg), lw=0.5, color="k")
        colors = {1: "tab:blue", 2: "tab:red"}
        for k, c in colors.items():
            ax.fill_between(
                t, ax.get_ylim()[0], ax.get_ylim()[1],
                where=labels == k, color=c, alpha=0.25, label=LABEL_NAMES[k],
            )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("A2 raw (offset removed)")
        ax.set_title("Check: shaded areas (your labels) should line up with bursts")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
