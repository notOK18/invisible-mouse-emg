"""
Live EMG clicking: index press -> LEFT click, full fist -> RIGHT click.
Run this alongside imu_mouse.py (they don't share any state).

Keys (global, work from any window):
    E = enable / disable clicking
    X = quit            (not Q: imu_mouse.py already uses Q, M and C)

Usage:
    py live_emg.py
    py live_emg.py --monitor            # shows live strength, for tuning
    py live_emg.py --x 2.5 --y 7        # override the saved thresholds
"""

import argparse
import json
import time

import keyboard
from pynput.mouse import Button, Controller

from emg_common import (
    ACQ_CHANNELS, ADC_COLUMN, CLICK_NAMES, COM_PORT, FS, HOP,
    FeatureStream, ThresholdClicker, connect,
)

START_ENABLED = True
BUTTONS = {1: Button.left, 2: Button.right}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/emg_thresholds.json")
    parser.add_argument("--port", default=COM_PORT)
    parser.add_argument("--x", type=float, default=None, help="press onset threshold")
    parser.add_argument("--y", type=float, default=None, help="fist threshold")
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()

    with open(args.model) as f:
        cfg = json.load(f)

    x = args.x if args.x is not None else cfg["x"]
    y = args.y if args.y is not None else cfg["y"]
    print(f"Thresholds: press onset x = {x:.2f}, fist y = {y:.2f}")

    mouse = Controller()
    stream = FeatureStream()
    clicker = ThresholdClicker(x, y)

    device = connect(args.port)

    enabled = START_ENABLED
    e_prev = False
    ready = False
    t0 = time.perf_counter()

    print()
    print("Keep the arm RELAXED for 3 s while the background level is learned.")
    print("Keys: E = enable/disable clicks, X = quit")

    device.start(FS, ACQ_CHANNELS)

    try:
        while True:
            data = device.read(HOP)

            for end, feats in stream.push(data[:, ADC_COLUMN]):
                if feats is None:
                    continue

                if not ready:
                    ready = True
                    print(f"\nReady. Clicking {'ENABLED' if enabled else 'DISABLED'}.")

                strength = feats[0]
                click = clicker.push(strength)

                if args.monitor:
                    print(f"\rstrength {strength:6.2f}  [{clicker.state:9s}]   ", end="")

                if click is not None and enabled:
                    mouse.click(BUTTONS[click])
                    print(
                        f"\n{time.perf_counter() - t0:7.1f}s  {CLICK_NAMES[click]} click"
                        f"   (peak {clicker.peak:.1f})"
                    )

            e_down = keyboard.is_pressed("e")
            if e_down and not e_prev:
                enabled = not enabled
                print("\nClicking:", "ENABLED" if enabled else "DISABLED")
            e_prev = e_down

            if keyboard.is_pressed("x"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        try:
            device.stop()
            device.close()
        except Exception:
            pass

    print("\nStopped.")


if __name__ == "__main__":
    main()
