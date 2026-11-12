import asyncio
import struct
import math
import time

import keyboard
from bleak import BleakClient, BleakScanner
from pynput.mouse import Controller


# ==================================================
# CONFIGURATION
# ==================================================

DEVICE_NAME_PREFIX = "WT901BLE"

# Corrected WitMotion BLE notification characteristic
NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"

# Accelerometer range (WitMotion default is commonly +/- 16g)
ACCEL_RANGE_G = 16.0

# ---- Threshold ----------------------------------------------------
# Tilt (in g, after removing neutral offset) that must be exceeded
# before the cursor starts moving at all. Higher = you must tilt more
# before anything happens. Lower = more sensitive.
THRESHOLD_G = 0.4


# ---- Speed curve --------------------------------------------------
# Cursor speed is NOT linear in tilt. Small tilts stay slow so you can
# make fine corrections; large tilts go fast for crossing the screen.
#
# MIN_SPEED_PX: speed (px/sec) at the very moment you cross the
#               threshold. This is your "micro-correction" speed.
#               Lower = finer control, but need more ramp for big moves.
MIN_SPEED_PX = 10.0

# MAX_SPEED_PX: speed (px/sec) at full tilt AND fully ramped.
#               This is your "cross the screen" speed.
MAX_SPEED_PX = 1000.0

# SPEED_CURVE_EXP: how aggressively speed grows with tilt.
#   1.0 = linear (joystick feel, don't use)
#   2.0 = gentle curve  (good starting point)
#   3.0 = very fine near threshold, punchy at extremes
SPEED_CURVE_EXP = 1.4


# ---- Time-based ramp ----------------------------------------------
# This is the key change from the old script: while you HOLD a tilt,
# speed climbs toward MAX_SPEED_PX. Short taps = small move.
# Sustained tilt = fast move. This is what makes it feel less like a
# joystick and more like an accelerated pointer.
#
# RAMP_SECONDS: how long you must hold a tilt to reach full speed.
#   Higher = slower to speed up = more precise, but long moves take
#            longer to get going.
#   Lower  = quicker acceleration = more responsive, less precise.
RAMP_SECONDS = 1

# RAMP_DECAY_SECONDS: how fast the ramp "resets" when you stop tilting.
#   Lower = cursor stops crisply when you center the sensor.
#   Higher = momentum feel (may overshoot).
RAMP_DECAY_SECONDS = 0.15

# RAMP_FLOOR: the fraction of full speed that is available even
# before the ramp has charged. 0.25 means "at the first frame of
# tilt you get 25% of the curved speed, then climb to 100%".
#   Lower = even finer initial control, but noticeably rampy feel.
#   Higher = more immediate response, closer to old behavior.
RAMP_FLOOR = 0.25


# ---- Smoothing ----------------------------------------------------
# Low-pass filter on the cursor velocity. 0 = no smoothing (raw,
# jittery). 0.9 = very smooth but laggy.
VELOCITY_ALPHA = 0.6


# ---- Calibration / scan ------------------------------------------
CALIBRATION_SECONDS = 3.0
SCAN_SECONDS = 10.0


# ==================================================
# STATE
# ==================================================

mouse_enabled = False

latest_accel = None              # (ax, ay, az) in g
neutral_accel = (0.0, 0.0, 0.0)  # offsets captured during calibration

smoothed_vx = 0.0
smoothed_vy = 0.0

ramp = 0.0                       # 0.0 .. 1.0 charge factor

last_update = time.perf_counter()

packet_buffer = bytearray()

mouse = Controller()

fractional_x = 0.0
fractional_y = 0.0


# ==================================================
# UTILITIES
# ==================================================

def clamp(value, low, high):
    return max(low, min(high, value))


# ==================================================
# ACCELEROMETER PARSING
# ==================================================

PACKET_LEN = 20


def parse_accel_packet(packet):
    # 0x55 0x61 | AxL AxH AyL AyH AzL AzH | gyro(6) | angle(6)
    ax_raw, ay_raw, az_raw = struct.unpack("<3h", packet[2:8])
    scale = ACCEL_RANGE_G / 32768.0
    return ax_raw * scale, ay_raw * scale, az_raw * scale


# ==================================================
# THRESHOLD -> NORMALIZED DIRECTION
# ==================================================

def tilt_to_direction(accel):
    """
    Returns (nx, ny) where each component is in [-1, 1].

    Magnitude 0   = inside dead-zone (no movement)
    Magnitude 1   = fully past threshold on an axis

    Direction is the same as before:
        y < -THRESHOLD  -> move up      (ny positive)
        y >  THRESHOLD  -> move down    (ny negative)
        x >  THRESHOLD  -> move left    (nx negative)
        x < -THRESHOLD  -> move right   (nx positive)

    Note this returns a *normalized* direction, not a speed. Speed is
    applied separately in update_mouse() after the ramp is computed.
    """
    ax = accel[0] - neutral_accel[0]
    ay = accel[1] - neutral_accel[1]
    # az ignored

    nx = 0.0
    ny = 0.0

    if ax > THRESHOLD_G:
        nx = -clamp((ax - THRESHOLD_G) / THRESHOLD_G, 0.0, 1.0)
    elif ax < -THRESHOLD_G:
        nx = clamp((-ax - THRESHOLD_G) / THRESHOLD_G, 0.0, 1.0)

    if ay < -THRESHOLD_G:
        ny = clamp((-ay - THRESHOLD_G) / THRESHOLD_G, 0.0, 1.0)
    elif ay > THRESHOLD_G:
        ny = -clamp((ay - THRESHOLD_G) / THRESHOLD_G, 0.0, 1.0)

    return nx, ny


# ==================================================
# MOUSE UPDATE
# ==================================================

def move_mouse_fractional(dx, dy):
    global fractional_x, fractional_y
    fractional_x += dx
    fractional_y += dy
    ix, iy = int(fractional_x), int(fractional_y)
    fractional_x -= ix
    fractional_y -= iy
    if ix != 0 or iy != 0:
        mouse.move(ix, iy)


def update_mouse(accel):
    """
    Full pipeline:
        1. tilt -> normalized direction (nx, ny), magnitude 0..1
        2. charge ramp based on how long the tilt is sustained
        3. curve magnitude (small tilts stay small)
        4. map magnitude -> px/sec between MIN_SPEED and MAX_SPEED
        5. apply ramp multiplier so speed climbs while held
        6. low-pass filter and move cursor
    """
    global smoothed_vx, smoothed_vy, last_update, ramp

    now = time.perf_counter()
    dt = clamp(now - last_update, 0.0, 0.05)
    last_update = now

    if not mouse_enabled:
        smoothed_vx = 0.0
        smoothed_vy = 0.0
        ramp = 0.0
        return

    nx, ny = tilt_to_direction(accel)
    magnitude = math.hypot(nx, ny)

    # ---- 1. Charge / decay the ramp ------------------------------
    if magnitude > 0.0:
        # Held tilt: charge the ramp toward 1.0
        ramp = min(1.0, ramp + dt / RAMP_SECONDS)
    else:
        # Released: fall back toward 0.0
        ramp = max(0.0, ramp - dt / RAMP_DECAY_SECONDS)

    if magnitude < 1e-6:
        smoothed_vx = 0.0
        smoothed_vy = 0.0
        return

    # ---- 2. Curve the magnitude ----------------------------------
    # Small tilts -> small curved value -> small speed.
    curved = magnitude ** SPEED_CURVE_EXP

    # ---- 3. Map to pixels per second -----------------------------
    speed_px = MIN_SPEED_PX + (MAX_SPEED_PX - MIN_SPEED_PX) * curved

    # ---- 4. Apply ramp (charged speed) ---------------------------
    # At ramp=0 we still give RAMP_FLOOR of the speed so the very
    # first frame isn't dead. At ramp=1 we give full speed.
    speed_px *= (RAMP_FLOOR + (1.0 - RAMP_FLOOR) * ramp)

    # ---- 5. Convert to a velocity vector -------------------------
    dirx = nx / magnitude
    diry = ny / magnitude

    target_vx = dirx * speed_px
    target_vy = diry * speed_px

    # ---- 6. Smooth + apply ---------------------------------------
    smoothed_vx = VELOCITY_ALPHA * smoothed_vx + (1.0 - VELOCITY_ALPHA) * target_vx
    smoothed_vy = VELOCITY_ALPHA * smoothed_vy + (1.0 - VELOCITY_ALPHA) * target_vy

    move_mouse_fractional(smoothed_vx * dt, smoothed_vy * dt)


# ==================================================
# BLE NOTIFICATION HANDLER
# ==================================================

def notification_handler(sender, data):
    global latest_accel
    packet_buffer.extend(data)

    while len(packet_buffer) >= PACKET_LEN:
        if packet_buffer[0] != 0x55:
            del packet_buffer[0]
            continue

        if packet_buffer[1] == 0x61:
            latest_accel = parse_accel_packet(bytes(packet_buffer[:PACKET_LEN]))
            del packet_buffer[:PACKET_LEN]
        else:
            del packet_buffer[0]


# ==================================================
# SENSOR DISCOVERY
# ==================================================

async def find_sensor():
    print("Scanning for WitMotion sensor...")
    devices = await BleakScanner.discover(timeout=SCAN_SECONDS)

    for device in devices:
        name = device.name or ""
        print(f"Found: {name} | {device.address}")
        if name.upper().startswith(DEVICE_NAME_PREFIX.upper()):
            return device

    return None


# ==================================================
# CALIBRATION
# ==================================================

async def calibrate():
    global neutral_accel, smoothed_vx, smoothed_vy, ramp

    print()
    print("Calibration starting.")
    print("Hold the sensor in your neutral position and keep it still.")

    samples = []
    start = time.perf_counter()

    while time.perf_counter() - start < CALIBRATION_SECONDS:
        if latest_accel is not None:
            samples.append(latest_accel)
        await asyncio.sleep(0.02)

    if len(samples) < 10:
        print("Not enough accelerometer samples.")
        return

    avg = tuple(
        sum(s[i] for s in samples) / len(samples)
        for i in range(3)
    )

    neutral_accel = avg
    smoothed_vx = 0.0
    smoothed_vy = 0.0
    ramp = 0.0

    print("Calibration complete.")
    print("Neutral accel:", tuple(round(x, 3) for x in neutral_accel))


# ==================================================
# MOUSE LOOP
# ==================================================

async def mouse_loop(rate_hz=100):
    global last_update
    period = 1.0 / rate_hz
    last_update = time.perf_counter()
    while True:
        if latest_accel is not None:
            update_mouse(latest_accel)
        await asyncio.sleep(period)


# ==================================================
# MAIN
# ==================================================

async def main():
    global mouse_enabled, smoothed_vx, smoothed_vy, ramp, last_update

    device = await find_sensor()
    if device is None:
        print("Sensor not found.")
        return

    print()
    print(f"Connecting to {device.name}")
    print(f"Address: {device.address}")

    async with BleakClient(device) as client:
        print("Connected.")

        await client.start_notify(NOTIFY_UUID, notification_handler)
        print("Receiving accelerometer data.")

        while latest_accel is None:
            await asyncio.sleep(0.1)

        await calibrate()

        mouse_task = asyncio.create_task(mouse_loop())

        print()
        print("Controls:")
        print("  M = Toggle cursor movement")
        print("  C = Recalibrate")
        print("  Q = Quit")
        print()
        print("Cursor movement starts disabled.")

        last_update = time.perf_counter()

        while True:
            if keyboard.is_pressed("m"):
                mouse_enabled = not mouse_enabled
                smoothed_vx = 0.0
                smoothed_vy = 0.0
                ramp = 0.0
                last_update = time.perf_counter()
                print("\nMouse:", "ENABLED" if mouse_enabled else "DISABLED")
                await asyncio.sleep(0.3)

            if keyboard.is_pressed("c"):
                mouse_enabled = False
                await calibrate()
                await asyncio.sleep(0.3)

            if keyboard.is_pressed("q"):
                break

            if latest_accel is not None:
                ax, ay, az = latest_accel
                print(
                    f"\rAccel: X={ax:6.2f}g Y={ay:6.2f}g Z={az:6.2f}g "
                    f"ramp={ramp:0.2f}",
                    end=""
                )

            await asyncio.sleep(0.01)

        mouse_task.cancel()
        try:
            await mouse_task
        except asyncio.CancelledError:
            pass
        await client.stop_notify(NOTIFY_UUID)

    print()
    print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(main())