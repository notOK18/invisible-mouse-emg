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

# Threshold in g — must exceed this on an axis to trigger movement
THRESHOLD_G = 0.5

# Cursor speed when threshold is exceeded, pixels/sec
SPEED = 800.0

# Smoothing of cursor velocity (0..1, higher = smoother/slower response)
VELOCITY_ALPHA = 0.6

# Calibration
CALIBRATION_SECONDS = 3.0

# BLE scan
SCAN_SECONDS = 10.0


# ==================================================
# STATE
# ==================================================

mouse_enabled = False

latest_accel = None           # (ax, ay, az) in g
neutral_accel = (0.0, 0.0, 0.0)  # offsets captured during calibration

smoothed_vx = 0.0
smoothed_vy = 0.0

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


def apply_dead_zone(value, dead_zone):
    if abs(value) <= dead_zone:
        return 0.0
    return value - dead_zone if value > 0 else value + dead_zone


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
# THRESHOLD -> CURSOR VELOCITY
# ==================================================

def tilt_to_velocity(accel):
    """
    Rule (after removing the calibrated neutral offset):

        y < -0.6g  -> move up      (vy positive)
        y >  0.6g  -> move down    (vy negative)
        x >  0.6g  -> move left    (vx negative)
        x < -0.6g  -> move right   (vx positive)

    z is ignored.
    """

    ax = accel[0] - neutral_accel[0]
    ay = accel[1] - neutral_accel[1]
    # az ignored

    vx = 0.0
    vy = 0.0

    # Horizontal
    if ax > THRESHOLD_G:
        vx = -SPEED * clamp((ax - THRESHOLD_G) / THRESHOLD_G, 0.0, 1.0)
    elif ax < -THRESHOLD_G:
        vx = SPEED * clamp((-ax - THRESHOLD_G) / THRESHOLD_G, 0.0, 1.0)

    # Vertical
    if ay < -THRESHOLD_G:
        vy = SPEED * clamp((-ay - THRESHOLD_G) / THRESHOLD_G, 0.0, 1.0)
    elif ay > THRESHOLD_G:
        vy = -SPEED * clamp((ay - THRESHOLD_G) / THRESHOLD_G, 0.0, 1.0)

    return vx, vy


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
    global smoothed_vx, smoothed_vy, last_update

    now = time.perf_counter()
    dt = now - last_update
    last_update = now
    dt = clamp(dt, 0.0, 0.05)

    if not mouse_enabled:
        smoothed_vx = 0.0
        smoothed_vy = 0.0
        return

    vx, vy = tilt_to_velocity(accel)

    smoothed_vx = VELOCITY_ALPHA * smoothed_vx + (1.0 - VELOCITY_ALPHA) * vx
    smoothed_vy = VELOCITY_ALPHA * smoothed_vy + (1.0 - VELOCITY_ALPHA) * vy

    if abs(smoothed_vx) < 1e-3 and abs(smoothed_vy) < 1e-3:
        return

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
    global neutral_accel, smoothed_vx, smoothed_vy

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
    global mouse_enabled, last_update

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
                    f"\rAccel: X={ax:6.2f}g Y={ay:6.2f}g Z={az:6.2f}g",
                    end=""
                )

            await asyncio.sleep(0.01)

        mouse_task.cancel()
        await client.stop_notify(NOTIFY_UUID)

    print()
    print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(main())