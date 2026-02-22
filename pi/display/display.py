"""
GOAT-PI Status Display
2.4" ILI9341 SPI LCD (240x320)

Pin wiring:
  VCC  -> Pin 17 (3.3V)
  GND  -> Pin 30 (Ground)
  DIN  -> Pin 19 (GPIO 10 / MOSI)
  CLK  -> Pin 23 (GPIO 11 / SCLK)
  CS   -> Pin 24 (GPIO 8  / CE0)
  DC   -> Pin 12 (GPIO 18)
  RST  -> Pin 13 (GPIO 27)
  BL   -> Pin 22 (GPIO 25)
"""

import time
import os
import sys
import subprocess
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

import digitalio # type: ignore
import board # type: ignore
import adafruit_rgb_display.ili9341 as ili9341 # type: ignore

# === PIN CONFIGURATION ===
CS_PIN = board.CE0       # GPIO 8  / Pin 24
DC_PIN = board.D18       # GPIO 18 / Pin 12
RST_PIN = board.D27      # GPIO 27 / Pin 13
BL_PIN = board.D25       # GPIO 25 / Pin 22

# === DISPLAY SETUP ===
SCREEN_W = 240
SCREEN_H = 320
ROTATION = 0  # 0=portrait, 90/180/270 to rotate

# === SYSTEM CONFIG ===
LOGO_PATH = "/home/pi/goat-capture/pi/display/boot_logo.png"

SENSOR_IDS = {
    'camera1': '28-0000007193ed',
    'camera2': '28-0000006f96b7',
    'camera3': '28-000000704cc8',
}

HEATER_PINS = {
    'camera1': 17,
    'camera2': 5,   # Moved from GPIO 27 to GPIO 5
    'camera3': 22,
}

CAMERA_DEVICES = ['/dev/video0', '/dev/video2', '/dev/video4']

# === COLORS ===
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (34, 197, 94)
ORANGE = (249, 115, 22)
RED = (239, 68, 68)
DIM = (80, 80, 80)


# === INIT DISPLAY ===
def init_display():
    cs = digitalio.DigitalInOut(CS_PIN)
    dc = digitalio.DigitalInOut(DC_PIN)
    rst = digitalio.DigitalInOut(RST_PIN)
    bl = digitalio.DigitalInOut(BL_PIN)
    bl.direction = digitalio.Direction.OUTPUT
    bl.value = True  # Backlight on

    spi = board.SPI()
    disp = ili9341.ILI9341(
        spi, cs=cs, dc=dc, rst=rst,
        width=SCREEN_W, height=SCREEN_H,
        rotation=ROTATION,
        baudrate=24000000,
    )
    return disp, bl


# === LOAD FONT ===
def load_font(size):
    """Try to load a good font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# === SYSTEM CHECKS ===
def read_temp_f(sensor_id):
    path = f'/sys/bus/w1/devices/{sensor_id}/temperature'
    try:
        with open(path, 'r') as f:
            temp_c = int(f.read().strip()) / 1000.0
            return round(temp_c * 9.0 / 5.0 + 32.0)
    except Exception:
        return None


def check_cameras():
    for dev in CAMERA_DEVICES:
        if not os.path.exists(dev):
            return False
    return True


def check_network():
    try:
        result = subprocess.run(
            ['iwconfig', 'wlan0'],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split('\n'):
            if 'Signal level' in line:
                parts = line.split('Signal level=')
                if len(parts) > 1:
                    dbm = int(parts[1].split(' ')[0])
                    return max(0, min(100, (dbm + 90) * 100 // 60))
        return 0
    except Exception:
        return 0


def check_servers():
    try:
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '2', '8.8.8.8'],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def check_heaters_active():
    for pin in HEATER_PINS.values():
        try:
            with open(f'/sys/class/gpio/gpio{pin}/value', 'r') as f:
                if f.read().strip() == '1':
                    return True
        except Exception:
            pass
    return False


# === DRAWING HELPERS ===
def draw_dot(draw, x, y, color, radius=11):
    """Draw a filled circle (status dot) with glow."""
    # Glow (larger, dimmer circle)
    glow_r = radius + 6
    glow_color = tuple(c // 3 for c in color)
    draw.ellipse(
        [x - glow_r, y - glow_r, x + glow_r, y + glow_r],
        fill=glow_color
    )
    # Solid dot
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color
    )


def draw_wifi(draw, x, y, strength):
    """Draw simple wifi bars — always white."""
    bar_w = 3
    gap = 2
    for i in range(4):
        bar_h = 4 + i * 3
        bx = x + i * (bar_w + gap)
        by = y - bar_h
        if strength > (i * 25):
            draw.rectangle([bx, by, bx + bar_w, y], fill=WHITE)
        else:
            draw.rectangle([bx, by, bx + bar_w, y], fill=(40, 40, 40))


# === BOOT SCREEN ===
def show_boot(disp):
    """Show eagle logo fading in, hold until services ready."""
    logo = None
    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA")
            # Scale to fit display width with padding
            scale = 160 / max(logo.width, logo.height)
            new_w = int(logo.width * scale)
            new_h = int(logo.height * scale)
            logo = logo.resize((new_w, new_h), Image.LANCZOS)
        except Exception as e:
            print(f"Logo load error: {e}")

    if not logo:
        # No logo — just show black for a moment
        img = Image.new("RGB", (SCREEN_W, SCREEN_H), BLACK)
        disp.image(img)
        time.sleep(2)
        return

    # Fade in
    for alpha_pct in range(0, 105, 5):
        img = Image.new("RGB", (SCREEN_W, SCREEN_H), BLACK)
        # Composite logo at current alpha
        logo_faded = logo.copy()
        # Adjust alpha channel
        r, g, b, a = logo_faded.split()
        a = a.point(lambda x: int(x * alpha_pct / 100))
        logo_faded = Image.merge("RGBA", (r, g, b, a))

        # Center logo
        lx = (SCREEN_W - logo_faded.width) // 2
        ly = (SCREEN_H - logo_faded.height) // 2
        img.paste(logo_faded, (lx, ly), logo_faded)
        disp.image(img)
        time.sleep(0.04)

    # Hold logo while checking services (up to 60 seconds)
    start = time.time()
    while time.time() - start < 60:
        servers = check_servers()
        cameras = check_cameras()
        temps_ok = any(read_temp_f(s) is not None for s in SENSOR_IDS.values())
        if servers or cameras or temps_ok:
            break
        time.sleep(2)

    # Fade out
    for alpha_pct in range(100, -5, -5):
        img = Image.new("RGB", (SCREEN_W, SCREEN_H), BLACK)
        logo_faded = logo.copy()
        r, g, b, a = logo_faded.split()
        a = a.point(lambda x: int(x * alpha_pct / 100))
        logo_faded = Image.merge("RGBA", (r, g, b, a))
        lx = (SCREEN_W - logo_faded.width) // 2
        ly = (SCREEN_H - logo_faded.height) // 2
        img.paste(logo_faded, (lx, ly), logo_faded)
        disp.image(img)
        time.sleep(0.04)


# === STATUS SCREEN ===
def draw_status(disp, font_big, font_med, font_sm):
    """Main status loop."""
    slow_interval = 10  # Network/server checks
    fast_interval = 1   # Camera/temp/heater checks
    last_slow = 0
    last_fast = 0

    # Cached state
    wifi = 0
    servers_ok = False
    cameras_ok = False
    heaters_on = False
    temps = {k: None for k in SENSOR_IDS}

    while True:
        now = time.time()

        # Fast checks every 1 second (instant reads)
        if now - last_fast >= fast_interval:
            cameras_ok = check_cameras()
            heaters_on = check_heaters_active()
            for name, sid in SENSOR_IDS.items():
                temps[name] = read_temp_f(sid)
            last_fast = now

        # Slow checks every 10 seconds (network calls)
        if now - last_slow >= slow_interval:
            wifi = check_network()
            servers_ok = check_servers()
            last_slow = now

        # Determine colors
        server_color = GREEN if servers_ok else RED
        camera_color = GREEN if cameras_ok else RED
        temps_all_ok = all(t is not None for t in temps.values())
        if not temps_all_ok:
            temp_color = RED
        elif heaters_on:
            temp_color = ORANGE
        else:
            temp_color = GREEN

        # === BUILD FRAME ===
        img = Image.new("RGB", (SCREEN_W, SCREEN_H), BLACK)
        draw = ImageDraw.Draw(img)

        # Top bar: time + wifi
        time_str = datetime.now().strftime("%-I:%M %p")
        tw = draw.textlength(time_str, font=font_sm)
        draw.text((SCREEN_W - tw - 30, 6), time_str, font=font_sm, fill=WHITE)
        draw_wifi(draw, SCREEN_W - 22, 18, wifi)

        # Status rows
        labels = [
            ("SERVERS", server_color),
            ("CAMERAS", camera_color),
            ("TEMPS", temp_color),
        ]

        start_y = 40
        row_h = 60
        dot_x = SCREEN_W - 26

        for i, (label, color) in enumerate(labels):
            y = start_y + i * row_h
            draw.text((12, y), label, font=font_big, fill=WHITE)
            draw_dot(draw, dot_x, y + 18, color)

        # Camera temps
        temp_y = start_y + 3 * row_h + 4
        cam_labels = ['CAM 1', 'CAM 2', 'CAM 3']
        cam_keys = ['camera1', 'camera2', 'camera3']

        for i, (label, key) in enumerate(zip(cam_labels, cam_keys)):
            y = temp_y + i * 28
            t = temps.get(key)
            t_str = f"{t}°F" if t is not None else "--°F"
            draw.text((16, y), label, font=font_med, fill=WHITE)
            tw = draw.textlength(t_str, font=font_med)
            draw.text((SCREEN_W - tw - 14, y), t_str, font=font_med, fill=WHITE)

        # Push to display
        disp.image(img)
        time.sleep(1)  # 1 FPS is fine for status


# === MAIN ===
def main():
    print("GOAT-PI Display starting...")

    disp, bl = init_display()

    # Load fonts
    font_big = load_font(30)
    font_med = load_font(20)
    font_sm = load_font(15)

    # Boot screen with eagle logo
    print("Showing boot screen...")
    show_boot(disp)

    # Main status loop
    print("Entering status loop...")
    try:
        draw_status(disp, font_big, font_med, font_sm)
    except KeyboardInterrupt:
        print("\nShutting down display...")
        bl.value = False  # Turn off backlight
        img = Image.new("RGB", (SCREEN_W, SCREEN_H), BLACK)
        disp.image(img)


if __name__ == '__main__':
    main()