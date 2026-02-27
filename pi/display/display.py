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
import json
import subprocess
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

import digitalio # type: ignore
import board # type: ignore
import adafruit_rgb_display.ili9341 as ili9341 # type: ignore

from dotenv import load_dotenv
load_dotenv("/home/pi/goatdev/pi/.env")

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
LOGO_PATH = "/home/pi/goatdev/pi/display/boot_logo.png"
EC2_API = os.environ.get('EC2_API')

SENSOR_IDS = {    # TODO: reconfirm sensor id's to physical markers on sensors
    'sensor1': '28-0000006d3eba',
    'sensor2': '28-0000007047ea',
    'sensor3': '28-0000007193ed'
}

HEATER_PINS = {
    'camera1': 5,
    'camera2': 6,
    'camera3': 17,
}

CAMERAS = {
    'SIDE': '/dev/camera_side',
    'TOP': '/dev/camera_top',
    'FRONT': '/dev/camera_front',
}

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
        baudrate=4000000,
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
    """Return dict of camera name -> status string.
    Only checks device existence/permissions. The camera proxy
    owns all camera handles — no more BUSY state needed."""
    result = {}
    for name, dev in CAMERAS.items():
        if not os.path.exists(dev):
            if os.path.islink(dev):
                result[name] = 'DANGLING'
            else:
                result[name] = 'MISSING'
        elif not os.access(dev, os.R_OK):
            result[name] = 'NO_PERM'
        else:
            result[name] = 'OK'
    return result


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


def _curl_ok(url, timeout=2):
    """Return True if curl gets a 2xx from url."""
    try:
        result = subprocess.run(
            ['curl', '-sf', '-o', '/dev/null', '-m', str(timeout), url],
            capture_output=True, timeout=timeout + 3
        )
        return result.returncode == 0
    except Exception:
        return False


def check_servers():
    """Return dict of server name -> True/False."""
    return {
        'PROD': _curl_ok('http://localhost:5000/health'),
        'CAM_PROXY': _curl_ok('http://localhost:8080/status'),
        'EC2': _curl_ok(f'{EC2_API}/health', timeout=6),
    }


def check_heater_status():
    """Check heater API for detailed state. Falls back to GPIO check."""
    try:
        result = subprocess.run(
            ['curl', '-sf', '-m', '2', 'http://localhost:5002/status'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            cameras = data.get('cameras', {})
            return {
                'any_on': any(c.get('heater_on') for c in cameras.values()),
                'any_failsafe': any(c.get('failsafe') for c in cameras.values()),
                'any_override': any(c.get('override') != 'auto' for c in cameras.values()),
                'details': {name: {
                    'on': c.get('heater_on', False),
                    'failsafe': c.get('failsafe', False),
                    'override': c.get('override', 'auto'),
                } for name, c in cameras.items()},
            }
    except Exception:
        pass

    # Fallback: direct GPIO check
    any_on = False
    for pin in HEATER_PINS.values():
        try:
            with open(f'/sys/class/gpio/gpio{pin}/value', 'r') as f:
                if f.read().strip() == '1':
                    any_on = True
        except Exception:
            pass
    return {'any_on': any_on, 'any_failsafe': False, 'any_override': False, 'details': {}}


# === DRAWING HELPERS ===
def draw_dot(draw, x, y, color, radius=11):
    """Draw a filled circle (status dot) with glow."""
    glow_r = radius + 6
    glow_color = tuple(c // 3 for c in color)
    draw.ellipse(
        [x - glow_r, y - glow_r, x + glow_r, y + glow_r],
        fill=glow_color
    )
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
            scale = 220 / max(logo.width, logo.height)
            new_w = int(logo.width * scale)
            new_h = int(logo.height * scale)
            logo = logo.resize((new_w, new_h), Image.LANCZOS)
        except Exception as e:
            print(f"Logo load error: {e}")

    if not logo:
        img = Image.new("RGB", (SCREEN_W, SCREEN_H), BLACK)
        try:
            disp.image(img)
        except Exception:
            pass
        time.sleep(2)
        return

    loading_font = load_font(14)

    # Fade in
    for alpha_pct in range(0, 105, 5):
        img = Image.new("RGB", (SCREEN_W, SCREEN_H), BLACK)
        draw = ImageDraw.Draw(img)

        logo_faded = logo.copy()
        r, g, b, a = logo_faded.split()
        a = a.point(lambda x: int(x * alpha_pct / 100))
        logo_faded = Image.merge("RGBA", (r, g, b, a))

        lx = (SCREEN_W - logo_faded.width) // 2
        ly = (SCREEN_H - logo_faded.height) // 2 - 20
        img.paste(logo_faded, (lx, ly), logo_faded)

        text = "Systems loading..."
        tw = draw.textlength(text, font=loading_font)
        text_color = tuple(int(255 * alpha_pct / 100) for _ in range(3))
        draw.text(((SCREEN_W - tw) // 2, ly + logo_faded.height + 12), text, font=loading_font, fill=text_color)

        try:
            disp.image(img)
        except Exception:
            pass
        time.sleep(0.04)

    # Hold logo while checking services (up to 20 seconds)
    img = Image.new("RGB", (SCREEN_W, SCREEN_H), BLACK)
    draw = ImageDraw.Draw(img)
    lx = (SCREEN_W - logo.width) // 2
    ly = (SCREEN_H - logo.height) // 2 - 20
    img.paste(logo, (lx, ly), logo)
    text = "Systems loading..."
    tw = draw.textlength(text, font=loading_font)
    draw.text(((SCREEN_W - tw) // 2, ly + logo.height + 12), text, font=loading_font, fill=WHITE)
    try:
        disp.image(img)
    except Exception:
        pass

    start = time.time()
    while time.time() - start < 20:
        servers = check_servers()
        if all(servers.values()):
            break
        time.sleep(2)


# === STATUS SCREEN ===
def draw_status(disp, font_big, font_med, font_sm, font_xs):
    """Main status loop."""
    slow_interval = 5
    fast_interval = 0.25
    last_slow = 0
    last_fast = 0
    frame_count = 0

    # Cached state
    wifi = 0
    server_status = {'PROD': False, 'CAM_PROXY': False, 'EC2': False}
    cam_status = {name: 'MISSING' for name in CAMERAS}
    heater_status = {'any_on': False, 'any_failsafe': False, 'any_override': False, 'details': {}}
    temps = {k: None for k in SENSOR_IDS}

    while True:
        now = time.time()

        # Fast checks every .25 second
        if now - last_fast >= fast_interval:
            cam_status = check_cameras()
            heater_status = check_heater_status()
            for name, sid in SENSOR_IDS.items():
                temps[name] = read_temp_f(sid)
            last_fast = now

        # Slow checks every 5 seconds
        if now - last_slow >= slow_interval:
            wifi = check_network()
            server_status = check_servers()
            last_slow = now

        # Determine colors
        all_servers = all(server_status.values())
        server_color = GREEN if all_servers else RED

        cam_proxy_up = server_status.get('CAM_PROXY', False)

        all_cams = all(v == 'OK' for v in cam_status.values())
        some_cams = any(v == 'OK' for v in cam_status.values())

        # If CAM_PROXY is down, cameras are effectively unusable -> treat as DOWN
        if not cam_proxy_up:
            camera_color = RED
        elif all_cams:
            camera_color = GREEN
        elif some_cams:
            camera_color = ORANGE
        else:
            camera_color = RED

        temps_all_ok = all(t is not None for t in temps.values())
        if heater_status['any_failsafe']:
            temp_color = RED
        elif not temps_all_ok:
            temp_color = RED
        elif heater_status['any_on']:
            temp_color = ORANGE
        else:
            temp_color = GREEN

        # === BUILD FRAME ===
        try:
            img = Image.new("RGB", (SCREEN_W, SCREEN_H), BLACK)
            draw = ImageDraw.Draw(img)

            # Top bar: heartbeat + time + wifi
            heartbeat = "●" if frame_count % 2 == 0 else " "
            draw.text((6, 6), heartbeat, font=font_sm, fill=GREEN)

            time_str = datetime.now().strftime("%-I:%M %p")
            tw = draw.textlength(time_str, font=font_sm)
            draw.text((SCREEN_W - tw - 30, 6), time_str, font=font_sm, fill=WHITE)
            draw_wifi(draw, SCREEN_W - 22, 18, wifi)

            # --- SERVERS ---
            y = 34
            draw.text((12, y), "SERVERS", font=font_big, fill=WHITE)
            draw_dot(draw, SCREEN_W - 26, y + 18, server_color)
            if not all_servers:
                down = [k for k, v in server_status.items() if not v]
                down_display = [k for k in down]
                line = "DOWN: " + ", ".join(down_display)
                if draw.textlength(line, font=font_xs) > SCREEN_W - 28:
                    draw.text((14, y + 34), "DOWN:", font=font_xs, fill=RED)
                    draw.text((14, y + 46), ", ".join(down_display), font=font_xs, fill=RED)
                else:
                    draw.text((14, y + 34), line, font=font_xs, fill=RED)
            else:
                draw.text((14, y + 34), "ALL OK", font=font_xs, fill=GREEN)

            # --- CAMERAS ---
            y = 100
            draw.text((12, y), "CAMERAS", font=font_big, fill=WHITE)
            draw_dot(draw, SCREEN_W - 26, y + 18, camera_color)

            cam_proxy_up = server_status.get('CAM_PROXY', False)

            if not cam_proxy_up:
                # Same style as SERVERS when something is down
                down_display = ["CAM_PROXY"]
                line = "DOWN: " + ", ".join(down_display)
                if draw.textlength(line, font=font_xs) > SCREEN_W - 28:
                    draw.text((14, y + 34), "DOWN:", font=font_xs, fill=RED)
                    draw.text((14, y + 46), ", ".join(down_display), font=font_xs, fill=RED)
                else:
                    draw.text((14, y + 34), line, font=font_xs, fill=RED)

            elif not all_cams:
                issues = [f"{k}:{v}" for k, v in cam_status.items() if v != 'OK']
                line = " ".join(issues)
                if draw.textlength(line, font=font_xs) > SCREEN_W - 28:
                    mid = len(issues) // 2 + 1
                    draw.text((14, y + 34), " ".join(issues[:mid]), font=font_xs, fill=RED)
                    draw.text((14, y + 46), " ".join(issues[mid:]), font=font_xs, fill=RED)
                else:
                    draw.text((14, y + 34), line, font=font_xs, fill=RED)
            else:
                draw.text((14, y + 34), "ALL OK", font=font_xs, fill=GREEN)

            # --- TEMPS ---
            y = 166
            draw.text((12, y), "TEMPS", font=font_big, fill=WHITE)
            draw_dot(draw, SCREEN_W - 26, y + 18, temp_color)
            if heater_status['any_failsafe']:
                fs = [k for k, v in heater_status['details'].items() if v.get('failsafe')]
                line = "FAILSAFE: " + ", ".join(fs)
                if draw.textlength(line, font=font_xs) > SCREEN_W - 28:
                    draw.text((14, y + 34), "FAILSAFE:", font=font_xs, fill=RED)
                    draw.text((14, y + 46), ", ".join(fs), font=font_xs, fill=RED)
                else:
                    draw.text((14, y + 34), line, font=font_xs, fill=RED)
            elif heater_status['any_override']:
                ov = [k for k, v in heater_status['details'].items() if v.get('override') != 'auto']
                line = "OVERRIDE: " + ", ".join(ov)
                if draw.textlength(line, font=font_xs) > SCREEN_W - 28:
                    draw.text((14, y + 34), "OVERRIDE:", font=font_xs, fill=ORANGE)
                    draw.text((14, y + 46), ", ".join(ov), font=font_xs, fill=ORANGE)
                else:
                    draw.text((14, y + 34), line, font=font_xs, fill=ORANGE)
            elif heater_status['any_on']:
                draw.text((14, y + 34), "HEATERS ON", font=font_xs, fill=ORANGE)
            elif not temps_all_ok:
                missing = [k for k, v in temps.items() if v is None]
                line = "ERR: " + ", ".join(missing)
                if draw.textlength(line, font=font_xs) > SCREEN_W - 28:
                    mid = len(missing) // 2 + 1
                    draw.text((14, y + 34), "ERR: " + ", ".join(missing[:mid]), font=font_xs, fill=RED)
                    draw.text((14, y + 46), ", ".join(missing[mid:]), font=font_xs, fill=RED)
                else:
                    draw.text((14, y + 34), line, font=font_xs, fill=RED)
            else:
                draw.text((14, y + 34), "ALL OK", font=font_xs, fill=GREEN)

            # --- Camera temps ---
            temp_y = 235
            cam_labels = ['CAM 1', 'CAM 2', 'CAM 3']
            cam_keys = ['sensor1', 'sensor2', 'sensor3']

            for i, (label, key) in enumerate(zip(cam_labels, cam_keys)):
                y = temp_y + i * 28
                t = temps.get(key)
                t_str = f"{t}°F" if t is not None else "--°F"
                t_color = WHITE if t is not None else RED
                draw.text((16, y), label, font=font_med, fill=WHITE)
                tw = draw.textlength(t_str, font=font_med)
                draw.text((SCREEN_W - tw - 14, y), t_str, font=font_med, fill=t_color)

            # Push to display
            disp.image(img)
            frame_count += 1

        except Exception as e:
            print(f"Display error: {e}")

        time.sleep(1)


# === MAIN ===
def main():
    while True:
        try:
            disp, bl = init_display()

            font_big = load_font(28)
            font_med = load_font(18)
            font_sm = load_font(15)
            font_xs = load_font(11)

            show_boot(disp)

            draw_status(disp, font_big, font_med, font_sm, font_xs)

        except KeyboardInterrupt:
            try:
                bl.value = False
                img = Image.new("RGB", (SCREEN_W, SCREEN_H), BLACK)
                disp.image(img)
            except Exception:
                pass
            break

        except Exception as e:
            print(f"Fatal display error: {e}")
            time.sleep(5)


if __name__ == '__main__':
    main()