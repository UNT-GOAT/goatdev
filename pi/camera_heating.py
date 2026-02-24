import RPi.GPIO as GPIO # type: ignore
import time

# === CONFIGURATION ===
TURN_ON_BELOW = 40    # Heater ON when temp drops below (°F)
TURN_OFF_ABOVE = 70   # Heater OFF when temp rises above (°F)
CHECK_INTERVAL = 5   # Seconds between checks

# Hardcoded sensor-to-camera mappings
CAMERAS = {
    'camera1': {
        'sensor': '/sys/bus/w1/devices/28-0000006d3eba/temperature',
        'heater_pin': 17,  # GPIO 17, physical pin 11
    },
    'camera2': {
        'sensor': '/sys/bus/w1/devices/28-0000007047ea/temperature',
        'heater_pin': 5,  # GPIO 5, physical pin 29
    },
    'camera3': {
        'sensor': '/sys/bus/w1/devices/28-0000007193ed/temperature',
        'heater_pin': 6,  # GPIO 6, physical pin 31
    },
}

# === SETUP ===
GPIO.setmode(GPIO.BCM)
for name, cfg in CAMERAS.items():
    GPIO.setup(cfg['heater_pin'], GPIO.OUT)
    GPIO.output(cfg['heater_pin'], GPIO.LOW)

def read_temp_f(sensor_path):
    try:
        with open(sensor_path, 'r') as f:
            temp_c = int(f.read().strip()) / 1000.0
            temp_f = temp_c * 9.0 / 5.0 + 32.0
            return temp_f
    except Exception as e:
        print(f"Sensor error: {e}")
        return None

# === MAIN LOOP ===
print("Heater control starting...")
print(f"ON below {TURN_ON_BELOW}°F, OFF above {TURN_OFF_ABOVE}°F")

# Verify all sensors are reachable at startup
for name, cfg in CAMERAS.items():
    temp = read_temp_f(cfg['sensor'])
    if temp is not None:
        print(f"  {name}: sensor OK ({temp:.1f}°F)")
    else:
        print(f"  {name}: WARNING - sensor not responding!")

try:
    while True:
        for name, cfg in CAMERAS.items():
            temp = read_temp_f(cfg['sensor'])
            pin = cfg['heater_pin']

            if temp is None:
                print(f"{name}: SENSOR READ FAILED - skipping")
                continue

            state = GPIO.input(pin)

            if temp < TURN_ON_BELOW and not state:
                GPIO.output(pin, GPIO.HIGH)
                print(f"{name}: {temp:.1f}°F - HEATER ON")
            elif temp > TURN_OFF_ABOVE and state:
                GPIO.output(pin, GPIO.LOW)
                print(f"{name}: {temp:.1f}°F - HEATER OFF")
            else:
                s = "ON" if state else "OFF"
                print(f"{name}: {temp:.1f}°F - heater {s}")

        print("-" * 40)
        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("Shutting down...")
    for cfg in CAMERAS.values():
        GPIO.output(cfg['heater_pin'], GPIO.LOW)
    GPIO.cleanup()
    print("All heaters off.")