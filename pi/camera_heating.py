"""
GOAT-PI Heater Control Service

Controls 3 MOSFET-driven heater circuits based on DS18B20 temperature readings.
Logs all state changes and heartbeats to CloudWatch under /goatdev -> heating.

Thresholds: ON below 40°F, OFF above 70°F
Fail-safe: If sensor fails 10 consecutive reads, heater forced ON
Safety: If forced-on for 2+ hours with no sensor recovery, logs CRITICAL

Remote Override API:
  GET  http://pi-ip:5002/status        - All heater/sensor states
  POST http://pi-ip:5002/override      - Force heater on/off
       Body: {"camera": "camera1", "state": "on"|"off"|"auto"}
  GET  http://pi-ip:5002/history       - Recent state changes

GPIO Pins:
  camera1: GPIO 17 (pin 11)
  camera2: GPIO 5  (pin 29)
  camera3: GPIO 6  (pin 31)

Sensor IDs:
  camera1: 28-0000006d3eba
  camera2: 28-0000007047ea
  camera3: 28-0000007193ed
"""

import RPi.GPIO as GPIO # type: ignore
import time
import sys
import os
import json
import threading
from datetime import datetime, timedelta
from collections import deque
from flask_cors import CORS

sys.path.insert(0, '/home/pi/goatdev/pi')
from logger.pi_cloudwatch import Logger

# === CONFIGURATION ===
TURN_ON_BELOW = 40      # °F — heater ON when temp drops below
TURN_OFF_ABOVE = 70     # °F — heater OFF when temp rises above
CHECK_INTERVAL = 2      # Seconds between sensor reads
HEARTBEAT_INTERVAL = 300 # Seconds between heartbeat logs (5 min)
FAIL_THRESHOLD = 10     # Consecutive failures before fail-safe ON
FAILSAFE_MAX_HOURS = 2  # Log CRITICAL if forced-on this long
BOGUS_TEMP_C = 85.0     # DS18B20 power-on default — filter this out

CAMERAS = {
    'camera1': {
        'sensor': '/sys/bus/w1/devices/28-0000006d3eba/temperature',
        'heater_pin': 5,
    },
    'camera2': {
        'sensor': '/sys/bus/w1/devices/28-0000007047ea/temperature',
        'heater_pin': 6,
    },
    'camera3': {
        'sensor': '/sys/bus/w1/devices/28-0000007193ed/temperature',
        'heater_pin': 17,
    },
}

API_PORT = 5002

# === STATE TRACKING ===
state = {}
history = deque(maxlen=100)
log = Logger('heating')


def init_state():
    """Initialize tracking state for each camera."""
    for name in CAMERAS:
        state[name] = {
            'heater_on': False,
            'temp_f': None,
            'last_good_temp': None,
            'fail_count': 0,
            'failsafe_on': False,
            'failsafe_since': None,
            'override': 'auto',
            'last_change': None,
        }


def log_history(camera, event, **kwargs):
    """Record a state change for the history API."""
    entry = {
        'time': datetime.now().isoformat(),
        'camera': camera,
        'event': event,
        **kwargs,
    }
    history.append(entry)


# === SENSOR READING ===
def read_temp_f(sensor_path):
    """Read DS18B20 sensor. Returns °F or None on failure."""
    try:
        with open(sensor_path, 'r') as f:
            raw = f.read().strip()
            if not raw:
                return None
            temp_c = int(raw) / 1000.0

            # Filter bogus 85°C power-on reading
            if abs(temp_c - BOGUS_TEMP_C) < 0.5:
                return None

            return round(temp_c * 9.0 / 5.0 + 32.0, 1)
    except Exception:
        return None


# === HEATER CONTROL ===
def set_heater(name, on):
    """Set heater GPIO and verify."""
    pin = CAMERAS[name]['heater_pin']
    GPIO.output(pin, GPIO.HIGH if on else GPIO.LOW)

    # Verify GPIO actually changed
    actual = GPIO.input(pin)
    expected = 1 if on else 0
    if actual != expected:
        log.error(f'gpio:{name}', 'GPIO verify failed',
                  pin=pin, expected=expected, actual=actual)

    state[name]['heater_on'] = on
    state[name]['last_change'] = datetime.now().isoformat()


def control_loop():
    """Main heater control loop."""
    last_heartbeat = 0

    while True:
        now = time.time()

        for name, cfg in CAMERAS.items():
            s = state[name]
            temp = read_temp_f(cfg['sensor'])

            # --- Handle override ---
            if s['override'] != 'auto':
                want_on = s['override'] == 'on'
                if s['heater_on'] != want_on:
                    set_heater(name, want_on)
                    action = 'ON' if want_on else 'OFF'
                    log.info(f'override:{name}', f'Heater forced {action}',
                             override=s['override'])
                    log_history(name, f'override_{action.lower()}')
                # Still read sensor for monitoring
                if temp is not None:
                    s['temp_f'] = temp
                    s['last_good_temp'] = temp
                    s['fail_count'] = 0
                continue

            # --- Handle sensor read ---
            if temp is not None:
                s['temp_f'] = temp
                s['last_good_temp'] = temp

                # Recover from failsafe
                if s['failsafe_on']:
                    log.info(f'failsafe:{name}', 'Sensor recovered, resuming auto control',
                             temp=temp, was_forced_minutes=round(_failsafe_minutes(s)))
                    log_history(name, 'failsafe_recovered', temp=temp)
                    s['failsafe_on'] = False
                    s['failsafe_since'] = None

                s['fail_count'] = 0

                # --- Thermostat logic ---
                if temp < TURN_ON_BELOW and not s['heater_on']:
                    set_heater(name, True)
                    log.info(f'heater:{name}', 'Heater ON', temp=temp,
                             threshold=TURN_ON_BELOW)
                    log_history(name, 'heater_on', temp=temp)

                elif temp > TURN_OFF_ABOVE and s['heater_on']:
                    set_heater(name, False)
                    log.info(f'heater:{name}', 'Heater OFF', temp=temp,
                             threshold=TURN_OFF_ABOVE)
                    log_history(name, 'heater_off', temp=temp)

            else:
                # Sensor failed
                s['temp_f'] = None
                s['fail_count'] += 1

                if s['fail_count'] == FAIL_THRESHOLD:
                    # Engage fail-safe: force heater ON
                    log.error(f'failsafe:{name}',
                              f'Sensor failed {FAIL_THRESHOLD} consecutive reads, forcing heater ON',
                              last_good_temp=s['last_good_temp'],
                              fail_count=s['fail_count'])
                    log_history(name, 'failsafe_on',
                                last_good_temp=s['last_good_temp'])
                    set_heater(name, True)
                    s['failsafe_on'] = True
                    s['failsafe_since'] = datetime.now()

                elif s['fail_count'] > FAIL_THRESHOLD and s['failsafe_on']:
                    # Check if forced-on too long
                    minutes = _failsafe_minutes(s)
                    if minutes > FAILSAFE_MAX_HOURS * 60:
                        if s['fail_count'] % 30 == 0:  # Don't spam
                            log.critical(f'failsafe:{name}',
                                         f'Sensor dead for {minutes:.0f} min, heater still forced ON',
                                         last_good_temp=s['last_good_temp'])

                elif s['fail_count'] % 10 == 0 and s['fail_count'] > 0:
                    log.warn(f'sensor:{name}',
                             f'Sensor read failed ({s["fail_count"]} consecutive)',
                             last_good_temp=s['last_good_temp'])

        # --- Heartbeat ---
        if now - last_heartbeat >= HEARTBEAT_INTERVAL:
            temps = {n: s['temp_f'] for n, s in state.items()}
            heaters = {n: 'ON' if s['heater_on'] else 'OFF' for n, s in state.items()}
            overrides = {n: s['override'] for n, s in state.items() if s['override'] != 'auto'}
            failsafes = {n: True for n, s in state.items() if s['failsafe_on']}

            log.info('heartbeat', 'Status check',
                     temps=json.dumps(temps),
                     heaters=json.dumps(heaters),
                     overrides=json.dumps(overrides) if overrides else None,
                     failsafes=json.dumps(failsafes) if failsafes else None)
            last_heartbeat = now

        time.sleep(CHECK_INTERVAL)


def _failsafe_minutes(s):
    """Minutes since failsafe was engaged."""
    if s['failsafe_since']:
        return (datetime.now() - s['failsafe_since']).total_seconds() / 60
    return 0


# === OVERRIDE API ===
def start_api():
    """Simple Flask API for remote heater control."""
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    CORS(app)

    @app.route('/status')
    def api_status():
        result = {}
        for name, s in state.items():
            result[name] = {
                'temp_f': s['temp_f'],
                'heater_on': s['heater_on'],
                'override': s['override'],
                'failsafe': s['failsafe_on'],
                'fail_count': s['fail_count'],
                'last_good_temp': s['last_good_temp'],
                'last_change': s['last_change'],
            }
        return jsonify({
            'status': 'ok',
            'thresholds': {'on_below': TURN_ON_BELOW, 'off_above': TURN_OFF_ABOVE},
            'cameras': result,
            'timestamp': datetime.now().isoformat(),
        })

    @app.route('/override', methods=['POST'])
    def api_override():
        data = request.get_json()
        camera = data.get('camera')
        new_state = data.get('state')

        if camera not in state:
            return jsonify({'error': f'Unknown camera: {camera}'}), 400
        if new_state not in ('on', 'off', 'auto'):
            return jsonify({'error': 'State must be on, off, or auto'}), 400

        old = state[camera]['override']
        state[camera]['override'] = new_state

        log.info(f'override:{camera}', f'Override changed: {old} -> {new_state}')
        log_history(camera, 'override_set', old=old, new=new_state)

        return jsonify({'ok': True, 'camera': camera, 'override': new_state})

    @app.route('/history')
    def api_history():
        return jsonify({'history': list(history)})

    # Suppress Flask startup banner
    import logging as _logging
    _logging.getLogger('werkzeug').setLevel(_logging.WARNING)

    app.run(host='0.0.0.0', port=API_PORT, threaded=True)


# === STARTUP ===
def startup_check():
    """Verify all sensors and GPIOs at startup."""
    log.info('startup', 'Heater control starting',
             on_below=TURN_ON_BELOW, off_above=TURN_OFF_ABOVE,
             check_interval=CHECK_INTERVAL)

    all_ok = True
    for name, cfg in CAMERAS.items():
        # Test sensor
        temp = read_temp_f(cfg['sensor'])
        if temp is not None:
            state[name]['temp_f'] = temp
            state[name]['last_good_temp'] = temp
            log.info(f'startup:{name}', 'Sensor OK', temp=temp)
        else:
            log.error(f'startup:{name}', 'Sensor NOT responding',
                      path=cfg['sensor'])
            all_ok = False

        # Test GPIO toggle
        pin = cfg['heater_pin']
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.05)
        high_ok = GPIO.input(pin) == 1
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.05)
        low_ok = GPIO.input(pin) == 0

        if high_ok and low_ok:
            log.info(f'startup:{name}', 'GPIO OK', pin=pin)
        else:
            log.error(f'startup:{name}', 'GPIO toggle failed',
                      pin=pin, high_ok=high_ok, low_ok=low_ok)
            all_ok = False

    if all_ok:
        log.info('startup', 'All systems OK')
    else:
        log.warn('startup', 'Some systems failed — starting anyway')

    log_history('system', 'startup', all_ok=all_ok)


# === MAIN ===
def main():
    GPIO.setmode(GPIO.BCM)
    for name, cfg in CAMERAS.items():
        GPIO.setup(cfg['heater_pin'], GPIO.OUT)
        GPIO.output(cfg['heater_pin'], GPIO.LOW)

    init_state()
    startup_check()

    # Start API in background thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    log.info('startup', 'Override API started', port=API_PORT)

    try:
        control_loop()
    except KeyboardInterrupt:
        log.info('shutdown', 'Shutting down — turning all heaters off')
        for name in CAMERAS:
            set_heater(name, False)
        # Don't call GPIO.cleanup() — it kills display pins


if __name__ == '__main__':
    main()