"""
GOAT-PI Heater Control Service

Controls 3 MOSFET-driven heater circuits based on DS18B20 temperature readings.
Logs all state changes and heartbeats to CloudWatch under /goatdev -> heating.

Thresholds: ON below 40°F, OFF above 70°F
Fail-safe: If sensor fails 30 consecutive reads, heater forced ON only if last
           known temp was below safety threshold (otherwise stays off).
Safety: If forced-on for 2+ hours with no sensor recovery, logs CRITICAL

Sensor filtering:
  - Rejects DS18B20 85°C power-on default
  - Rejects temps outside -20°F to 150°F (physically impossible in enclosure)
  - Rejects spikes >10°F change per read cycle (EMI noise)
  - Requires 3 consecutive agreeing reads before acting on large changes

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
FAIL_THRESHOLD = 30     # Consecutive failures before fail-safe (60 seconds at 2s interval)
FAILSAFE_MAX_HOURS = 2  # Log CRITICAL if forced-on this long
BOGUS_TEMP_C = 85.0     # DS18B20 power-on default — filter this out

# Sensor sanity filtering
TEMP_MIN_F = -20.0      # Reject anything below (physically impossible in enclosure)
TEMP_MAX_F = 150.0      # Reject anything above (physically impossible in enclosure)
SPIKE_THRESHOLD_F = 10.0 # Max °F change per read cycle (real temp can't jump this fast)
CONFIRM_READS = 3        # Consecutive agreeing reads required after a rejected spike

# Failsafe behavior
FAILSAFE_COLD_THRESHOLD = 50.0  # Only failsafe-ON if last known temp was below this

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
            # Spike detection state
            'spike_confirm_count': 0,
            'spike_candidate': None,
            'rejected_spike_count': 0,
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
def read_temp_raw(sensor_path):
    """Read DS18B20 sensor. Returns °F or None on hardware failure."""
    try:
        with open(sensor_path, 'r') as f:
            raw = f.read().strip()
            if not raw:
                return None
            temp_c = int(raw) / 1000.0

            # Filter bogus 85°C power-on reading
            if abs(temp_c - BOGUS_TEMP_C) < 0.5:
                return None

            temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)

            # Reject physically impossible values
            if temp_f < TEMP_MIN_F or temp_f > TEMP_MAX_F:
                return None

            return temp_f
    except Exception:
        return None


def filter_temp(name, raw_temp):
    """Apply spike detection filter. Returns accepted temp or None.

    If a reading jumps more than SPIKE_THRESHOLD_F from the last known good
    temp, it's likely EMI noise. We reject it and require CONFIRM_READS
    consecutive readings near the new value before accepting it.
    This prevents single-read glitches from triggering heater changes.
    """
    s = state[name]

    if raw_temp is None:
        # Do NOT reset spike tracking on transient read failures.
        # Just report no valid reading; spike confirmation will continue
        # on the next successful read.
        return None

    last = s['last_good_temp']

    # First reading ever — accept it
    if last is None:
        return raw_temp

    delta = abs(raw_temp - last)

    if delta <= SPIKE_THRESHOLD_F:
        # Normal reading, within expected range — reset spike tracking
        s['spike_confirm_count'] = 0
        s['spike_candidate'] = None
        return raw_temp
    else:
        # Suspicious jump — could be EMI or real temp change
        if s['spike_candidate'] is not None and abs(raw_temp - s['spike_candidate']) <= SPIKE_THRESHOLD_F:
            # Consistent with previous spike candidate
            s['spike_confirm_count'] += 1
            if s['spike_confirm_count'] >= CONFIRM_READS:
                # Confirmed — this is a real temp change, not noise
                s['spike_confirm_count'] = 0
                s['spike_candidate'] = None
                log.info(f'sensor:{name}',
                         f'Large temp change confirmed after {CONFIRM_READS} reads',
                         old_temp=last, new_temp=raw_temp)
                return raw_temp
            else:
                # Still confirming
                return None
        else:
            # New spike candidate
            s['spike_candidate'] = raw_temp
            s['spike_confirm_count'] = 1
            s['rejected_spike_count'] += 1
            if s['rejected_spike_count'] % 10 == 0:
                log.warn(f'sensor:{name}',
                         f'Rejected {s["rejected_spike_count"]} spikes total',
                         last_good=last, spike_value=raw_temp)
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
            raw_temp = read_temp_raw(cfg['sensor'])
            temp = filter_temp(name, raw_temp)

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
                # Sensor failed or reading was filtered out
                s['temp_f'] = None
                s['fail_count'] += 1

                if s['fail_count'] == FAIL_THRESHOLD:
                    # Engage fail-safe — but only heat if it was actually cold
                    last_temp = s['last_good_temp']

                    if last_temp is not None and last_temp >= FAILSAFE_COLD_THRESHOLD:
                        # Last known temp was warm — no need to heat
                        log.warn(f'failsafe:{name}',
                                 f'Sensor failed {FAIL_THRESHOLD} reads but last temp was warm, heater stays OFF',
                                 last_good_temp=last_temp,
                                 threshold=FAILSAFE_COLD_THRESHOLD)
                        log_history(name, 'failsafe_warm_skip',
                                    last_good_temp=last_temp)
                        s['failsafe_on'] = True
                        s['failsafe_since'] = datetime.now()
                        # Don't turn heater on
                    else:
                        # Last known temp was cold (or unknown) — heat to be safe
                        log.error(f'failsafe:{name}',
                                  f'Sensor failed {FAIL_THRESHOLD} reads, last temp was cold, forcing heater ON',
                                  last_good_temp=last_temp,
                                  fail_count=s['fail_count'])
                        log_history(name, 'failsafe_on',
                                    last_good_temp=last_temp)
                        set_heater(name, True)
                        s['failsafe_on'] = True
                        s['failsafe_since'] = datetime.now()

                elif s['fail_count'] > FAIL_THRESHOLD and s['failsafe_on']:
                    # Check if forced-on too long
                    minutes = _failsafe_minutes(s)
                    if minutes > FAILSAFE_MAX_HOURS * 60:
                        if s['fail_count'] % 30 == 0:  # Don't spam
                            log.critical(f'failsafe:{name}',
                                         f'Sensor dead for {minutes:.0f} min, heater still in failsafe',
                                         last_good_temp=s['last_good_temp'],
                                         heater_on=s['heater_on'])

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
            spikes = {n: s['rejected_spike_count'] for n, s in state.items() if s['rejected_spike_count'] > 0}

            log.info('heartbeat', 'Status check',
                     temps=json.dumps(temps),
                     heaters=json.dumps(heaters),
                     overrides=json.dumps(overrides) if overrides else None,
                     failsafes=json.dumps(failsafes) if failsafes else None,
                     rejected_spikes=json.dumps(spikes) if spikes else None)
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
                'rejected_spikes': s['rejected_spike_count'],
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
             check_interval=CHECK_INTERVAL,
             fail_threshold=FAIL_THRESHOLD,
             spike_threshold=SPIKE_THRESHOLD_F,
             failsafe_cold_threshold=FAILSAFE_COLD_THRESHOLD)

    all_ok = True
    for name, cfg in CAMERAS.items():
        # Test sensor — use raw read for startup (no spike filter yet)
        temp = read_temp_raw(cfg['sensor'])
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