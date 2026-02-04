"""
Pi Production Server
Connectivity testing and health checks
"""

from flask import Flask, jsonify
import subprocess
import requests
import time
import os
from datetime import datetime

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

EC2_IP = os.environ.get('EC2_IP', '3.16.96.182')
EC2_API = f'http://{EC2_IP}:8000'
PING_TIMEOUT_SEC = 15
REQUEST_TIMEOUT_SEC = 10

# ============================================================
# LOGGING
# ============================================================

import sys
sys.path.insert(0, '/home/pi/goat-capture')
from logger.pi_cloudwatch import Logger

log = Logger('pi/prod')

# ============================================================
# HELPERS
# ============================================================

def ping_host(host: str, count: int = 3) -> dict:
    """Ping a host and return result with latency."""
    try:
        result = subprocess.run(
            ['ping', '-c', str(count), '-W', '5', host],
            capture_output=True, text=True, timeout=PING_TIMEOUT_SEC
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'avg' in line:
                    latency = line.split('/')[4]
                    return {'status': 'ok', 'latency_ms': latency}
            return {'status': 'ok', 'latency_ms': 'unknown'}
        else:
            return {
                'status': 'error',
                'error': 'Ping failed',
                'stderr': result.stderr[:200] if result.stderr else None
            }
    except subprocess.TimeoutExpired:
        return {'status': 'error', 'error': 'Timeout'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def check_ec2_api() -> dict:
    """Check EC2 API health endpoint."""
    try:
        response = requests.get(f'{EC2_API}/health', timeout=REQUEST_TIMEOUT_SEC)
        if response.status_code == 200:
            return {'status': 'ok', 'response': response.json()}
        else:
            return {
                'status': 'error',
                'status_code': response.status_code,
                'error': response.text[:100]
            }
    except requests.exceptions.ConnectTimeout:
        return {'status': 'error', 'error': 'Connection timeout'}
    except requests.exceptions.ConnectionError:
        return {'status': 'error', 'error': 'Connection refused'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def get_system_info() -> dict:
    """Get Pi system diagnostics."""
    info = {}
    
    # CPU temp
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            info['cpu_temp_c'] = round(int(f.read().strip()) / 1000, 1)
    except:
        info['cpu_temp_c'] = None
    
    # Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    info['mem_available_mb'] = int(line.split()[1]) // 1024
                    break
    except:
        info['mem_available_mb'] = None
    
    # Load
    try:
        info['load_avg'] = round(os.getloadavg()[0], 2)
    except:
        info['load_avg'] = None
    
    # Uptime
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_sec = float(f.read().split()[0])
            info['uptime_hours'] = round(uptime_sec / 3600, 1)
    except:
        info['uptime_hours'] = None
    
    return info


# ============================================================
# ROUTES
# ============================================================

@app.route('/health')
def health():
    """Quick health check."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'ec2_target': EC2_API
    })


@app.route('/diagnostics')
def diagnostics():
    """Detailed system diagnostics."""
    log.info('diag', 'Running diagnostics')
    
    diag = {
        'timestamp': datetime.now().isoformat(),
        'system': get_system_info(),
        'ec2_target': EC2_API
    }
    
    log.info('diag', 'Complete',
             cpu_temp=diag['system'].get('cpu_temp_c'),
             mem_mb=diag['system'].get('mem_available_mb'),
             load=diag['system'].get('load_avg'))
    
    return jsonify(diag)


@app.route('/test')
def test_connectivity():
    """Full connectivity test to internet and EC2."""
    log.info('test', 'Connectivity test started')
    start = time.time()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Test 1: Internet (ping Google DNS)
    log.info('test:internet', 'Pinging 8.8.8.8')
    internet = ping_host('8.8.8.8')
    results['tests']['internet'] = internet
    
    if internet['status'] == 'ok':
        log.info('test:internet', 'OK', latency_ms=internet.get('latency_ms'))
    else:
        internet['fix'] = 'Check ethernet cable or WiFi connection'
        log.error('test:internet', 'Failed',
                  error=internet.get('error'),
                  fix=internet['fix'])
    
    # Test 2: EC2 Ping
    log.info('test:ec2_ping', 'Pinging EC2', ip=EC2_IP)
    ec2_ping = ping_host(EC2_IP)
    results['tests']['ec2_ping'] = ec2_ping
    
    if ec2_ping['status'] == 'ok':
        log.info('test:ec2_ping', 'OK', latency_ms=ec2_ping.get('latency_ms'))
    else:
        ec2_ping['fix'] = 'Check EC2 is running and security group allows ICMP'
        log.error('test:ec2_ping', 'Failed',
                  error=ec2_ping.get('error'),
                  fix=ec2_ping['fix'],
                  ip=EC2_IP)
    
    # Test 3: EC2 API
    log.info('test:ec2_api', 'Checking EC2 API', url=EC2_API)
    ec2_api = check_ec2_api()
    results['tests']['ec2_api'] = ec2_api
    
    if ec2_api['status'] == 'ok':
        log.info('test:ec2_api', 'OK')
    else:
        if ec2_api.get('error') == 'Connection refused':
            ec2_api['fix'] = 'EC2 API container may be down. SSH to EC2 and run: docker ps'
        elif ec2_api.get('error') == 'Connection timeout':
            ec2_api['fix'] = 'Check EC2 security group allows port 8000 from Pi IP'
        else:
            ec2_api['fix'] = 'Check EC2 API logs in CloudWatch: /goatdev/ec2/api'
        log.error('test:ec2_api', 'Failed',
                  error=ec2_api.get('error'),
                  fix=ec2_api['fix'],
                  url=EC2_API)
    
    # Summary
    all_ok = all(t.get('status') == 'ok' for t in results['tests'].values())
    results['summary'] = 'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'
    results['duration_sec'] = round(time.time() - start, 2)
    
    log.info('test', 'Complete',
             summary=results['summary'],
             passed=sum(1 for t in results['tests'].values() if t.get('status') == 'ok'),
             total=len(results['tests']),
             duration_sec=results['duration_sec'])
    
    return jsonify(results)


# ============================================================
# STARTUP
# ============================================================

def run_startup_checks():
    """Run all startup checks and log results."""
    log.info('startup', '=' * 50)
    log.info('startup', 'PROD PI SERVER STARTING')
    log.info('startup', 'Configuration', ec2_ip=EC2_IP, ec2_api=EC2_API)
    
    # System info
    sys_info = get_system_info()
    log.info('startup:system', 'System info',
             cpu_temp=sys_info.get('cpu_temp_c'),
             mem_mb=sys_info.get('mem_available_mb'),
             load=sys_info.get('load_avg'),
             uptime_hrs=sys_info.get('uptime_hours'))
    
    # Check for high CPU temp
    if sys_info.get('cpu_temp_c') and sys_info['cpu_temp_c'] > 70:
        log.warn('startup:system', 'CPU temperature high',
                 temp_c=sys_info['cpu_temp_c'],
                 fix='Check Pi ventilation')
    
    # Check for low memory
    if sys_info.get('mem_available_mb') and sys_info['mem_available_mb'] < 200:
        log.warn('startup:system', 'Low memory',
                 available_mb=sys_info['mem_available_mb'],
                 fix='Restart Pi or check for memory leaks')
    
    # Test internet
    log.info('startup:network', 'Testing internet connectivity')
    internet = ping_host('8.8.8.8', count=1)
    if internet['status'] == 'ok':
        log.info('startup:network', 'Internet OK', latency_ms=internet.get('latency_ms'))
    else:
        log.error('startup:network', 'No internet connection',
                  error=internet.get('error'),
                  fix='Check ethernet cable or WiFi')
    
    # Test EC2
    log.info('startup:ec2', 'Testing EC2 connectivity')
    ec2_ping = ping_host(EC2_IP, count=1)
    if ec2_ping['status'] == 'ok':
        log.info('startup:ec2', 'EC2 ping OK', latency_ms=ec2_ping.get('latency_ms'))
    else:
        log.warn('startup:ec2', 'EC2 ping failed',
                 error=ec2_ping.get('error'),
                 ip=EC2_IP)
    
    # Test EC2 API
    ec2_api = check_ec2_api()
    if ec2_api['status'] == 'ok':
        log.info('startup:ec2', 'EC2 API OK', url=EC2_API)
    else:
        log.warn('startup:ec2', 'EC2 API not reachable',
                 error=ec2_api.get('error'),
                 url=EC2_API,
                 fix='This may be OK if EC2 is still starting')
    
    log.info('startup', 'Startup checks complete')
    log.info('startup', '=' * 50)


if __name__ == '__main__':
    run_startup_checks()
    
    log.info('startup', 'Server listening', host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=5000)