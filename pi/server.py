"""
Pi Connectivity Test Server
"""

from flask import Flask, jsonify
import subprocess
import requests
from datetime import datetime

app = Flask(__name__)

EC2_IP = '3.16.96.182'
EC2_API = f'http://{EC2_IP}:8000'


import sys
sys.path.insert(0, '/home/pi/goat-capture')
from logger.pi_cloudwatch import SimpleLogger
log = SimpleLogger('pi/capture')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'message': 'Pi is alive!'
    })


@app.route('/test')
def test_connectivity():
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Test 1: Internet (ping Google DNS)
    log("Testing internet...")
    try:
        result = subprocess.run(['ping', '-c', '3', '8.8.8.8'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'avg' in line:
                    results['tests']['internet'] = {'status': 'ok', 'latency_ms': line.split('/')[4]}
                    break
        else:
            results['tests']['internet'] = {'status': 'error', 'error': result.stderr[:200]}
    except Exception as e:
        results['tests']['internet'] = {'status': 'error', 'error': str(e)}
    
    # Test 2: Ping EC2
    log(f"Pinging EC2...")
    try:
        result = subprocess.run(['ping', '-c', '3', EC2_IP], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'avg' in line:
                    results['tests']['ec2_ping'] = {'status': 'ok', 'latency_ms': line.split('/')[4]}
                    break
        else:
            results['tests']['ec2_ping'] = {'status': 'error', 'error': result.stderr[:200]}
    except Exception as e:
        results['tests']['ec2_ping'] = {'status': 'error', 'error': str(e)}
    
    # Test 3: EC2 API
    log("Testing EC2 API...")
    try:
        response = requests.get(f'{EC2_API}/health', timeout=10)
        results['tests']['ec2_api'] = {
            'status': 'ok' if response.status_code == 200 else 'error',
            'status_code': response.status_code,
            'response': response.json() if response.status_code == 200 else response.text[:200]
        }
    except requests.exceptions.ConnectionError:
        results['tests']['ec2_api'] = {'status': 'error', 'error': 'Connection refused'}
    except Exception as e:
        results['tests']['ec2_api'] = {'status': 'error', 'error': str(e)}
    
    all_ok = all(t.get('status') == 'ok' for t in results['tests'].values())
    results['summary'] = 'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'
    
    log(f"Done: {results['summary']}")
    return jsonify(results)


if __name__ == '__main__':
    log("Pi Test Server starting on port 5000")
    log(f"EC2 target: {EC2_API}")
    app.run(host='0.0.0.0', port=5000)