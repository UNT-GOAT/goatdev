#!/usr/bin/env python3
"""
Heartbeat script - runs via cron every 2 minutes
Logs errors when unhealthy, logs OK once when recovered
"""

import subprocess
import sys
import os

sys.path.insert(0, '/home/pi/goat-capture')
from logger.pi_cloudwatch import Logger

log = Logger('pi/heartbeat')
STATE_FILE = '/tmp/heartbeat_unhealthy'

def check_service(name: str) -> bool:
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', name],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() == 'active'
    except:
        return False

def check_port(port: int) -> bool:
    try:
        result = subprocess.run(
            ['ss', '-tln', f'sport = :{port}'],
            capture_output=True, text=True, timeout=5
        )
        return str(port) in result.stdout
    except:
        return False

def main():
    services = {
        'goat-prod': {'service': 'goat-prod', 'port': 5000},
    }
    
    issues = []
    
    for name, config in services.items():
        service_up = check_service(config['service'])
        port_up = check_port(config['port'])
        
        if not service_up:
            issues.append(f"{name}: service down")
        elif not port_up:
            issues.append(f"{name}: service active but port {config['port']} not listening")
    
    was_unhealthy = os.path.exists(STATE_FILE)
    
    if issues:
        # Create state file to track we're unhealthy
        open(STATE_FILE, 'w').close()
        log.error('heartbeat', 'Services unhealthy',
                  issues='; '.join(issues),
                  fix='SSH to Pi and run: sudo systemctl restart goat-prod')
    elif was_unhealthy:
        # Just recovered - log once and clear state
        os.remove(STATE_FILE)
        log.info('heartbeat', 'Services recovered')

if __name__ == '__main__':
    main()