"""
Heartbeat script - runs via cron every 2 minutes
Only logs if Pi is up but services are down
"""

import subprocess
import sys
sys.path.insert(0, '/home/pi/goat-capture')
from logger.pi_cloudwatch import Logger

log = Logger('pi/heartbeat')

def check_service(name: str) -> bool:
    """Check if a systemd service is active."""
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', name],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() == 'active'
    except:
        return False

def check_port(port: int) -> bool:
    """Check if something is listening on a port."""
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
        'goat-training': {'service': 'goat-training', 'port': 5001},
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
    
    if issues:
        # Something is wrong - log error
        log.error('heartbeat', 'Services unhealthy',
                  issues='; '.join(issues),
                  fix='SSH to Pi and run: sudo systemctl restart goat-training goat-prod')
    else:
        # All good - log quietly (for CloudWatch alarm to track)
        log.info('heartbeat', 'All services healthy')

if __name__ == '__main__':
    main()