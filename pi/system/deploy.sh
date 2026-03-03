#!/bin/bash
set -e
REPO_ROOT="/home/pi/goatdev"
SYSTEM_DIR="$REPO_ROOT/pi/system"

echo "=== GOAT-PI System Deploy ==="

echo "Installing systemd services..."
for service in goat-prod.service goat-training.service goat-display.service camera-heating.service camera-proxy.service git-sync-on-boot.service; do
    if [ -f "$SYSTEM_DIR/$service" ]; then
        cp "$SYSTEM_DIR/$service" /etc/systemd/system/
        echo "  ✓ $service"
    fi
done

echo "Installing udev rules..."
if [ -f "$SYSTEM_DIR/99-cameras.rules" ]; then
    cp "$SYSTEM_DIR/99-cameras.rules" /etc/udev/rules.d/
    udevadm control --reload-rules
    echo "  ✓ 99-cameras.rules"
fi

echo "Installing cron jobs..."
if [ -f "$SYSTEM_DIR/goat-heartbeat.cron" ]; then
    crontab -u pi "$SYSTEM_DIR/goat-heartbeat.cron"
    echo "  ✓ goat-heartbeat.cron"
fi

echo "Installing boot sync script..."
if [ -f "$SYSTEM_DIR/sync-on-boot.sh" ]; then
    cp "$SYSTEM_DIR/sync-on-boot.sh" /home/pi/sync-on-boot.sh
    chmod +x /home/pi/sync-on-boot.sh
    echo "  ✓ sync-on-boot.sh"
fi

echo "Installing Caddyfile..."
if [ -f "$SYSTEM_DIR/Caddyfile" ]; then
    sudo cp "$SYSTEM_DIR/Caddyfile" /etc/caddy/Caddyfile
    echo "  ✓ Caddyfile"
    sudo systemctl reload caddy || sudo systemctl restart caddy
fi

systemctl daemon-reload
systemctl enable goat-prod.service
systemctl enable goat-training.service
systemctl enable goat-display.service
systemctl enable camera-proxy.service
systemctl enable git-sync-on-boot.service

echo "=== Deploy complete ==="