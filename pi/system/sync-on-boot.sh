#!/bin/bash
set -e

cd /home/pi/goatdev
git fetch origin
git reset --hard origin/main

source /home/pi/venv/bin/activate
pip install -r pi/requirements.txt --quiet

sudo bash /home/pi/goatdev/pi/system/deploy.sh

echo "Boot sync completed at $(date)" >> /home/pi/boot-sync.log
git log -1 --oneline >> /home/pi/boot-sync.log