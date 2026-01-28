#!/bin/bash
# EC2 First-Time Setup Script
# Run this once after creating the EC2 instance
#
# Usage: ssh into EC2 and run:
#   curl -fsSL https://raw.githubusercontent.com/YOUR_REPO/main/scripts/ec2-setup.sh | bash

set -e

echo "============================================"
echo "  Goat API - EC2 Setup"
echo "============================================"

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "Installing Python and system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    nginx \
    certbot \
    python3-certbot-nginx

# Create app directory
echo "Setting up application directory..."
cd /home/ubuntu
if [ -d "goat-api" ]; then
    echo "goat-api directory already exists, pulling latest..."
    cd goat-api
    git pull origin main
else
    echo "Cloning repository..."
    # TODO: Replace with your actual repo URL
    git clone https://github.com/YOUR_USERNAME/goatdev.git goat-api
    cd goat-api
fi

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit /home/ubuntu/goat-api/.env with your actual values!"
    echo ""
fi

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/goat-api.service > /dev/null <<EOF
[Unit]
Description=Goat Measurement API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/goat-api
Environment=PATH=/home/ubuntu/goat-api/venv/bin
EnvironmentFile=/home/ubuntu/goat-api/.env
ExecStart=/home/ubuntu/goat-api/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
echo "Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable goat-api
sudo systemctl start goat-api

# Check service status
echo ""
echo "Service status:"
sudo systemctl status goat-api --no-pager

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Edit /home/ubuntu/goat-api/.env with your database password"
echo "2. Restart the service: sudo systemctl restart goat-api"
echo "3. Test the API: curl http://localhost:8000/health"
echo ""
echo "Useful commands:"
echo "  View logs:     journalctl -u goat-api -f"
echo "  Restart:       sudo systemctl restart goat-api"
echo "  Stop:          sudo systemctl stop goat-api"
echo ""
