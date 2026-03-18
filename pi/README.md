# Raspberry Pi Capture System

Raspberry Pi 4 running 6 services for goat image capture, (sending off to ec2)-grading, training data collection, camera management, environmental monitoring, and system health display. All services accessible remotely via Tailscale VPN and Caddy HTTPS proxy.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Raspberry Pi 4 (4GB)                          │
│                                                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐ │
│  │  Prod Server  │  │ Training Srvr │  │     Camera Proxy          │ │
│  │   Port 5000   │  │   Port 5001   │  │      Port 8080            │ │
│  │               │  │               │  │                           │ │
│  │  Capture +    │  │  20-img burst │  │  Owns all 3 USB cameras   │ │
│  │  EC2 grading  │  │  + S3 upload  │  │  Pair rotation (2 at a    │ │
│  │               │  │               │  │  time), ~8fps per cam     │ │
│  └──────┬────────┘  └──────┬────────┘  │  MJPEG streams + capture  │ │
│         │                  │           └─────────┬─────────────────┘ │
│         └──────────────────┴─────────────────────┘                   │
│                          HTTP to proxy                               │
│                                                                      │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────────┐  │
│  │ Heater Ctrl  │  │ Status Display│  │     Caddy Proxy           │  │
│  │  Port 5002   │  │  SPI LCD      │  │      Port 8443            │  │
│  │              │  │               │  │                           │  │
│  │  DS18B20     │  │  Server/cam/  │  │  HTTPS consolidation      │  │
│  │  thermostat  │  │  temp status  │  │  for Tailscale Funnel     │  │
│  │  + failsafe  │  │  + boot logo  │  │  → mobile browser         │  │
│  └──────────────┘  └───────────────┘  └───────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │              3x Arducam 16MP USB Cameras (udev)              │    │
│  │  /dev/camera_side    /dev/camera_top    /dev/camera_front    │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────┐  ┌───────────────┐                                 │
│  │  CloudWatch  │  │  Heartbeat    │                                 │
│  │  Logger      │  │  Cron (2min)  │                                 │
│  └──────────────┘  └───────────────┘                                 │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │ Tailscale VPN
                              ▼
                ┌──────────────────────────┐
                │      Remote Access       │
                │                          │
                │  SSH, curl, browser      │
                │  EC2 API ◄── grading     │
                │  S3      ◄── training    │
                └──────────────────────────┘
```

## Services

### Camera Proxy — Port 8080

Single process that owns all 3 USB cameras. The Pi 4's USB controller can't handle 3 concurrent full-resolution reads, so the proxy keeps all 3 cameras open but only reads from 2 at a time in rotating pairs. This yields ~8fps per camera at full 4656×3496 resolution with zero open/close overhead.

All other services (prod, training) get camera frames via HTTP from the proxy instead of touching USB devices directly. This eliminates all "device busy" conflicts.

Runs under gunicorn with gevent for concurrent MJPEG streaming to multiple clients.

| Endpoint                   | Method   | Description                       |
| -------------------------- | -------- | --------------------------------- |
| `/stream/<camera>`         | GET      | MJPEG preview stream (640×480)    |
| `/capture/<camera>`        | GET      | Latest full-res JPEG              |
| `/capture`                 | GET/POST | Capture all 3, save to /tmp       |
| `/status`                  | GET      | Per-camera health, FPS, frame age |
| `/focus/<camera>/<val>`    | GET      | Set manual focus                  |
| `/autofocus/<camera>/<on>` | GET      | Toggle autofocus                  |
| `/save`                    | POST     | Persist focus settings            |
| `/settings`                | GET      | Load saved focus settings         |

### Production Server — Port 5000

Single-shot capture for live grading. Captures one image per camera from the proxy, sends all three to the EC2 API, returns a grade result.

| Endpoint       | Method | Description                                        |
| -------------- | ------ | -------------------------------------------------- |
| `/health`      | GET    | Quick status check                                 |
| `/diagnostics` | GET    | Detailed system info                               |
| `/grade`       | POST   | Capture + grade (body: `serial_id`, `live_weight`) |
| `/grade/test`  | POST   | Grade with uploaded images (skip capture)          |
| `/test`        | GET    | Full connectivity test                             |
| `/status`      | GET    | Current capture state                              |
| `/cancel`      | POST   | Emergency stop                                     |

### Training Server — Port 5001

Multi-image capture for model training. Captures 20 images per camera at 1fps from the proxy, tars per camera, uploads to S3 with metadata.

| Endpoint       | Method | Description                                             |
| -------------- | ------ | ------------------------------------------------------- |
| `/health`      | GET    | Quick status check                                      |
| `/diagnostics` | GET    | Detailed system info                                    |
| `/record`      | POST   | Start capture (body: `goat_id`, `goat_data`, `is_test`) |
| `/status`      | GET    | Current capture state                                   |
| `/cancel`      | POST   | Emergency stop                                          |

### Heater Control — Port 5002

Thermostat for 3 MOSFET-driven heater circuits inside the camera enclosures. Reads DS18B20 temperature sensors every 2 seconds, heats below 40°F, stops above 70°F.

Includes EMI noise filtering (spike rejection, bogus 85°C detection, physically impossible value rejection) and smart failsafe that only heats on sensor failure if the last known temperature was actually cold. Sensor wires are also wrapped in EMI protection tape at the physcial camera enclosure wire entrance where they're all pinched together.

| Endpoint    | Method | Description                                        |
| ----------- | ------ | -------------------------------------------------- |
| `/status`   | GET    | All heater/sensor states                           |
| `/override` | POST   | Force heater on/off/auto (body: `camera`, `state`) |
| `/history`  | GET    | Recent state changes                               |

### Status Display

2.4" ILI9341 SPI LCD showing real-time system health. Boot sequence shows logo with fade-in, then transitions to status screen showing server health (including camera proxy), camera device status, temperature readings per enclosure, heater state, WiFi strength, and time.

### Caddy Proxy — Port 8443

Reverse proxy that consolidates all services behind a single HTTPS endpoint for Tailscale Funnel access from mobile browsers.

| Path               | Routes to              |
| ------------------ | ---------------------- |
| `/api/capture/*`   | Training server (5001) |
| `/api/heater/*`    | Heater control (5002)  |
| `/api/viewfocus/*` | Camera proxy (8080)    |
| Default            | Training server (5001) |

## Directory Structure

```
pi/
├── servers/
│   ├── camera_proxy.py        # Camera proxy service
│   ├── prod_server.py         # Production grading service
│   ├── training_server.py     # Training capture service
│   └── camera_heating.py      # Heater thermostat service
├── display/
│   ├── display.py             # LCD status display
│   └── boot_logo.png          # Boot screen logo
├── logger/
│   ├── pi_cloudwatch.py       # Shared CloudWatch logging module
│   └── pi_heartbeat_cron.py   # Heartbeat health check (cron)
├── system/
│   ├── camera-proxy.service   # Systemd unit (gunicorn + gevent)
│   ├── goat-prod.service      # Systemd unit
│   ├── goat-training.service  # Systemd unit
│   ├── camera-heating.service # Systemd unit
│   ├── goat-display.service   # Systemd unit
│   ├── setup-proxy.service    # Caddy systemd unit
│   ├── Caddyfile              # Caddy reverse proxy config
│   ├── 99-cameras.rules       # udev rules for camera symlinks
│   ├── deploy.sh              # Copies services/rules, enables units
│   ├── sync-on-boot.sh        # Git pull + deploy on boot
│   ├── git-sync-on-boot.service
│   └── goat-heartbeat.cron    # Cron schedule for heartbeat
├── .env                       # Environment variables (EC2_IP, etc.)
└── requirements.txt           # Python dependencies
```

## `/servers`

All servers use Flask and talk to cameras exclusively through the camera proxy at `http://127.0.0.1:8080`. None of them open USB devices directly.

**camera_proxy.py** is the only process that touches `/dev/camera_*`. It opens all 3 cameras once at startup, reads from 2 at a time in rotating pairs (side+top → side+front → top+front), and stores raw JPEG frames in thread-safe buffers. Other services fetch frames via HTTP. Runs under gunicorn with a single gevent worker for concurrent MJPEG streaming.

**prod_server.py** handles the live grading workflow. On a `/grade` POST, it fetches one frame per camera from the proxy, validates image sizes, sends all three to the EC2 API's `/analyze` endpoint, and returns the grade.

**training_server.py** handles burst captures for training data. On a `/record` POST, it pulls 20 frames per camera at 1-second intervals from the proxy, tars each camera's images, and uploads to S3. Supports test mode (capture but don't upload) and real mode (all 3 cameras required, uploads to S3).

**camera_heating.py** runs a thermostat loop reading DS18B20 sensors every 2 seconds. Filters out EMI noise (85°C power-on default, physically impossible values, sudden spikes) and requires 3 consecutive confirming reads before acting on suspicious temperature changes. Failsafe engages after 30 consecutive read failures but only turns heaters on if the last known temp was below 50°F. Exposes an override API for manual control.

## `/display`

**display.py** drives a 2.4" ILI9341 SPI LCD (240×320) connected via GPIO. On boot it shows `boot_logo.png` (a sweet unt logo) with a fade-in animation and holds until services are healthy (up to 20 seconds). Then transitions to a status dashboard that updates every second.

The status screen monitors:

- **Servers** — PROD, CAM_PROXY, EC2 health checks (green/red dot)
- **Cameras** — device existence and readability, plus proxy liveness
- **Temps** — per-enclosure temperature with heater state (on/failsafe/override)
- **WiFi** — signal strength bars
- **Time** — current time with heartbeat indicator

Pin wiring: VCC→3.3V, GND→Pin30, DIN→GPIO10 (MOSI), CLK→GPIO11 (SCLK), CS→GPIO8 (CE0), DC→GPIO18, RST→GPIO27, BL→GPIO25.

## `/logger`

**pi_cloudwatch.py** provides a structured `Logger` class used by all services. Logs to both CloudWatch (under `/goatdev` log group) and console. Format: `[LEVEL] [component] message | key=value`. Thread-safe with a lock around all log calls.

| CloudWatch Stream | Source            |
| ----------------- | ----------------- |
| `camera-proxy`    | Camera proxy      |
| `pi/prod`         | Production server |
| `pi/training`     | Training server   |
| `heating`         | Heater control    |
| `pi/heartbeat`    | Heartbeat cron    |

**pi_heartbeat_cron.py** runs every 2 minutes via cron. Checks that `goat-prod` is active and port 5000 is listening. Only logs to CloudWatch when unhealthy, plus a single "recovered" message when health returns. Uses a state file at `/tmp/heartbeat_unhealthy` to track transitions.

## `/system`

All systemd services, udev rules, and deployment scripts. Everything here gets installed to the system by `deploy.sh` and can be updated via git push (the CI pipeline SSHes in and runs deploy).

### Service Dependency Order

```
network.target
    └── camera-proxy.service     (must start first — owns cameras)
         ├── goat-prod.service   (After=camera-proxy)
         ├── goat-training.service (After=camera-proxy)
         └── setup-proxy.service (Caddy, after all services)
    └── camera-heating.service   (independent, reads sensors directly)
    └── goat-display.service     (starts at sysinit, before network)
```

### Boot Sequence

1. `goat-display.service` starts immediately at sysinit (shows boot logo)
2. `git-sync-on-boot.service` runs `sync-on-boot.sh` — pulls latest code, installs deps, runs `deploy.sh`
3. `camera-proxy.service` starts (opens cameras, begins reading)
4. `goat-prod.service` and `goat-training.service` start (depend on proxy)
5. `camera-heating.service` starts (reads sensors, controls heaters)
6. `setup-proxy.service` starts Caddy (reverse proxy for Tailscale Funnel)

### udev Rules

`99-cameras.rules` maps USB cameras to stable symlinks by vendor ID and kernel USB path:

| Symlink             | USB Path | Physical Port |
| ------------------- | -------- | ------------- |
| `/dev/camera_side`  | `*1.1*`  | Port 1        |
| `/dev/camera_top`   | `*1.3*`  | Port 3        |
| `/dev/camera_front` | `*1.4*`  | Port 4        |

### deploy.sh

Copies all `.service` files to `/etc/systemd/system/`, installs udev rules, sets up cron, enables all services, and reloads systemd. Run automatically on boot via `sync-on-boot.sh` or manually via `sudo bash deploy.sh`.

## Hardware

### Cameras

3x Arducam 16MP (IMX298) USB 2.0 cameras. Resolution: 4656×3496 MJPEG. All three share the Pi 4's VL805 USB controller — only 2 can be read concurrently at full resolution without triggering `select()` timeouts.

### Temperature Sensors

3x DS18B20 one-wire sensors, one per camera enclosure. Connected to the Pi's 1-Wire bus with a 4.7kΩ pullup resistor.

| Sensor ID         | Assignment |
| ----------------- | ---------- |
| `28-0000006d3eba` | Camera 1   |
| `28-0000007047ea` | Camera 2   |
| `28-0000007193ed` | Camera 3   |

### Heater MOSFETs

3x N-channel MOSFETs driven by GPIO pins, one per enclosure heater, but everything soldered together on a small perfboard on the main compute enclosure.

| Camera  | GPIO Pin | Physical Pin |
| ------- | -------- | ------------ |
| camera1 | GPIO 5   | Pin 29       |
| camera2 | GPIO 6   | Pin 31       |
| camera3 | GPIO 17  | Pin 11       |

## Remote Access

### Tailscale

All services accessible over Tailscale mesh VPN. The Pi appears as `goat-pi` at `100.xxx.xxx.xxx`.

```bash
ssh pi@100.xxx.xxx.xxx
curl http://100.xxx.xxx.xxx:5000/health
curl http://100.xxx.xxx.xxx:8080/status
```

### Tailscale Funnel

Caddy on port 8443 provides HTTPS for mobile browser access via Tailscale Funnel. The frontend at `herd-sync.com` connects to the Pi through this endpoint.

### Quick Reference

| Task                 | Command                                 |
| -------------------- | --------------------------------------- |
| SSH to Pi            | `ssh pi@100.xxx.xxx.xxx`                |
| Camera proxy status  | `curl http://localhost:8080/status`     |
| Restart camera proxy | `sudo systemctl restart camera-proxy`   |
| Restart prod         | `sudo systemctl restart goat-prod`      |
| Restart training     | `sudo systemctl restart goat-training`  |
| Restart heater       | `sudo systemctl restart camera-heating` |
| View proxy logs      | `journalctl -u camera-proxy -f`         |
| View prod logs       | `journalctl -u goat-prod -f`            |
| View heater logs     | `journalctl -u camera-heating -f`       |
| Check cameras        | `ls -la /dev/camera_*`                  |
| Check memory         | `free -h`                               |
| Check disk           | `df -h /tmp`                            |
| Tailscale status     | `tailscale status`                      |
