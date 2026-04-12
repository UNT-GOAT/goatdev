# Raspberry Pi - Edge Device

All software running on the Raspberry Pi 4 at the livestock facility. Owns cameras, temperature sensors, heater circuits, and an LCD status display. Connected to the cloud via Tailscale VPN for both API traffic and SSH deployment.

## Services

Six application services run as systemd units, plus Caddy as the reverse proxy:

| Service         | Port     | Unit File              | Description                                           |
| --------------- | -------- | ---------------------- | ----------------------------------------------------- |
| camera-proxy    | 8080     | camera-proxy.service   | Owns all USB cameras, MJPEG streams, full-res capture |
| prod_server     | 5000     | goat-prod.service      | Production grading: capture → send to EC2 → grade     |
| training_server | 5001     | goat-training.service  | Training data: burst capture → S3 upload              |
| camera_heating  | 5002     | camera-heating.service | Thermostat control for camera enclosure heaters       |
| auth_verifier   | 5003     | auth-verifier.service  | Local JWT verification for Caddy forward_auth         |
| display         | -        | goat-display.service   | SPI LCD status display (no network port)              |
| Caddy           | 443/8444 | caddy.service          | Reverse proxy, TLS, auth gating                       |

## Architecture

```
EC2 Caddy → Tailscale → Pi Caddy (:8444)
                            ├── /api/viewfocus/* → camera-proxy (:8080)
                            ├── /api/prod/*      → prod_server (:5000)
                            ├── /api/capture/*   → training_server (:5001)
                            ├── /api/heater/*    → camera_heating (:5002)
                            └── forward_auth     → auth_verifier (:5003)

Tailscale VPN → Caddy (:443, Tailscale TLS)
                  └── (same routing as above)
```

Every request through Caddy (except OPTIONS preflight and WebSocket upgrades) goes through `forward_auth` to the auth verifier, which validates the JWT against the cached public key from EC2's JWKS endpoint. The verifier also issues short-lived opaque tickets for browser stream and debug-image requests so JWTs do not need to ride in query strings.

## Directory Structure

```
pi/
├── requirements.txt                # Python deps for all Pi services
├── README.md
├── servers/
│   ├── camera_proxy.py             # Camera ownership: multiprocess (HardwareWorker + Flask/gevent)
│   ├── prod_server.py              # Grading workflow: capture all → send to EC2 /analyze
│   ├── training_server.py          # Training capture: concurrent burst → S3 upload
│   ├── camera_heating.py           # Thermostat: DS18B20 sensors, GPIO heater control, failsafe
│   └── auth_verifier.py            # JWT verifier: JWKS fetch + cache, Caddy forward_auth
├── display/
│   ├── display.py                  # SPI LCD: servers, cameras, temps, WiFi, heartbeat
│   └── boot_logo.png               # Eagle logo shown during boot
├── logger/
│   ├── pi_cloudwatch.py            # Structured logger: CloudWatch + console, [LEVEL] [component] format
│   └── pi_heartbeat_cron.py        # Cron health check: logs errors when services are down
└── system/
    ├── deploy.sh                   # Installs services, udev rules, cron, Caddyfile
    ├── Caddyfile                   # Reverse proxy config (Tailscale TLS + HTTP on 8444)
    ├── 99-cameras.rules            # udev rules for stable /dev/camera_* symlinks
    ├── camera-proxy.service        # systemd unit (gunicorn + geventwebsocket worker, root)
    ├── goat-prod.service           # systemd unit (python3, user=pi)
    ├── goat-training.service       # systemd unit (python3, user=pi)
    ├── camera-heating.service      # systemd unit (python3, user=pi, needs GPIO)
    ├── auth-verifier.service       # systemd unit (python3, user=pi)
    ├── goat-display.service        # systemd unit (python3, root, SPI access)
    ├── goat-heartbeat.cron         # Cron: health check every 2 minutes
    ├── git-sync-on-boot.service    # Pull latest code on boot
    └── sync-on-boot.sh             # Boot script: git pull, pip install, deploy.sh
```

## Service Details

### camera_proxy.py

The most complex service. Uses a **multiprocess architecture** to isolate USB camera I/O from the web server:

- **HardwareWorker** (separate process, no gevent): owns all OpenCV/V4L2 camera handles. Reads sequentially at 640x480 for streaming. On capture signal, switches resolution in-place to 4656x3496 per camera (no release/reopen), grabs the frame, switches back.
- **Flask/gevent** (main process): reads frames from shared memory for MJPEG `/stream` endpoints. Signals the hardware worker for full-res `/capture` requests.
- **Shared memory**: 6MB per camera for frame data, plus stats and capture event signaling via SHM flags.
- **Watchdog**: health monitor thread detects frozen cameras (>30s stale) and restarts the hardware process.

This separation exists because gevent monkeypatches threading, and `cv2.VideoCapture.read()` is a blocking C call that freezes the gevent event loop. The hardware worker runs in a real OS process with no gevent.

### prod_server.py

Production grading workflow:

1. Pre-checks: verify EC2 reachable, all cameras connected
2. Signals camera proxy for full-res capture of all 3 cameras
3. Pulls individual frames from SHM via `/capture/frame/<camera>`
4. Sends all 3 images + metadata to EC2's `/analyze` endpoint over Tailscale
5. EC2 runs YOLO inference, returns grade + measurements
6. EC2 handles S3 archival (Pi does not write to S3 for grading)

Also provides `/grade/test` for uploading images directly (bypasses cameras).

### training_server.py

Training data collection:

1. Triggers concurrent burst captures on all 3 cameras via proxy
2. Each burst: 20 frames at 1.5s intervals, returned as tar.gz
3. Uploads tar.gz files directly to S3 training bucket
4. Supports test mode (capture + S3 capability check, no upload)

### camera_heating.py

Thermostat for camera enclosure heaters:

- Reads DS18B20 1-Wire temperature sensors every 2 seconds
- Hysteresis: ON below 40°F, OFF above 70°F
- EMI spike filtering: rejects >10°F jumps, requires 3 consecutive agreeing reads
- Failsafe: after 30 consecutive sensor failures, forces heater ON only if last known temp was cold
- Writes state to `/tmp/heater_state.json` atomically for the display to read
- Flask API on port 5002 for remote override and history

### auth_verifier.py

Lightweight JWT verifier for Caddy's `forward_auth`:

- Fetches RSA public key from EC2's JWKS endpoint via Tailscale on startup
- Caches key, refreshes hourly
- Validates JWT from `Authorization: Bearer` header or `?token=` query param (for MJPEG streams loaded via `<img>` tags)
- Returns 200 (allow) or 401 (deny) - Caddy only checks the status code

### display.py

SPI LCD status display (2.4" ILI9341, 240x320):

- Boot screen with fade-in logo, waits up to 20s for services
- Main loop: servers (PROD/CAM_PROXY/EC2), cameras, temps, WiFi signal
- Network checks run on a background thread pool to prevent UI freezing
- Reads heater state from shared file (no HTTP, no 1-Wire bus contention)

## Hardware

- **Cameras**: 3x Arducam 16MP (IMX298) USB, mapped via udev rules to `/dev/camera_side`, `/dev/camera_top`, `/dev/camera_front`
- **Temperature sensors**: 3x DS18B20 on 1-Wire bus with 4.7kΩ pullup
- **Heaters**: 3x N-channel MOSFET circuits on GPIO 5, 6, 17
- **Display**: 2.4" ILI9341 SPI LCD on SPI0 (GPIO 8, 10, 11, 18, 25, 27)

## Environment Variables

Stored in `/home/pi/goatdev/pi/.env`:

| Variable             | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| `EC2_IP`             | Tailscale IP of the EC2 instance                                  |
| `EC2_GOAT_API`       | Full URL to goat-api over Tailscale                               |
| `API_KEY`            | API key matching goat-api's `API_KEY` (sent via X-API-Key header) |
| `S3_TRAINING_BUCKET` | S3 bucket for training data uploads                               |
| `AUTH_JWKS_URL`      | Legacy single JWKS endpoint (fallback if `AUTH_JWKS_URLS` unset)  |
| `AUTH_JWKS_URLS`     | Ordered JWKS endpoints, private/Tailscale first and public second |
| `ALLOW_LEGACY_QUERY_BEARER` | Temporary rollout flag for `?token=` query bearer support |
| `SITE_DOMAIN`        | Caddy domain for Tailscale TLS cert                               |

## Deployment

Automated via GitHub Actions (`.github/workflows/deploy-pi.yml`):

1. Connects to Pi via Tailscale VPN
2. `git fetch && git reset --hard origin/main`
3. `pip install -r pi/requirements.txt`
4. `deploy.sh` copies systemd units, udev rules, cron, Caddyfile
5. Ordered service restart: stop consumers → restart camera-proxy → wait for health → start consumers → restart display + heating
