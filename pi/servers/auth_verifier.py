"""
Pi-local JWT Verifier

Tiny service that validates JWTs for Caddy's forward_auth directive.
Fetches the public key from EC2's JWKS endpoint on startup, caches it,
and refreshes periodically. No shared secrets — just RS256 public key.

Caddy sends every API request here first. If this returns 200, Caddy
forwards to the backend. If 401, Caddy rejects the request.

Port: 5003

Why a separate service instead of middleware on each Flask app?
  - One place to maintain auth logic
  - Four Flask services (prod, training, heating, camera) stay untouched
  - Caddy config is simple and standard
  - Public key caching in one place
"""

from flask import Flask, request, jsonify
import jwt as pyjwt # type: ignore
import requests
import threading
import time
import sys
import os
import json
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import base64

app = Flask(__name__)

sys.path.insert(0, '/home/pi/goatdev/pi')
from logger.pi_cloudwatch import Logger

log = Logger('pi/auth')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Auth service on EC2 — accessed via CloudFront or direct
AUTH_JWKS_URL = os.environ.get('AUTH_JWKS_URL')
KEY_REFRESH_INTERVAL_SEC = 3600  # Refresh public key every hour
KEY_FETCH_TIMEOUT_SEC = 10

PORT = 5003

# =============================================================================
# PUBLIC KEY CACHE
# =============================================================================

_public_key = None
_public_key_lock = threading.Lock()
_last_fetch = 0


def _b64url_decode(data):
    """Decode base64url without padding."""
    padding = 4 - len(data) % 4
    if padding != 4:
        data += '=' * padding
    return base64.urlsafe_b64decode(data)


def fetch_public_key():
    """Fetch RSA public key from JWKS endpoint."""
    global _public_key, _last_fetch

    if not AUTH_JWKS_URL:
        log.error('jwks', 'AUTH_JWKS_URL not set')
        return False

    try:
        resp = requests.get(AUTH_JWKS_URL, timeout=KEY_FETCH_TIMEOUT_SEC)
        if resp.status_code != 200:
            log.error('jwks', 'JWKS fetch failed', status=resp.status_code)
            return False

        jwks = resp.json()
        key_data = jwks['keys'][0]

        # Reconstruct RSA public key from JWK components
        n = int.from_bytes(_b64url_decode(key_data['n']), byteorder='big')
        e = int.from_bytes(_b64url_decode(key_data['e']), byteorder='big')

        pub_numbers = RSAPublicNumbers(e, n)
        pub_key = pub_numbers.public_key()

        with _public_key_lock:
            _public_key = pub_key
            _last_fetch = time.time()

        log.info('jwks', 'Public key fetched and cached')
        return True

    except Exception as ex:
        log.error('jwks', 'Failed to fetch JWKS', error=str(ex))
        return False


def get_public_key():
    """Get cached public key, refreshing if stale."""
    global _last_fetch

    with _public_key_lock:
        key = _public_key

    # Refresh if stale (but don't block — use cached key)
    if time.time() - _last_fetch > KEY_REFRESH_INTERVAL_SEC:
        threading.Thread(target=fetch_public_key, daemon=True).start()

    return key


def key_refresh_loop():
    """Background thread to periodically refresh the public key."""
    while True:
        time.sleep(KEY_REFRESH_INTERVAL_SEC)
        fetch_public_key()


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/verify')
def verify():
    """
    Validate JWT from Authorization header.
    Called by Caddy forward_auth on every /api/* request.
    Returns 200 if valid, 401 if not.
    """
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return jsonify({'error': 'Missing token'}), 401

    token = auth[7:]
    key = get_public_key()

    if key is None:
        # No public key yet — can't validate. Reject.
        log.error('verify', 'No public key available')
        return jsonify({'error': 'Auth service not ready'}), 503

    try:
        payload = pyjwt.decode(token, key, algorithms=['RS256'])
        if payload.get('type') != 'access':
            return jsonify({'error': 'Invalid token type'}), 401
        return jsonify({'status': 'ok', 'user': payload.get('username')}), 200
    except pyjwt.ExpiredSignatureError:
        return jsonify({'error': 'Token expired'}), 401
    except pyjwt.InvalidTokenError as ex:
        return jsonify({'error': 'Invalid token'}), 401


@app.route('/health')
def health():
    """Health check."""
    key_ok = _public_key is not None
    key_age = round(time.time() - _last_fetch) if _last_fetch > 0 else None

    return jsonify({
        'status': 'ok' if key_ok else 'degraded',
        'public_key_cached': key_ok,
        'key_age_sec': key_age,
        'jwks_url': AUTH_JWKS_URL,
    })


# =============================================================================
# STARTUP
# =============================================================================

if __name__ == '__main__':
    log.info('startup', '=' * 50)
    log.info('startup', 'PI AUTH VERIFIER STARTING')
    log.info('startup', 'Configuration',
             jwks_url=AUTH_JWKS_URL,
             refresh_interval=KEY_REFRESH_INTERVAL_SEC,
             port=PORT)

    # Fetch key on startup — retry a few times
    for attempt in range(5):
        if fetch_public_key():
            break
        log.warn('startup', f'JWKS fetch attempt {attempt + 1}/5 failed, retrying in 5s')
        time.sleep(5)

    if _public_key is None:
        log.error('startup', 'Could not fetch public key — verifier will reject all requests until key is available')
    else:
        log.info('startup', 'Public key ready')

    # Start background refresh
    refresh_thread = threading.Thread(target=key_refresh_loop, daemon=True)
    refresh_thread.start()

    log.info('startup', 'Verifier ready', port=PORT)
    log.info('startup', '=' * 50)

    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    app.run(host='127.0.0.1', port=PORT)