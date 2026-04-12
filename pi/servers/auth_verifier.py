"""
Pi-local JWT verifier and ticket issuer.

This service backs Caddy's forward_auth flow for Pi routes and also issues
short-lived opaque tickets for browser image requests that cannot attach
Authorization headers directly.
"""

from urllib.parse import urlparse, parse_qs
from flask import Flask, request, jsonify
import base64
import os
import secrets
import sys
import threading
import time
from typing import Optional

import jwt as pyjwt  # type: ignore
import requests
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers

app = Flask(__name__)

sys.path.insert(0, "/home/pi/goatdev/pi")
from logger.pi_cloudwatch import Logger

log = Logger("pi/auth")


# =============================================================================
# CONFIGURATION
# =============================================================================

AUTH_JWKS_URL = (os.environ.get("AUTH_JWKS_URL") or "").strip()
AUTH_JWKS_URLS = [
    url.strip()
    for url in (os.environ.get("AUTH_JWKS_URLS") or "").split(",")
    if url.strip()
]
if not AUTH_JWKS_URLS and AUTH_JWKS_URL:
    AUTH_JWKS_URLS = [AUTH_JWKS_URL]

ALLOW_LEGACY_QUERY_BEARER = (
    os.environ.get("ALLOW_LEGACY_QUERY_BEARER", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

KEY_REFRESH_INTERVAL_SEC = 3600
KEY_FETCH_TIMEOUT_SEC = 10
TICKET_TTL_SEC = 120
MAX_SERIAL_ID_LEN = 50

PORT = 5003
VALID_VIEWS = {"side", "top", "front"}


# =============================================================================
# PUBLIC KEY CACHE
# =============================================================================

_public_key = None
_public_key_lock = threading.Lock()
_last_fetch = 0.0
_last_success_at = 0.0
_last_error = None
_active_source = None


# =============================================================================
# TICKET CACHE
# =============================================================================

_tickets = {}
_ticket_lock = threading.Lock()


def _b64url_decode(data):
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def _ticket_resource(kind: str, view: str, serial_id: Optional[str] = None) -> str:
    if kind == "stream":
        return f"/api/viewfocus/stream/{view}"
    if kind == "debug" and serial_id:
        return f"/api/prod/debug/{serial_id}/{view}"
    raise ValueError("Invalid ticket resource")


def _sanitize_serial_id(serial_id: str) -> str:
    serial_id = (serial_id or "").strip()
    if not serial_id:
        raise ValueError("serial_id is required")
    if len(serial_id) > MAX_SERIAL_ID_LEN:
        raise ValueError(f"serial_id must be {MAX_SERIAL_ID_LEN} chars or fewer")
    cleaned = "".join(ch for ch in serial_id if ch.isalnum() or ch in {"_", "-"})
    if cleaned != serial_id:
        raise ValueError("serial_id must be alphanumeric, underscore, or dash")
    return cleaned


def _cleanup_expired_tickets(now: Optional[float] = None):
    now = now or time.time()
    expired = []
    with _ticket_lock:
        for ticket, entry in _tickets.items():
            if entry["expires_at"] <= now:
                expired.append(ticket)
        for ticket in expired:
            _tickets.pop(ticket, None)


def _issue_ticket(resource: str) -> dict:
    now = time.time()
    _cleanup_expired_tickets(now)
    ticket = secrets.token_urlsafe(16)
    expires_at = now + TICKET_TTL_SEC
    with _ticket_lock:
        _tickets[ticket] = {
            "resource": resource,
            "expires_at": expires_at,
        }
    return {
        "ticket": ticket,
        "expires_in": TICKET_TTL_SEC,
        "resource": resource,
    }


def _validate_ticket(ticket: str, requested_resource: str):
    _cleanup_expired_tickets()
    with _ticket_lock:
        entry = _tickets.get(ticket)
    if not entry:
        return False, "Invalid or expired ticket"
    if entry["resource"] != requested_resource:
        return False, "Ticket is not valid for this resource"
    return True, None


def _requested_resource_from_forwarded_uri():
    forwarded_uri = request.headers.get("X-Forwarded-Uri", "")
    if not forwarded_uri:
        return None
    parsed = urlparse(forwarded_uri)
    path = parsed.path or ""
    parts = path.strip("/").split("/")

    if len(parts) == 4 and parts[:3] == ["api", "viewfocus", "stream"] and parts[3] in VALID_VIEWS:
        return f"/api/viewfocus/stream/{parts[3]}"

    if (
        len(parts) == 5
        and parts[:3] == ["api", "prod", "debug"]
        and parts[4] in VALID_VIEWS
    ):
        try:
            serial_id = _sanitize_serial_id(parts[3])
        except ValueError:
            return None
        return f"/api/prod/debug/{serial_id}/{parts[4]}"

    return None


def fetch_public_key():
    global _public_key, _last_fetch, _last_success_at, _last_error, _active_source

    if not AUTH_JWKS_URLS:
        _last_error = "AUTH_JWKS_URLS not set"
        log.error("jwks", _last_error)
        return False

    last_error = None
    for jwks_url in AUTH_JWKS_URLS:
        try:
            resp = requests.get(jwks_url, timeout=KEY_FETCH_TIMEOUT_SEC)
            if resp.status_code != 200:
                last_error = f"JWKS fetch failed from {jwks_url} (status {resp.status_code})"
                log.error("jwks", "JWKS fetch failed", source=jwks_url, status=resp.status_code)
                continue

            jwks = resp.json()
            key_data = jwks["keys"][0]
            n = int.from_bytes(_b64url_decode(key_data["n"]), byteorder="big")
            e = int.from_bytes(_b64url_decode(key_data["e"]), byteorder="big")

            pub_key = RSAPublicNumbers(e, n).public_key()
            now = time.time()
            with _public_key_lock:
                _public_key = pub_key
                _last_fetch = now
                _last_success_at = now
                _active_source = jwks_url
                _last_error = None

            log.info("jwks", "Public key fetched and cached", source=jwks_url)
            return True
        except Exception as ex:
            last_error = str(ex)
            log.error("jwks", "Failed to fetch JWKS", source=jwks_url, error=last_error)

    _last_error = last_error
    return False


def get_public_key():
    with _public_key_lock:
        key = _public_key
        last_fetch = _last_fetch

    if key is not None and (time.time() - last_fetch) > KEY_REFRESH_INTERVAL_SEC:
        threading.Thread(target=fetch_public_key, daemon=True).start()

    return key


def key_refresh_loop():
    while True:
        time.sleep(KEY_REFRESH_INTERVAL_SEC)
        fetch_public_key()


def _decode_access_token(token: str, *, retry_refresh: bool = True):
    key = get_public_key()
    if key is None:
        return None, ("Auth service not ready", 503)

    try:
        payload = pyjwt.decode(token, key, algorithms=["RS256"])
    except pyjwt.ExpiredSignatureError:
        return None, ("Token expired", 401)
    except pyjwt.InvalidTokenError:
        if retry_refresh and fetch_public_key():
            return _decode_access_token(token, retry_refresh=False)
        return None, ("Invalid token", 401)

    if payload.get("type") != "access":
        return None, ("Invalid token type", 401)
    return payload, None


def _extract_bearer_token():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


# =============================================================================
# ROUTES
# =============================================================================


@app.route("/verify")
def verify():
    """
    Validate an access token from Authorization, or an opaque ticket for the
    exact stream/debug resource in the original forwarded URI.
    """
    bearer = _extract_bearer_token()
    if bearer:
        payload, error = _decode_access_token(bearer)
        if error:
            return jsonify({"error": error[0]}), error[1]
        return jsonify({"status": "ok", "user": payload.get("username")}), 200

    forwarded_uri = request.headers.get("X-Forwarded-Uri", "")
    params = parse_qs(urlparse(forwarded_uri).query) if "?" in forwarded_uri else {}
    requested_resource = _requested_resource_from_forwarded_uri()

    ticket_values = params.get("ticket")
    if ticket_values:
        if not requested_resource:
            return jsonify({"error": "Ticket is not valid for this resource"}), 401
        ok, detail = _validate_ticket(ticket_values[0], requested_resource)
        if not ok:
            return jsonify({"error": detail}), 401
        return jsonify({"status": "ok", "resource": requested_resource}), 200

    if ALLOW_LEGACY_QUERY_BEARER:
        token_values = params.get("token")
        if token_values:
            payload, error = _decode_access_token(token_values[0])
            if error:
                return jsonify({"error": error[0]}), error[1]
            return jsonify({"status": "ok", "user": payload.get("username")}), 200

    return jsonify({"error": "Missing token or ticket"}), 401


@app.route("/tickets", methods=["POST"])
def issue_ticket():
    bearer = _extract_bearer_token()
    if not bearer:
        return jsonify({"error": "Missing Authorization header"}), 401

    payload, error = _decode_access_token(bearer)
    if error:
        return jsonify({"error": error[0]}), error[1]

    body = request.get_json(force=True, silent=True) or {}
    kind = (body.get("kind") or "").strip().lower()
    view = (body.get("view") or "").strip().lower()
    serial_id = body.get("serial_id")

    if view not in VALID_VIEWS:
        return jsonify({"error": "view must be one of side, top, or front"}), 400

    try:
        if kind == "stream":
            resource = _ticket_resource("stream", view)
        elif kind == "debug":
            resource = _ticket_resource("debug", view, _sanitize_serial_id(str(serial_id or "")))
        else:
            return jsonify({"error": "kind must be stream or debug"}), 400
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400

    ticket = _issue_ticket(resource)
    ticket["user"] = payload.get("username")
    return jsonify(ticket), 201


@app.route("/health")
def health():
    key_ok = _public_key is not None
    key_age = round(time.time() - _last_fetch) if _last_fetch > 0 else None
    stale = key_age is None or key_age > (KEY_REFRESH_INTERVAL_SEC * 2)
    status = "ok" if key_ok and not stale else "degraded"

    return jsonify(
        {
            "status": status,
            "public_key_cached": key_ok,
            "key_age_sec": key_age,
            "last_success_at": round(_last_success_at) if _last_success_at else None,
            "active_source": _active_source,
            "jwks_urls": AUTH_JWKS_URLS,
            "last_error": _last_error,
            "ticket_ttl_sec": TICKET_TTL_SEC,
            "allow_legacy_query_bearer": ALLOW_LEGACY_QUERY_BEARER,
        }
    )


# =============================================================================
# STARTUP
# =============================================================================


if __name__ == "__main__":
    log.info("startup", "=" * 50)
    log.info("startup", "PI AUTH VERIFIER STARTING")
    log.info(
        "startup",
        "Configuration",
        jwks_urls=AUTH_JWKS_URLS,
        refresh_interval=KEY_REFRESH_INTERVAL_SEC,
        port=PORT,
        allow_legacy_query_bearer=ALLOW_LEGACY_QUERY_BEARER,
    )

    for attempt in range(5):
        if fetch_public_key():
            break
        log.warn("startup", f"JWKS fetch attempt {attempt + 1}/5 failed, retrying in 5s")
        time.sleep(5)

    if _public_key is None:
        log.error("startup", "Could not fetch public key — verifier will reject bearer requests until a key is available")
    else:
        log.info("startup", "Public key ready", source=_active_source)

    refresh_thread = threading.Thread(target=key_refresh_loop, daemon=True)
    refresh_thread.start()

    log.info("startup", "Verifier ready", port=PORT)
    log.info("startup", "=" * 50)

    import logging

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(host="127.0.0.1", port=PORT)
