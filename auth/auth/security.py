"""
Security utilities.

JWT:
  - RS256 (asymmetric). Private key signs, public key verifies.
  - Key pair auto-generated on first run if not present.
  - Public key served via JWKS endpoint so any service can validate
    without a shared secret.

Passwords:
  - bcrypt via passlib. Slow by design — brute force resistant.

Tokens:
  - Access token: short-lived (15 min), used for API auth
  - Refresh token: long-lived (30 days), stored hashed in DB,
    used to get new access tokens without re-entering password
"""

import os
import hashlib
import secrets
import threading
from datetime import datetime, timedelta
from typing import Optional

import jwt # type: ignore
from passlib.context import CryptContext # type: ignore
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from .config import (
    JWT_PRIVATE_KEY_PATH, JWT_PUBLIC_KEY_PATH,
    ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
)


# =============================================================================
# PASSWORD HASHING
# =============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# =============================================================================
# RSA KEY MANAGEMENT
# =============================================================================

_private_key = None
_public_key = None
_public_key_pem = None
_key_id = None


def _compute_kid(public_key_pem: bytes) -> str:
    """Derive a stable key ID from the public key (SHA-256 truncated)."""
    return hashlib.sha256(public_key_pem).hexdigest()[:16]


def get_key_id() -> str:
    """Return the current key ID."""
    _ensure_keys()
    return _key_id

def _ensure_keys():
    """Load or generate RSA key pair."""
    global _private_key, _public_key, _public_key_pem

    if _private_key is not None:
        return

    if os.path.exists(JWT_PRIVATE_KEY_PATH) and os.path.exists(JWT_PUBLIC_KEY_PATH):
        # Load existing keys
        with open(JWT_PRIVATE_KEY_PATH, "rb") as f:
            _private_key = serialization.load_pem_private_key(f.read(), password=None)
        with open(JWT_PUBLIC_KEY_PATH, "rb") as f:
            _public_key_pem = f.read()
            _public_key = serialization.load_pem_public_key(_public_key_pem)
        _key_id = _compute_kid(_public_key_pem)
    else:
        # Generate new key pair
        _private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        _public_key = _private_key.public_key()

        private_pem = _private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        _public_key_pem = _public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        os.makedirs(os.path.dirname(JWT_PRIVATE_KEY_PATH), exist_ok=True)
        with open(JWT_PRIVATE_KEY_PATH, "wb") as f:
            f.write(private_pem)
        os.chmod(JWT_PRIVATE_KEY_PATH, 0o600)

        with open(JWT_PUBLIC_KEY_PATH, "wb") as f:
            f.write(_public_key_pem)

        _key_id = _compute_kid(_public_key_pem)


def get_public_key_pem() -> bytes:
    """Return the public key in PEM format (for JWKS endpoint)."""
    _ensure_keys()
    return _public_key_pem


# =============================================================================
# JWT TOKEN CREATION & VALIDATION
# =============================================================================

def create_access_token(user_id: int, username: str, role: str) -> str:
    """Create a short-lived access token."""
    _ensure_keys()
    now = datetime.utcnow()
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, _private_key, algorithm="RS256", headers={"kid": _key_id})


def create_refresh_token() -> tuple[str, datetime]:
    """
    Create a refresh token.
    Returns (raw_token, expires_at).
    The raw token is sent to the client. Only the hash is stored in DB.
    """
    raw_token = secrets.token_urlsafe(48)
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return raw_token, expires_at


def hash_refresh_token(raw_token: str) -> str:
    """SHA-256 hash of refresh token for DB storage."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and validate an access token.
    Returns the payload dict or None if invalid/expired.
    """
    _ensure_keys()
    try:
        payload = jwt.decode(token, _public_key, algorithms=["RS256"])
        if payload.get("type") != "access":
            return None
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    


# =============================================================================
# DEACTIVATED USER TRACKING
# =============================================================================
# In-memory set of user IDs whose tokens should be rejected immediately,
# even if the JWT is technically valid. Populated when an admin deactivates
# a user. Checked by the verify endpoint on every request.
#
# Cleared on restart — but that just means a deactivated user gets the
# normal 15-min expiry window until the next deactivation event repopulates
# the set. Acceptable tradeoff vs. a DB query on every single API request.

_deactivated_users: set[str] = set()
_deactivated_lock = threading.Lock()


def mark_user_deactivated(user_id: int):
    """Add a user ID to the deactivated set."""
    with _deactivated_lock:
        _deactivated_users.add(str(user_id))


def mark_user_reactivated(user_id: int):
    """Remove a user ID from the deactivated set."""
    with _deactivated_lock:
        _deactivated_users.discard(str(user_id))


def is_user_deactivated(user_id: str) -> bool:
    """Check if a user ID is in the deactivated set."""
    with _deactivated_lock:
        return user_id in _deactivated_users
