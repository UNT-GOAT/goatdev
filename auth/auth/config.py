"""
Auth Service Configuration

All settings are loaded from environment variables.
"""

import os

# Database
DATABASE_URL = os.environ.get("DATABASE_URL")  # postgres://user:pass@host:5432/dbname

# JWT Settings
# RSA key pair for RS256 signing. Private key stays here (signs tokens).
# Public key is served via /auth/.well-known/jwks.json so consumers (Pi, EC2)
# can validate tokens without a shared secret.
JWT_PRIVATE_KEY_PATH = os.environ.get("JWT_PRIVATE_KEY_PATH", "/app/keys/private.pem")
JWT_PUBLIC_KEY_PATH = os.environ.get("JWT_PUBLIC_KEY_PATH", "/app/keys/public.pem")

# Token lifetimes
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

# Initial admin account — seeded on first run if no users exist
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")  # Required on first run

# CORS — CloudFront domain + dev
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://dqh1843col5pu.cloudfront.net"
).split(",")
