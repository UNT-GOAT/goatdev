"""
API Key Authentication Middleware

Pi-to-EC2 requests are service-to-service — no user JWT involved.
A shared API key (set via API_KEY env var) gates all non-health endpoints.

If API_KEY is not set, ALL requests are rejected except health checks.
This is intentional — fail closed, not open.
"""

import os
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api_auth")

API_KEY = os.environ.get("API_KEY")

# Endpoints that must remain open (health checks for deploy verification)
PUBLIC_PATHS = frozenset({"/health", "/health/deep"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        if not API_KEY:
            # No key configured = reject everything. Don't silently degrade.
            logger.error("API_KEY not set — rejecting request to %s", request.url.path)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "API key not configured",
                    "error_code": "AUTH_NOT_CONFIGURED",
                    "fix": "Set API_KEY environment variable on EC2",
                },
            )

        provided = request.headers.get("X-API-Key", "")
        if provided != API_KEY:
            logger.warning(
                "Invalid API key for %s from %s",
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key", "error_code": "UNAUTHORIZED"},
            )

        return await call_next(request)