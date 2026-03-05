"""
herdsync-db-proxy

Authenticated gateway to herdsync-db.

External:
  /db/*

Internal:
  http://db:8002/*
"""

import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from jose import jwt

AUTH_JWKS_URL = os.environ.get(
    "AUTH_JWKS_URL",
)

DB_INTERNAL = os.environ.get(
    "DB_INTERNAL",
)

jwks = None

app = FastAPI(title="herdsync-db-proxy")


# --------------------------------------------------
# JWKS
# --------------------------------------------------

async def load_jwks():
    global jwks
    async with httpx.AsyncClient() as client:
        r = await client.get(AUTH_JWKS_URL)
        jwks = r.json()


async def verify_token(token: str):

    if jwks is None:
        await load_jwks()

    headers = jwt.get_unverified_header(token)

    key = next(
        (k for k in jwks["keys"] if k["kid"] == headers["kid"]),
        None
    )

    if not key:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            options={"verify_aud": False}
        )
        return payload

    except Exception:
        raise HTTPException(status_code=401, detail="Token invalid")


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{DB_INTERNAL}/health")
            return r.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# --------------------------------------------------
# PROXY
# --------------------------------------------------

@app.api_route("/db/{path:path}", methods=["GET","POST","PUT","PATCH","DELETE"])
async def proxy(path: str, request: Request):

    auth = request.headers.get("Authorization")

    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth.split(" ")[1]

    await verify_token(token)

    url = f"{DB_INTERNAL}/{path}"

    body = await request.body()

    async with httpx.AsyncClient() as client:
        resp = await client.request(
            request.method,
            url,
            content=body,
            headers={"content-type": request.headers.get("content-type","application/json")}
        )

    return resp.json()