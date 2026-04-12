"""
Shared cross-service capture ownership lock for the Pi.

Prod grading and training capture run in separate processes, so in-memory flags
are not enough to coordinate access. This helper uses flock on a single lock
file plus a small JSON metadata file so callers can reject overlapping work with
operator-readable context.
"""

from __future__ import annotations

import fcntl
import json
import os
from typing import Optional

LOCK_PATH = "/tmp/herdsync-capture.lock"
META_PATH = "/tmp/herdsync-capture.lock.json"


def _write_metadata(metadata: dict) -> None:
    tmp_path = f"{META_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    os.replace(tmp_path, META_PATH)


def read_capture_owner() -> Optional[dict]:
    try:
        with open(META_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def acquire_capture_lock(
    service: str,
    owner_field: str,
    owner_value: str,
    *,
    started_at: str,
    progress: str,
):
    handle = open(LOCK_PATH, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None, read_capture_owner()

    metadata = {
        "service": service,
        owner_field: owner_value,
        "started_at": started_at,
        "progress": progress,
    }
    _write_metadata(metadata)
    return handle, metadata


def update_capture_lock_metadata(handle, service: str, **fields):
    if handle is None:
        return None

    metadata = read_capture_owner() or {"service": service}
    metadata["service"] = service
    metadata.update(fields)
    _write_metadata(metadata)
    return metadata


def release_capture_lock(handle) -> None:
    if handle is None:
        return

    try:
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
