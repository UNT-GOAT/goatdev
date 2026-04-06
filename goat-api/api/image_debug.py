"""
Debug Image Management

Handles saving debug overlay images to disk, pruning old directories,
and serving debug images via API endpoints.
"""

import os
import shutil
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from .logger import log
from .image_validation import sanitize_serial_id

DEBUG_DIR_BASE = '/app/data/debug'
MAX_DEBUG_SERIAL_IDS = 100

router = APIRouter()


# =============================================================================
# SAVING & CLEANUP
# =============================================================================

def save_debug_images(
    serial_id: str,
    debug_images: Dict[str, np.ndarray],
) -> Dict[str, str]:
    """
    Save debug overlay images to disk and prune old directories.

    Args:
        serial_id: Goat identifier (used as subdirectory name)
        debug_images: Dict of {view_name: cv2_image}

    Returns:
        Dict of {view_name: filepath} for images that were saved successfully
    """
    debug_dir = f'{DEBUG_DIR_BASE}/{serial_id}'
    os.makedirs(debug_dir, exist_ok=True)

    saved = {}
    for view_name, debug_img in debug_images.items():
        if debug_img is None:
            continue
        debug_path = f'{debug_dir}/{view_name}_debug.jpg'
        try:
            cv2.imwrite(debug_path, debug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved[view_name] = debug_path
            log.info('debug', 'Saved debug image',
                     serial_id=serial_id, view=view_name, path=debug_path)
        except Exception as e:
            log.error('debug', 'Failed to save debug image',
                      serial_id=serial_id, view=view_name, error=str(e))

    # Prune old directories
    _cleanup_old_debug_dirs()

    return saved


def _cleanup_old_debug_dirs():
    """
    Remove oldest debug directories when count exceeds MAX_DEBUG_SERIAL_IDS.

    Keeps only the most recent MAX_DEBUG_SERIAL_IDS serial IDs worth of
    debug images, pruning the oldest (by directory mtime) after each new
    analysis. Runs synchronously — it's fast since it only does a listdir +
    stat on the debug base dir, and only removes directories when over limit.
    """
    try:
        if not os.path.isdir(DEBUG_DIR_BASE):
            return

        entries = []
        for name in os.listdir(DEBUG_DIR_BASE):
            full_path = os.path.join(DEBUG_DIR_BASE, name)
            if os.path.isdir(full_path):
                try:
                    mtime = os.path.getmtime(full_path)
                    entries.append((mtime, full_path, name))
                except OSError:
                    continue

        if len(entries) <= MAX_DEBUG_SERIAL_IDS:
            return

        # Sort by mtime ascending (oldest first)
        entries.sort(key=lambda x: x[0])
        to_remove = len(entries) - MAX_DEBUG_SERIAL_IDS

        for i in range(to_remove):
            _, dir_path, dir_name = entries[i]
            try:
                shutil.rmtree(dir_path)
                log.info('debug_cleanup', 'Removed old debug dir',
                         serial_id=dir_name)
            except Exception as e:
                log.warn('debug_cleanup', 'Failed to remove debug dir',
                         serial_id=dir_name, error=str(e))

        log.info('debug_cleanup', 'Cleanup complete',
                 removed=to_remove, remaining=MAX_DEBUG_SERIAL_IDS)

    except Exception as e:
        # Never let cleanup failure affect the grading response
        log.warn('debug_cleanup', 'Cleanup error (non-fatal)', error=str(e))


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/debug/{serial_id}/{view}")
async def get_debug_image(serial_id: str, view: str):
    """
    Get debug image for a specific serial_id and view.

    Args:
        serial_id: Goat identifier
        view: One of 'side', 'top', 'front'

    Returns:
        JPEG image with measurement overlays
    """
    from fastapi.responses import FileResponse

    if view not in ['side', 'top', 'front']:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid view: {view}",
                "error_code": "INVALID_VIEW",
                "fix": "View must be one of: side, top, front"
            }
        )

    serial_id, error = sanitize_serial_id(serial_id)
    if error:
        raise HTTPException(status_code=400, detail=error)

    debug_path = f'{DEBUG_DIR_BASE}/{serial_id}/{view}_debug.jpg'

    if not os.path.exists(debug_path):
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Debug image not found for {serial_id}/{view}",
                "error_code": "NOT_FOUND",
                "fix": "Ensure the goat was analyzed and debug images were generated"
            }
        )

    return FileResponse(
        debug_path,
        media_type="image/jpeg",
        filename=f"{serial_id}_{view}_debug.jpg"
    )


@router.get("/debug/{serial_id}")
async def list_debug_images(serial_id: str):
    """
    List available debug images for a serial_id.

    Returns:
        List of available views with URLs
    """
    serial_id, error = sanitize_serial_id(serial_id)
    if error:
        raise HTTPException(status_code=400, detail=error)

    debug_dir = f'{DEBUG_DIR_BASE}/{serial_id}'

    if not os.path.exists(debug_dir):
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"No debug images found for serial_id: {serial_id}",
                "error_code": "NOT_FOUND"
            }
        )

    available_views = []
    for view in ['side', 'top', 'front']:
        debug_path = f'{debug_dir}/{view}_debug.jpg'
        if os.path.exists(debug_path):
            available_views.append({
                "view": view,
                "url": f"/debug/{serial_id}/{view}",
                "filename": f"{view}_debug.jpg"
            })

    return {
        "serial_id": serial_id,
        "debug_images": available_views,
        "count": len(available_views)
    }