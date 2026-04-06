"""
S3 Archival

EC2 owns ALL S3 archival. The Pi does not write to S3.

Responsibility:
  goat-captures bucket  — raw uploaded images (as received from Pi)
  goat-processed bucket — debug overlay images

Archival only happens on successful grades (all views OK).
Runs in a background thread so it never blocks the API response.
"""

import os
import threading

from .logger import log
from .config import S3_CAPTURES_BUCKET, S3_PROCESSED_BUCKET

_s3_client = None
_s3_lock = threading.Lock()


def get_s3():
    """Lazy-init S3 client. Thread-safe."""
    global _s3_client
    if _s3_client is None:
        with _s3_lock:
            if _s3_client is None:
                import boto3
                _s3_client = boto3.client('s3')
    return _s3_client


def _upload_bytes(bucket: str, key: str, data: bytes, content_type: str = 'image/jpeg'):
    """Upload bytes to S3. Logs warning on failure, never raises."""
    try:
        get_s3().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        log.info('s3', 'Uploaded', bucket=bucket, key=key, size_bytes=len(data))
    except Exception as e:
        log.warn('s3', 'Upload failed (non-fatal)', bucket=bucket, key=key, error=str(e))


def _upload_file(bucket: str, key: str, filepath: str, content_type: str = 'image/jpeg'):
    """Upload a local file to S3. Logs warning on failure, never raises."""
    try:
        get_s3().upload_file(filepath, bucket, key, ExtraArgs={'ContentType': content_type})
        log.info('s3', 'Uploaded file', bucket=bucket, key=key,
                 size_bytes=os.path.getsize(filepath))
    except Exception as e:
        log.warn('s3', 'File upload failed (non-fatal)', bucket=bucket, key=key, error=str(e))


def archive_to_s3(
    serial_id: str,
    raw_images: dict,
    debug_image_paths: dict,
):
    """
    Archive a successful grade to S3. Intended to run in a background thread.

    Writes:
      goat-captures/{serial_id}/{view}.jpg        — raw images
      goat-processed/{serial_id}/{view}_debug.jpg  — debug overlays

    Args:
        serial_id: Goat identifier
        raw_images: Dict of {view_name: bytes} for the raw uploaded images
        debug_image_paths: Dict of {view_name: filepath} for debug images on disk
    """
    try:
        # Raw images → captures bucket
        if S3_CAPTURES_BUCKET:
            for view_name, raw_bytes in raw_images.items():
                _upload_bytes(
                    S3_CAPTURES_BUCKET,
                    f'{serial_id}/{view_name}.jpg',
                    raw_bytes
                )
        else:
            log.warn('s3:captures', 'S3_CAPTURES_BUCKET not set, skipping')

        # Debug images → processed bucket
        if S3_PROCESSED_BUCKET:
            for view_name, debug_path in debug_image_paths.items():
                if os.path.exists(debug_path):
                    _upload_file(
                        S3_PROCESSED_BUCKET,
                        f'{serial_id}/{view_name}_debug.jpg',
                        debug_path
                    )
        else:
            log.warn('s3:processed', 'S3_PROCESSED_BUCKET not set, skipping')

        log.info('s3', 'Archival complete', serial_id=serial_id)

    except Exception as e:
        # Catch-all so the background thread never crashes silently
        log.warn('s3', 'Archival failed (non-fatal)', serial_id=serial_id, error=str(e))


def archive_in_background(serial_id: str, raw_images: dict, debug_image_paths: dict):
    """Spawn a daemon thread to archive to S3 without blocking the response."""
    thread = threading.Thread(
        target=archive_to_s3,
        args=(serial_id, raw_images, debug_image_paths),
        daemon=True
    )
    thread.start()
    log.info('s3', 'Archival started in background', serial_id=serial_id)