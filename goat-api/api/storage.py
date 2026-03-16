"""
Storage module for persisting analysis results

Currently uses local JSON file. Can be extended to use S3 Tables or other storage.

_write_results() now uses atomic writes (write to temp file, then
os.replace to final path) to prevent corruption if the process crashes
mid-write. Previously a crash during json.dump() would leave a truncated
or empty results.json, losing all historical data.
"""

import json
import os
import tempfile
import threading
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .logger import log
from .config import RESULTS_FILE, DATA_DIR


class ResultsStorage:
    """
    Thread-safe storage for analysis results.
    
    Uses a local JSON file with file locking for concurrent access.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._initialized = False
    
    def initialize(self) -> tuple[bool, Optional[str]]:
        """
        Initialize storage, creating directories and files if needed.
        
        Returns:
            (success, error_message)
        """
        try:
            # Create data directory
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Create or validate results file
            if not RESULTS_FILE.exists():
                self._write_results([])
                log.info('storage', 'Created results file', path=str(RESULTS_FILE))
            else:
                # Validate existing file
                try:
                    results = self._read_results()
                    log.info('storage', 'Loaded existing results',
                            path=str(RESULTS_FILE), count=len(results))
                except json.JSONDecodeError as e:
                    # Backup corrupted file and create new one
                    backup_path = RESULTS_FILE.with_suffix('.json.bak')
                    shutil.copy(RESULTS_FILE, backup_path)
                    self._write_results([])
                    log.warn('storage', 'Results file corrupted, created backup',
                            backup=str(backup_path), error=str(e))
            
            # Check disk space
            disk_usage = shutil.disk_usage(DATA_DIR)
            free_mb = disk_usage.free // (1024 * 1024)
            
            if free_mb < 100:
                log.warn('storage', 'Low disk space', free_mb=free_mb,
                        fix='Free up disk space on the server')
            
            self._initialized = True
            return True, None
            
        except PermissionError as e:
            error = f"Permission denied: {e}"
            log.error('storage', 'Cannot access data directory',
                     error=error, path=str(DATA_DIR),
                     fix='Check directory permissions')
            return False, error
            
        except Exception as e:
            error = f"Storage initialization failed: {e}"
            log.exception('storage', 'Initialization failed', error=str(e))
            return False, error
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def _read_results(self) -> List[Dict]:
        """Read results from file (no locking, internal use)"""
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    
    def _write_results(self, results: List[Dict]):
        """
        Write results to file atomically (no locking, internal use).

        Uses write-to-temp-then-rename pattern so that a crash
        mid-write never leaves a truncated/empty results.json. os.replace()
        is atomic on POSIX filesystems when src and dst are on the same
        filesystem (guaranteed here since both are in DATA_DIR).

        Previously this did a direct open('w') + json.dump(), meaning a
        crash or OOM kill during the write would corrupt the file and lose
        all historical results.
        """
        # Write to a temp file in the same directory (same filesystem = atomic rename)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(DATA_DIR),
            suffix='.json.tmp',
            prefix='results_'
        )
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(results, f, indent=2)
            # Atomic replace — old file is replaced in one operation
            os.replace(tmp_path, str(RESULTS_FILE))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    
    def save_result(self, result: Dict) -> tuple[bool, Optional[str]]:
        """
        Save an analysis result.
        
        Args:
            result: Analysis result dict
            
        Returns:
            (success, error_message)
        """
        if not self._initialized:
            return False, "Storage not initialized"
        
        serial_id = result.get('serial_id', 'unknown')
        
        try:
            with self._lock:
                results = self._read_results()
                
                # Check for duplicate
                existing_idx = None
                for i, r in enumerate(results):
                    if r.get('serial_id') == serial_id:
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    # Update existing
                    results[existing_idx] = result
                    log.info('storage', 'Updated existing result', serial_id=serial_id)
                else:
                    # Append new
                    results.append(result)
                    log.info('storage', 'Saved new result',
                            serial_id=serial_id, total=len(results))
                
                self._write_results(results)
            
            return True, None
            
        except Exception as e:
            error = f"Failed to save result: {e}"
            log.exception('storage', 'Save failed', serial_id=serial_id, error=str(e))
            return False, error
    
    def get_result(self, serial_id: str) -> Optional[Dict]:
        """Get a single result by serial_id"""
        if not self._initialized:
            return None
        
        try:
            with self._lock:
                results = self._read_results()
                
            for r in results:
                if r.get('serial_id') == serial_id:
                    return r
            
            return None
            
        except Exception as e:
            log.error('storage', 'Read failed', serial_id=serial_id, error=str(e))
            return None
    
    def get_all_results(self) -> List[Dict]:
        """Get all results"""
        if not self._initialized:
            return []
        
        try:
            with self._lock:
                return self._read_results()
        except Exception as e:
            log.error('storage', 'Read all failed', error=str(e))
            return []
    
    def delete_result(self, serial_id: str) -> bool:
        """Delete a result by serial_id"""
        if not self._initialized:
            return False
        
        try:
            with self._lock:
                results = self._read_results()
                original_count = len(results)
                
                results = [r for r in results if r.get('serial_id') != serial_id]
                
                if len(results) < original_count:
                    self._write_results(results)
                    log.info('storage', 'Deleted result', serial_id=serial_id)
                    return True
                else:
                    log.warn('storage', 'Result not found for deletion', serial_id=serial_id)
                    return False
                    
        except Exception as e:
            log.error('storage', 'Delete failed', serial_id=serial_id, error=str(e))
            return False


# Global storage instance
storage = ResultsStorage()