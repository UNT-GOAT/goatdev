"""
Structured logging for Goat Grading API

Format: [LEVEL] [component] message | key=value key2=value2

Matches the Pi logging standards for consistent CloudWatch querying.
"""

import logging
import sys
import threading
import traceback
from typing import Optional


class Logger:
    """
    Structured logger with component tags and key=value formatting.
    
    Usage:
        log = Logger('api')
        log.info('startup', 'Server started', port=8000)
        log.error('model:side', 'Inference failed', error='OOM', fix='Reduce batch size')
    """
    
    def __init__(self, name: str = 'api'):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._lock = threading.Lock()
        
        # Only add handler if none exist
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self._logger.addHandler(handler)
    
    def _format(self, level: str, component: str, message: str, **kwargs) -> str:
        """Format: [LEVEL] [component] message | key=value key2=value2"""
        parts = [f"[{level}]", f"[{component}]", message]
        base = " ".join(parts)
        
        if kwargs:
            # Filter None values, truncate long strings
            kv_pairs = []
            for k, v in kwargs.items():
                if v is None:
                    continue
                if isinstance(v, str) and len(v) > 200:
                    v = v[:200] + "..."
                # Escape any pipe characters in values
                if isinstance(v, str):
                    v = v.replace('|', '\\|')
                kv_pairs.append(f"{k}={v}")
            if kv_pairs:
                return f"{base} | {' '.join(kv_pairs)}"
        return base
    
    def info(self, component: str, message: str, **kwargs):
        with self._lock:
            self._logger.info(self._format("INFO", component, message, **kwargs))
    
    def warn(self, component: str, message: str, **kwargs):
        with self._lock:
            self._logger.warning(self._format("WARN", component, message, **kwargs))
    
    def error(self, component: str, message: str, **kwargs):
        """
        Log an error. Always include 'error' kwarg with the actual error.
        Optionally include 'fix' kwarg with action to take.
        """
        with self._lock:
            self._logger.error(self._format("ERROR", component, message, **kwargs))
    
    def critical(self, component: str, message: str, **kwargs):
        """For errors that stop everything."""
        with self._lock:
            self._logger.critical(self._format("CRITICAL", component, message, **kwargs))
    
    def debug(self, component: str, message: str, **kwargs):
        with self._lock:
            self._logger.debug(self._format("DEBUG", component, message, **kwargs))
    
    def exception(self, component: str, message: str, **kwargs):
        """Log error with stack trace."""
        tb = traceback.format_exc()
        # Only include last 500 chars of traceback
        if len(tb) > 500:
            tb = "..." + tb[-500:]
        kwargs['traceback'] = tb.replace('\n', ' | ')
        self.error(component, message, **kwargs)


# Global logger instance
log = Logger('api')
