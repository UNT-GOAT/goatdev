# logger/pi_cloudwatch.py

import logging
import boto3
import time
import traceback
import threading

try:
    import watchtower
    WATCHTOWER_AVAILABLE = True
except ImportError:
    WATCHTOWER_AVAILABLE = False


def get_cloudwatch_logger(stream_name: str, level=logging.DEBUG):
    logger = logging.getLogger(stream_name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    if WATCHTOWER_AVAILABLE:
        try:
            client = boto3.client('logs', region_name='us-east-2')
            cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group='/goatdev',
                stream_name=stream_name,
                boto3_client=client,
                create_log_group=True,
                log_group_retention_days=30,
            )
            cloudwatch_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(cloudwatch_handler)
        except Exception as e:
            print(f"[WARN] [logger] CloudWatch init failed | error={e}")
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console_handler)
    
    return logger


class Logger:
    """
    Structured logger for field debugging.
    
    Format: [LEVEL] [component] message | key=value key2=value2
    
    Usage:
        log = Logger('pi/training')
        log.info('startup', 'Server started', port=5001)
        log.error('camera:side', 'Capture failed', error='device busy', fix='Unplug and replug USB')
    """
    
    def __init__(self, stream_name: str):
        self._logger = get_cloudwatch_logger(stream_name)
        self._lock = threading.Lock()
    
    def _format(self, level: str, component: str, message: str, **kwargs) -> str:
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
        Optionally include 'fix' kwarg with physical action to take.
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


# Backwards compatibility
class SimpleLogger:
    def __init__(self, stream_name: str):
        self.logger = Logger(stream_name)
    
    def __call__(self, msg: str):
        self.logger.info("app", msg)
    
    def error(self, msg: str):
        self.logger.error("app", msg)
    
    def warning(self, msg: str):
        self.logger.warn("app", msg)
    
    def debug(self, msg: str):
        self.logger.debug("app", msg)