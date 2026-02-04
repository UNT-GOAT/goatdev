import logging
import watchtower
import boto3
from datetime import datetime

def get_cloudwatch_logger(stream_name: str, level=logging.INFO):
    """
    Create a logger that sends to both CloudWatch and stdout.
    
    Args:
        stream_name: The log stream name (e.g., 'pi/capture', 'pi/training')
        level: Logging level (default INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(stream_name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    # CloudWatch handler
    try:
        client = boto3.client(region_name='us-east-2')
        cloudwatch_handler = watchtower.CloudWatchLogHandler(
            log_group='/goatdev',
            stream_name=stream_name,
            boto3_client=client,
            create_log_group=True,
            log_group_retention_days=30,  # Auto-set retention
        )
        cloudwatch_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(cloudwatch_handler)
    except Exception as e:
        print(f"[WARNING] CloudWatch logging failed to initialize: {e}")
        print("[WARNING] Falling back to stdout only")
    
    # Console handler (keeps journalctl working)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)
    
    return logger


# Convenience wrapper that mimics your existing log() function.
class SimpleLogger:
    """Drop-in replacement for existing log() function pattern."""
    
    def __init__(self, stream_name: str):
        self.logger = get_cloudwatch_logger(stream_name)
    
    def __call__(self, msg: str):
        """Call like: log("message") """
        self.logger.info(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)