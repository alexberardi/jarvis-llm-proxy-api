"""
RQ worker entrypoint for LLM Proxy jobs.
Usage:
    python scripts/queue_worker.py

NOTE: Do NOT import jarvis-log-client directly here. RQ workers fork child
processes (work horses) and the jarvis-log-client's network connections don't
survive fork. Instead, we use an HTTP handler to forward logs to the model
service, which then forwards them to jarvis-logs.
"""

import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

from rq import Worker


# ============================================================================
# HTTP Log Handler - forwards logs to model service
# ============================================================================

class HttpLogHandler(logging.Handler):
    """
    Logging handler that forwards log records to the model service via HTTP.

    Uses a background thread and queue for non-blocking log delivery.
    Falls back silently if the model service is unavailable.
    """

    def __init__(
        self,
        service_url: str,
        token: Optional[str] = None,
        level: int = logging.DEBUG,
        batch_size: int = 10,
        flush_interval: float = 2.0,
    ):
        super().__init__(level)
        self.service_url = service_url.rstrip("/") + "/internal/log"
        self.token = token
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: Queue = Queue()
        self._shutdown = False
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        """Queue a log record for async delivery."""
        if self._shutdown:
            return
        try:
            entry = {
                "level": record.levelname,
                "message": self.format(record),
                "logger_name": record.name,
            }
            self._queue.put(entry)
        except (RuntimeError, ValueError, AttributeError):
            pass  # Never fail on logging

    def _worker(self) -> None:
        """Background worker that sends logs to the model service."""
        import httpx
        import time

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Internal-Token"] = self.token

        while not self._shutdown:
            try:
                # Wait for an entry or timeout
                try:
                    entry = self._queue.get(timeout=self.flush_interval)
                except (TimeoutError, EOFError):
                    continue

                # Send the log entry
                try:
                    with httpx.Client(timeout=5.0) as client:
                        client.post(self.service_url, json=entry, headers=headers)
                except (httpx.HTTPError, OSError):
                    pass  # Silently ignore failures

            except (RuntimeError, ValueError):
                time.sleep(1)  # Back off on unexpected errors

    def close(self) -> None:
        """Shutdown the handler."""
        self._shutdown = True
        super().close()


# ============================================================================
# Logging setup
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "queue_worker.log"

# Create formatters
log_format = "[worker] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
formatter = logging.Formatter(log_format)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# File handler with rotation
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=10*1024*1024, backupCount=5  # 10MB, keep 5 backups
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

# HTTP handler disabled for now - causes issues with RQ fork model
# TODO: Re-enable after confirming core worker functionality
http_handler: Optional[HttpLogHandler] = None

# Configure root logger
handlers = [console_handler, file_handler]

logging.basicConfig(
    level=logging.DEBUG,
    handlers=handlers,
)
logger = logging.getLogger(__name__)

logger.info(f"🌐 MODEL_SERVICE_URL={os.getenv('MODEL_SERVICE_URL')}")
logger.info(f"🔐 MODEL_SERVICE_TOKEN set: {'yes' if os.getenv('MODEL_SERVICE_TOKEN') else 'no'}")
logger.info(f"🔐 LLM_PROXY_INTERNAL_TOKEN set: {'yes' if os.getenv('LLM_PROXY_INTERNAL_TOKEN') else 'no'}")
logger.info(f"🔐 JARVIS_AUTH_BASE_URL={os.getenv('JARVIS_AUTH_BASE_URL')}")
logger.info(f"📝 Log file: {LOG_FILE}")

# Add project root to path for imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Suppress noisy BrokenPipe traces when stdout/stderr pipes close during shutdown
logging.raiseExceptions = False
# Ignore SIGPIPE instead of terminating - prevents work horse from dying on broken pipes
signal.signal(signal.SIGPIPE, signal.SIG_IGN)

from queues.redis_queue import get_redis_connection


def main():
    conn = get_redis_connection()
    queue_name = os.getenv("LLM_PROXY_QUEUE_NAME", "llm_proxy_jobs")
    logger.info(f"👷 Starting worker on queue '{queue_name}'")
    worker = Worker([queue_name], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
