"""Logging configuration.

Sets up console logging and optional remote logging to jarvis-logs server.
"""

import logging
import os

_jarvis_handler = None


def setup_console_logging() -> logging.Logger:
    """Set up console logging for local development.

    Returns:
        The configured uvicorn logger.
    """
    console_level = os.getenv("JARVIS_LOG_CONSOLE_LEVEL", "WARNING")
    logging.basicConfig(
        level=getattr(logging, console_level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger("uvicorn")


def setup_remote_logging() -> None:
    """Set up remote logging to jarvis-logs server.

    This is called during application startup to enable centralized logging.
    """
    global _jarvis_handler
    try:
        from jarvis_log_client import init as init_log_client, JarvisLogHandler

        app_id = os.getenv("JARVIS_APP_ID", "llm-proxy")
        app_key = os.getenv("JARVIS_APP_KEY")
        if not app_key:
            return

        init_log_client(app_id=app_id, app_key=app_key)

        remote_level = os.getenv("JARVIS_LOG_REMOTE_LEVEL", "DEBUG")
        _jarvis_handler = JarvisLogHandler(
            service="llm-proxy",
            level=getattr(logging, remote_level.upper(), logging.DEBUG),
        )

        for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
            logging.getLogger(logger_name).addHandler(_jarvis_handler)

        logging.getLogger("uvicorn").info("Remote logging enabled to jarvis-logs")
    except ImportError:
        pass


def print_startup_info() -> None:
    """Print critical endpoint configs for visibility at startup."""
    print(f"MODEL_SERVICE_URL={os.getenv('MODEL_SERVICE_URL')}")
    print(
        f"MODEL_SERVICE_TOKEN set: {'yes' if os.getenv('MODEL_SERVICE_TOKEN') else 'no'}"
    )
    print(
        f"LLM_PROXY_INTERNAL_TOKEN set: {'yes' if os.getenv('LLM_PROXY_INTERNAL_TOKEN') else 'no'}"
    )
