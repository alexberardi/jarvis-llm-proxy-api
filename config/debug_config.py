"""Debug configuration.

Sets up debugpy for remote debugging when DEBUG=true.
"""

import logging
import os
import subprocess
import time

from services.settings_helpers import get_bool_setting, get_int_setting

logger = logging.getLogger("uvicorn")


def setup_debugpy() -> None:
    """Set up debugpy for remote debugging.

    Only enables when DEBUG=true environment variable is set.
    Skips setup for worker processes (LLM_PROXY_PROCESS_ROLE=worker).
    """
    debug_enabled = get_bool_setting("debug.enabled", "DEBUG", False)
    skip_debugpy = os.getenv("LLM_PROXY_PROCESS_ROLE", "").lower() == "worker"
    debug_port = get_int_setting("debug.port", "DEBUG_PORT", 5678)

    if not debug_enabled or skip_debugpy:
        return

    try:
        import debugpy

        debugpy.listen(("0.0.0.0", debug_port))
        logger.info(f"Debugger listening on port {debug_port}")
    except ImportError:
        logger.warning("debugpy is not installed, but DEBUG is set to true")
    except OSError as e:
        if "Address already in use" in str(e):
            logger.warning(f"Debug port {debug_port} in use, attempting to free it...")
            try:
                subprocess.run(
                    f"lsof -ti:{debug_port} | xargs kill -9",
                    shell=True,
                    capture_output=True,
                    timeout=5,
                )
                time.sleep(0.5)
                import debugpy

                debugpy.listen(("0.0.0.0", debug_port))
                logger.info(f"Debugger listening on port {debug_port} (after clearing)")
            except Exception as retry_err:
                logger.error(f"Could not start debugger after retry: {retry_err}")
        else:
            logger.error(f"Could not start debugger on port {debug_port}: {e}")
    except Exception as e:
        logger.error(f"Could not start debugger on port {debug_port}: {e}")
