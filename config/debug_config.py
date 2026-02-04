"""Debug configuration.

Sets up debugpy for remote debugging when DEBUG=true.
"""

import os
import subprocess
import time


def setup_debugpy() -> None:
    """Set up debugpy for remote debugging.

    Only enables when DEBUG=true environment variable is set.
    Skips setup for worker processes (LLM_PROXY_PROCESS_ROLE=worker).
    """
    debug_enabled = os.getenv("DEBUG", "false").lower() == "true"
    skip_debugpy = os.getenv("LLM_PROXY_PROCESS_ROLE", "").lower() == "worker"
    debug_port = int(os.getenv("DEBUG_PORT", "5678"))

    if not debug_enabled or skip_debugpy:
        return

    try:
        import debugpy

        debugpy.listen(("0.0.0.0", debug_port))
        print(f"Debugger listening on port {debug_port}")
    except ImportError:
        print("debugpy is not installed, but DEBUG is set to true")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Debug port {debug_port} in use, attempting to free it...")
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
                print(f"Debugger listening on port {debug_port} (after clearing)")
            except Exception as retry_err:
                print(f"Could not start debugger after retry: {retry_err}")
        else:
            print(f"Could not start debugger on port {debug_port}: {e}")
    except Exception as e:
        print(f"Could not start debugger on port {debug_port}: {e}")
