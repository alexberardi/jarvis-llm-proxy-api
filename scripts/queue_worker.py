"""
RQ worker entrypoint for LLM Proxy jobs.
Usage:
    python scripts/queue_worker.py
"""

import os

import sys
import signal
import logging
from pathlib import Path

from rq import Worker

print(f"🌐 MODEL_SERVICE_URL={os.getenv('MODEL_SERVICE_URL')}")
print(f"🔐 MODEL_SERVICE_TOKEN set: {'yes' if os.getenv('MODEL_SERVICE_TOKEN') else 'no'}")
print(f"🔐 LLM_PROXY_INTERNAL_TOKEN set: {'yes' if os.getenv('LLM_PROXY_INTERNAL_TOKEN') else 'no'}")
print(f"🔐 JARVIS_AUTH_BASE_URL={os.getenv('JARVIS_AUTH_BASE_URL')}")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Suppress noisy BrokenPipe traces when stdout/stderr pipes close during shutdown
logging.raiseExceptions = False
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

from queues.redis_queue import get_redis_connection


def main():
    conn = get_redis_connection()
    queue_name = os.getenv("LLM_PROXY_QUEUE_NAME", "llm_proxy_jobs")
    print(f"👷 Starting worker on queue '{queue_name}'")
    worker = Worker([queue_name], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()

