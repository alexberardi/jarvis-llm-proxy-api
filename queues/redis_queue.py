import os
import time
from typing import Any, Dict, Optional, Tuple

import redis
from rq import Queue

from services.settings_helpers import get_setting

def get_redis_connection() -> redis.Redis:
    url = os.getenv("REDIS_URL")
    if url:
        return redis.from_url(url)

    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    password = os.getenv("REDIS_PASSWORD")
    return redis.Redis(host=host, port=port, db=db, password=password)


def get_queue(name: Optional[str] = None) -> Queue:
    queue_name = name or get_setting(
        "queue.name", "LLM_PROXY_QUEUE_NAME", "llm_proxy_jobs"
    )
    conn = get_redis_connection()
    return Queue(queue_name, connection=conn)


def dedupe_key(job_id: str, idempotency_key: str) -> str:
    return f"llmproxy:dedupe:{job_id}:{idempotency_key}"


def mark_deduped(conn: redis.Redis, job_id: str, idempotency_key: str, ttl_seconds: int) -> bool:
    """
    Returns True if we successfully set the dedupe key (not seen before).
    Returns False if the key already exists (duplicate).
    """
    key = dedupe_key(job_id, idempotency_key)
    return bool(conn.set(key, "1", nx=True, ex=ttl_seconds))


def existing_dedup(conn: redis.Redis, job_id: str, idempotency_key: str) -> bool:
    return bool(conn.exists(dedupe_key(job_id, idempotency_key)))


def enqueue_job(payload: Dict[str, Any], ttl_seconds: int, queue_name: Optional[str] = None) -> Tuple[Queue, Any]:
    """
    Enqueue job with provided payload. Returns (queue, job) where job is the RQ Job instance.
    """
    q = get_queue(queue_name)
    job = q.enqueue("queues.tasks.process_llm_job", payload, job_timeout=ttl_seconds)
    return q, job


def current_timestamp_ms() -> int:
    return int(time.time() * 1000)

