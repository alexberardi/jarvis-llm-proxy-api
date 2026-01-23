import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AdapterTrainingError(RuntimeError):
    pass


def _validate_model_id(base_model_id: str) -> str:
    """Validate that base_model_id is a usable HuggingFace model ID or local path.
    
    Valid formats:
    - HuggingFace model ID: "meta-llama/Llama-3.2-3B-Instruct" (contains "/")
    - Local path: "/path/to/model" or "./models/llama" or ".models/llama"
    
    Raises AdapterTrainingError if the model ID is not valid.
    """
    # Check if it's a HuggingFace model ID (contains "/")
    if "/" in base_model_id:
        return base_model_id
    
    # Check if it's a local path that exists
    if base_model_id.startswith((".", "/", "~")):
        expanded = os.path.expanduser(base_model_id)
        if os.path.isdir(expanded):
            return expanded
        raise AdapterTrainingError(
            f"base_model_id '{base_model_id}' looks like a local path but directory not found"
        )
    
    # Not a valid format - raise clear error
    raise AdapterTrainingError(
        f"Invalid base_model_id: '{base_model_id}'. "
        f"Expected a HuggingFace model ID (e.g. 'meta-llama/Llama-3.2-3B-Instruct') "
        f"or a local path (e.g. '.models/llama-3.2-3b-instruct'). "
        f"The JCC should send the full model identifier."
    )


def _stable_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _dataset_hash(dataset_ref: Any) -> str:
    payload = _stable_json_dumps(dataset_ref)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _zip_dir(source_dir: Path, dest_zip: Path) -> None:
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source_dir))


def _build_paths(
    node_id: str,
    base_model_id: str,
    dataset_hash: str,
    job_id: str,
) -> Tuple[Path, Path, Path, Path]:
    # Use same env var as adapter_storage for consistency
    adapter_dir = Path(os.getenv("LLM_PROXY_ADAPTER_DIR", "/tmp/jarvis-adapters"))
    work_root = adapter_dir / "work" / job_id
    output_dir = work_root / "adapter"
    # Flat structure: adapters stored directly by hash
    artifact_dir = adapter_dir / dataset_hash
    artifact_path = artifact_dir / "adapter.zip"
    return work_root, output_dir, artifact_dir, artifact_path


def _artifact_url(artifact_path: Path) -> str:
    prefix = os.getenv("JARVIS_ADAPTER_PUBLIC_URL_PREFIX")
    if prefix:
        rel = artifact_path.as_posix()
        return prefix.rstrip("/") + "/" + rel.lstrip("/")
    return f"file://{artifact_path.as_posix()}"


def _detect_adapter_format(artifact_path: Path) -> str:
    """Detect adapter format from artifact zip.

    NOTE: GGUF LoRA is currently disabled. All adapters are PEFT format.
    This function is kept for forward compatibility in case GGUF support
    is re-enabled in the future.
    """
    # GGUF LoRA is disabled - always return peft_lora
    return "peft_lora"


def _get_db_session() -> Optional[Session]:
    """Get a database session if database is configured."""
    try:
        from db.session import SessionLocal
        return SessionLocal()
    except Exception as e:
        logger.warning(f"Database not available: {e}")
        return None


def _create_training_job(
    db: Session,
    job_id: str,
    dataset_hash: str,
    node_id: str,
    base_model_id: str,
) -> None:
    """Create a new training job record in PENDING status."""
    try:
        from db.models import TrainingJob
        job = TrainingJob(
            job_id=job_id,
            dataset_hash=dataset_hash,
            node_id=node_id,
            base_model_id=base_model_id,
            status="PENDING",
        )
        db.add(job)
        db.commit()
        logger.info(f"Created training job {job_id} with status PENDING")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create training job: {e}")


def _update_training_job_status(
    db: Session,
    job_id: str,
    status: str,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    artifact_path: Optional[str] = None,
    error_message: Optional[str] = None,
    progress_pct: Optional[float] = None,
) -> None:
    """Update a training job's status and related fields."""
    try:
        from db.models import TrainingJob
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            logger.warning(f"Training job {job_id} not found for status update")
            return

        job.status = status
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at
        if artifact_path is not None:
            job.artifact_path = artifact_path
        if error_message is not None:
            job.error_message = error_message
        if progress_pct is not None:
            job.progress_pct = progress_pct

        db.commit()
        logger.info(f"Updated training job {job_id} to status {status}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update training job status: {e}")


def _run_training_command(env: Dict[str, str], timeout_s: int) -> None:
    cmd = os.getenv("JARVIS_ADAPTER_TRAIN_CMD")
    if not cmd:
        raise AdapterTrainingError("JARVIS_ADAPTER_TRAIN_CMD is not set")
    args = shlex.split(cmd)
    if args:
        executable = shutil.which(args[0])
        if executable is None and args[0] == "python":
            args[0] = sys.executable
    log_path = env.get("JARVIS_TRAIN_LOG_PATH")
    try:
        if log_path:
            with open(log_path, "a", encoding="utf-8") as handle:
                subprocess.run(
                    args,
                    check=True,
                    env=env,
                    timeout=timeout_s,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        else:
            subprocess.run(
                args,
                check=True,
                env=env,
                timeout=timeout_s,
            )
    except subprocess.TimeoutExpired as exc:
        raise AdapterTrainingError(f"Training timed out after {timeout_s}s") from exc
    except subprocess.CalledProcessError as exc:
        if log_path and Path(log_path).exists():
            try:
                log_text = Path(log_path).read_text(encoding="utf-8")[-8000:]
                if log_text.strip():
                    print("📄 Adapter training output (tail):")
                    print(log_text)
                else:
                    print("📄 Adapter training output is empty.")
            except Exception as log_exc:
                print(f"⚠️  Failed to read training log: {log_exc}")
        raise AdapterTrainingError(f"Training command failed with exit code {exc.returncode}") from exc


def _wait_for_gpu_memory(min_free_gb: float = 6.0, timeout_s: int = 30) -> bool:
    """Wait for GPU memory to be freed before training."""
    import sys
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU memory wait", flush=True)
            return True
        
        import time
        start = time.time()
        print(f"🔍 Starting GPU memory check (need {min_free_gb} GiB free, timeout {timeout_s}s)", flush=True)
        
        while time.time() - start < timeout_s:
            free_mem_gb = torch.cuda.mem_get_info()[0] / (1024**3)
            total_mem_gb = torch.cuda.mem_get_info()[1] / (1024**3)
            
            if free_mem_gb >= min_free_gb:
                print(f"✅ GPU memory ready: {free_mem_gb:.2f}/{total_mem_gb:.2f} GiB free", flush=True)
                return True
            
            print(f"⏳ Waiting for GPU memory... {free_mem_gb:.2f}/{total_mem_gb:.2f} GiB free (need {min_free_gb} GiB)", flush=True)
            sys.stdout.flush()
            time.sleep(2)
        
        # Timeout - log warning but continue anyway
        free_mem_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        print(f"⚠️  GPU memory wait timeout. {free_mem_gb:.2f} GiB free, may be insufficient", flush=True)
        return False
    except ImportError:
        print("⚠️  torch not available, skipping GPU memory wait", flush=True)
        return True
    except Exception as e:
        print(f"⚠️  GPU memory check failed: {e}", flush=True)
        return True


def _debug_pause_model_service() -> None:
    debug_enabled = os.getenv("DEBUG", "false").lower() == "true"
    model_service_url = os.getenv("MODEL_SERVICE_URL")
    if not debug_enabled or not model_service_url:
        print("⏭️  Debug pause skipped (DEBUG not enabled or MODEL_SERVICE_URL not set)", flush=True)
        return

    token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    headers = {}
    if token:
        headers["X-Internal-Token"] = token

    unload_url = model_service_url.rstrip("/") + "/internal/model/unload"
    print(f"🔌 Requesting model unload from {unload_url}...", flush=True)
    try:
        with httpx.Client(timeout=60.0) as client:  # Longer timeout for vLLM cleanup
            resp = client.post(unload_url, headers=headers)
            if resp.status_code != 200:
                print(f"⚠️  Debug pause: unload failed {resp.status_code}: {resp.text}", flush=True)
            else:
                print("🧊 Debug pause: model service unloaded successfully", flush=True)
                # Wait for GPU memory to actually be freed
                _wait_for_gpu_memory(min_free_gb=6.0, timeout_s=30)
    except Exception as exc:
        print(f"⚠️  Debug pause: unload request failed: {exc}", flush=True)


def _debug_resume_model_service() -> None:
    debug_enabled = os.getenv("DEBUG", "false").lower() == "true"
    model_service_url = os.getenv("MODEL_SERVICE_URL")
    if not debug_enabled or not model_service_url:
        print("⏭️  Debug resume skipped (DEBUG not enabled or MODEL_SERVICE_URL not set)", flush=True)
        return

    # Force garbage collection to release training memory
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("🧹 Forced CUDA cache clear after training", flush=True)
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  CUDA cleanup warning: {e}", flush=True)

    # Wait for training's GPU memory to be released before attempting reload
    # vLLM at 75% utilization on a 12GB card needs ~9GB free
    min_free_for_vllm = 9.0  # Conservative estimate for 3B model on 12GB card
    print(f"🧹 Waiting for training memory to be released (need ~{min_free_for_vllm} GiB for vLLM)...", flush=True)
    _wait_for_gpu_memory(min_free_gb=min_free_for_vllm, timeout_s=60)

    token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    headers = {}
    if token:
        headers["X-Internal-Token"] = token

    reload_url = model_service_url.rstrip("/") + "/internal/model/reload"
    print(f"🔄 Requesting model reload from {reload_url}...", flush=True)
    try:
        with httpx.Client(timeout=120.0) as client:  # Longer timeout for model loading
            resp = client.post(reload_url, headers=headers)
            if resp.status_code != 200:
                print(f"⚠️  Debug resume: reload failed {resp.status_code}: {resp.text}", flush=True)
            else:
                print("🔥 Debug resume: model service reloaded successfully", flush=True)
    except Exception as exc:
        print(f"⚠️  Debug resume: reload request failed: {exc}", flush=True)


def run_adapter_training(request: Dict[str, Any], job_id: str, ttl_seconds: int) -> Dict[str, Any]:
    node_id = request.get("node_id")
    base_model_id = request.get("base_model_id")
    dataset_ref = request.get("dataset_ref")
    params = request.get("params") or {}
    if not node_id or not base_model_id:
        raise AdapterTrainingError("node_id and base_model_id are required")
    if dataset_ref is None:
        raise AdapterTrainingError("dataset_ref is required")

    dataset_hash = request.get("dataset_hash") or _dataset_hash(dataset_ref)
    work_root, output_dir, artifact_dir, artifact_path = _build_paths(
        node_id, base_model_id, dataset_hash, job_id
    )
    _ensure_dir(work_root)
    _ensure_dir(output_dir)
    _ensure_dir(artifact_dir)

    if artifact_path.exists():
        adapter_format = _detect_adapter_format(artifact_path)
        return {
            "artifact_url": _artifact_url(artifact_path),
            "artifact_metadata": {
                "node_id": node_id,
                "base_model_id": base_model_id,
                "dataset_hash": dataset_hash,
                "adapter_format": adapter_format,
                "artifact_path": str(artifact_path),
                "artifact_size_bytes": artifact_path.stat().st_size,
                "cached": True,
            },
        }

    dataset_path = work_root / "dataset.json"
    params_path = work_root / "params.json"
    dataset_path.write_text(_stable_json_dumps(dataset_ref), encoding="utf-8")
    params_path.write_text(_stable_json_dumps(params), encoding="utf-8")

    timeout_s = int(os.getenv("JARVIS_ADAPTER_TRAIN_TIMEOUT_SECONDS", str(ttl_seconds)))
    timeout_s = max(60, min(timeout_s, ttl_seconds))

    # Validate model ID - must be a HF model ID or local path
    training_model_id = _validate_model_id(base_model_id)
    
    env = os.environ.copy()
    if not env.get("HF_TOKEN") and env.get("HUGGINGFACE_HUB_TOKEN"):
        env["HF_TOKEN"] = env["HUGGINGFACE_HUB_TOKEN"]
    env.update(
        {
            "JARVIS_TRAIN_JOB_ID": job_id,
            "JARVIS_TRAIN_NODE_ID": node_id,
            "JARVIS_TRAIN_BASE_MODEL_ID": training_model_id,
            "JARVIS_TRAIN_DATASET_HASH": dataset_hash,
            "JARVIS_TRAIN_DATASET_PATH": str(dataset_path),
            "JARVIS_TRAIN_PARAMS_PATH": str(params_path),
            "JARVIS_TRAIN_OUTPUT_DIR": str(output_dir),
            "JARVIS_TRAIN_LOG_PATH": str(work_root / "train.log"),
            "PYTHONUNBUFFERED": "1",
        }
    )

    # Initialize database tracking
    db = _get_db_session()
    if db:
        _create_training_job(db, job_id, dataset_hash, node_id, base_model_id)

    started = time.time()
    _debug_pause_model_service()
    try:
        # Update status to RUNNING
        if db:
            _update_training_job_status(
                db, job_id, "RUNNING", started_at=datetime.now(timezone.utc)
            )

        _run_training_command(env, timeout_s)
    except Exception as e:
        # Update status to FAILED on error
        if db:
            _update_training_job_status(
                db, job_id, "FAILED", error_message=str(e)
            )
            db.close()
        _debug_resume_model_service()
        raise
    finally:
        _debug_resume_model_service()

    if not output_dir.exists() or not any(output_dir.iterdir()):
        error_msg = "Training completed but output directory is empty"
        if db:
            _update_training_job_status(db, job_id, "FAILED", error_message=error_msg)
            db.close()
        raise AdapterTrainingError(error_msg)

    _zip_dir(output_dir, artifact_path)
    adapter_format = _detect_adapter_format(artifact_path)
    duration_s = time.time() - started

    # Update status to COMPLETE
    if db:
        _update_training_job_status(
            db,
            job_id,
            "COMPLETE",
            completed_at=datetime.now(timezone.utc),
            artifact_path=str(artifact_path),
        )
        db.close()

    return {
        "artifact_url": _artifact_url(artifact_path),
        "artifact_metadata": {
            "node_id": node_id,
            "base_model_id": base_model_id,
            "dataset_hash": dataset_hash,
            "adapter_format": adapter_format,
            "artifact_path": str(artifact_path),
            "artifact_size_bytes": artifact_path.stat().st_size,
            "train_duration_seconds": round(duration_s, 2),
            "cached": False,
        },
    }
