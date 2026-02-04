"""Training job status API routes.

Endpoints for checking training job status.
"""

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/v1/training", tags=["training"])


@router.get("/status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status by job_id or dataset_hash.

    Args:
        job_id: Either the job_id or dataset_hash to look up

    Returns:
        Training job status information
    """
    try:
        from db.session import SessionLocal
        from db.models import TrainingJob
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"Database module not available: {e}"
        )

    db = SessionLocal()
    try:
        # Query by job_id or dataset_hash
        job = (
            db.query(TrainingJob)
            .filter(
                (TrainingJob.job_id == job_id) | (TrainingJob.dataset_hash == job_id)
            )
            .first()
        )

        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        return {
            "job_id": job.job_id,
            "dataset_hash": job.dataset_hash,
            "node_id": job.node_id,
            "base_model_id": job.base_model_id,
            "status": job.status,
            "progress_pct": job.progress_pct,
            "artifact_path": job.artifact_path,
            "error_message": job.error_message,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
    finally:
        db.close()
