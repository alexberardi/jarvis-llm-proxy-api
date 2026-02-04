"""Training job service.

Provides utilities for creating and managing training job records in the database.
"""

import logging
from typing import Optional

logger = logging.getLogger("uvicorn")


def create_queued_training_job(
    job_id: str,
    node_id: str,
    base_model_id: str,
    dataset_hash: Optional[str] = None,
) -> bool:
    """Create a training job record with QUEUED status at enqueue time.

    Returns True if record was created/exists, False if DB unavailable.
    Handles race condition where worker might create the record first.
    """
    try:
        from sqlalchemy.exc import IntegrityError

        from db.session import SessionLocal
        from db.models import TrainingJob

        db = SessionLocal()
        try:
            # First check if job already exists (worker may have created it)
            existing = (
                db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            )
            if existing:
                logger.info(
                    f"Training job {job_id} already exists with status {existing.status}"
                )
                return True

            job = TrainingJob(
                job_id=job_id,
                dataset_hash=dataset_hash or "",
                node_id=node_id,
                base_model_id=base_model_id,
                status="QUEUED",
            )
            db.add(job)
            db.commit()
            logger.info(f"Created training job {job_id} with status QUEUED")
            return True
        except IntegrityError:
            # Race condition: worker created the record between our check and insert
            db.rollback()
            logger.info(f"Training job {job_id} created by worker (race condition)")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create queued training job: {e}")
            return False
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Database not available for job tracking: {e}")
        return False
