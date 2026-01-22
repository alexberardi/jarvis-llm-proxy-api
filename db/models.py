"""Database models for training job tracking."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class TrainingJob(Base):
    """Training job status tracking."""

    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    dataset_hash: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    node_id: Mapped[str] = mapped_column(String(255), nullable=False)
    base_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="PENDING")  # PENDING, RUNNING, COMPLETE, FAILED
    progress_pct: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0-100 if trackable
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
