"""Database models for training job tracking and settings."""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
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


class Setting(Base):
    """Runtime settings that can be modified without restarting the service.

    Settings are organized by category and support type coercion.
    If a setting is not in the database, it falls back to the original
    environment variable (env_fallback).
    """

    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    key: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON-encoded for complex types
    value_type: Mapped[str] = mapped_column(String(50), nullable=False)  # string, int, float, bool, json
    category: Mapped[str] = mapped_column(String(100), index=True, nullable=False)  # model.main, inference.vllm, etc.
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    requires_reload: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_secret: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # Mask in responses
    env_fallback: Mapped[str | None] = mapped_column(String(255), nullable=True)  # Original env var name
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
