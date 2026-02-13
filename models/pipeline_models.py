"""Pydantic models for the pipeline API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PipelineStep(str, Enum):
    """Individual steps in the build pipeline."""

    generate = "generate"
    train = "train"
    validate = "validate"
    merge = "merge"
    convert_gguf = "convert_gguf"
    convert_mlx = "convert_mlx"


class PipelineState(str, Enum):
    """Overall pipeline state."""

    idle = "idle"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class BuildConfig(BaseModel):
    """Configuration for a pipeline build run."""

    base_model: str = Field(
        default=".models/llama-3.1-8b-instruct",
        description="Path to the HuggingFace-format base model",
    )
    adapter_dir: str = Field(
        default="adapters/jarvis",
        description="Adapter output directory",
    )
    output_name: str | None = Field(
        default=None,
        description="Output model name (default: {base_model}-jarvis)",
    )
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=64)
    lora_r: int = Field(default=16, ge=4, le=256)
    optim: str = Field(default="adamw_8bit")
    gguf_quant: str = Field(default="Q4_K_M")
    mlx_bits: int = Field(default=4)
    formats: list[str] = Field(default=["gguf"])


class BuildRequest(BaseModel):
    """Request to start a pipeline build."""

    steps: list[PipelineStep] = Field(
        default=[
            PipelineStep.generate,
            PipelineStep.train,
            PipelineStep.validate,
            PipelineStep.merge,
            PipelineStep.convert_gguf,
        ],
        description="Steps to execute (can be a subset)",
    )
    config: BuildConfig = Field(default_factory=BuildConfig)


class StepStatus(BaseModel):
    """Status of an individual pipeline step."""

    step: PipelineStep
    state: PipelineState
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class PipelineStatus(BaseModel):
    """Current pipeline status."""

    state: PipelineState
    current_step: PipelineStep | None = None
    steps: list[StepStatus] = []
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class ArtifactInfo(BaseModel):
    """Info about a single artifact on disk."""

    name: str
    path: str
    size_gb: float | None = None


class AdapterInfo(BaseModel):
    """Info about a trained adapter."""

    name: str
    path: str
    has_config: bool = False


class TrainingDataInfo(BaseModel):
    """Info about training data file."""

    path: str
    num_examples: int = 0
    size_kb: float = 0


class ArtifactsResponse(BaseModel):
    """Response for GET /v1/pipeline/artifacts."""

    base_models: list[ArtifactInfo] = []
    adapters: list[AdapterInfo] = []
    merged_models: list[ArtifactInfo] = []
    gguf_models: list[ArtifactInfo] = []
    mlx_models: list[ArtifactInfo] = []
    training_data: TrainingDataInfo | None = None
