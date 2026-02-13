"""Pipeline orchestration service.

Runs build_jarvis_model.py (or individual scripts) as subprocesses,
tracks state, captures logs, and pushes to SSE subscribers.
Only one pipeline run at a time (mutex).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from models.pipeline_models import (
    AdapterInfo,
    ArtifactInfo,
    ArtifactsResponse,
    BuildConfig,
    BuildRequest,
    PipelineState,
    PipelineStatus,
    PipelineStep,
    StepStatus,
    TrainingDataInfo,
)

logger = logging.getLogger("uvicorn")

PROJECT_ROOT = Path(__file__).parent.parent

# Step → (script path, description, argument builder)
STEP_SCRIPTS: dict[PipelineStep, str] = {
    PipelineStep.generate: "scripts/generate_jarvis_training_data.py",
    PipelineStep.train: "scripts/train_jarvis_adapter.py",
    PipelineStep.validate: "scripts/validate_jarvis_adapter.py",
    PipelineStep.merge: "scripts/merge_adapter.py",
    PipelineStep.convert_gguf: "scripts/convert_to_gguf.py",
    PipelineStep.convert_mlx: "scripts/convert_to_mlx.py",
}


def _build_step_args(step: PipelineStep, config: BuildConfig) -> list[str]:
    """Build CLI arguments for each pipeline step."""
    base_model = config.base_model
    adapter_dir = config.adapter_dir
    output_name = config.output_name or f"{Path(base_model).name}-jarvis"
    merged_dir = str(Path(base_model).parent / output_name)

    if step == PipelineStep.generate:
        return []
    elif step == PipelineStep.train:
        return [
            "--base-model", base_model,
            "--output-dir", adapter_dir,
            "--epochs", str(config.epochs),
            "--batch-size", str(config.batch_size),
            "--optim", config.optim,
            "--lora-r", str(config.lora_r),
        ]
    elif step == PipelineStep.validate:
        return ["--adapter-path", adapter_dir]
    elif step == PipelineStep.merge:
        return [
            "--base-model", base_model,
            "--adapter", adapter_dir,
            "--output", merged_dir,
        ]
    elif step == PipelineStep.convert_gguf:
        quant_type = config.gguf_quant or "f16"
        if quant_type.lower() == "f16":
            gguf_output = f"{merged_dir}.gguf"
        else:
            gguf_output = f"{merged_dir}-{quant_type}.gguf"
        return [
            "--model", merged_dir,
            "--output", gguf_output,
            "--quant-type", quant_type,
        ]
    elif step == PipelineStep.convert_mlx:
        mlx_output = f"{merged_dir}-mlx-{config.mlx_bits}bit"
        return [
            "--model", merged_dir,
            "--output", mlx_output,
            "--bits", str(config.mlx_bits),
        ]
    return []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineService:
    """Singleton pipeline orchestrator."""

    def __init__(self) -> None:
        self._state: PipelineState = PipelineState.idle
        self._current_step: PipelineStep | None = None
        self._steps: list[StepStatus] = []
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._error: str | None = None
        self._logs: deque[str] = deque(maxlen=5000)
        self._subscribers: list[asyncio.Queue[str | None]] = []
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._lock = asyncio.Lock()

    def get_status(self) -> PipelineStatus:
        return PipelineStatus(
            state=self._state,
            current_step=self._current_step,
            steps=list(self._steps),
            started_at=self._started_at,
            finished_at=self._finished_at,
            error=self._error,
        )

    async def start_build(self, request: BuildRequest) -> PipelineStatus:
        """Start a pipeline build. Raises if already running."""
        async with self._lock:
            if self._state == PipelineState.running:
                raise RuntimeError("Pipeline is already running")

            self._state = PipelineState.running
            self._current_step = None
            self._steps = [
                StepStatus(step=s, state=PipelineState.idle) for s in request.steps
            ]
            self._started_at = _now_iso()
            self._finished_at = None
            self._error = None
            self._logs.clear()

            self._task = asyncio.create_task(
                self._run_pipeline(request.steps, request.config)
            )

        return self.get_status()

    async def cancel(self) -> PipelineStatus:
        """Cancel a running pipeline."""
        if self._state != PipelineState.running:
            return self.get_status()

        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()

        if self._task and not self._task.done():
            self._task.cancel()

        self._state = PipelineState.cancelled
        self._finished_at = _now_iso()
        self._push_log("[pipeline] Cancelled by user")

        return self.get_status()

    def subscribe_logs(self) -> asyncio.Queue[str | None]:
        """Subscribe to live log events. Returns a queue that receives log lines."""
        q: asyncio.Queue[str | None] = asyncio.Queue()
        # Send existing logs as backfill
        for line in self._logs:
            q.put_nowait(line)
        self._subscribers.append(q)
        return q

    def unsubscribe_logs(self, q: asyncio.Queue[str | None]) -> None:
        """Remove a log subscriber."""
        if q in self._subscribers:
            self._subscribers.remove(q)

    def _push_log(self, line: str) -> None:
        """Push a log line to the buffer and all subscribers."""
        self._logs.append(line)
        for q in self._subscribers:
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                pass

    async def _run_pipeline(
        self, steps: list[PipelineStep], config: BuildConfig
    ) -> None:
        """Execute pipeline steps sequentially."""
        try:
            for i, step in enumerate(steps):
                self._current_step = step
                self._steps[i].state = PipelineState.running
                self._steps[i].started_at = _now_iso()
                self._push_log(f"[pipeline] Starting step: {step.value}")

                success = await self._run_step(step, config)

                if success:
                    self._steps[i].state = PipelineState.completed
                    self._steps[i].finished_at = _now_iso()
                    self._push_log(f"[pipeline] Completed step: {step.value}")
                else:
                    self._steps[i].state = PipelineState.failed
                    self._steps[i].finished_at = _now_iso()
                    self._steps[i].error = f"Step {step.value} failed"
                    self._push_log(f"[pipeline] Failed step: {step.value}")
                    self._state = PipelineState.failed
                    self._error = f"Step {step.value} failed"
                    self._finished_at = _now_iso()
                    self._notify_done()
                    return

            self._state = PipelineState.completed
            self._current_step = None
            self._finished_at = _now_iso()
            self._push_log("[pipeline] All steps completed successfully")
            self._notify_done()

        except asyncio.CancelledError:
            self._state = PipelineState.cancelled
            self._finished_at = _now_iso()
            self._push_log("[pipeline] Cancelled")
            self._notify_done()
        except Exception as e:
            self._state = PipelineState.failed
            self._error = str(e)
            self._finished_at = _now_iso()
            self._push_log(f"[pipeline] Error: {e}")
            self._notify_done()

    def _notify_done(self) -> None:
        """Send None sentinel to all subscribers to signal stream end."""
        for q in self._subscribers:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def _run_step(self, step: PipelineStep, config: BuildConfig) -> bool:
        """Run a single pipeline step as a subprocess."""
        script = STEP_SCRIPTS.get(step)
        if not script:
            self._push_log(f"[pipeline] Unknown step: {step.value}")
            return False

        script_path = PROJECT_ROOT / script
        if not script_path.exists():
            self._push_log(f"[pipeline] Script not found: {script}")
            return False

        args = _build_step_args(step, config)
        cmd = [sys.executable, str(script_path)] + args

        self._push_log(f"[pipeline] Running: {' '.join(cmd)}")

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )

        if self._process.stdout:
            async for line_bytes in self._process.stdout:
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._push_log(line)

        returncode = await self._process.wait()
        self._process = None
        return returncode == 0

    def get_artifacts(self) -> ArtifactsResponse:
        """Scan disk for models, adapters, GGUF/MLX files, and training data."""
        models_dir = PROJECT_ROOT / ".models"
        adapters_dir = PROJECT_ROOT / "adapters"
        data_dir = PROJECT_ROOT / "data"

        base_models: list[ArtifactInfo] = []
        merged_models: list[ArtifactInfo] = []
        gguf_models: list[ArtifactInfo] = []
        mlx_models: list[ArtifactInfo] = []
        adapters: list[AdapterInfo] = []

        # Scan .models/ directory
        if models_dir.exists():
            for item in sorted(models_dir.iterdir()):
                if item.is_file() and item.suffix == ".gguf":
                    size_gb = item.stat().st_size / (1024**3)
                    gguf_models.append(
                        ArtifactInfo(name=item.name, path=str(item), size_gb=round(size_gb, 2))
                    )
                elif item.is_dir():
                    size_gb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024**3)
                    name = item.name
                    if "-jarvis" in name and "-mlx-" in name:
                        mlx_models.append(
                            ArtifactInfo(name=name, path=str(item), size_gb=round(size_gb, 2))
                        )
                    elif "-jarvis" in name:
                        merged_models.append(
                            ArtifactInfo(name=name, path=str(item), size_gb=round(size_gb, 2))
                        )
                    else:
                        base_models.append(
                            ArtifactInfo(name=name, path=str(item), size_gb=round(size_gb, 2))
                        )

        # Scan adapters/ directory
        if adapters_dir.exists():
            for item in sorted(adapters_dir.iterdir()):
                if item.is_dir():
                    has_config = (item / "adapter_config.json").exists()
                    adapters.append(
                        AdapterInfo(name=item.name, path=str(item), has_config=has_config)
                    )

        # Training data
        training_data: TrainingDataInfo | None = None
        training_file = data_dir / "jarvis_training.jsonl"
        if training_file.exists():
            size_kb = training_file.stat().st_size / 1024
            num_examples = sum(1 for _ in training_file.open())
            training_data = TrainingDataInfo(
                path=str(training_file),
                num_examples=num_examples,
                size_kb=round(size_kb, 1),
            )

        return ArtifactsResponse(
            base_models=base_models,
            adapters=adapters,
            merged_models=merged_models,
            gguf_models=gguf_models,
            mlx_models=mlx_models,
            training_data=training_data,
        )

    async def stream_logs(self) -> AsyncGenerator[str, None]:
        """Async generator that yields log lines for SSE."""
        q = self.subscribe_logs()
        try:
            while True:
                line = await q.get()
                if line is None:
                    # Pipeline finished
                    break
                yield line
        finally:
            self.unsubscribe_logs(q)


# Module-level singleton
_pipeline_service: PipelineService | None = None


def get_pipeline_service() -> PipelineService:
    """Get or create the singleton pipeline service."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service
