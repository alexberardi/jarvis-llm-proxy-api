"""vLLM must not be built into the image by default.

It is the heaviest dependency in the stack -- torch, triton, CUDA kernels,
several GB -- and nothing selects it: prod serves GGUF models through the REST
backend (llama.cpp), chosen because vLLM had a split-GPU problem on that box.

Nobody ever switched it on deliberately; it was simply installed
unconditionally, and because the jarvis CLI captures `docker compose up`
output, the cost was invisible. install-e2e-quickstart spent ~115 minutes per
run compiling it, printed nothing, and was killed by its job timeout every
night from 2026-09-08 -- which read as a hang rather than a build.

So: opt-in via a build arg, and a clear error if the engine is selected on an
image that does not have it.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text()


class TestTheImage:
    def test_vllm_is_behind_a_build_arg(self):
        dockerfile = _read("Dockerfile")
        assert "ARG INSTALL_VLLM" in dockerfile, "vLLM has no opt-in switch"

    def test_the_default_is_off(self):
        # A default of true would keep the two hours and merely make it
        # configurable, which is not the point.
        assert "ARG INSTALL_VLLM=false" in _read("Dockerfile")

    def test_the_install_is_conditional(self):
        dockerfile = _read("Dockerfile")
        # The bare form -- `RUN pip install ... -r requirements-vllm.txt` with no
        # guard -- is what cost the lane its nights.
        for line in dockerfile.splitlines():
            stripped = line.strip()
            if stripped.startswith("RUN pip install") and "requirements-vllm" in stripped:
                pytest.fail(f"vLLM installed unconditionally: {stripped}")
        assert "INSTALL_VLLM" in dockerfile and "requirements-vllm.txt" in dockerfile


class TestTheVersionFloors:
    @pytest.mark.parametrize(
        "requirements,package",
        [
            ("requirements-base.txt", "torch"),
            ("requirements-transformers.txt", "torch"),
            ("requirements-vllm.txt", "vllm"),
        ],
    )
    def test_the_heavy_packages_have_an_upper_bound(self, requirements, package):
        """An unbounded floor means every rebuild takes whatever is newest.

        That is how this image grew without anyone choosing it -- the same shape
        as `minio/minio:latest` and `pip install --upgrade pip`, both of which
        broke something else this week. The Dockerfile already pins
        llama-cpp-python for exactly this reason.
        """
        line = next(
            l for l in _read(requirements).splitlines()
            if l.strip().startswith(package)
        )
        assert "<" in line, f"{package} in {requirements} has no upper bound: {line.strip()!r}"


class TestSelectingItWithoutIt:
    def test_the_error_says_how_to_get_vllm(self):
        from backends.vllm_availability import load_vllm_client

        # Simulate the lean image: importing vllm fails. Both the package and any
        # cached backend module are blanked, or a previous import would satisfy it.
        with patch.dict(sys.modules, {"vllm": None, "backends.vllm_backend": None}):
            with pytest.raises(RuntimeError) as exc:
                load_vllm_client()

        message = str(exc.value)
        assert "INSTALL_VLLM=true" in message, "the error does not say how to fix it"
        assert "REST" in message, "the error does not offer the alternative prod uses"

    def test_it_is_a_runtime_error_not_an_import_error(self):
        # The caller has already chosen vLLM by then; the interesting fact is
        # what to do about the image, not which module was missing.
        from backends.vllm_availability import load_vllm_client

        with patch.dict(sys.modules, {"vllm": None, "backends.vllm_backend": None}):
            with pytest.raises(RuntimeError):
                load_vllm_client()
