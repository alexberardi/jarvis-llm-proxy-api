"""Loading the vLLM client, with an error that says what to do about it.

vLLM is an opt-in layer in the image (see the Dockerfile's INSTALL_VLLM arg):
it is the heaviest dependency in the stack, and nothing selects it by default --
prod serves GGUF models through the REST backend. So an image legitimately may
not have it, and three separate call sites used to discover that as a bare

    ModuleNotFoundError: No module named 'vllm'

surfacing from inside a lazy import, with nothing to say whether that was a
packaging bug, a broken install, or a deliberate build choice.
"""
import logging

logger = logging.getLogger(__name__)

VLLM_MISSING_MESSAGE = (
    "The vllm engine was selected but vLLM is not installed in this image. "
    "It is opt-in because it pulls torch, triton and CUDA kernels (several GB) "
    "and nothing selects it by default. Either rebuild the image with "
    "`--build-arg INSTALL_VLLM=true`, or pick another engine -- the REST "
    "backend serves GGUF models through llama.cpp, which is what prod uses."
)


def load_vllm_client():
    """Return the VLLMClient class, or raise with instructions.

    RuntimeError rather than letting ImportError through: the caller has already
    decided to use vLLM by this point, so the interesting fact is what to do
    about the image, not which module was missing.
    """
    try:
        from .vllm_backend import VLLMClient
    except ImportError as exc:
        logger.error("vLLM unavailable: %s", exc)
        raise RuntimeError(VLLM_MISSING_MESSAGE) from exc
    return VLLMClient
