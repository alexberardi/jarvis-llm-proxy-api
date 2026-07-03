"""Discrete-GPU auto-select for the llama.cpp / whisper.cpp GPU backends.

At process startup — BEFORE the GPU runtime initializes — this pins the backend
to the DISCRETE GPU chosen by device *type*, not by enumeration index, via the
appropriate visibility env var:

  * Vulkan   -> GGML_VK_VISIBLE_DEVICES = <index of the DISCRETE_GPU>
  * ROCm/HIP -> HIP_VISIBLE_DEVICES     = <index of the non-integrated device>

Why: on a box with a discrete GPU + an integrated GPU (e.g. an RX 9070 / gfx1201
next to a Ryzen iGPU / gfx1036) the iGPU can enumerate as device 0, so a hardcoded
"device 0" binds the wrong adapter — slow, or a hard crash on the kernel-less iGPU.
Selecting by type (DISCRETE_GPU / the HIP `integrated` flag) is index-order
independent, so it always lands on the real dGPU and always excludes the iGPU.

Contract:
  * Stdlib only (os, ctypes, subprocess, shutil, re, logging). NEVER imports
    llama_cpp / pywhispercpp / torch.
  * Idempotent and NEVER raises: any failure, missing tooling, or "no discrete GPU
    found" is a no-op that leaves the backend's own default in place.
  * An operator-set visibility env var (GGML_VK_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES
    / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES) is ALWAYS respected — we skip.
  * On CPU / CUDA / Metal images there is no AMD/Vulkan tooling, so it no-ops.
"""

import ctypes
import logging
import os
import re
import shutil
import subprocess

log = logging.getLogger("gpu-select")

# If any of these is already set, an operator (or the compose) chose the device;
# respect that and do nothing.
_OVERRIDE_VARS = (
    "GGML_VK_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)

_VK_DISCRETE = "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"

# gfx targets that are Ryzen/APU iGPUs (Raphael/Phoenix/Strix/etc.) — used only by
# the rocminfo fallback, which can't read the HIP `integrated` flag directly.
_APU_ARCHS = (
    "gfx1036", "gfx1037", "gfx1035", "gfx1103", "gfx1150", "gfx1151",
    "gfx90c", "gfx1013", "gfx103c", "gfx1033", "gfx1034",
)


# ---------------------------------------------------------------------------
# Vulkan
# ---------------------------------------------------------------------------
def _vk_via_vulkaninfo() -> list[tuple[int, str]]:
    """[(index, deviceName)] for DISCRETE Vulkan GPUs, in enumeration order.

    vulkaninfo GPU<n> == vkEnumeratePhysicalDevices index == the index that
    GGML_VK_VISIBLE_DEVICES uses (verified against llama.cpp ggml-vulkan).
    """
    if not shutil.which("vulkaninfo"):
        return []
    try:
        out = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    cur: int | None = None
    types: dict[int, str] = {}
    names: dict[int, str] = {}
    for line in out.splitlines():
        s = line.strip()
        m = re.match(r"GPU(\d+):", s)
        if m:
            cur = int(m.group(1))
            continue
        if cur is None:
            continue
        if "deviceType" in s and "=" in s:
            types[cur] = s.split("=", 1)[1].strip()
        elif "deviceName" in s and "=" in s:
            names[cur] = s.split("=", 1)[1].strip()
    return [(i, names.get(i, "")) for i in sorted(types) if types[i] == _VK_DISCRETE]


def _vk_via_libvulkan() -> list[tuple[int, str]]:
    """Fallback with no vulkan-tools dependency. VkPhysicalDeviceType 2 = DISCRETE_GPU."""
    try:
        vk = ctypes.CDLL("libvulkan.so.1")
    except OSError:
        return []

    class ICI(ctypes.Structure):
        _fields_ = [
            ("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p),
            ("flags", ctypes.c_uint32), ("pApplicationInfo", ctypes.c_void_p),
            ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.c_void_p),
            ("enabledExtensionCount", ctypes.c_uint32), ("ppEnabledExtensionNames", ctypes.c_void_p),
        ]

    inst = ctypes.c_void_p()
    ici = ICI(sType=1)  # VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    if vk.vkCreateInstance(ctypes.byref(ici), None, ctypes.byref(inst)) != 0:
        return []
    try:
        count = ctypes.c_uint32(0)
        vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), None)
        if count.value == 0:
            return []
        arr = (ctypes.c_void_p * count.value)()
        vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), arr)
        # VkPhysicalDeviceProperties: apiVersion,driverVersion,vendorID,deviceID
        # (4x u32) then deviceType (u32 @ offset 16), deviceName char[256] @ 20.
        props = (ctypes.c_uint8 * 824)()
        out: list[tuple[int, str]] = []
        for i in range(count.value):
            vk.vkGetPhysicalDeviceProperties(arr[i], props)
            dtype = int.from_bytes(bytes(props[16:20]), "little")
            if dtype == 2:  # DISCRETE_GPU
                name = bytes(props[20:20 + 256]).split(b"\x00", 1)[0].decode("utf-8", "replace")
                out.append((i, name))
        return out
    finally:
        vk.vkDestroyInstance(inst, None)


def select_discrete_vulkan_index() -> int | None:
    discrete = _vk_via_vulkaninfo() or _vk_via_libvulkan()
    if not discrete:
        return None
    prefer = os.getenv("JARVIS_GPU_PREFER")
    if prefer and len(discrete) > 1:
        for idx, name in discrete:
            if prefer.lower() in name.lower():
                return idx
    if len(discrete) > 1:
        log.warning("Multiple discrete Vulkan GPUs %s; pinning index %d. "
                    "Set GGML_VK_VISIBLE_DEVICES to override.", discrete, discrete[0][0])
    log.info("Discrete Vulkan GPU: index %d (%s)", discrete[0][0], discrete[0][1])
    return discrete[0][0]


# ---------------------------------------------------------------------------
# ROCm / HIP
# ---------------------------------------------------------------------------
class _HipDevicePropR0600(ctypes.Structure):
    # Mirror hipDeviceProp_tR0600 up to `integrated`; ctypes computes padding.
    # A tail pad lets the runtime write the full (larger) struct into our buffer.
    # `integrated` lands at offset 396 on x86_64 (guarded by a unit test).
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("uuid", ctypes.c_byte * 16),
        ("luid", ctypes.c_char * 8),
        ("luidDeviceNodeMask", ctypes.c_uint),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("textureAlignment", ctypes.c_size_t),
        ("texturePitchAlignment", ctypes.c_size_t),
        ("deviceOverlap", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("kernelExecTimeoutEnabled", ctypes.c_int),
        ("integrated", ctypes.c_int),
        ("_tail_pad", ctypes.c_byte * 4096),
    ]


def _rocm_via_hip() -> int | None:
    """Read hipDeviceProp_t.integrated (0=discrete, 1=APU/iGPU). Enumeration only —
    launches no kernels. ROCm 6/7 export the versioned symbol hipGetDevicePropertiesR0600."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        return None
    get_props = getattr(hip, "hipGetDevicePropertiesR0600", None) \
        or getattr(hip, "hipGetDeviceProperties", None)
    if get_props is None:
        return None
    get_props.restype = ctypes.c_int
    get_props.argtypes = [ctypes.POINTER(_HipDevicePropR0600), ctypes.c_int]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    n = ctypes.c_int(0)
    if hip.hipGetDeviceCount(ctypes.byref(n)) != 0 or n.value == 0:
        return None
    for idx in range(n.value):  # HIP index space == HIP_VISIBLE_DEVICES space
        p = _HipDevicePropR0600()
        if get_props(ctypes.byref(p), idx) != 0:
            continue
        name = p.name.decode("utf-8", "replace")
        log.info("HIP dev %d: %s integrated=%d", idx, name, p.integrated)
        if p.integrated == 0:  # 0 = discrete, 1 = APU/iGPU
            return idx
    return None


def _rocm_via_rocminfo() -> int | None:
    if not shutil.which("rocminfo"):
        return None
    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=15, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpu_idx = -1
    # rocminfo prints one block per HSA agent, separated by rows of asterisks.
    for block in re.split(r"\*{4,}", out):
        m_type = re.search(r"Device Type:\s*(\w+)", block)
        if not m_type or m_type.group(1) != "GPU":
            continue
        gpu_idx += 1  # GPU-agent order (assumed == HIP visible-device order here)
        m_name = re.search(r"Name:\s*(gfx\w+)", block)
        arch = m_name.group(1) if m_name else ""
        if any(arch.startswith(a) for a in _APU_ARCHS):
            continue  # integrated APU (e.g. gfx1036)
        return gpu_idx  # first non-APU discrete GPU
    return None


def select_discrete_rocm_index() -> int | None:
    idx = _rocm_via_hip()
    if idx is None:
        idx = _rocm_via_rocminfo()
    return idx


# ---------------------------------------------------------------------------
# CUDA topology (for GGUF split-mode auto)
# ---------------------------------------------------------------------------
def cuda_device_count() -> int:
    """Number of CUDA GPUs visible to this process. Never raises; 0 on any failure.

    An operator-set CUDA_VISIBLE_DEVICES is respected: comma-separated non-empty
    entries are counted (index or GPU-UUID form — both select a device), stopping
    at (and excluding) any "-1" entry, because the CUDA runtime truncates the
    visible list at the first invalid entry. Set-but-empty means "no GPUs".

    Otherwise the count comes from `nvidia-smi --list-gpus`. Gating on the
    nvidia-smi TOOL (injected into CUDA containers by nvidia-container-toolkit)
    follows this module's probe rule: tooling presence, NEVER a merely-loadable
    shared library — a ctypes probe of libcuda would repeat the Mesa-llvmpipe
    libvulkan SIGSEGV footgun on CPU images (see _detect_backend).
    """
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            count = 0
            for entry in visible.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if entry == "-1":
                    break
                count += 1
            return count
        if not shutil.which("nvidia-smi"):
            return 0
        out = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout
        return len(re.findall(r"^GPU \d+: ", out, flags=re.MULTILINE))
    except Exception:  # contract: never raises
        return 0


def auto_gguf_split_mode() -> int:
    """Resolve the GGUF split-mode AUTO sentinel (-1) from the visible topology.

    Returns 1 (LLAMA_SPLIT_MODE_LAYER) when >=2 CUDA GPUs are visible — the only
    topology that genuinely wants layer split (e.g. the dual-RTX-3090 prod box,
    where single-GPU mode piles both models onto GPU0 and OOMs at boot) — and
    0 (SPLIT_MODE_NONE) otherwise.

    Why NVIDIA-only: blanket auto-split across every visible device is a footgun
    on AMD boxes where a kernel-less iGPU (e.g. gfx1036) enumerates next to the
    dGPU — llama.cpp faults trying to place tensors on it. But those boxes are
    already visibility-pinned to the single discrete GPU by select_discrete_gpu()
    (HIP_VISIBLE_DEVICES / GGML_VK_VISIBLE_DEVICES), so exactly one device is
    visible there and split mode is irrelevant. CPU/Metal images have no
    nvidia-smi, count 0, and stay single-GPU.
    """
    return 1 if cuda_device_count() >= 2 else 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _detect_backend() -> str | None:
    hint = (os.getenv("JARVIS_GPU_BACKEND") or "").strip().lower()
    if hint in ("none", "cpu"):
        return None
    if hint in ("rocm", "vulkan"):
        return hint
    # Gate on GPU TOOLING / paths, NEVER on a merely-loadable lib. CPU images ship
    # Mesa's llvmpipe libvulkan.so.1 (software Vulkan) — probing it via ctypes
    # SIGSEGVs, and a native crash bypasses select_discrete_gpu()'s try/except.
    # vulkaninfo (vulkan-tools) / rocminfo / /opt/rocm exist ONLY on the real GPU
    # image variants, so they are the safe "this is actually a GPU image" signal.
    if os.path.isdir("/opt/rocm") or shutil.which("rocminfo"):
        return "rocm"
    if shutil.which("vulkaninfo"):
        return "vulkan"
    return None


def select_discrete_gpu() -> None:
    """Pin the GPU backend to the discrete GPU. Idempotent; never raises."""
    try:
        for var in _OVERRIDE_VARS:
            if os.environ.get(var):
                log.info("%s already set (%s); leaving GPU selection to the operator.",
                         var, os.environ[var])
                return
        backend = _detect_backend()
        if backend == "rocm":
            idx = select_discrete_rocm_index()
            if idx is not None:
                os.environ["HIP_VISIBLE_DEVICES"] = str(idx)  # HIP only; NOT ROCR (avoid double-remap)
                log.info("Auto-selected discrete AMD GPU: HIP_VISIBLE_DEVICES=%d", idx)
            else:
                log.warning("ROCm detected but no discrete GPU identified; leaving backend default.")
        elif backend == "vulkan":
            idx = select_discrete_vulkan_index()
            if idx is not None:
                os.environ["GGML_VK_VISIBLE_DEVICES"] = str(idx)
                log.info("Auto-selected discrete Vulkan GPU: GGML_VK_VISIBLE_DEVICES=%d", idx)
            else:
                log.warning("Vulkan detected but no discrete GPU identified; leaving backend default.")
        # else: CPU / CUDA / Metal image — no AMD/Vulkan tooling present, nothing to do.
    except Exception as e:  # never break process startup
        log.warning("discrete-GPU auto-select skipped (%s: %s)", type(e).__name__, e)
