"""Unit tests for the discrete-GPU auto-select helper.

These run anywhere (no GPU): tooling (vulkaninfo / rocminfo / libvulkan / libamdhip64)
is mocked, so only the parsing + selection logic + the never-raise / respect-override
contract are exercised. The decisive assertion is that an iGPU-first enumeration
selects the DISCRETE index (1), NOT the hardcoded 0 — the whole point of the fix.
"""

import types

import gpu_select


# vulkaninfo --summary with the integrated GPU enumerating FIRST (index 0) and the
# discrete RX 9070 SECOND (index 1) — the exact case that broke the hardcoded "=0".
VULKANINFO_IGPU_FIRST = """\
Devices:
========
GPU0:
\tapiVersion         = 1.4.305
\tvendorID           = 0x1002
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName         = AMD Radeon Graphics (RADV GFX1036)
GPU1:
\tapiVersion         = 1.4.305
\tvendorID           = 0x1002
\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName         = AMD Radeon RX 9070 (RADV GFX1201)
"""

# rocminfo with a CPU agent, then the iGPU (gfx1036), then the discrete gfx1201.
ROCMINFO_DUAL = """\
*******
Agent 1
*******
  Name:                    AMD Ryzen 7 7700X
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1036
  Marketing Name:          AMD Radeon Graphics
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon RX 9070
  Device Type:             GPU
"""


def _fake_run(stdout):
    return lambda *a, **k: types.SimpleNamespace(stdout=stdout, returncode=0)


def _which(*present):
    return lambda name: f"/usr/bin/{name}" if name in present else None


# --------------------------------------------------------------------------
# Vulkan
# --------------------------------------------------------------------------
def test_vulkaninfo_selects_discrete_not_index_zero(monkeypatch):
    monkeypatch.setattr(gpu_select.shutil, "which", _which("vulkaninfo"))
    monkeypatch.setattr(gpu_select.subprocess, "run", _fake_run(VULKANINFO_IGPU_FIRST))

    discrete = gpu_select._vk_via_vulkaninfo()
    assert [i for i, _ in discrete] == [1]  # the DISCRETE GPU, not the iGPU at 0
    assert gpu_select.select_discrete_vulkan_index() == 1


def test_vulkaninfo_missing_tool_returns_empty(monkeypatch):
    monkeypatch.setattr(gpu_select.shutil, "which", _which())  # nothing present
    assert gpu_select._vk_via_vulkaninfo() == []


def test_vulkaninfo_garbage_never_raises(monkeypatch):
    monkeypatch.setattr(gpu_select.shutil, "which", _which("vulkaninfo"))
    monkeypatch.setattr(gpu_select.subprocess, "run", _fake_run("not vulkan output at all"))
    assert gpu_select._vk_via_vulkaninfo() == []
    assert gpu_select.select_discrete_vulkan_index() is None


# --------------------------------------------------------------------------
# ROCm
# --------------------------------------------------------------------------
def test_rocminfo_skips_apu_picks_discrete(monkeypatch):
    monkeypatch.setattr(gpu_select.shutil, "which", _which("rocminfo"))
    monkeypatch.setattr(gpu_select.subprocess, "run", _fake_run(ROCMINFO_DUAL))
    # GPU agents in order: gfx1036 (APU, skipped) -> gfx1201 (discrete) at index 1.
    assert gpu_select._rocm_via_rocminfo() == 1


def test_hip_struct_integrated_offset_is_stable():
    # Guards the ctypes ABI mirror: `integrated` must sit at offset 396 on x86_64.
    # A field-order edit that moves it would silently misread the flag.
    assert gpu_select._HipDevicePropR0600.integrated.offset == 396


# --------------------------------------------------------------------------
# Entry point contract
# --------------------------------------------------------------------------
def test_respects_operator_override(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7")
    monkeypatch.delenv("GGML_VK_VISIBLE_DEVICES", raising=False)
    # Even if a discrete GPU is detectable, an operator-set var wins.
    monkeypatch.setattr(gpu_select.shutil, "which", _which("vulkaninfo"))
    monkeypatch.setattr(gpu_select.subprocess, "run", _fake_run(VULKANINFO_IGPU_FIRST))

    gpu_select.select_discrete_gpu()

    import os
    assert os.environ["HIP_VISIBLE_DEVICES"] == "7"  # untouched
    assert "GGML_VK_VISIBLE_DEVICES" not in os.environ


def test_noop_when_no_gpu_tooling(monkeypatch):
    for var in gpu_select._OVERRIDE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gpu_select.shutil, "which", _which())  # no vulkaninfo/rocminfo
    monkeypatch.setattr(gpu_select.os.path, "isdir", lambda p: False)  # no /opt/rocm

    gpu_select.select_discrete_gpu()  # must be a silent no-op

    import os
    assert "GGML_VK_VISIBLE_DEVICES" not in os.environ
    assert "HIP_VISIBLE_DEVICES" not in os.environ


def test_noop_on_cpu_image_with_software_vulkan(monkeypatch):
    # Regression: CPU images ship Mesa llvmpipe's libvulkan.so.1 but NO vulkaninfo.
    # The old code treated a loadable libvulkan as a GPU signal and ctypes-probed it
    # -> native SIGSEGV (uncatchable). Now: no vulkaninfo/rocminfo tool -> no-op, and
    # the dangerous libvulkan probe is never reached.
    for var in gpu_select._OVERRIDE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gpu_select.shutil, "which", _which())  # no vulkaninfo / rocminfo
    monkeypatch.setattr(gpu_select.os.path, "isdir", lambda p: False)  # no /opt/rocm

    def _must_not_probe():
        raise AssertionError("must not ctypes-probe software Vulkan on a CPU image")

    monkeypatch.setattr(gpu_select, "_vk_via_libvulkan", _must_not_probe)

    gpu_select.select_discrete_gpu()  # no-op; the probe is never entered

    import os
    assert "GGML_VK_VISIBLE_DEVICES" not in os.environ
    assert "HIP_VISIBLE_DEVICES" not in os.environ


def test_noop_when_backend_forced_none(monkeypatch):
    # An explicit JARVIS_GPU_BACKEND=none opt-out wins even if tooling is present.
    for var in gpu_select._OVERRIDE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("JARVIS_GPU_BACKEND", "none")
    monkeypatch.setattr(gpu_select.shutil, "which", _which("vulkaninfo"))
    monkeypatch.setattr(gpu_select.subprocess, "run", _fake_run(VULKANINFO_IGPU_FIRST))

    gpu_select.select_discrete_gpu()

    import os
    assert "GGML_VK_VISIBLE_DEVICES" not in os.environ


def test_vulkan_backend_sets_ggml_visible_devices(monkeypatch):
    for var in gpu_select._OVERRIDE_VARS:
        monkeypatch.delenv(var, raising=False)
    # Vulkan image: vulkaninfo present (that's the GPU-image signal), no ROCm.
    monkeypatch.setattr(gpu_select.shutil, "which", _which("vulkaninfo"))
    monkeypatch.setattr(gpu_select.os.path, "isdir", lambda p: False)
    monkeypatch.setattr(gpu_select.subprocess, "run", _fake_run(VULKANINFO_IGPU_FIRST))

    gpu_select.select_discrete_gpu()

    import os
    assert os.environ.get("GGML_VK_VISIBLE_DEVICES") == "1"
    assert "HIP_VISIBLE_DEVICES" not in os.environ


def test_select_discrete_gpu_never_raises(monkeypatch):
    for var in gpu_select._OVERRIDE_VARS:
        monkeypatch.delenv(var, raising=False)

    def _boom(*a, **k):
        raise RuntimeError("exploding tool")

    monkeypatch.setattr(gpu_select.shutil, "which", _which("vulkaninfo"))
    monkeypatch.setattr(gpu_select.subprocess, "run", _boom)
    # Should swallow the error and leave env untouched.
    gpu_select.select_discrete_gpu()

    import os
    assert "GGML_VK_VISIBLE_DEVICES" not in os.environ
