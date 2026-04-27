#!/usr/bin/env python3
"""
Train date key LoRA adapters for all supported models using RunPod.

Provisions an H100 pod, uploads training scripts and data, trains adapters
for each target model sequentially, downloads the results, and terminates.

Usage:
    # Train all models
    python scripts/runpod/train_date_adapters.py --api-key <KEY>

    # Train specific models
    python scripts/runpod/train_date_adapters.py --api-key <KEY> --models qwen2.5-3b-instruct,llama-3.2-3b-instruct

    # Resume on existing pod
    python scripts/runpod/train_date_adapters.py --api-key <KEY> --pod-id <POD_ID>

    # Dry run
    python scripts/runpod/train_date_adapters.py --api-key <KEY> --dry-run

Requires: pip install runpod paramiko scp
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import runpod
except ImportError:
    print("Install runpod SDK: pip install runpod")
    sys.exit(1)

try:
    import paramiko
    from scp import SCPClient
except ImportError:
    print("Install SSH tools: pip install paramiko scp")
    sys.exit(1)


# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
REMOTE_TRAIN_SCRIPT = SCRIPT_DIR / "remote_train.py"
TRAINING_DATA = PROJECT_ROOT / "data" / "jarvis_training.jsonl"
REQUIREMENTS = SCRIPT_DIR / "requirements-remote.txt"
LOCAL_ADAPTERS_DIR = PROJECT_ROOT / "adapters" / "date-keys"

# All target models (same order as the plan)
ALL_MODELS = [
    "qwen2.5-3b-instruct",
    "qwen2.5-7b-instruct",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "hermes-3-llama-3.1-8b",
    "llama-3.1-8b-instruct",
    "llama-3.2-3b-instruct",
    "gemma-2-9b-it",
    "gemma-3-12b-it",
]

# RunPod pod configuration
POD_CONFIG = {
    "name": "jarvis-date-adapter-training",
    "image_name": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "gpu_type_id": "NVIDIA H100 80GB HBM3",
    "gpu_count": 1,
    "volume_in_gb": 100,
    "container_disk_in_gb": 50,
    "min_vcpu_count": 8,
    "min_memory_in_gb": 64,
    "ports": "22/tcp",
    "docker_args": "",
    "env": {},
}

# Fallback GPU types if H100 is unavailable
FALLBACK_GPU_TYPES = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 PCIe",
    "NVIDIA A100-SXM4-40GB",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train date key adapters on RunPod")
    p.add_argument("--api-key", required=True, help="RunPod API key")
    p.add_argument("--models", default=None,
                   help="Comma-separated model slugs (default: all)")
    p.add_argument("--pod-id", default=None,
                   help="Resume on existing pod instead of creating new one")
    p.add_argument("--dry-run", action="store_true",
                   help="Show plan without executing")
    p.add_argument("--keep-pod", action="store_true",
                   help="Don't terminate pod after training")
    p.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_HUB_TOKEN"),
                   help="HuggingFace token for gated models")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--ssh-key", default=None,
                   help="Path to SSH private key (default: ~/.ssh/id_ed25519)")
    return p.parse_args()


def wait_for_pod_ready(pod_id: str, timeout: int = 300) -> dict:
    """Wait for pod to be running and return pod info."""
    print(f"   Waiting for pod {pod_id} to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "unknown")
        runtime = pod.get("runtime", {})
        if status == "RUNNING" and runtime:
            ports = runtime.get("ports", [])
            if ports:
                print(f"   ✅ Pod ready! Status: {status}")
                return pod
        print(f"   ⏳ Status: {status}, waiting...", end="\r")
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")


def get_ssh_connection(pod: dict, ssh_key_path: str | None) -> paramiko.SSHClient:
    """Establish SSH connection to the pod."""
    runtime = pod.get("runtime", {})
    ports = runtime.get("ports", [])

    # Find SSH port
    ssh_port = None
    ssh_host = None
    for port_info in ports:
        if port_info.get("privatePort") == 22:
            ssh_host = port_info.get("ip")
            ssh_port = port_info.get("publicPort")
            break

    if not ssh_host or not ssh_port:
        raise ConnectionError("Could not find SSH port in pod info")

    print(f"   Connecting to {ssh_host}:{ssh_port}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Try SSH key first, then fall back to RunPod's default
    key_paths = []
    if ssh_key_path:
        key_paths.append(ssh_key_path)
    key_paths.extend([
        os.path.expanduser("~/.ssh/id_ed25519"),
        os.path.expanduser("~/.ssh/id_rsa"),
    ])

    for key_path in key_paths:
        if os.path.exists(key_path):
            try:
                client.connect(
                    hostname=ssh_host,
                    port=ssh_port,
                    username="root",
                    key_filename=key_path,
                    timeout=30,
                )
                print(f"   ✅ SSH connected via {key_path}")
                return client
            except paramiko.AuthenticationException:
                continue

    raise ConnectionError("SSH authentication failed with all available keys")


def ssh_exec(client: paramiko.SSHClient, cmd: str, timeout: int = 3600) -> tuple[str, str, int]:
    """Execute command via SSH, streaming stdout."""
    print(f"   $ {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out_lines = []
    for line in stdout:
        line = line.rstrip()
        out_lines.append(line)
        print(f"      {line}")
    err = stderr.read().decode()
    exit_code = stdout.channel.recv_exit_status()
    return "\n".join(out_lines), err, exit_code


def upload_files(client: paramiko.SSHClient) -> None:
    """Upload training scripts and data to the pod."""
    print("\n📤 Uploading files...")
    with SCPClient(client.get_transport()) as scp:
        # Create workspace
        ssh_exec(client, "mkdir -p /workspace/data /workspace/scripts /workspace/adapters")

        # Upload training script
        scp.put(str(REMOTE_TRAIN_SCRIPT), "/workspace/scripts/remote_train.py")
        print(f"   ✅ remote_train.py")

        # Upload training data
        scp.put(str(TRAINING_DATA), "/workspace/data/jarvis_training.jsonl")
        print(f"   ✅ jarvis_training.jsonl")

        # Upload requirements
        if REQUIREMENTS.exists():
            scp.put(str(REQUIREMENTS), "/workspace/scripts/requirements-remote.txt")
            print(f"   ✅ requirements-remote.txt")


def install_deps(client: paramiko.SSHClient) -> None:
    """Install Python dependencies on the pod."""
    print("\n📦 Installing dependencies...")
    req_path = "/workspace/scripts/requirements-remote.txt"

    # Check if requirements file exists, otherwise install directly
    _, _, code = ssh_exec(client, f"test -f {req_path} && echo exists || echo missing", timeout=10)

    if code == 0:
        ssh_exec(client, f"pip install -r {req_path}", timeout=600)
    else:
        ssh_exec(
            client,
            "pip install torch transformers peft trl datasets bitsandbytes accelerate",
            timeout=600,
        )


def train_model(
    client: paramiko.SSHClient,
    slug: str,
    epochs: int,
    hf_token: str | None,
) -> bool:
    """Train adapter for a single model. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"🏋️ Training: {slug}")
    print(f"{'='*60}")

    cmd = (
        f"cd /workspace && python scripts/remote_train.py"
        f" --model {slug}"
        f" --output /workspace/adapters/{slug}"
        f" --training-data /workspace/data/jarvis_training.jsonl"
        f" --epochs {epochs}"
    )
    if hf_token:
        cmd += f" --hf-token {hf_token}"

    _, err, exit_code = ssh_exec(client, cmd, timeout=7200)  # 2 hour timeout per model

    if exit_code != 0:
        print(f"   ❌ Training failed for {slug} (exit code {exit_code})")
        if err:
            print(f"   stderr: {err[-500:]}")
        return False

    print(f"   ✅ Training complete for {slug}")
    return True


def download_adapters(
    client: paramiko.SSHClient,
    models: list[str],
) -> list[str]:
    """Download trained adapters from the pod."""
    print(f"\n📥 Downloading adapters...")
    LOCAL_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []

    with SCPClient(client.get_transport()) as scp:
        for slug in models:
            remote_dir = f"/workspace/adapters/{slug}"

            # Check if adapter exists
            _, _, code = ssh_exec(
                client,
                f"test -f {remote_dir}/adapter_config.json && echo exists",
                timeout=10,
            )
            if code != 0:
                print(f"   ⚠�� No adapter found for {slug}")
                continue

            local_dir = LOCAL_ADAPTERS_DIR / slug
            local_dir.mkdir(parents=True, exist_ok=True)

            # Download key files
            files_to_download = [
                "adapter_config.json",
                "metadata.json",
            ]

            # Check for GGUF adapter
            _, _, gguf_code = ssh_exec(
                client,
                f"test -f {remote_dir}/gguf/adapter.gguf && echo exists",
                timeout=10,
            )
            if gguf_code == 0:
                (local_dir / "gguf").mkdir(exist_ok=True)
                files_to_download.append("gguf/adapter.gguf")

            # Also grab PEFT safetensors as backup
            _, _, peft_code = ssh_exec(
                client,
                f"test -f {remote_dir}/adapter_model.safetensors && echo exists",
                timeout=10,
            )
            if peft_code == 0:
                files_to_download.append("adapter_model.safetensors")

            for fname in files_to_download:
                remote_path = f"{remote_dir}/{fname}"
                local_path = str(local_dir / fname)
                try:
                    scp.get(remote_path, local_path)
                    size_mb = os.path.getsize(local_path) / (1024 * 1024)
                    print(f"   ✅ {slug}/{fname} ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"   ⚠️ Failed to download {slug}/{fname}: {e}")

            downloaded.append(slug)

    return downloaded


def generate_manifest(downloaded: list[str]) -> None:
    """Generate manifest.json from downloaded adapters."""
    from scripts.runpod.remote_train import MODEL_REGISTRY

    manifest = {"version": "1.0", "adapters": {}}

    for slug in downloaded:
        entry = MODEL_REGISTRY.get(slug, {})
        hf_id = entry.get("hf_model_id", slug)

        # Build GGUF filename patterns
        # e.g., "Qwen/Qwen3-14B" → ["Qwen3-14B*", "qwen3-14b*"]
        model_name = hf_id.split("/")[-1]
        patterns = [f"{model_name}*"]
        lower = model_name.lower()
        if lower != model_name:
            patterns.append(f"{lower}*")

        manifest["adapters"][slug] = {
            "hf_model_ids": [hf_id],
            "gguf_patterns": patterns,
        }

    manifest_path = LOCAL_ADAPTERS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n📋 Manifest written: {manifest_path}")


def main() -> int:
    args = parse_args()
    runpod.api_key = args.api_key

    # Resolve models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
        invalid = [m for m in models if m not in ALL_MODELS]
        if invalid:
            print(f"❌ Unknown models: {invalid}")
            print(f"   Available: {ALL_MODELS}")
            return 1
    else:
        models = ALL_MODELS

    # Dry run
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — Training plan")
        print("=" * 60)
        print(f"  Models ({len(models)}):")
        for m in models:
            print(f"    - {m}")
        print(f"  Epochs:     {args.epochs}")
        print(f"  GPU:        {POD_CONFIG['gpu_type_id']}")
        print(f"  Est. cost:  ~${len(models) * 0.5:.0f}-${len(models) * 2:.0f}")
        print(f"  Est. time:  ~{len(models) * 15}-{len(models) * 45} minutes")
        print(f"  Output:     {LOCAL_ADAPTERS_DIR}")
        return 0

    pod_id = args.pod_id
    created_pod = False

    try:
        # 1. Provision or connect to pod
        if pod_id:
            print(f"📡 Connecting to existing pod: {pod_id}")
            pod = wait_for_pod_ready(pod_id)
        else:
            print("🚀 Creating RunPod instance...")
            try:
                pod = runpod.create_pod(**POD_CONFIG)
                pod_id = pod["id"]
                created_pod = True
                print(f"   Pod created: {pod_id}")
            except Exception as e:
                # Try fallback GPU types
                print(f"   Primary GPU unavailable: {e}")
                for gpu_type in FALLBACK_GPU_TYPES:
                    try:
                        config = {**POD_CONFIG, "gpu_type_id": gpu_type}
                        pod = runpod.create_pod(**config)
                        pod_id = pod["id"]
                        created_pod = True
                        print(f"   Pod created with {gpu_type}: {pod_id}")
                        break
                    except Exception:
                        continue
                else:
                    print("❌ No GPU available on RunPod")
                    return 1

            pod = wait_for_pod_ready(pod_id)

        # 2. Connect via SSH
        client = get_ssh_connection(pod, args.ssh_key)

        # 3. Upload files
        upload_files(client)

        # 4. Install deps
        install_deps(client)

        # 5. Train each model
        t0 = time.time()
        results: dict[str, bool] = {}

        for i, slug in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Training {slug}...")
            success = train_model(client, slug, args.epochs, args.hf_token)
            results[slug] = success

        total_time = time.time() - t0

        # 6. Download adapters
        successful = [slug for slug, ok in results.items() if ok]
        if successful:
            downloaded = download_adapters(client, successful)
            generate_manifest(downloaded)
        else:
            print("\n⚠��� No models trained successfully")

        # 7. Summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Total time: {total_time / 60:.1f} minutes")
        print(f"  Results:")
        for slug, ok in results.items():
            status = "✅" if ok else "❌"
            print(f"    {status} {slug}")
        if successful:
            print(f"\n  Adapters saved to: {LOCAL_ADAPTERS_DIR}")
        print("=" * 60)

        client.close()

    finally:
        # 8. Terminate pod (unless --keep-pod)
        if created_pod and not args.keep_pod and pod_id:
            print(f"\n🗑️ Terminating pod {pod_id}...")
            try:
                runpod.terminate_pod(pod_id)
                print("   Pod terminated")
            except Exception as e:
                print(f"   ⚠️ Failed to terminate pod: {e}")
                print(f"   Manually terminate: https://www.runpod.io/console/pods")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
