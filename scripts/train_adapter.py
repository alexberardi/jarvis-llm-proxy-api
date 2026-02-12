#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_examples(dataset_ref: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    data = dataset_ref.get("data") if isinstance(dataset_ref, dict) else None
    payload = data if isinstance(data, dict) else dataset_ref
    commands = payload.get("commands", []) if isinstance(payload, dict) else []
    examples: List[Tuple[str, Dict[str, Any]]] = []
    for cmd in commands:
        for ex in cmd.get("examples", []):
            voice = ex.get("voice_command")
            tool_call = ex.get("expected_tool_call")
            if voice and tool_call:
                examples.append((voice, tool_call))
    return examples


def _format_prompt(voice_command: str) -> str:
    return (
        "You are a tool router. Return JSON only.\n"
        f"User: {voice_command}\n"
        "Assistant:"
    )


@dataclass
class TokenizedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples: List[Tuple[str, Dict[str, Any]]], max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items: List[TokenizedExample] = []
        for voice, tool_call in examples:
            prompt = _format_prompt(voice)
            # Wrap tool_call in OpenAI-compatible response format
            response = {
                "message": "",
                "tool_calls": [tool_call],
                "error": None
            }
            completion = json.dumps(response, ensure_ascii=False)
            full_text = prompt + " " + completion
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=max_len)
            input_ids = full["input_ids"]
            attention_mask = full["attention_mask"]
            labels = input_ids.copy()
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100
            self.items.append(TokenizedExample(input_ids, attention_mask, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        return {
            "input_ids": torch.tensor(item.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(item.attention_mask, dtype=torch.long),
            "labels": torch.tensor(item.labels, dtype=torch.long),
        }


class PadCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        padded = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        max_len = padded["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.shape[0]] = lab
        padded["labels"] = padded_labels
        return padded


def _get_param(params: Dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key) if params and key in params else default


def _is_quantized_model(model_path: str) -> bool:
    """Check if a local model is pre-quantized (AWQ, GPTQ, compressed-tensors, etc.)."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quant_config = config.get("quantization_config")
        if quant_config:
            # Check for common quantization methods
            quant_method = quant_config.get("quant_method", "")
            if quant_method in ("awq", "gptq", "compressed-tensors", "marlin"):
                return True
            # Also check if format indicates quantization
            fmt = quant_config.get("format", "")
            if "quantized" in fmt.lower() or "pack" in fmt.lower():
                return True
        return False
    except (json.JSONDecodeError, KeyError, FileNotFoundError, TypeError):
        return False


def main() -> int:
    output_dir = os.getenv("JARVIS_TRAIN_OUTPUT_DIR")
    dataset_path = os.getenv("JARVIS_TRAIN_DATASET_PATH")
    params_path = os.getenv("JARVIS_TRAIN_PARAMS_PATH")
    base_model_id = os.getenv("JARVIS_TRAIN_BASE_MODEL_ID")

    if not output_dir or not dataset_path or not params_path or not base_model_id:
        print("Missing required env vars for training.", flush=True)
        return 2

    dataset_ref = _read_json(dataset_path)
    params = _read_json(params_path)
    examples = _extract_examples(dataset_ref)
    if not examples:
        print("No training examples found in dataset_ref.", flush=True)
        return 2

    hf_base_model_id = _get_param(params, "hf_base_model_id", os.getenv("JARVIS_ADAPTER_HF_BASE_MODEL_ID"))
    training_model_id = base_model_id

    # Check if base model is GGUF or pre-quantized - these require the original HF model for training
    needs_hf_model = base_model_id.endswith(".gguf") or _is_quantized_model(base_model_id)
    if needs_hf_model:
        if not hf_base_model_id:
            if base_model_id.endswith(".gguf"):
                print(
                    "GGUF base model detected. Provide hf_base_model_id in params "
                    "or set JARVIS_ADAPTER_HF_BASE_MODEL_ID.",
                    flush=True,
                )
            else:
                print(
                    "Pre-quantized model detected (AWQ/GPTQ/compressed-tensors). "
                    "Cannot train directly on quantized weights.\n"
                    "Provide hf_base_model_id in params or set JARVIS_ADAPTER_HF_BASE_MODEL_ID "
                    "to point to the original (non-quantized) HuggingFace model.\n"
                    f"Example: hf_base_model_id='meta-llama/Llama-3.1-8B-Instruct'",
                    flush=True,
                )
            return 2
        print(f"ℹ️  Using HuggingFace model for training: {hf_base_model_id}", flush=True)
        print(f"ℹ️  Adapter will be compatible with: {base_model_id}", flush=True)
        training_model_id = hf_base_model_id

    max_seq_len = int(_get_param(params, "max_seq_len", os.getenv("JARVIS_ADAPTER_MAX_SEQ_LEN", "2048")))
    batch_size = int(_get_param(params, "batch_size", os.getenv("JARVIS_ADAPTER_BATCH_SIZE", "1")))
    grad_accum = int(_get_param(params, "grad_accum", os.getenv("JARVIS_ADAPTER_GRAD_ACCUM", "4")))
    epochs = float(_get_param(params, "epochs", os.getenv("JARVIS_ADAPTER_EPOCHS", "1")))
    lr = float(_get_param(params, "learning_rate", os.getenv("JARVIS_ADAPTER_LEARNING_RATE", "2e-4")))
    lora_r = int(_get_param(params, "lora_r", os.getenv("JARVIS_ADAPTER_LORA_R", "16")))
    lora_alpha = int(_get_param(params, "lora_alpha", os.getenv("JARVIS_ADAPTER_LORA_ALPHA", "32")))
    lora_dropout = float(_get_param(params, "lora_dropout", os.getenv("JARVIS_ADAPTER_LORA_DROPOUT", "0.05")))
    target_modules = _get_param(
        params,
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    dtype_env = os.getenv("JARVIS_ADAPTER_TRAIN_DTYPE", "auto").lower()
    if dtype_env == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_env == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = "auto"

    load_in_4bit = os.getenv("JARVIS_ADAPTER_TRAIN_LOAD_IN_4BIT", "false").lower() == "true"
    load_in_8bit = os.getenv("JARVIS_ADAPTER_TRAIN_LOAD_IN_8BIT", "false").lower() == "true"
    if load_in_4bit and load_in_8bit:
        print("Both 4bit and 8bit flags are set; defaulting to 4bit.", flush=True)
        load_in_8bit = False

    # Device map configuration for model loading
    # Options: "auto", "0", "1", etc. for single GPU, or JSON for custom mapping
    # Default "0" = {"": 0} which is most compatible with accelerate
    device_map_env = os.getenv("JARVIS_ADAPTER_TRAIN_DEVICE_MAP", "0")
    if device_map_env == "auto":
        device_map = "auto"
    elif device_map_env.startswith("{"):
        # JSON device map for multi-GPU or custom configurations
        device_map = json.loads(device_map_env)
    else:
        # Single GPU index (e.g., "0", "1")
        device_map = {"": int(device_map_env)}

    tokenizer = AutoTokenizer.from_pretrained(training_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if load_in_4bit or load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype == torch.bfloat16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Load model with configured device map
    # Default device_map={"": 0} is most compatible with accelerate for single-GPU
    model = AutoModelForCausalLM.from_pretrained(
        training_model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype if torch_dtype != "auto" else None,
        quantization_config=quant_config,
        device_map=device_map,
    )
    if quant_config:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = PromptDataset(tokenizer, examples, max_seq_len)
    collator = PadCollator(tokenizer)

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=(torch_dtype == torch.float16),
        bf16=(torch_dtype == torch.bfloat16),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    # GGUF LoRA conversion: convert PEFT adapter to GGUF format for llama.cpp
    if base_model_id.endswith(".gguf") and hf_base_model_id:
        gguf_convert_cmd = os.getenv("JARVIS_ADAPTER_GGUF_CONVERT_CMD", "").strip()
        if not gguf_convert_cmd:
            # Default: use converter from vendored llama.cpp sparse checkout
            vendor_script = Path(__file__).parent / "vendor" / "llama.cpp" / "convert_lora_to_gguf.py"
            if vendor_script.is_file():
                gguf_convert_cmd = f"{sys.executable} {vendor_script}"

        if gguf_convert_cmd:
            gguf_out_dir = out / "gguf"
            gguf_out_dir.mkdir(parents=True, exist_ok=True)
            gguf_out_file = gguf_out_dir / "adapter.gguf"
            convert_args = (
                f"{gguf_convert_cmd} {out} "
                f"--base {hf_base_model_id} "
                f"--outfile {gguf_out_file}"
            )
            print(f"🔄 Converting PEFT adapter to GGUF: {convert_args}", flush=True)
            try:
                subprocess.run(
                    convert_args,
                    shell=True,
                    check=True,
                    timeout=600,
                    capture_output=True,
                    text=True,
                )
                print(f"✅ GGUF adapter saved: {gguf_out_file}", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"⚠️  GGUF conversion failed (non-fatal): {e.stderr[-500:] if e.stderr else e}", flush=True)
            except subprocess.TimeoutExpired:
                print("⚠️  GGUF conversion timed out (non-fatal)", flush=True)
        else:
            print("ℹ️  GGUF conversion skipped: no converter found", flush=True)
    elif base_model_id.endswith(".gguf"):
        print("ℹ️  GGUF conversion skipped: hf_base_model_id not provided", flush=True)

    # Cleanup: destroy any distributed process groups and clear CUDA cache
    # This prevents "destroy_process_group() was not called" warnings and
    # ensures clean GPU state for subsequent vLLM reinitialization
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            print("🧹 Destroyed torch distributed process group", flush=True)
    except Exception as e:
        print(f"⚠️  Process group cleanup warning: {e}", flush=True)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("🧹 CUDA cache cleared", flush=True)
    except Exception as e:
        print(f"⚠️  CUDA cleanup warning: {e}", flush=True)

    # Delete model and trainer to release GPU memory before subprocess exits
    del trainer
    del model

    import gc
    gc.collect()

    print(f"✅ Adapter training complete. Output: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
