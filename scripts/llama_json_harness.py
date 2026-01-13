"""
Harness to reproduce and debug llama-cpp JSON grammar behavior.

Features:
- Loads a model and grammar, fails fast if grammar missing or rejected.
- Runs repeated generations with deterministic (debug) or stable (prod) decoding.
- Logs raw outputs, parse results, and config per trial to logs/.
- Performs a smoke test to verify grammar application before running trials.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError as exc:
    print("llama_cpp is required. Install with: pip install llama-cpp-python", file=sys.stderr)
    raise

LOG_DIR = Path("logs")
DEFAULT_GRAMMAR_PATH = Path("tools/json_grammars/tool_call.gbnf")
DEFAULT_STOP_SENTINEL = "\n<END_JSON>"


@dataclass
class HarnessConfig:
    model_path: Path
    chat_format: str
    n_ctx: int
    n_gpu_layers: int
    n_trials: int
    grammar_path: Path
    mode: str  # debug | prod
    stop_sentinel: str
    prompt_variant: str  # good | bad


def get_env(name: str, default: Any, cast) -> Any:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except Exception:
        return default


def load_config() -> HarnessConfig:
    model_path = Path(os.getenv("MODEL_PATH", "")).expanduser()
    if not model_path:
        print("MODEL_PATH is required.", file=sys.stderr)
        sys.exit(1)

    cfg = HarnessConfig(
        model_path=model_path,
        chat_format=os.getenv("CHAT_FORMAT", "llama-3"),
        n_ctx=get_env("CTX", 8192, int),
        n_gpu_layers=get_env("N_GPU_LAYERS", -1, int),
        n_trials=get_env("N_TRIALS", 10, int),
        grammar_path=Path(os.getenv("GRAMMAR_PATH", str(DEFAULT_GRAMMAR_PATH))).expanduser(),
        mode=os.getenv("MODE", "debug").lower(),
        stop_sentinel=os.getenv("STOP_SENTINEL", DEFAULT_STOP_SENTINEL),
        prompt_variant=os.getenv("PROMPT_VARIANT", "good").lower(),
    )

    if cfg.mode not in {"debug", "prod"}:
        print(f"Invalid MODE={cfg.mode}. Use debug|prod.", file=sys.stderr)
        sys.exit(1)
    if cfg.prompt_variant not in {"good", "bad"}:
        print(f"Invalid PROMPT_VARIANT={cfg.prompt_variant}. Use good|bad.", file=sys.stderr)
        sys.exit(1)
    if not cfg.model_path.exists():
        print(f"Model file not found: {cfg.model_path}", file=sys.stderr)
        sys.exit(1)
    if not cfg.grammar_path.exists():
        print(f"Grammar file not found: {cfg.grammar_path}", file=sys.stderr)
        sys.exit(1)
    return cfg


def read_grammar(grammar_path: Path) -> Tuple[LlamaGrammar, str, str]:
    try:
        data = grammar_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Failed to read grammar: {exc}", file=sys.stderr)
        sys.exit(1)

    checksum = hashlib.sha256(data.encode("utf-8")).hexdigest()
    preview = data[:100].replace("\n", "\\n")

    try:
        grammar = LlamaGrammar.from_string(data)
    except Exception as exc:
        print(f"Grammar rejected by llama-cpp: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Loaded grammar from {grammar_path}")
    print(f"   SHA256: {checksum}")
    print(f"   Preview: {preview}")
    return grammar, checksum, preview


def filter_kwargs(kwargs: Dict[str, Any], func) -> Dict[str, Any]:
    supported = set(inspect.signature(func).parameters.keys())
    filtered = {}
    for k, v in kwargs.items():
        if k in supported and v is not None:
            filtered[k] = v
        elif v is not None:
            print(f"⚠️  Parameter '{k}' not supported by {func.__name__}; omitting.")
    return filtered


def build_llama(cfg: HarnessConfig, seed: int) -> Llama:
    init_kwargs = {
        "model_path": str(cfg.model_path),
        "n_ctx": cfg.n_ctx,
        "n_gpu_layers": cfg.n_gpu_layers,
        "seed": seed,
        "verbose": False,
        "chat_format": cfg.chat_format,
    }
    init_kwargs = filter_kwargs(init_kwargs, Llama.__init__)
    print(f"🦙 Loading model {cfg.model_path} with args: {init_kwargs}")
    try:
        return Llama(**init_kwargs)
    except Exception as exc:
        print(f"Failed to initialize Llama: {exc}", file=sys.stderr)
        sys.exit(1)


def prompts(cfg: HarnessConfig, use_sentinel: bool) -> List[Dict[str, str]]:
    sentinel_note = ""
    if use_sentinel:
        sentinel_note = (
            f"\nAfter the JSON, output a newline followed by {cfg.stop_sentinel} and nothing else."
        )

    good_prompt = [
        {"role": "system", "content": "Return ONLY JSON matching the schema. No prose." + sentinel_note},
        {"role": "user", "content": "Turn on the kitchen lights."},
    ]

    bad_prompt = [
        {"role": "system", "content": "Return JSON and also explain your reasoning." + sentinel_note},
        {"role": "user", "content": "Turn on the kitchen lights and tell me why."},
    ]

    return good_prompt if cfg.prompt_variant == "good" else bad_prompt


def decoding_params(cfg: HarnessConfig) -> Tuple[Dict[str, Any], int]:
    if cfg.mode == "debug":
        params = {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repeat_penalty": 1.0,
            "mirostat_mode": 0,
            "max_tokens": 512,
        }
        seed = get_env("SEED", 42, int)
    else:
        params = {
            "temperature": get_env("TEMP", 0.3, float),
            "top_k": get_env("TOP_K", 20, int),
            "top_p": get_env("TOP_P", 0.9, float),
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repeat_penalty": 1.0,
            "mirostat_mode": 0,
            "max_tokens": get_env("MAX_TOKENS", 512, int),
        }
        seed = get_env("SEED", int(time.time()), int)
    return params, seed


def attempt_json_parse(text: str) -> Tuple[bool, str | None]:
    try:
        json.loads(text)
        return True, None
    except Exception as exc:
        return False, str(exc)


def recover_braces(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def run_once(
    llama: Llama,
    grammar: LlamaGrammar,
    cfg: HarnessConfig,
    gen_params: Dict[str, Any],
    stop_sequences: List[str],
    trial_idx: int,
    use_recovery: bool,
) -> Dict[str, Any]:
    # Construct messages
    use_sentinel = bool(stop_sequences)
    messages = prompts(cfg, use_sentinel)

    kwargs = {
        "messages": messages,
        "grammar": grammar,
        "stop": stop_sequences or None,
    }
    kwargs.update(gen_params)
    kwargs = filter_kwargs(kwargs, llama.create_chat_completion)

    start_time = time.time()
    try:
        resp = llama.create_chat_completion(**kwargs)
    except Exception as exc:
        return {
            "trial": trial_idx,
            "raw_output": "",
            "raw_parse_ok": False,
            "parse_error": f"generation_error: {exc}",
            "recovery_used": False,
            "recovered_output": None,
            "elapsed_seconds": time.time() - start_time,
        }

    content = resp["choices"][0]["message"]["content"] if resp and "choices" in resp else ""
    elapsed = time.time() - start_time

    raw_ok, raw_err = attempt_json_parse(content)
    recovery_used = False
    recovered_output = None
    final_ok = raw_ok
    final_err = raw_err

    if not raw_ok and use_recovery:
        recovered_output = recover_braces(content)
        if recovered_output:
            recovery_used = True
            final_ok, final_err = attempt_json_parse(recovered_output)

    return {
        "trial": trial_idx,
        "raw_output": content,
        "raw_parse_ok": raw_ok,
        "parse_error": raw_err,
        "recovery_used": recovery_used,
        "recovered_output": recovered_output,
        "final_parse_ok": final_ok,
        "final_parse_error": final_err,
        "elapsed_seconds": elapsed,
    }


def write_log(entry: Dict[str, Any], cfg: HarnessConfig, meta: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_slug = cfg.model_path.stem.replace(" ", "_")
    fname = f"{timestamp}_{model_slug}_{cfg.mode}_trial{entry['trial']}.json"
    payload = {**meta, **entry}
    path = LOG_DIR / fname
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"📝 Wrote log: {path}")


def smoke_test(
    llama: Llama,
    grammar: LlamaGrammar,
    cfg: HarnessConfig,
    gen_params: Dict[str, Any],
    stop_sequences: List[str],
) -> None:
    print("🔎 Running grammar smoke test...")
    result = run_once(
        llama,
        grammar,
        cfg,
        {**gen_params, "max_tokens": min(gen_params.get("max_tokens", 256), 256)},
        stop_sequences,
        trial_idx=0,
        use_recovery=False,
    )
    if not result.get("raw_parse_ok"):
        print(f"❌ Smoke test failed: {result.get('parse_error')}", file=sys.stderr)
        sys.exit(1)
    print("✅ Smoke test passed (raw parse succeeded).")


def main() -> None:
    cfg = load_config()
    gen_params, seed = decoding_params(cfg)
    llama = build_llama(cfg, seed)
    grammar, checksum, preview = read_grammar(cfg.grammar_path)

    # Decide whether stop is supported and should be used
    supports_stop = "stop" in inspect.signature(llama.create_chat_completion).parameters
    stop_sequences: List[str] = [cfg.stop_sentinel] if supports_stop and cfg.stop_sentinel else []
    if stop_sequences:
        print(f"⏹️  Using stop sequence: {stop_sequences}")
    else:
        print("⏹️  Stop sequences not supported or disabled; running without sentinel.")

    smoke_test(llama, grammar, cfg, gen_params, stop_sequences)

    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": cfg.mode,
        "prompt_variant": cfg.prompt_variant,
        "model_path": str(cfg.model_path),
        "model_basename": cfg.model_path.name,
        "chat_format": cfg.chat_format,
        "entrypoint": "create_chat_completion",
        "grammar_path": str(cfg.grammar_path),
        "grammar_checksum": checksum,
        "grammar_preview": preview,
        "decoding_params": gen_params,
        "stop_sequences": stop_sequences,
        "seed": seed,
    }

    use_recovery = cfg.mode == "debug"
    raw_passes = 0
    final_passes = 0

    for i in range(1, cfg.n_trials + 1):
        result = run_once(llama, grammar, cfg, gen_params, stop_sequences, i, use_recovery)
        write_log(result, cfg, meta)
        if result.get("raw_parse_ok"):
            raw_passes += 1
        if result.get("final_parse_ok"):
            final_passes += 1
        status = "PASS" if result.get("raw_parse_ok") else "FAIL"
        print(f"Trial {i}/{cfg.n_trials}: {status} (raw). Final pass: {result.get('final_parse_ok')}")

    print("=" * 60)
    print(f"Mode: {cfg.mode} | Trials: {cfg.n_trials}")
    print(f"Raw parses: {raw_passes}/{cfg.n_trials}")
    print(f"Final parses (after optional debug recovery): {final_passes}/{cfg.n_trials}")
    if cfg.mode == "prod" and raw_passes < max(cfg.n_trials - 1, int(0.9 * cfg.n_trials)):
        print("❌ Production stability target not met (>=9/10 required).", file=sys.stderr)
        sys.exit(1)
    if cfg.mode == "debug" and raw_passes < cfg.n_trials:
        print("❌ Debug mode expects 10/10 raw passes.", file=sys.stderr)
        sys.exit(1)
    print("✅ Harness run completed within targets.")


if __name__ == "__main__":
    main()

