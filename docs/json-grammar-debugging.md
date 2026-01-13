# JSON Grammar Debugging Harness

This harness reproduces and debugs llama-cpp JSON grammar behavior. It loads a grammar, runs repeated generations, and logs parse outcomes per trial.

## Quick start
- Install deps: `pip install -r requirements.txt` (ensures llama-cpp-python).
- Set env:
  - `MODEL_PATH` (required) – path to your GGUF model.
  - Optional: `MODE=debug|prod` (default `debug`), `GRAMMAR_PATH`, `N_TRIALS`, `CTX`, `N_GPU_LAYERS`, `CHAT_FORMAT`, `PROMPT_VARIANT=good|bad`, `STOP_SENTINEL`.
- Run: `python scripts/llama_json_harness.py`

## What it does
- Loads the grammar, prints preview + SHA256, fails fast if missing/rejected.
- Runs a smoke test (raw JSON parse required).
- Executes N trials (default 10) with deterministic `debug` or stable `prod` decoding.
- Logs each trial to `logs/{timestamp}_{model}_{mode}_{trial}.json` with model info, decoding params, grammar checksum, raw output, parse result, and whether recovery was attempted.
- In `debug`, brace-based recovery is attempted **but never counted as a pass**; `prod` rejects any non-raw-valid JSON.

## Decoding presets
- `debug` (deterministic): temp=0, top_k=1, top_p=1, min_p=0, presence_penalty=0, frequency_penalty=0, repeat_penalty=1.0, mirostat_mode=0, seed fixed, max_tokens=512.
- `prod` (stable): temp≈0.3, top_k≈20, top_p=0.9, min_p=0, presence_penalty=0, frequency_penalty=0, repeat_penalty=1.0, mirostat_mode=0, seed varied, max_tokens configurable.
- Unsupported params are omitted (never silently passed).

## Stops and prompting
- Default stop sentinel: `\n<END_JSON>` if the backend supports `stop`. The system prompt instructs emitting the sentinel; the stop sequence trims it so raw output stays JSON-only.
- If `stop` is unsupported, the sentinel is disabled and not mentioned.
- Two prompt variants:
  - `good`: system “Return ONLY JSON…”; user “Turn on the kitchen lights.”
  - `bad`: system “Return JSON and also explain…”; user “…and tell me why.”
  - Select via `PROMPT_VARIANT=good|bad`.

## Acceptance targets
- `debug`: 10/10 raw parses (no recovery) required.
- `prod`: ≥9/10 raw parses; failures still log details.

## Smoke test
- One short generation is run before trials; any parse failure aborts the run to signal grammar/application issues early.

## Logs
- Written to `logs/` (gitignored). Each entry includes model path/basename, chat_format, entrypoint, grammar path + checksum + preview, decoding params, stop sequences, seed, raw output, parse errors, recovery_used, recovered_output, and timing.

## Troubleshooting tips
- If smoke test fails: verify grammar path, chat_format compatibility, and that `grammar` is supported in your llama-cpp version.
- If loops/invalid JSON persist: lower temperature/top_k, keep repeat_penalty at 1.0, disable mirostat, ensure stop sentinel is active, and prefer `good` prompt.
- If `stop` is unsupported, consider upgrading llama-cpp-python or rely on grammar + deterministic decoding.***

