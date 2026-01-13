Codex Task: Fix llama-cpp-python JSON grammar loops + add a reproducible harness

Goal

We are using llama-cpp-python (Metal backend) and JSON-grammar constrained generation for tool-calling. We are seeing a failure mode where generation gets stuck in a loop producing invalid JSON (or repeatedly failing to complete valid JSON).

Your job:
	1.	Create a small reproducible harness in Python that:
	•	loads a model via llama-cpp-python
	•	runs the same prompt multiple times
	•	enforces JSON output via grammar (GBNF) or json schema (if supported)
	•	logs the raw output and whether it parses
	•	The harness must fail fast if the grammar file is missing, unreadable, or rejected by llama-cpp.
	2.	Provide a recommended “boring mode” decoding config (temp=0, etc.) for debugging.
	3.	Provide a production config that still stays stable.
	4.	Provide guidance + code changes to avoid the loop (prompting + grammar string rules + stop sequences).
	5.	Include CLI-friendly output and clear instructions.

Constraints / Notes
	•	We are on macOS with Metal enabled (n_gpu_layers=-1 typical).
	•	Context is ~8192.
	•	We want valid JSON only (no markdown, no explanation).
	•	We have an existing tool calling schema like:
	•	{"tool":"...", "arguments": {...}} OR similar.
	•	We suspect the grammar might not be applied correctly, or the grammar’s JSON string rules are wrong, or the sampler is causing rejection churn.

What to implement

A) tools/json_grammars/

Create a folder with at least:
	•	json_object.gbnf: a known-good JSON grammar that supports:
	•	objects, arrays, numbers, booleans, null
	•	proper JSON string escapes (\", \\, \/, \b, \f, \n, \r, \t, \uXXXX)
	•	tool_call.gbnf: a narrower grammar for:
    {
        "tool": "string",
        "arguments": { "any": "json" }
    }

    (or arguments can be any JSON value)

Make sure the grammar is actually compatible with llama.cpp GBNF.

B) scripts/llama_json_harness.py

Write a script that:
	•	Loads config from env vars:
	•	MODEL_PATH
	•	CTX=8192
	•	N_GPU_LAYERS=-1
	•	N_TRIALS=10
	•	GRAMMAR_PATH=tools/json_grammars/tool_call.gbnf
	•	Uses llama-cpp-python Llama(...) to load the model.
	•	Runs a prompt like:
	•	system: “Return ONLY JSON matching schema…”
	•	user: “Turn on the kitchen lights”
	•	Generates using the grammar and deterministic decoding (see below).
	•	After each run:
	•	prints raw output
	•	attempts json.loads(output)
	•	prints PASS/FAIL + error message
	•	If the output contains any leading/trailing non-JSON, optionally attempt a brace-based extraction **ONLY in debug mode**, but this must be logged as `recovery_used=true` and **MUST NOT count as a pass**. Production mode must reject any non-raw-valid JSON.
	•	Saves a logs/ file with each trial output and parse result.

C) Recommended decoding settings

Provide two configs and code to select via env var MODE=debug|prod:

debug mode:
	•	temperature=0
	•	top_k=1
	•	top_p=1
	•	min_p=0 (if supported)
	•	presence_penalty=0
	•	frequency_penalty=0
	•	repeat_penalty=1.0 (explicitly note that higher values can cause grammar rejection churn)
	•	disable mirostat
	•	seed=fixed
	•	set max_tokens reasonably (e.g. 512)
	•	include stop sequences that do not conflict with JSON (avoid } etc.)

prod mode:
	•	slightly relaxed (e.g. temp 0.2–0.4, top_p 0.9)
	•	presence_penalty=0
	•	frequency_penalty=0
	•	repeat_penalty=1.0 (explicitly note that higher values can cause grammar rejection churn)
	•	seed=varied
	•	still must be stable and JSON-valid

State clearly that unsupported parameters must be omitted, not silently ignored.

Define a default sentinel stop sequence `\n<END_JSON>` that is only used if the prompt instructs the model to emit it.

Stops must never include JSON-significant characters like `{`, `}`, `]`, or `"`.

C.1 Grammar + sampler interaction notes

Overly restrictive sampling parameters (such as high repeat_penalty, mirostat enabled, or aggressive top_k/top_p settings) can cause infinite rejection loops even when the grammar is correct. Careful tuning of decoding parameters is necessary to avoid sampler churn that prevents valid JSON output.

D) Ensure grammar is applied

Add explicit printing of:
	•	which grammar file is loaded
	•	the first 100 chars of grammar for sanity (or checksum)
	•	Compute and log a SHA256 checksum of the grammar file.
	•	Perform a required “grammar smoke test” run (e.g., 1 short generation) to verify constrained output.
	•	Log whether grammar was successfully passed to llama-cpp (or inferred via constrained output if no explicit flag exists).

E) Prompting rules

In the harness include:
	•	a “bad prompt” example that triggers loops
	•	a “good prompt” example:
	•	no markdown
	•	no extra prose
	•	no dual modes (“json and explanation”)

F) Output / docs

Create docs/json-grammar-debugging.md that explains:
	•	how to run the harness
	•	how to interpret failures
	•	typical causes (grammar not loaded, bad string escapes, prompt conflict, sampler rejection churn)
	•	how to tune until 10/10 passes

Expand logging requirements:
	•	Log files must be saved with a filename pattern like `logs/{timestamp}_{model}_{mode}_{trial}.json`.
	•	Logs must include: model path, quant, entrypoint used, chat_format, grammar path + checksum, decoding params, seed, raw output, parse result, and whether recovery parsing was used.

Acceptance criteria
	•	Running python scripts/llama_json_harness.py with a valid model path produces:
	•	10/10 parses in debug mode for at least a simple tool-call, where “10/10 parses” means *raw output parses with no recovery*.
	•	In production mode, target ≥9/10 passes raw parse; any fails must produce failure logs.
	•	The harness makes it obvious when grammar is not applied.
	•	The JSON grammar correctly supports escape sequences so it won’t choke on quotes/newlines.
	•	The docs are clear and copy/pastable.

Repo layout expectation

If unsure, create:
	•	scripts/
	•	tools/json_grammars/
	•	docs/
	•	logs/ (gitignored)

Also add a .gitignore entry for logs/.
