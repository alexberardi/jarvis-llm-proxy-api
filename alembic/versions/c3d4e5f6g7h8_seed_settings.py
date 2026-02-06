"""Seed default settings

Revision ID: c3d4e5f6g7h8
Revises: b2c3d4e5f6g7
Create Date: 2026-02-05 17:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = 'c3d4e5f6g7h8'
down_revision = 'b2c3d4e5f6g7'
branch_labels = None
depends_on = None


# Settings definitions from services/settings_service.py
# EXCLUDED: model paths (fallback to env), s3 URLs (use config-service)
# INCLUDED: inference parameters, training parameters, logging levels
SETTINGS = [
    # model.main - only non-path settings
    {
        "key": "model.main.backend",
        "value": "GGUF",
        "value_type": "string",
        "category": "model.main",
        "description": "Main model backend: GGUF, VLLM, TRANSFORMERS, REST, MOCK",
        "env_fallback": "JARVIS_MODEL_BACKEND",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "model.main.chat_format",
        "value": "llama3",
        "value_type": "string",
        "category": "model.main",
        "description": "Chat template format: llama3, chatml, mistral, etc.",
        "env_fallback": "JARVIS_MODEL_CHAT_FORMAT",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "model.main.context_window",
        "value": "8192",
        "value_type": "int",
        "category": "model.main",
        "description": "Maximum context window size in tokens",
        "env_fallback": "JARVIS_MODEL_CONTEXT_WINDOW",
        "requires_reload": True,
        "is_secret": False,
    },
    # model.lightweight - context window only
    {
        "key": "model.lightweight.context_window",
        "value": "8192",
        "value_type": "int",
        "category": "model.lightweight",
        "description": "Lightweight model context window",
        "env_fallback": "JARVIS_LIGHTWEIGHT_MODEL_CONTEXT_WINDOW",
        "requires_reload": True,
        "is_secret": False,
    },
    # model.vision - context window and chat format
    {
        "key": "model.vision.context_window",
        "value": "131072",
        "value_type": "int",
        "category": "model.vision",
        "description": "Vision model context window",
        "env_fallback": "JARVIS_VISION_MODEL_CONTEXT_WINDOW",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "model.vision.chat_format",
        "value": "qwen",
        "value_type": "string",
        "category": "model.vision",
        "description": "Vision model chat format (e.g., qwen, chatml)",
        "env_fallback": "JARVIS_VISION_MODEL_CHAT_FORMAT",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "model.vision.n_gpu_layers",
        "value": "0",
        "value_type": "int",
        "category": "model.vision",
        "description": "Number of layers to offload to GPU for GGUF vision models",
        "env_fallback": "JARVIS_VISION_N_GPU_LAYERS",
        "requires_reload": True,
        "is_secret": False,
    },
    # model.cloud - context window only
    {
        "key": "model.cloud.backend",
        "value": "REST",
        "value_type": "string",
        "category": "model.cloud",
        "description": "Cloud model backend (typically REST)",
        "env_fallback": "JARVIS_CLOUD_MODEL_BACKEND",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "model.cloud.context_window",
        "value": "4096",
        "value_type": "int",
        "category": "model.cloud",
        "description": "Cloud model context window",
        "env_fallback": "JARVIS_CLOUD_MODEL_CONTEXT_WINDOW",
        "requires_reload": True,
        "is_secret": False,
    },
    # inference.vllm
    {
        "key": "inference.vllm.gpu_memory_utilization",
        "value": "0.9",
        "value_type": "float",
        "category": "inference.vllm",
        "description": "GPU memory utilization (0.0-1.0)",
        "env_fallback": "JARVIS_VLLM_GPU_MEMORY_UTILIZATION",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.vllm.tensor_parallel_size",
        "value": "1",
        "value_type": "int",
        "category": "inference.vllm",
        "description": "Number of GPUs for tensor parallelism",
        "env_fallback": "JARVIS_VLLM_TENSOR_PARALLEL_SIZE",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.vllm.max_batched_tokens",
        "value": "8192",
        "value_type": "int",
        "category": "inference.vllm",
        "description": "Maximum batched tokens for vLLM",
        "env_fallback": "JARVIS_VLLM_MAX_BATCHED_TOKENS",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.vllm.max_num_seqs",
        "value": "256",
        "value_type": "int",
        "category": "inference.vllm",
        "description": "Maximum number of sequences",
        "env_fallback": "JARVIS_VLLM_MAX_NUM_SEQS",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.vllm.max_lora_rank",
        "value": "64",
        "value_type": "int",
        "category": "inference.vllm",
        "description": "Maximum LoRA rank for adapters",
        "env_fallback": "JARVIS_VLLM_MAX_LORA_RANK",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.vllm.max_loras",
        "value": "1",
        "value_type": "int",
        "category": "inference.vllm",
        "description": "Maximum concurrent LoRA adapters",
        "env_fallback": "JARVIS_VLLM_MAX_LORAS",
        "requires_reload": True,
        "is_secret": False,
    },
    # inference.gguf
    {
        "key": "inference.gguf.n_gpu_layers",
        "value": "-1",
        "value_type": "int",
        "category": "inference.gguf",
        "description": "GPU layers (-1=all, 0=CPU only)",
        "env_fallback": "JARVIS_N_GPU_LAYERS",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.gguf.n_batch",
        "value": "512",
        "value_type": "int",
        "category": "inference.gguf",
        "description": "Batch size for llama.cpp",
        "env_fallback": "JARVIS_N_BATCH",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.gguf.n_ubatch",
        "value": "512",
        "value_type": "int",
        "category": "inference.gguf",
        "description": "Micro-batch size",
        "env_fallback": "JARVIS_N_UBATCH",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.gguf.n_threads",
        "value": "10",
        "value_type": "int",
        "category": "inference.gguf",
        "description": "Number of CPU threads",
        "env_fallback": "JARVIS_N_THREADS",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.gguf.flash_attn",
        "value": "true",
        "value_type": "bool",
        "category": "inference.gguf",
        "description": "Enable flash attention",
        "env_fallback": "JARVIS_FLASH_ATTN",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.gguf.f16_kv",
        "value": "true",
        "value_type": "bool",
        "category": "inference.gguf",
        "description": "Use FP16 for KV cache",
        "env_fallback": "JARVIS_F16_KV",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.gguf.mul_mat_q",
        "value": "true",
        "value_type": "bool",
        "category": "inference.gguf",
        "description": "Enable quantized matrix multiplication",
        "env_fallback": "JARVIS_MUL_MAT_Q",
        "requires_reload": True,
        "is_secret": False,
    },
    # inference.transformers
    {
        "key": "inference.transformers.device",
        "value": "auto",
        "value_type": "string",
        "category": "inference.transformers",
        "description": "Device: auto, cuda, mps, cpu",
        "env_fallback": "JARVIS_DEVICE",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.transformers.torch_dtype",
        "value": "auto",
        "value_type": "string",
        "category": "inference.transformers",
        "description": "Torch dtype: auto, float16, float32, bfloat16",
        "env_fallback": "JARVIS_TORCH_DTYPE",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.transformers.use_quantization",
        "value": "false",
        "value_type": "bool",
        "category": "inference.transformers",
        "description": "Enable bitsandbytes quantization",
        "env_fallback": "JARVIS_USE_QUANTIZATION",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.transformers.quantization_type",
        "value": "4bit",
        "value_type": "string",
        "category": "inference.transformers",
        "description": "Quantization type: 4bit, 8bit",
        "env_fallback": "JARVIS_QUANTIZATION_TYPE",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.transformers.device_map",
        "value": "auto",
        "value_type": "string",
        "category": "inference.transformers",
        "description": "Device map for transformers: auto, none",
        "env_fallback": "JARVIS_TRANSFORMERS_DEVICE_MAP",
        "requires_reload": True,
        "is_secret": False,
    },
    # inference.general
    {
        "key": "inference.general.engine",
        "value": "llama_cpp",
        "value_type": "string",
        "category": "inference.general",
        "description": "Default inference engine: llama_cpp, vllm, transformers",
        "env_fallback": "JARVIS_INFERENCE_ENGINE",
        "requires_reload": True,
        "is_secret": False,
    },
    {
        "key": "inference.general.max_tokens",
        "value": "7000",
        "value_type": "int",
        "category": "inference.general",
        "description": "Default max generation tokens",
        "env_fallback": "JARVIS_MAX_TOKENS",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "inference.general.top_p",
        "value": "0.95",
        "value_type": "float",
        "category": "inference.general",
        "description": "Top-P sampling value",
        "env_fallback": "JARVIS_TOP_P",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "inference.general.top_k",
        "value": "40",
        "value_type": "int",
        "category": "inference.general",
        "description": "Top-K sampling value",
        "env_fallback": "JARVIS_TOP_K",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "inference.general.repeat_penalty",
        "value": "1.1",
        "value_type": "float",
        "category": "inference.general",
        "description": "Repetition penalty",
        "env_fallback": "JARVIS_REPEAT_PENALTY",
        "requires_reload": False,
        "is_secret": False,
    },
    # training
    {
        "key": "training.batch_size",
        "value": "1",
        "value_type": "int",
        "category": "training",
        "description": "Training batch size",
        "env_fallback": "JARVIS_ADAPTER_BATCH_SIZE",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "training.grad_accum",
        "value": "4",
        "value_type": "int",
        "category": "training",
        "description": "Gradient accumulation steps",
        "env_fallback": "JARVIS_ADAPTER_GRAD_ACCUM",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "training.epochs",
        "value": "1",
        "value_type": "int",
        "category": "training",
        "description": "Training epochs",
        "env_fallback": "JARVIS_ADAPTER_EPOCHS",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "training.learning_rate",
        "value": "0.0002",
        "value_type": "float",
        "category": "training",
        "description": "Training learning rate",
        "env_fallback": "JARVIS_ADAPTER_LEARNING_RATE",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "training.lora_r",
        "value": "16",
        "value_type": "int",
        "category": "training",
        "description": "LoRA rank",
        "env_fallback": "JARVIS_ADAPTER_LORA_R",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "training.lora_alpha",
        "value": "32",
        "value_type": "int",
        "category": "training",
        "description": "LoRA alpha scaling",
        "env_fallback": "JARVIS_ADAPTER_LORA_ALPHA",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "training.lora_dropout",
        "value": "0.05",
        "value_type": "float",
        "category": "training",
        "description": "LoRA dropout rate",
        "env_fallback": "JARVIS_ADAPTER_LORA_DROPOUT",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "training.max_seq_len",
        "value": "2048",
        "value_type": "int",
        "category": "training",
        "description": "Maximum sequence length for training",
        "env_fallback": "JARVIS_ADAPTER_MAX_SEQ_LEN",
        "requires_reload": False,
        "is_secret": False,
    },
    # storage - only bucket/prefix names (NOT urls)
    {
        "key": "storage.s3_region",
        "value": "us-east-1",
        "value_type": "string",
        "category": "storage",
        "description": "S3 region",
        "env_fallback": "S3_REGION",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "storage.adapter_bucket",
        "value": "jarvis-llm-proxy",
        "value_type": "string",
        "category": "storage",
        "description": "S3 bucket for adapters",
        "env_fallback": "LLM_PROXY_ADAPTER_BUCKET",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "storage.adapter_prefix",
        "value": "adapters",
        "value_type": "string",
        "category": "storage",
        "description": "S3 prefix for adapters",
        "env_fallback": "LLM_PROXY_ADAPTER_PREFIX",
        "requires_reload": False,
        "is_secret": False,
    },
    # logging
    {
        "key": "logging.console_level",
        "value": "WARNING",
        "value_type": "string",
        "category": "logging",
        "description": "Console log level",
        "env_fallback": "JARVIS_LOG_CONSOLE_LEVEL",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "logging.remote_level",
        "value": "DEBUG",
        "value_type": "string",
        "category": "logging",
        "description": "Remote (jarvis-logs) log level",
        "env_fallback": "JARVIS_LOG_REMOTE_LEVEL",
        "requires_reload": False,
        "is_secret": False,
    },
]


def upgrade() -> None:
    conn = op.get_bind()
    is_postgres = conn.dialect.name == 'postgresql'

    for setting in SETTINGS:
        if is_postgres:
            conn.execute(
                sa.text("""
                    INSERT INTO settings (key, value, value_type, category, description,
                                         env_fallback, requires_reload, is_secret,
                                         household_id, node_id, user_id)
                    VALUES (:key, :value, :value_type, :category, :description,
                           :env_fallback, :requires_reload, :is_secret,
                           NULL, NULL, NULL)
                    ON CONFLICT (key, household_id, node_id, user_id) DO NOTHING
                """),
                setting
            )
        else:
            conn.execute(
                sa.text("""
                    INSERT OR IGNORE INTO settings (key, value, value_type, category, description,
                                                   env_fallback, requires_reload, is_secret,
                                                   household_id, node_id, user_id)
                    VALUES (:key, :value, :value_type, :category, :description,
                           :env_fallback, :requires_reload, :is_secret,
                           NULL, NULL, NULL)
                """),
                setting
            )


def downgrade() -> None:
    conn = op.get_bind()
    for setting in SETTINGS:
        conn.execute(
            sa.text("""
                DELETE FROM settings
                WHERE key = :key
                  AND household_id IS NULL
                  AND node_id IS NULL
                  AND user_id IS NULL
            """),
            {"key": setting["key"]}
        )
