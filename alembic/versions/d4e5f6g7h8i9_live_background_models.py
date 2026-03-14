"""Replace lightweight/vision/cloud model settings with live/background

Revision ID: d4e5f6g7h8i9
Revises: c3d4e5f6g7h8
Create Date: 2026-03-14 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "d4e5f6g7h8i9"
down_revision = "c3d4e5f6g7h8"
branch_labels = None
depends_on = None

# Keys to delete (old 4-model system)
DELETED_KEYS = [
    "model.lightweight.context_window",
    "model.vision.context_window",
    "model.vision.chat_format",
    "model.vision.n_gpu_layers",
    "model.cloud.backend",
    "model.cloud.context_window",
]

# Rename model.main → model.live (copy value, then leave main as legacy fallback)
# We don't delete model.main.* because model_manager still reads them as fallback.


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Delete obsolete model settings
    for key in DELETED_KEYS:
        conn.execute(
            sa.text("""
                DELETE FROM settings
                WHERE key = :key
                  AND household_id IS NULL
                  AND node_id IS NULL
                  AND user_id IS NULL
            """),
            {"key": key},
        )


def downgrade() -> None:
    """Re-seed deleted settings with their original defaults."""
    conn = op.get_bind()
    is_postgres = conn.dialect.name == "postgresql"

    restore = [
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
    ]

    for setting in restore:
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
                setting,
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
                setting,
            )
