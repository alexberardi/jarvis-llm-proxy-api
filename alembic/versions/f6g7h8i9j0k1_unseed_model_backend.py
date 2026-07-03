"""Unseed model.main.backend so the installer's env choice resolves

Revision ID: f6g7h8i9j0k1
Revises: e5f6g7h8i9j0
Create Date: 2026-07-03 08:50:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "f6g7h8i9j0k1"
down_revision = "e5f6g7h8i9j0"
branch_labels = None
depends_on = None

# The seed migration (c3d4e5f6g7h8) inserted a global 'VLLM' row for
# model.main.backend. Settings resolve DB row -> env var -> definition
# default, and model.live/background.backend fall back to model.main.backend
# — so on every FRESH install that row silently overrode the
# JARVIS_MODEL_BACKEND=GGUF the installer's generated compose sets. On the
# GGUF-only images (-cuda/-rocm/-vulkan/-cpu) the model service then tried to
# boot vLLM against the local GGUF: EngineCore crashes, :7705 never binds,
# and chat 500s behind a green-looking container. Caught live by the GPU
# install-e2e lane (jarvis#5, run 28633465160 onward) on a rented RTX 3090.
#
# Only the still-default 'VLLM' row is deleted. Unlike the flash_attn unseed
# (e5f6g7h8i9j0), a deliberate VLLM row is NOT behaviorally identical to the
# definition default (GGUF) — but it is indistinguishable from the seed, and
# leaving it bricks every fresh install. A vLLM deployment that loses the row
# still resolves correctly via its JARVIS_MODEL_BACKEND env or one settings
# write; a fresh GGUF install with the row has no recourse at all.

BACKEND_SEED = {
    "key": "model.main.backend",
    "value": "VLLM",
    "value_type": "string",
    "category": "model.main",
    "description": "Main model backend: GGUF, VLLM, TRANSFORMERS, REST, MOCK",
    "env_fallback": "JARVIS_MODEL_BACKEND",
    "requires_reload": True,
    "is_secret": False,
}


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text("""
            DELETE FROM settings
            WHERE key = :key
              AND value = :value
              AND household_id IS NULL
              AND node_id IS NULL
              AND user_id IS NULL
        """),
        {"key": BACKEND_SEED["key"], "value": BACKEND_SEED["value"]},
    )


def downgrade() -> None:
    """Best-effort re-insert of the seeded row (values mirror c3d4e5f6g7h8).

    NOT EXISTS guards the global scope instead of ON CONFLICT — uq_setting_scope
    treats NULL scope columns as distinct, so a conflict clause can't prevent a
    duplicate global row (same reasoning as e5f6g7h8i9j0's downgrade).
    """
    conn = op.get_bind()
    conn.execute(
        sa.text("""
            INSERT INTO settings (key, value, value_type, category, description,
                                 env_fallback, requires_reload, is_secret,
                                 household_id, node_id, user_id)
            SELECT :key, :value, :value_type, :category, :description,
                   :env_fallback, :requires_reload, :is_secret,
                   NULL, NULL, NULL
            WHERE NOT EXISTS (
                SELECT 1 FROM settings
                WHERE key = :key
                  AND household_id IS NULL
                  AND node_id IS NULL
                  AND user_id IS NULL
            )
        """),
        BACKEND_SEED,
    )
