"""Unseed model.main.chat_format + context_window (complete the f6g7 fix)

Revision ID: g7h8i9j0k1l2
Revises: f6g7h8i9j0k1
Create Date: 2026-07-03 22:10:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "g7h8i9j0k1l2"
down_revision = "f6g7h8i9j0k1"
branch_labels = None
depends_on = None

# f6g7h8i9j0k1 unseeded model.main.backend, but the seed migration
# (c3d4e5f6g7h8) shadows ALL THREE env vars the installer's generated compose
# sets — proven live by the GPU install-e2e lane immediately after the
# backend fix landed:
#   - model.main.chat_format = "llama3": overrides JARVIS_MODEL_CHAT_FORMAT
#     AND is not even a valid llama-cpp-python handler name (theirs is
#     "llama-3") — every GGUF model load fails with "Invalid chat handler:
#     llama3" on a fresh install
#   - model.main.context_window = "8192": silently clamps the installer's
#     JARVIS_MODEL_CONTEXT_WINDOW=32768
# Same fix shape as f6g7/e5f6: delete only the still-default seeded rows so a
# user's deliberate different value survives; env/definition then resolves.

SEEDS = [
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
]


def upgrade() -> None:
    conn = op.get_bind()
    for seed in SEEDS:
        conn.execute(
            sa.text("""
                DELETE FROM settings
                WHERE key = :key
                  AND value = :value
                  AND household_id IS NULL
                  AND node_id IS NULL
                  AND user_id IS NULL
            """),
            {"key": seed["key"], "value": seed["value"]},
        )


def downgrade() -> None:
    """Best-effort re-insert of the seeded rows (values mirror c3d4e5f6g7h8);
    NOT EXISTS guards the global scope (see e5f6g7h8i9j0's downgrade note)."""
    conn = op.get_bind()
    for seed in SEEDS:
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
            seed,
        )
