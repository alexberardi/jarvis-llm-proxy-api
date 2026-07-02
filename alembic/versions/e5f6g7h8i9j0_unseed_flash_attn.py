"""Unseed inference.gguf.flash_attn so env/definition default resolves

Revision ID: e5f6g7h8i9j0
Revises: d4e5f6g7h8i9
Create Date: 2026-07-01 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "e5f6g7h8i9j0"
down_revision = "d4e5f6g7h8i9"
branch_labels = None
depends_on = None

# The seed migration (c3d4e5f6g7h8) inserted a global 'true' row for
# inference.gguf.flash_attn. Settings resolve DB row -> env var -> definition
# default, so that row silently overrode JARVIS_FLASH_ATTN=false baked into
# Dockerfile.rocm (flash attention's HIP kernel faults natively on RDNA4 /
# gfx1201 — uncatchable). With no row, AMD images get env 'false' while
# CUDA/Metal keep the definition default 'true'.
#
# Only the still-default 'true' row is deleted: a user who deliberately
# stored 'false' keeps their override. A deliberate 'true' row is
# indistinguishable from the seed but behaviorally identical to the
# definition default, so removing it is safe.

FLASH_ATTN_SEED = {
    "key": "inference.gguf.flash_attn",
    "value": "true",
    "value_type": "bool",
    "category": "inference.gguf",
    "description": "Enable flash attention",
    "env_fallback": "JARVIS_FLASH_ATTN",
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
        {"key": FLASH_ATTN_SEED["key"], "value": FLASH_ATTN_SEED["value"]},
    )


def downgrade() -> None:
    """Best-effort re-insert of the seeded row (values mirror c3d4e5f6g7h8).

    The seed's ON CONFLICT (key, household_id, node_id, user_id) DO NOTHING
    cannot guard the global scope: uq_setting_scope treats NULL scope columns
    as distinct, so a preserved user 'false' row would not conflict and a
    duplicate 'true' row would be inserted. NOT EXISTS guards it instead
    (valid on both PostgreSQL and SQLite).
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
        FLASH_ATTN_SEED,
    )
