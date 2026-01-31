"""add settings table

Revision ID: a1b2c3d4e5f6
Revises: 4f3cb0c3cf28
Create Date: 2026-01-29 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '4f3cb0c3cf28'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(length=255), nullable=False),
        sa.Column('value', sa.Text(), nullable=True),
        sa.Column('value_type', sa.String(length=50), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('requires_reload', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_secret', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('env_fallback', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_settings_id'), 'settings', ['id'], unique=False)
    op.create_index(op.f('ix_settings_key'), 'settings', ['key'], unique=True)
    op.create_index(op.f('ix_settings_category'), 'settings', ['category'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_settings_category'), table_name='settings')
    op.drop_index(op.f('ix_settings_key'), table_name='settings')
    op.drop_index(op.f('ix_settings_id'), table_name='settings')
    op.drop_table('settings')
