"""add multitenant settings columns

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-05 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b2c3d4e5f6g7'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add multi-tenant scoping columns
    op.add_column('settings', sa.Column('household_id', sa.String(length=255), nullable=True))
    op.add_column('settings', sa.Column('node_id', sa.String(length=255), nullable=True))
    op.add_column('settings', sa.Column('user_id', sa.Integer(), nullable=True))

    # Add indexes for the new columns
    op.create_index(op.f('ix_settings_household_id'), 'settings', ['household_id'], unique=False)
    op.create_index(op.f('ix_settings_node_id'), 'settings', ['node_id'], unique=False)
    op.create_index(op.f('ix_settings_user_id'), 'settings', ['user_id'], unique=False)

    # Drop the unique constraint on key (it was too strict)
    op.drop_index(op.f('ix_settings_key'), table_name='settings')

    # Create a new unique constraint that includes scope columns
    op.create_index(
        'ix_settings_key_nonunique',
        'settings',
        ['key'],
        unique=False
    )
    op.create_unique_constraint(
        'uq_setting_scope',
        'settings',
        ['key', 'household_id', 'node_id', 'user_id']
    )


def downgrade() -> None:
    # Drop the unique constraint
    op.drop_constraint('uq_setting_scope', 'settings', type_='unique')
    op.drop_index('ix_settings_key_nonunique', table_name='settings')

    # Restore original unique key index
    op.create_index(op.f('ix_settings_key'), 'settings', ['key'], unique=True)

    # Drop the scope indexes
    op.drop_index(op.f('ix_settings_user_id'), table_name='settings')
    op.drop_index(op.f('ix_settings_node_id'), table_name='settings')
    op.drop_index(op.f('ix_settings_household_id'), table_name='settings')

    # Drop the scope columns
    op.drop_column('settings', 'user_id')
    op.drop_column('settings', 'node_id')
    op.drop_column('settings', 'household_id')
