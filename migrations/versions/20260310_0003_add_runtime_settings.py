"""add runtime settings table

Revision ID: 20260310_0003
Revises: 20260306_0002
Create Date: 2026-03-10 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260310_0003"
down_revision = "20260306_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runtime_settings",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value_text", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("updated_by_key_id", sa.Text(), nullable=True),
    )
    op.create_index("idx_runtime_settings_updated_at", "runtime_settings", ["updated_at"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_runtime_settings_updated_at", table_name="runtime_settings")
    op.drop_table("runtime_settings")
