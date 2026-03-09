"""add ground truth files table

Revision ID: 20260306_0002
Revises: 20260226_0001
Create Date: 2026-03-06 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260306_0002"
down_revision = "20260226_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ground_truth_files",
        sa.Column("gt_id", sa.Text(), primary_key=True),
        sa.Column("owner_key_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("s3_rel", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.UniqueConstraint("s3_rel", name="uq_ground_truth_files_s3_rel"),
    )
    op.create_index(
        "idx_ground_truth_files_owner_created",
        "ground_truth_files",
        ["owner_key_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_ground_truth_files_owner_created", table_name="ground_truth_files")
    op.drop_table("ground_truth_files")
