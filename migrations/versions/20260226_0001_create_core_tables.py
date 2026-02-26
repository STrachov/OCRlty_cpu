"""create core tables

Revision ID: 20260226_0001
Revises:
Create Date: 2026-02-26 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260226_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("api_key_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("key_id", sa.Text(), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("scopes", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("last_used_at", sa.Text(), nullable=True),
        sa.Column("revoked_at", sa.Text(), nullable=True),
        sa.UniqueConstraint("key_id", name="uq_api_keys_key_id"),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
        sa.CheckConstraint("is_active IN (0,1)", name="ck_api_keys_is_active"),
    )
    op.create_index("idx_api_keys_hash", "api_keys", ["key_hash"], unique=False)
    op.create_index("idx_api_keys_keyid", "api_keys", ["key_id"], unique=False)

    op.create_table(
        "jobs",
        sa.Column("job_id", sa.Text(), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("started_at", sa.Text(), nullable=True),
        sa.Column("finished_at", sa.Text(), nullable=True),
        sa.Column("owner_key_id", sa.Text(), nullable=True),
        sa.Column("owner_role", sa.Text(), nullable=True),
        sa.Column("owner_scopes", sa.Text(), nullable=False),
        sa.Column("cancel_requested", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("request_json", sa.Text(), nullable=False),
        sa.Column("progress_json", sa.Text(), nullable=True),
        sa.Column("error_json", sa.Text(), nullable=True),
        sa.Column("result_ref", sa.Text(), nullable=True),
        sa.Column("result_meta_json", sa.Text(), nullable=True),
        sa.Column("result_bytes", sa.Integer(), nullable=True),
        sa.Column("result_sha256", sa.Text(), nullable=True),
        sa.Column("error_ref", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued','running','succeeded','failed','canceled')",
            name="ck_jobs_status",
        ),
        sa.CheckConstraint("cancel_requested IN (0,1)", name="ck_jobs_cancel_requested"),
    )
    op.create_index("idx_jobs_owner", "jobs", ["owner_key_id"], unique=False)
    op.create_index("idx_jobs_status", "jobs", ["status"], unique=False)
    op.create_index("idx_jobs_created", "jobs", ["created_at"], unique=False)

    op.create_table(
        "artifact_index",
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("artifact_id", sa.Text(), nullable=False),
        sa.Column("owner_key_id", sa.Text(), nullable=True),
        sa.Column("day", sa.Text(), nullable=True),
        sa.Column("storage", sa.Text(), nullable=False),
        sa.Column("full_ref", sa.Text(), nullable=False),
        sa.Column("rel_ref", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("kind", "artifact_id", name="pk_artifact_index"),
    )
    op.create_index(
        "idx_artifact_index_owner_kind_day",
        "artifact_index",
        ["owner_key_id", "kind", "day"],
        unique=False,
    )
    op.create_index(
        "idx_artifact_index_kind_created_id",
        "artifact_index",
        ["kind", "created_at", "artifact_id"],
        unique=False,
    )
    op.create_index(
        "idx_artifact_index_owner_kind_created_id",
        "artifact_index",
        ["owner_key_id", "kind", "created_at", "artifact_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_artifact_index_owner_kind_created_id", table_name="artifact_index")
    op.drop_index("idx_artifact_index_kind_created_id", table_name="artifact_index")
    op.drop_index("idx_artifact_index_owner_kind_day", table_name="artifact_index")
    op.drop_table("artifact_index")

    op.drop_index("idx_jobs_created", table_name="jobs")
    op.drop_index("idx_jobs_status", table_name="jobs")
    op.drop_index("idx_jobs_owner", table_name="jobs")
    op.drop_table("jobs")

    op.drop_index("idx_api_keys_keyid", table_name="api_keys")
    op.drop_index("idx_api_keys_hash", table_name="api_keys")
    op.drop_table("api_keys")
