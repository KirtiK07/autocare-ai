"""predictions and chat_messages

Revision ID: 0001
Revises:
Create Date: 2026-11-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("inputs", JSONType, nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("label", sa.String(16), nullable=False),
        sa.Column("risk_band", sa.String(16), nullable=False),
        sa.Column("top_features", JSONType, nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("ix_predictions_risk_band", "predictions", ["risk_band"])
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "prediction_id", sa.Integer(), sa.ForeignKey("predictions.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", JSONType, nullable=False),
        sa.Column("supported", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chat_messages_prediction_id", "chat_messages", ["prediction_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_messages_prediction_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_predictions_risk_band", table_name="predictions")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_table("predictions")
