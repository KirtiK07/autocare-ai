"""ORM tables: predictions and chat_messages."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, JSONType


def _now() -> datetime:
    return datetime.now(UTC)


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSONType)
    probability: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(16))
    risk_band: Mapped[str] = mapped_column(String(16), index=True)
    top_features: Mapped[list[dict[str, Any]]] = mapped_column(JSONType)
    model_version: Mapped[str] = mapped_column(String(64))

    messages: Mapped[list[ChatMessageRow]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", order_by="ChatMessageRow.id"
    )


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), index=True, nullable=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, default=list)
    supported: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    prediction: Mapped[PredictionRow | None] = relationship(back_populates="messages")
