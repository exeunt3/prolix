from __future__ import annotations

import json
import os
from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.models import Hop, TraceRecord, VectorDomain


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./prolix.db")
engine = create_engine(DATABASE_URL, future=True)


class Base(DeclarativeBase):
    pass


class TraceRow(Base):
    __tablename__ = "traces"

    trace_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    object_label: Mapped[str] = mapped_column(String(128), nullable=False)
    vector_domain: Mapped[str] = mapped_column(String(64), nullable=False)
    concept_path: Mapped[str] = mapped_column(Text, nullable=False)
    paragraph_text: Mapped[str] = mapped_column(Text, nullable=False)
    ending_type: Mapped[str] = mapped_column(String(16), nullable=False)
    safety_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    dark_flag: Mapped[bool] = mapped_column(Boolean, default=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


class TraceStore:
    def __init__(self) -> None:
        init_db()

    def insert_trace(self, record: TraceRecord) -> TraceRecord:
        with Session(engine) as session:
            row = TraceRow(
                trace_id=str(record.trace_id),
                created_at=record.created_at,
                object_label=record.object_label,
                vector_domain=record.vector_domain.value,
                concept_path=json.dumps([hop.model_dump() for hop in record.concept_path]),
                paragraph_text=record.paragraph_text,
                ending_type=record.ending_type,
                safety_flag=record.safety_flag,
                dark_flag=record.dark_flag,
            )
            session.add(row)
            session.commit()
        return record

    def get_trace(self, trace_id: UUID) -> TraceRecord | None:
        with Session(engine) as session:
            row = session.get(TraceRow, str(trace_id))
            if row is None:
                return None
            return TraceRecord(
                trace_id=UUID(row.trace_id),
                created_at=row.created_at,
                object_label=row.object_label,
                vector_domain=VectorDomain(row.vector_domain),
                concept_path=[Hop(**hop) for hop in json.loads(row.concept_path)],
                paragraph_text=row.paragraph_text,
                ending_type=row.ending_type,
                safety_flag=row.safety_flag,
                dark_flag=row.dark_flag,
            )
