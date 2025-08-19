from __future__ import annotations
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from .db import SessionLocal, engine
from .models import Message, Base

# Provides functions to save analysis results to the database and query them.

# Create tables on import
Base.metadata.create_all(bind=engine)

# takes a text string and analysis the result and optional message ID. 
# creates message obj with text and results 
# saves it to db using a session and returns msg ID.
def save_result(text: str, result: Dict[str, Any], message_id: Optional[str] = None) -> str:
    """Insert a row into messages and return the id."""
    if not message_id:
        message_id = str(uuid.uuid4())

    with SessionLocal() as db:
        msg = Message(
            id=message_id,
            text=text,
            language=result.get("language", "unknown"),
            sentiment_label=(result.get("sentiment") or {}).get("label"),
            sentiment_score=(result.get("sentiment") or {}).get("score"),
            toxicity_is_toxic=(result.get("toxicity") or {}).get("is_toxic"),
            toxicity_scores=(result.get("toxicity") or {}).get("scores"),
            intent_label=(result.get("intent") or {}).get("label"),
            intent_method=(result.get("intent") or {}).get("method"),
            ner_entities=(result.get("ner") or {}).get("entities"),
            total_ms=result.get("total_ms"),
        )
        db.add(msg)
        db.commit()
    return message_id

# retrieves messages from db with optional filters
# supports pagination and returns a list of dictionaries with msg details

def query_messages(
    *,
    limit: int = 50,
    offset: int = 0,
    sentiment: Optional[str] = None,
    intent: Optional[str] = None,
    toxic: Optional[bool] = None,
    language: Optional[str] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with SessionLocal() as db:
        stmt = select(Message).order_by(Message.created_at.desc())

        conditions = []
        if sentiment:
            conditions.append(Message.sentiment_label == sentiment)
        if intent:
            conditions.append(Message.intent_label == intent)
        if toxic is not None:
            conditions.append(Message.toxicity_is_toxic == toxic)
        if language:
            conditions.append(Message.language == language)
        if q:
            like = f"%{q}%"
            # SQLite LIKE is case-insensitive by default; fine for dev.
            conditions.append(Message.text.like(like))

        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = stmt.limit(limit).offset(offset)

        rows = db.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "text": r.text,
                "language": r.language,
                "sentiment_label": r.sentiment_label,
                "sentiment_score": r.sentiment_score,
                "toxicity_is_toxic": r.toxicity_is_toxic,
                "intent_label": r.intent_label,
                "intent_method": r.intent_method,
                "ner_entities": r.ner_entities,
                "total_ms": r.total_ms,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "toxicity_scores": r.toxicity_scores,
            }
            for r in rows
        ]
