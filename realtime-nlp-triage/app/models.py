from sqlalchemy import Column, String, Float, Boolean, JSON, DateTime, Text
from sqlalchemy.sql import func
from .db import Base

# Message class represents table message : has the following columns id -
# id - unique id for each mwg
# text - input text
# language - detected lang
# sentiment_label , sentiment_score - sentiment
# toxicity_is_toxic , toxicity_scores - if the text is toxic and toxicity details
# intenet_label, intent_method - intent of the text
# ner_entities - named entities
# total_ms - time taken to analyze the text
# created_at - timestamp when the record is created 

class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True)
    text = Column(Text, nullable=False)

    language = Column(String(8), index=True)

    sentiment_label = Column(String(16), index=True)
    sentiment_score = Column(Float)

    toxicity_is_toxic = Column(Boolean, index=True)
    toxicity_scores = Column(JSON)

    intent_label = Column(String(64), index=True)
    intent_method = Column(String(16))

    ner_entities = Column(JSON)

    total_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)