from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000, description="Raw maintenance work order text")

    model_config = {
        "json_schema_extra": {"example": {"text": "Responded to high vibration alarm on P-104. Found bearing wear after inspection. Root cause: inadequate lubrication. Replaced mechanical seal and bearing set. Returned to service."}}
    }


class SimilarCase(BaseModel):
    work_order_id: str
    text: str
    failure_category: str
    similarity_score: float


class ClassifyResponse(BaseModel):
    category:       str   = Field(..., description="Predicted failure category")
    confidence:     float = Field(..., ge=0, le=1)
    all_scores:     dict[str, float]
    model_used:     str
    similar_cases:  Optional[list[SimilarCase]] = None
    extracted_fields: Optional[dict] = Field(None, description="ETL-extracted structured fields")


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    embeddings_loaded: bool
    model_mode: str
    extractor_mode: str
    corpus_size: int
    version: str = "0.1.0"
