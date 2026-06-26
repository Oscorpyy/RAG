"""
models.py — RAG against the machine
Shared Pydantic data models used across ingestion, retrieval and evaluation.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Chunk of a source file with its character-level position."""

    file_path: str = Field(..., description="Relative path of the source file.")
    first_character_index: int = Field(
        ..., ge=0, description="Start index (inclusive) in the original file."
    )
    last_character_index: int = Field(
        ..., ge=0, description="End index (exclusive) in the original file."
    )

    @property
    def length(self) -> int:
        return self.last_character_index - self.first_character_index


class UnansweredQuestion(BaseModel):
    """A question from the evaluation dataset that has not yet been answered."""

    id: str = Field(..., description="Unique identifier for the question.")
    question: str = Field(..., description="The natural-language question text.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata attached to the question.",
    )


class SearchResult(BaseModel):
    """Single retrieval result for one question."""

    question_id: str = Field(..., description="ID of the originating question.")
    sources: list[MinimalSource] = Field(
        ..., description="Top-k retrieved sources, ordered by relevance."
    )


class StudentSearchResults(BaseModel):
    """Aggregated retrieval results for a full question dataset."""

    results: list[SearchResult] = Field(
        default_factory=list,
        description="One SearchResult per question.",
    )
