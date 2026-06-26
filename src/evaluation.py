"""
evaluation.py — RAG against the machine
Evaluates the quality of retrieval by calculating Recall@k metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import fire
from pydantic import BaseModel, Field
from tqdm import tqdm

from .models import (
    MinimalSource,
    SearchResult,
    StudentSearchResults,
    UnansweredQuestion,
)

# ---------------------------------------------------------------------------
# Additional Models (Required for Evaluation)
# ---------------------------------------------------------------------------

class AnsweredQuestion(BaseModel):
    """A question from the evaluation dataset with known correct sources."""

    id: str = Field(..., description="Unique identifier for the question.")
    question: str = Field(..., description="The natural-language question text.")
    correct_sources: List[MinimalSource] = Field(
        ..., description="List of source segments that contain the answer."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadata (e.g., 'category': 'code' or 'documentation')."
    )


class RagDataset(BaseModel):
    """A collection of answered questions for evaluation."""

    questions: List[AnsweredQuestion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Handles the calculation of retrieval metrics like Recall@k.
    """

    def __init__(self, overlap_threshold: float = 0.05):
        self.overlap_threshold = overlap_threshold

    def has_sufficient_overlap(self, source_a: MinimalSource, source_b: MinimalSource) -> bool:
        """
        Checks if two sources overlap by at least the threshold percentage.
        The overlap is calculated relative to the smaller of the two segments.
        """
        # Ensure we are comparing the same file
        if source_a.file_path != source_b.file_path:
            return False

        # Calculate intersection of the two character intervals [start, end)
        overlap_start = max(source_a.first_character_index, source_b.first_character_index)
        overlap_end = min(source_a.last_character_index, source_b.last_character_index)
        overlap_len = max(0, overlap_end - overlap_start)

        if overlap_len == 0:
            return False

        # Rule: 5% character overlap relative to the smaller segment 
        # (or ground truth depending on interpretation; here we use the min length)
        min_len = min(source_a.length, source_b.length)
        return (overlap_len / min_len) >= self.overlap_threshold

    def calculate_recall_at_k(
        self, 
        k: int, 
        ground_truth: AnsweredQuestion, 
        retrieved: SearchResult
    ) -> float:
        """
        Calculates Recall@k for a single question.
        Formula: (Correct sources found in top-k) / (Total correct sources)
        """
        correct_sources = ground_truth.correct_sources
        if not correct_sources:
            return 0.0

        # Limit retrieved sources to top-k
        top_k_sources = retrieved.sources[:k]
        
        found_count = 0
        for gold in correct_sources:
            # A gold source is 'found' if ANY of the top-k sources overlap it sufficiently
            if any(self.has_sufficient_overlap(gold, ret) for ret in top_k_sources):
                found_count += 1

        return found_count / len(correct_sources)


# ---------------------------------------------------------------------------
# CLI Implementation
# ---------------------------------------------------------------------------

class EvaluationCLI:
    """
    CLI for evaluating RAG retrieval performance.

    Example:
        python evaluation.py run \
            --results_path=data/results.json \
            --ground_truth_path=data/ground_truth.json
    """

    def run(
        self,
        results_path: str,
        ground_truth_path: str,
    ) -> None:
        """
        Load results and ground truth, then print Recall@k for k=1, 3, 5, 10.
        """
        logger.info(f"Loading results from {results_path}...")
        with open(results_path, "r") as f:
            results_data = json.load(f)
        student_results = StudentSearchResults(**results_data)

        logger.info(f"Loading ground truth from {ground_truth_path}...")
        with open(ground_truth_path, "r") as f:
            gt_data = json.load(f)
        dataset = RagDataset(**gt_data)

        # Map results by question_id for O(1) lookup
        results_map: Dict[str, SearchResult] = {
            res.question_id: res for res in student_results.results
        }

        evaluator = Evaluator()
        k_values = [1, 3, 5, 10]
        
        # To track performance by category (doc vs code)
        category_scores: Dict[str, Dict[int, List[float]]] = {}

        logger.info("Calculating metrics...")
        for q in tqdm(dataset.questions, desc="Evaluating questions"):
            if q.id not in results_map:
                logger.warning(f"Question ID {q.id} not found in results. Skipping.")
                continue

            res = results_map[q.id]
            category = q.metadata.get("category", "general")
            
            if category not in category_scores:
                category_scores[category] = {k: [] for k in k_values}

            for k in k_values:
                score = evaluator.calculate_recall_at_k(k, q, res)
                category_scores[category][k].append(score)

        # --- Final Summary Reporting ---
        print("\n" + "="*50)
        print("RETRIEVAL EVALUATION SUMMARY")
        print("="*50)

        for category, metrics in category_scores.items():
            print(f"\nCategory: {category.upper()}")
            for k in k_values:
                scores = metrics[k]
                avg_recall = sum(scores) / len(scores) if scores else 0.0
                
                # Target highlighting
                target = 0.0
                if category == "documentation" and k == 5: target = 0.80
                elif category == "code" and k == 5: target = 0.50
                
                target_str = f" (Target: {target:.0%})" if target > 0 else ""
                status = "✅" if avg_recall >= target else "❌" if target > 0 else ""
                
                print(f"Recall@{k}: {avg_recall:.4f}{target_str} {status}")
        
        print("="*50)
