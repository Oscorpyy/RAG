"""
evaluation.py — RAG against the machine
Evaluation module to measure retrieval quality using Recall@k metric with 5%
character overlap rule.

The overlap rule ensures retrieved sources are only counted as "found" when
there is at least 5% character overlap with ground-truth source segments.

Usage CLI (via Fire) :
    python -m student evaluate_dataset \\
        --student_search_results_path data/results_docs.json \\
        --ground_truth_path \
datasets_public/public/AnsweredQuestions/ \\
        --dataset_type docs \\
        --overlap_threshold 0.05

    python -m student evaluate_dataset \\
        --student_search_results_path data/results_code.json \\
        --ground_truth_path \
datasets_public/public/AnsweredQuestions/ \\
        --dataset_type code \\
        --overlap_threshold 0.05
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import NamedTuple

import fire
from tqdm import tqdm

from .models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OVERLAP_THRESHOLD: float = 0.05  # 5% character overlap
DEFAULT_K_VALUES: list[int] = [1, 3, 5, 10]
TARGET_RECALL_DOCS: float = 0.80  # 80% for documentation
TARGET_RECALL_CODE: float = 0.50  # 50% for code


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class RecallMetrics(NamedTuple):
    """Recall metrics for a single dataset."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    average_recall: float
    total_questions: int
    total_sources_found: int
    total_sources_expected: int


class QuestionEvaluation(NamedTuple):
    """Evaluation result for a single question."""

    question_id: str
    question: str
    sources_found: int
    sources_expected: int
    recall: float
    k_used: int


# ---------------------------------------------------------------------------
# Overlap Calculation
# ---------------------------------------------------------------------------


def calculate_overlap_ratio(seg_a: str, seg_b: str) -> float:
    """
    Calculate character overlap ratio between two text segments.

    Uses the intersection of character sets divided by the minimum length
    to get a simple overlap percentage. This captures semantic proximity.

    Args:
        seg_a: First segment.
        seg_b: Second segment.

    Returns:
        Overlap ratio (0.0 to 1.0).
    """
    if not seg_a or not seg_b:
        return 0.0

    len_a = len(seg_a)
    len_b = len(seg_b)
    min_len = min(len_a, len_b)

    if min_len == 0:
        return 0.0

    # Count common characters (order-independent)
    set_a = set(seg_a)
    set_b = set(seg_b)
    intersection = len(set_a & set_b)

    return intersection / min_len


def extract_segment(
    file_path: str, start_idx: int, end_idx: int, repo_path: str
) -> str | None:
    """
    Extract a text segment from a file given character indices.

    Args:
        file_path: Path within repo (e.g., "vllm/engine/llm_engine.py").
        start_idx: Start character index (inclusive).
        end_idx:   End character index (inclusive).
        repo_path: Root path to the repository.

    Returns:
        Extracted segment string, or None if file not found.
    """
    full_path = Path(repo_path) / file_path
    if not full_path.exists():
        logger.warning(
            "File not found: %s (resolved: %s)", file_path, full_path
        )
        return None

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
            if end_idx >= len(content):
                logger.warning(
                    "Index out of bounds for %s: [%d:%d] vs file len %d",
                    file_path,
                    start_idx,
                    end_idx,
                    len(content),
                )
                return content[start_idx:]
            return content[start_idx:end_idx + 1]
    except (IOError, UnicodeDecodeError) as e:
        logger.warning("Failed to read %s: %s", file_path, e)
        return None


def is_source_found(
    retrieved: MinimalSource,
    ground_truth_sources: list[MinimalSource],
    threshold: float,
    repo_path: str,
) -> bool:
    """
    Check if a retrieved source overlaps with any ground-truth source.

    A source is considered "found" if the character overlap ratio with
    at least one ground-truth source meets or exceeds the threshold.

    Args:
        retrieved:           Retrieved source to validate.
        ground_truth_sources: List of ground-truth sources for the question.
        threshold:           Minimum overlap ratio (default 0.05 = 5%).
        repo_path:           Root path to extract segments from.

    Returns:
        True if found (overlap >= threshold with any ground-truth source).
    """
    retrieved_segment = extract_segment(
        retrieved.file_path,
        retrieved.first_character_index,
        retrieved.last_character_index,
        repo_path,
    )

    if retrieved_segment is None:
        return False

    for ground_truth in ground_truth_sources:
        gt_segment = extract_segment(
            ground_truth.file_path,
            ground_truth.first_character_index,
            ground_truth.last_character_index,
            repo_path,
        )

        if gt_segment is None:
            continue

        overlap = calculate_overlap_ratio(retrieved_segment, gt_segment)
        if overlap >= threshold:
            return True

    return False


# ---------------------------------------------------------------------------
# Loading Data
# ---------------------------------------------------------------------------


def load_student_results(
    results_path: str,
) -> StudentSearchResults:
    """
    Load student search results from JSON file.

    Args:
        results_path: Path to StudentSearchResults JSON file.

    Returns:
        Parsed StudentSearchResults model.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError:        If JSON is invalid or schema mismatch.
    """
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Student results file not found: '{path.resolve()}'"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return StudentSearchResults(**raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to parse {path}: {e}") from e


def load_ground_truth(
    truth_path: str,
) -> dict[str, AnsweredQuestion]:
    """
    Load ground-truth answered questions from JSON file.

    Supports both bare list format and wrapped format (with 'rag_questions').

    Args:
        truth_path: Path to ground-truth JSON file.

    Returns:
        Dictionary mapping question_id -> AnsweredQuestion.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError:        If JSON format is unrecognized.
    """
    path = Path(truth_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: '{path.resolve()}'"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    # Normalize to list
    if isinstance(raw, dict):
        if "rag_questions" in raw:
            raw = raw["rag_questions"]
        elif "questions" in raw:
            raw = raw["questions"]
        else:
            raise ValueError(
                f"Unrecognized JSON structure in {path}. "
                f"Expected 'rag_questions' or 'questions' key."
            )

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected list in {path}, got {type(raw).__name__}"
        )

    result: dict[str, AnsweredQuestion] = {}
    for item in raw:
        try:
            # Try parsing as AnsweredQuestion (has sources/answer)
            q = AnsweredQuestion(**item)
            result[q.question_id] = q
        except Exception as e:
            logger.warning(
                "Skipping item (not a valid AnsweredQuestion): %s", e
            )

    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_single_question(
    student_result: MinimalSearchResults,
    ground_truth: AnsweredQuestion,
    overlap_threshold: float,
    repo_path: str,
) -> QuestionEvaluation:
    """
    Evaluate recall for a single question.

    Args:
        student_result:  Student's search result for the question.
        ground_truth:    Ground-truth sources and answer.
        overlap_threshold: Minimum overlap ratio.
        repo_path:       Root path to repository.

    Returns:
        QuestionEvaluation with recall score.
    """
    sources_expected = len(ground_truth.sources)
    sources_found = 0

    for retrieved in student_result.retrieved_sources:
        if is_source_found(
            retrieved,
            ground_truth.sources,
            overlap_threshold,
            repo_path,
        ):
            sources_found += 1

    recall = (
        sources_found / sources_expected if sources_expected > 0 else 1.0
    )

    return QuestionEvaluation(
        question_id=student_result.question_id,
        question=student_result.question,
        sources_found=sources_found,
        sources_expected=sources_expected,
        recall=recall,
        k_used=len(student_result.retrieved_sources),
    )


def evaluate_dataset(
    student_results: StudentSearchResults,
    ground_truth: dict[str, AnsweredQuestion],
    overlap_threshold: float,
    repo_path: str,
) -> tuple[list[QuestionEvaluation], RecallMetrics]:
    """
    Evaluate recall for entire dataset.

    Args:
        student_results: All search results from student.
        ground_truth:    All ground-truth answered questions.
        overlap_threshold: Minimum overlap ratio.
        repo_path:       Root path to repository.

    Returns:
        (list of question evaluations, overall RecallMetrics).
    """
    evaluations: list[QuestionEvaluation] = []
    total_sources_found = 0
    total_sources_expected = 0

    pbar = tqdm(
        student_results.search_results,
        desc="Evaluating",
        unit="q",
    )

    for student_result in pbar:
        if student_result.question_id not in ground_truth:
            logger.warning(
                "Question %s not in ground truth, skipping",
                student_result.question_id,
            )
            continue

        gt = ground_truth[student_result.question_id]
        evaluation = evaluate_single_question(
            student_result,
            gt,
            overlap_threshold,
            repo_path,
        )
        evaluations.append(evaluation)
        total_sources_found += evaluation.sources_found
        total_sources_expected += evaluation.sources_expected

    # Calculate recall@k
    if evaluations:
        avg_recall = sum(e.recall for e in evaluations) / len(evaluations)
    else:
        avg_recall = 0.0

    metrics = RecallMetrics(
        recall_at_1=avg_recall,  # Approximation; for exact need k info
        recall_at_3=avg_recall,
        recall_at_5=avg_recall,
        recall_at_10=avg_recall,
        average_recall=avg_recall,
        total_questions=len(evaluations),
        total_sources_found=total_sources_found,
        total_sources_expected=total_sources_expected,
    )

    return evaluations, metrics


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------


def print_evaluation_summary(
    metrics: RecallMetrics,
    dataset_type: str = "unknown",
    target_recall: float | None = None,
) -> None:
    """
    Print a formatted summary of evaluation metrics.

    Args:
        metrics:      RecallMetrics to display.
        dataset_type: Type of dataset (e.g., "docs", "code").
        target_recall: Target recall threshold for comparison.
    """
    print("\n" + "=" * 70)
    print(f"📊 Evaluation Summary — {dataset_type.upper()}")
    print("=" * 70)

    print(f"\n✓ Total questions evaluated  : {metrics.total_questions}")
    print(
        f"✓ Total sources found       : {metrics.total_sources_found} / "
        f"{metrics.total_sources_expected}"
    )

    print("\n📈 Recall Metrics:")
    print(f"   Recall@1               : {metrics.recall_at_1:.2%}")
    print(f"   Recall@3               : {metrics.recall_at_3:.2%}")
    print(f"   Recall@5               : {metrics.recall_at_5:.2%}")
    print(f"   Recall@10              : {metrics.recall_at_10:.2%}")
    print(f"   Average Recall         : {metrics.average_recall:.2%}")

    if target_recall is not None:
        status = "✅" if metrics.average_recall >= target_recall else "❌"
        print(
            f"\n{status} Target (>= {target_recall:.0%})"
            f"  : {metrics.average_recall:.2%}"
        )

    print("\n" + "=" * 70)


def save_evaluation_results(
    evaluations: list[QuestionEvaluation],
    metrics: RecallMetrics,
    output_path: str,
) -> None:
    """
    Save detailed evaluation results to JSON file.

    Args:
        evaluations: Per-question evaluation results.
        metrics:     Overall metrics.
        output_path: Output JSON file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metrics": {
            "total_questions": metrics.total_questions,
            "total_sources_found": metrics.total_sources_found,
            "total_sources_expected": metrics.total_sources_expected,
            "recall_at_1": metrics.recall_at_1,
            "recall_at_3": metrics.recall_at_3,
            "recall_at_5": metrics.recall_at_5,
            "recall_at_10": metrics.recall_at_10,
            "average_recall": metrics.average_recall,
        },
        "evaluations": [
            {
                "question_id": e.question_id,
                "question": e.question,
                "sources_found": e.sources_found,
                "sources_expected": e.sources_expected,
                "recall": e.recall,
                "k_used": e.k_used,
            }
            for e in evaluations
        ],
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation results saved → %s", out)


# ---------------------------------------------------------------------------
# CLI (Fire)
# ---------------------------------------------------------------------------


class EvaluationCLI:
    """
    CLI for evaluation phase of RAG against the machine.

    Commands
    --------
    evaluate_dataset  Evaluate retrieval quality against ground truth.

    Examples
    --------
    ::

    python -m student evaluate_dataset \\
        --student_search_results_path data/results_docs.json \\
        --ground_truth_path \\
        datasets_public/public/AnsweredQuestions/\\
        dataset_docs_public.json \\
            --dataset_type docs

    python -m student evaluate_dataset \\
        --student_search_results_path data/results_code.json \\
        --ground_truth_path \\
        datasets_public/public/AnsweredQuestions/\\
        dataset_code_public.json \\
            --dataset_type code \\
            --overlap_threshold 0.10 \\
            --repo_path ./data/row/vllm-0.10.1/vllm \\
            --output_path data/eval_results_code.json
    """

    def evaluate_dataset(
        self,
        student_search_results_path: str,
        ground_truth_path: str,
        dataset_type: str = "unknown",
        overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
        repo_path: str = "./data/row/vllm-0.10.1/vllm",
        output_path: str | None = None,
    ) -> None:
        """
        Evaluate student search results against ground truth.

        Calculates Recall@k metric with configurable overlap threshold.
        Results can be saved to a JSON file for further analysis.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            ground_truth_path: Path to ground-truth AnsweredQuestions JSON.
            dataset_type: Type of dataset (e.g., "docs", "code") for display.
            overlap_threshold: Min overlap ratio (0.0–1.0, default 0.05).
            repo_path: Root path to repository for segment extraction.
            output_path: Optional path to save detailed results JSON.
        """
        t_start = time.perf_counter()

        logger.info(
            "Loading student results from %s", student_search_results_path
        )
        student_results = load_student_results(student_search_results_path)

        logger.info(
            "Loading ground truth from %s", ground_truth_path
        )
        ground_truth = load_ground_truth(ground_truth_path)

        logger.info(
            "Evaluating %d questions with overlap threshold %.0f%%",
            len(student_results.search_results),
            overlap_threshold * 100,
        )

        evaluations, metrics = evaluate_dataset(
            student_results,
            ground_truth,
            overlap_threshold,
            repo_path,
        )

        elapsed = time.perf_counter() - t_start

        # Determine target recall based on dataset type
        target = (
            TARGET_RECALL_DOCS
            if dataset_type.lower() == "docs"
            else TARGET_RECALL_CODE
            if dataset_type.lower() == "code"
            else None
        )

        print_evaluation_summary(metrics, dataset_type, target)

        if output_path:
            save_evaluation_results(evaluations, metrics, output_path)

        print(f"\n⏱️  Evaluation completed in {elapsed:.2f}s\n")


if __name__ == "__main__":
    fire.Fire(EvaluationCLI)
