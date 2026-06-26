"""
retrieval.py — RAG against the machine
Semantic search over the BM25 index produced by the ingestion phase.

Usage (CLI via Fire):
    python retrieval.py search  --query="how does vllm schedule requests" --top_k=5
    python retrieval.py search_dataset \\
        --dataset_path=data/questions.json \\
        --output_path=data/results.json \\
        --top_k=5
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import fire
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .models import (
    MinimalSource,
    SearchResult,
    StudentSearchResults,
    UnansweredQuestion,
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

DEFAULT_INDEX_DIR: str = "data/processed"
DEFAULT_INDEX_FILE: str = "index.pkl"
DEFAULT_TOP_K: int = 5


# ---------------------------------------------------------------------------
# Tokeniser (must match the one used during ingestion)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """
    Tokenise text into lowercase alphanumeric+underscore tokens.

    Must be identical to the tokeniser used in ingestion.py so that
    query tokens are in the same vocabulary as the index.
    """
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Index loader  (cold-start ≤ 60 s on vLLM corpus)
# ---------------------------------------------------------------------------


def load_index(
    index_path: Optional[str] = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    index_file: str = DEFAULT_INDEX_FILE,
) -> tuple[BM25Okapi, list[MinimalSource]]:
    """
    Load a BM25 index and its associated sources from disk.

    Resolution order for the index path:
      1. Explicit ``index_path`` argument.
      2. ``<index_dir>/<index_file>`` (default: data/processed/index.pkl).

    Args:
        index_path:  Absolute or relative path to the .pkl file. Overrides
                     ``index_dir`` / ``index_file`` when provided.
        index_dir:   Directory that contains the index file.
        index_file:  Filename of the pickled index (default: index.pkl).

    Returns:
        (bm25_index, list_of_MinimalSource)

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        KeyError:          If the pickle payload is missing expected keys.
    """
    resolved = Path(index_path) if index_path else Path(index_dir) / index_file

    if not resolved.exists():
        raise FileNotFoundError(
            f"Index not found at '{resolved.resolve()}'. "
            "Run the ingestion phase first: "
            "python -m student index --repo_path=./vllm"
        )

    logger.info("Loading index from %s …", resolved)
    t0 = time.perf_counter()

    with open(resolved, "rb") as fh:
        payload = pickle.load(fh)

    if "index" not in payload or "sources" not in payload:
        raise KeyError(
            f"Unexpected pickle format in '{resolved}'. "
            "Expected keys: 'index', 'sources'."
        )

    bm25: BM25Okapi = payload["index"]
    sources: list[MinimalSource] = [
        MinimalSource(**s) if isinstance(s, dict) else s
        for s in payload["sources"]
    ]

    elapsed = time.perf_counter() - t0
    logger.info(
        "Index loaded in %.2fs — %d chunks, vocab: %d terms",
        elapsed,
        len(sources),
        len(bm25.idf),
    )
    return bm25, sources


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------


def search(
    query: str,
    bm25: BM25Okapi,
    sources: list[MinimalSource],
    top_k: int = DEFAULT_TOP_K,
) -> list[MinimalSource]:
    """
    Retrieve the top-k most relevant chunks for a single query.

    The function tokenises the query with the same tokeniser used at
    index time, calls ``BM25Okapi.get_scores``, and returns the
    ``top_k`` sources ordered by descending BM25 score.

    Args:
        query:   Natural-language or code search query.
        bm25:    Pre-loaded BM25Okapi index.
        sources: List of MinimalSource objects aligned with the index.
        top_k:   Number of results to return (default: 5).

    Returns:
        Ordered list of up to ``top_k`` MinimalSource objects.
        Every item is guaranteed to contain file_path,
        first_character_index, and last_character_index.
    """
    if not query.strip():
        logger.warning("Empty query received — returning no results.")
        return []

    tokens = _tokenize(query)
    if not tokens:
        logger.warning("Query '%s' produced no tokens — returning no results.", query)
        return []

    scores = bm25.get_scores(tokens)

    # argsort descending; limit to min(top_k, len(sources))
    k = min(top_k, len(sources))
    # numpy is a transitive dep of rank-bm25, so scores is an ndarray
    top_indices = scores.argsort()[::-1][:k]

    results: list[MinimalSource] = [sources[i] for i in top_indices]
    return results


# ---------------------------------------------------------------------------
# Batch search
# ---------------------------------------------------------------------------


def search_dataset(
    questions: list[UnansweredQuestion],
    bm25: BM25Okapi,
    sources: list[MinimalSource],
    top_k: int = DEFAULT_TOP_K,
) -> StudentSearchResults:
    """
    Run search for every question in a dataset and aggregate results.

    Args:
        questions: List of UnansweredQuestion models to process.
        bm25:      Pre-loaded BM25Okapi index.
        sources:   List of MinimalSource objects aligned with the index.
        top_k:     Number of results per question (default: 5).

    Returns:
        StudentSearchResults containing one SearchResult per question.
    """
    results: list[SearchResult] = []

    for question in tqdm(questions, desc="Searching", unit="q"):
        try:
            top_sources = search(question.question, bm25, sources, top_k)
            results.append(
                SearchResult(
                    question_id=question.id,
                    sources=top_sources,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Search failed for question id=%s: %s", question.id, exc
            )
            results.append(
                SearchResult(question_id=question.id, sources=[])
            )

    return StudentSearchResults(results=results)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_questions(dataset_path: str) -> list[UnansweredQuestion]:
    """
    Load a JSON dataset of questions into UnansweredQuestion models.

    The JSON file must be either:
      - A JSON array of question objects, or
      - A JSON object with a ``"questions"`` key containing such an array.

    Args:
        dataset_path: Path to the JSON question file.

    Returns:
        List of UnansweredQuestion models.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the JSON structure is unrecognised.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: '{path.resolve()}'")

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # Accept both {"questions": [...]} and bare [...]
    if isinstance(raw, dict):
        if "questions" in raw:
            raw = raw["questions"]
        else:
            raise ValueError(
                f"Unrecognised JSON structure in '{path}'. "
                "Expected a list or a dict with key 'questions'."
            )

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON array in '{path}', got {type(raw).__name__}."
        )

    questions: list[UnansweredQuestion] = []
    for i, item in enumerate(raw):
        try:
            questions.append(UnansweredQuestion(**item))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed question at index %d: %s", i, exc)

    logger.info("Loaded %d question(s) from %s", len(questions), path)
    return questions


def save_results(results: StudentSearchResults, output_path: str) -> None:
    """
    Serialise a StudentSearchResults object to a JSON file.

    Args:
        results:     The aggregated search results to save.
        output_path: Destination path for the JSON file.
                     Parent directories are created if they do not exist.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results.model_dump(), fh, indent=2, ensure_ascii=False)

    logger.info("Results saved → %s", out)


# ---------------------------------------------------------------------------
# Retriever class  (stateful wrapper for reuse across CLI calls)
# ---------------------------------------------------------------------------


class Retriever:
    """
    Stateful wrapper that owns a loaded BM25 index.

    Preferred for repeated searches (avoids reloading the index on every
    call) and for testing (the index can be injected directly).

    Example::

        retriever = Retriever.from_disk("data/processed/index.pkl")
        sources   = retriever.search("PagedAttention memory layout", top_k=5)
    """

    def __init__(
        self,
        bm25: BM25Okapi,
        sources: list[MinimalSource],
    ) -> None:
        self._bm25 = bm25
        self._sources = sources

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_disk(
        cls,
        index_path: Optional[str] = None,
        index_dir: str = DEFAULT_INDEX_DIR,
        index_file: str = DEFAULT_INDEX_FILE,
    ) -> "Retriever":
        """Load index from disk and return a ready Retriever."""
        bm25, sources = load_index(index_path, index_dir, index_file)
        return cls(bm25, sources)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[MinimalSource]:
        """Single-query search. See module-level ``search()`` for details."""
        return search(query, self._bm25, self._sources, top_k)

    def search_dataset(
        self,
        questions: list[UnansweredQuestion],
        top_k: int = DEFAULT_TOP_K,
    ) -> StudentSearchResults:
        """Batch search. See module-level ``search_dataset()`` for details."""
        return search_dataset(questions, self._bm25, self._sources, top_k)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        """Number of indexed chunks."""
        return len(self._sources)

    @property
    def vocab_size(self) -> int:
        """Number of distinct tokens in the BM25 vocabulary."""
        return len(self._bm25.idf)


# ---------------------------------------------------------------------------
# CLI  (Fire)
# ---------------------------------------------------------------------------


class RetrievalCLI:
    """
    CLI for the retrieval phase of RAG against the machine.

    Commands
    --------
    search          Run a single query and print results.
    search_dataset  Batch-process a JSON question file and write results.

    Examples
    --------
    ::

        python retrieval.py search --query="vllm continuous batching" --top_k=5
        python retrieval.py search_dataset \\
            --dataset_path=data/questions.json \\
            --output_path=data/results.json   \\
            --top_k=5
    """

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        index_path: Optional[str] = None,
        index_dir: str = DEFAULT_INDEX_DIR,
        index_file: str = DEFAULT_INDEX_FILE,
    ) -> None:
        """
        Retrieve and print the top-k sources for a single query.

        Args:
            query:       The search query string.
            top_k:       Number of results to display (default: 5).
            index_path:  Optional direct path to the .pkl index file.
            index_dir:   Directory containing the index (default: data/processed).
            index_file:  Filename of the index (default: index.pkl).
        """
        retriever = Retriever.from_disk(index_path, index_dir, index_file)

        logger.info(
            "Index ready — %d chunks, %d vocab terms",
            retriever.corpus_size,
            retriever.vocab_size,
        )

        t0 = time.perf_counter()
        results = retriever.search(query, top_k)
        elapsed = time.perf_counter() - t0

        print(f"\n🔍 Query : {query!r}")
        print(f"   Top-{top_k} results  ({elapsed * 1000:.1f} ms)\n")

        if not results:
            print("   (no results)")
            return

        for rank, src in enumerate(results, start=1):
            print(
                f"   [{rank}] {src.file_path}"
                f"  [{src.first_character_index}:{src.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        output_path: str,
        top_k: int = DEFAULT_TOP_K,
        index_path: Optional[str] = None,
        index_dir: str = DEFAULT_INDEX_DIR,
        index_file: str = DEFAULT_INDEX_FILE,
    ) -> None:
        """
        Batch-process a JSON question file and write a StudentSearchResults JSON.

        Args:
            dataset_path: Path to the JSON file containing UnansweredQuestion objects.
            output_path:  Destination path for the StudentSearchResults JSON.
            top_k:        Number of results per question (default: 5).
            index_path:   Optional direct path to the .pkl index file.
            index_dir:    Directory containing the index (default: data/processed).
            index_file:   Filename of the index (default: index.pkl).
        """
        t_start = time.perf_counter()

        retriever = Retriever.from_disk(index_path, index_dir, index_file)
        questions = load_questions(dataset_path)

        if not questions:
            logger.error("No questions loaded — aborting.")
            return

        results = retriever.search_dataset(questions, top_k)
        save_results(results, output_path)

        total = time.perf_counter() - t_start
        per_q = (total / len(questions)) * 1000 if questions else 0

        print("\n✅ Batch search complete")
        print(f"   Questions processed : {len(questions)}")
        print(f"   Output              : {output_path}")
        print(f"   Total time          : {total:.2f}s")
        print(f"   Average / question  : {per_q:.1f} ms")

        if len(questions) > 0:
            projected = (total / len(questions)) * 1000
            status = "✅" if projected < 90 else "⚠️ "
            print(f"   Projected (1 000 q) : {projected:.1f}s  {status}")
