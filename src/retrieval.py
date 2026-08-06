"""
retrieval.py — RAG against the machine
BM25 search on the index produced by the ingestion phase.

The index is loaded via the native bm25s API (BM25.load) from the
`data/processed` directory — no pickle, no handcrafted files.

Usage CLI (via Fire) :
    python -m student search_dataset \
        --dataset_path=datasets_public/public/UnansweredQuestions/\
dataset_docs_public.json \
        --output_path=data/results_docs.json \
        --k=5

    python -m student search_dataset \
        --dataset_path=datasets_public/public/UnansweredQuestions/\
dataset_code_public.json \
        --output_path=data/results_code.json \
        --k=10

    python -m student search \
        --query="how does vllm schedule requests" \
        --k=5
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import bm25s
import ollama
from tqdm import tqdm

from .ingestion import BM25_K1, BM25_B, tokenize_query
from .models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)
from .parsing import parse_json_file, parse_questions

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

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DEFAULT_INDEX_DIR: str = str(
    PROJECT_ROOT / "data" / "processed"
)
DEFAULT_K: int = 5
DEFAULT_SEARCH_OUTPUT_DIR: str = (
    "data/output/search_results/UnansweredQuestions"
)
DEFAULT_SEARCH_DATASET_OUTPUT_PATH: str = (
    "data/output/search_results/UnansweredQuestions/"
    "dataset_docs_public.json"
)
DEFAULT_DOCS_DATASET_PATH: str = (
    "datasets_public/public/UnansweredQuestions/"
    "dataset_docs_public.json"
)
DEFAULT_CODE_DATASET_PATH: str = (
    "datasets_public/public/UnansweredQuestions/"
    "dataset_code_public.json"
)

# Ollama settings for query expansion
DEFAULT_EXPANSION_MODEL: str = "qwen3:0.6b"
DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"

LOCAL_QUERY_EXPANSION_TERMS: dict[str, tuple[str, ...]] = {
    "scheduler": ("schedule", "dispatch", "queue", "batch"),
    "schedule": ("scheduler", "dispatch", "queue", "batch"),
    "request": ("requests", "prompt", "token", "batch"),
    "requests": ("request", "prompt", "token", "batch"),
    "batch": ("batching", "request", "queue", "prefill"),
    "prefill": ("decode", "token", "kv", "cache"),
    "decode": ("prefill", "token", "generation"),
    "cache": ("kv", "memory", "block"),
    "token": ("tokens", "sampling", "vocab"),
    "gpu": ("cuda", "device", "kernel"),
    "cuda": ("gpu", "device", "kernel"),
    "model": ("models", "weights", "checkpoint"),
}

LOCAL_QUERY_EXPANSION_TERMS_EXTENDED: dict[str, tuple[str, ...]] = {
    "cli": ("command", "serve", "bench", "chat", "run-batch"),
    "benchmark": ("benchmarking", "latency", "throughput"),
    "benchmarking": ("benchmark", "latency", "throughput"),
    "tpu": ("hardware", "tpu_supported_models"),
    "v1": ("v1_guide", "redesign", "architecture"),
    "cutlass": ("scaled_mm", "kernel"),
    "compatibility": ("compatibility_matrix", "matrix"),
    "anchor": ("link", "section", "template"),
    "chat": ("template", "completions", "interface"),
    "multimodal": ("mm_processing", "vision", "vlm"),
    "quantization": ("nvfp4", "fp8", "awq", "gptq", "cutlass"),
    "serve": ("serving", "server", "deploy", "endpoint"),
    "serving": ("serve", "server", "deploy", "endpoint"),
    "api": ("endpoint", "openai", "server", "rest"),
    "endpoint": ("api", "route", "server"),
    "install": ("installation", "setup", "pip", "build"),
    "installation": ("install", "setup", "pip", "build"),
    "deploy": ("deployment", "serve", "docker", "kubernetes"),
    "deployment": ("deploy", "serve", "docker", "container"),
    "docker": ("container", "image", "deployment"),
    "quantize": ("quantization", "awq", "gptq", "precision"),
    "parallel": ("parallelism", "distributed", "tensor", "pipeline"),
    "parallelism": ("parallel", "distributed", "tensor", "pipeline"),
    "distributed": ("parallel", "multi-gpu", "cluster"),
    "lora": ("adapter", "finetune", "peft"),
    "adapter": ("lora", "finetune", "peft"),
    "memory": ("kv", "cache", "block", "allocation"),
    "attention": ("kv", "paged", "flashattention"),
    "config": ("configuration", "settings", "args", "parameters"),
    "configuration": ("config", "settings", "args", "parameters"),
    "engine": ("llm", "runtime", "core"),
    "speculative": ("draft", "spec-decode", "lookahead"),
    "error": ("exception", "failure", "issue", "troubleshooting"),
}


# ---------------------------------------------------------------------------
# Path Normalization Helper
# ---------------------------------------------------------------------------


def normalize_file_path(raw_path: str) -> str:
    """
    Normalize the file path to ensure it is a clean
    relative path starting exactly with data/raw/...
    """
    path_obj = Path(raw_path)
    try:
        if path_obj.is_absolute():
            path_obj = path_obj.relative_to(PROJECT_ROOT)
    except ValueError:
        pass

    path_str = path_obj.as_posix().replace("data/row/", "data/raw/")

    parts = [p for p in path_str.split("/") if p]
    if "data" in parts:
        idx = parts.index("data")
        parts = parts[idx:]
    elif "raw" in parts:
        idx = parts.index("raw")
        parts = ["data"] + parts[idx:]
    else:
        if parts and parts[0] == "vllm-0.10.1":
            parts = ["data", "raw"] + parts
        else:
            parts = ["data", "raw", "vllm-0.10.1"] + parts

    res = "/".join(parts)
    res = res.replace(
        "data/raw/vllm-0.10.1/vllm/docs/", "data/raw/vllm-0.10.1/docs/"
    )
    res = res.replace(
        "data/raw/vllm-0.10.1/vllm/vllm/", "data/raw/vllm-0.10.1/vllm/"
    )
    return res


# ---------------------------------------------------------------------------
# Index Loading
# ---------------------------------------------------------------------------


def load_index(
    index_dir: str = DEFAULT_INDEX_DIR,
) -> tuple[bm25s.BM25, list[MinimalSource]]:
    """
    Load the BM25 index and chunks from the directory produced by
    bm25s.save().
    """
    t0 = time.perf_counter()
    idx_path = Path(index_dir)

    retriever = bm25s.BM25.load(str(idx_path), load_corpus=True)

    indexed_k1, indexed_b = retriever.k1, retriever.b
    if (indexed_k1, indexed_b) != (BM25_K1, BM25_B):
        logger.warning(
            "On-disk index built with k1=%.2f b=%.2f, but "
            "ingestion.py currently defines BM25_K1=%.2f BM25_B=%.2f.",
            indexed_k1, indexed_b, BM25_K1, BM25_B,
        )

    retriever.k1 = BM25_K1
    retriever.b = BM25_B

    sources: list[MinimalSource] = []

    for doc in retriever.corpus:
        doc_dict = dict(doc)
        raw_path = str(doc_dict.get("file_path", ""))
        doc_dict["file_path"] = normalize_file_path(raw_path)
        sources.append(MinimalSource(**doc_dict))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Index loaded in %.2fs — %d chunks, %d terms in vocab",
        elapsed,
        len(sources),
        len(retriever.vocab_dict),
    )
    return retriever, sources


# ---------------------------------------------------------------------------
# Query Expansion via Ollama / Local
# ---------------------------------------------------------------------------


QUERY_EXPANSION_PROMPT = (
    "You are a technical search assistant for the vLLM codebase "
    "(a high-throughput LLM inference engine written in Python).\n"
    "Given the user question below, output EXACTLY 3 short technical "
    "keywords or synonyms (single words or hyphenated compounds) that "
    "would help a BM25 search engine retrieve relevant source code or "
    "documentation.\n"
    "Rules:\n"
    "- Output ONLY the 3 keywords separated by spaces, nothing else.\n"
    "- Do NOT repeat words already in the question.\n"
    "- Prefer code identifiers, class names, or domain-specific terms.\n"
    "- Do NOT output any explanation, numbering, or punctuation.\n"
)


async def expand_query_async(
    query: str,
    model: str = DEFAULT_EXPANSION_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
) -> str:
    try:
        client = ollama.AsyncClient(host=host)
        response = await client.generate(
            model=model,
            system=QUERY_EXPANSION_PROMPT,
            prompt=f"Question: {query}",
            options={
                "temperature": 0.0,
                "num_predict": 32,
                "num_ctx": 512,
            },
        )
        raw = str(response["response"]).strip()
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        keywords = " ".join(
            tok for tok in raw.split()
            if tok.replace("-", "").replace("_", "").isalnum()
        )
        if keywords:
            return f"{query} {keywords}"
    except Exception:
        pass
    return query


def expand_query(
    query: str,
    model: str = DEFAULT_EXPANSION_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
) -> str:
    expanded = _expand_query_locally_cached(query)
    return expanded


# ---------------------------------------------------------------------------
# Single Search and Deduplication
# ---------------------------------------------------------------------------


def _deduplicate_sources(
    sources: list[MinimalSource], max_per_file: int = 3
) -> list[MinimalSource]:
    dedup: list[MinimalSource] = []
    file_counts: dict[str, int] = {}
    for s in sources:
        count = file_counts.get(s.file_path, 0)
        if count < max_per_file:
            dedup.append(s)
            file_counts[s.file_path] = count + 1
    return dedup


def search(
    query: str,
    retriever: bm25s.BM25,
    sources: list[MinimalSource],
    k: int = DEFAULT_K,
    use_expansion: bool = True,
    corpus_documents: list[dict[str, Any]] | None = None,
    dataset_type: str | None = None,
) -> list[MinimalSource]:
    """BM25 search with query expansion and targeted reranking."""
    query = query.strip()
    if not query:
        logger.warning("Empty query — no results.")
        return []

    if use_expansion:
        query = expand_query(query)

    query_tokens = tokenize_query(query)
    if not query_tokens:
        return []

    k_fetch = min(max(k * 4, 30), len(sources))
    if k_fetch == 0:
        return []

    if corpus_documents is None:
        corpus_documents = [s.model_dump() for s in sources]

    results, _scores = retriever.retrieve(
        query_tokens,
        corpus=corpus_documents,
        k=k_fetch,
    )

    top_docs: list[dict[str, Any]] = results[0].tolist()
    candidates = [MinimalSource(**doc) for doc in top_docs]

    # Reranking based on dataset type (docs vs code)
    doc_exts = (".md", ".mdx", ".rst")
    if dataset_type == "docs":
        ranked = [
            s for s in candidates if s.file_path.endswith(doc_exts)
        ] + [
            s for s in candidates if not s.file_path.endswith(doc_exts)
        ]
    elif dataset_type == "code":
        ranked = [
            s for s in candidates if s.file_path.endswith(".py")
        ] + [
            s for s in candidates if not s.file_path.endswith(".py")
        ]
    else:
        ranked = candidates

    dedup_sources = _deduplicate_sources(ranked, max_per_file=3)
    return dedup_sources[:k]


# ---------------------------------------------------------------------------
# Question Loading
# ---------------------------------------------------------------------------


def load_questions(dataset_path: str) -> list[UnansweredQuestion]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: '{path.resolve()}'"
        )
    questions = parse_questions(path)
    logger.info("%d question(s) loaded from %s", len(questions), path)
    return questions


# ---------------------------------------------------------------------------
# Saving Results
# ---------------------------------------------------------------------------


def save_results(
    results: StudentSearchResults,
    output_path: str,
    dataset_path: str | None = None,
) -> None:
    out = Path(output_path)

    is_directory = (
        (out.exists() and out.is_dir())
        or (not out.suffix and not output_path.endswith('.json'))
    )

    if is_directory:
        if dataset_path:
            dataset_file = Path(dataset_path).stem
            filename = f"{dataset_file}.json"
        else:
            filename = "dataset_docs_public.json"
        out = out / filename

    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results.model_dump(by_alias=True), fh, indent=2,
                  ensure_ascii=False)

    logger.info("Saved student_search_results to %s", out)


def _search_dataset_to_path(
    dataset_path: str,
    output_path: str,
    retriever: bm25s.BM25,
    sources: list[MinimalSource],
    k: int,
) -> None:
    questions = load_questions(dataset_path)
    if not questions:
        logger.error("No questions loaded from %s", dataset_path)
        return

    results = run_search_dataset(
        questions, retriever, sources, k, dataset_path=dataset_path
    )
    save_results(results, output_path)


# ---------------------------------------------------------------------------
# Batch Search
# ---------------------------------------------------------------------------


def run_search_dataset(
    questions: list[UnansweredQuestion],
    retriever: bm25s.BM25,
    sources: list[MinimalSource],
    k: int = DEFAULT_K,
    dataset_path: str | None = None,
    dataset_type: str | None = None,
) -> StudentSearchResults:
    if dataset_type is None and dataset_path is not None:
        path_lower = dataset_path.lower()
        if "docs" in path_lower:
            dataset_type = "docs"
        elif "code" in path_lower:
            dataset_type = "code"

    corpus_documents = [source.model_dump() for source in sources]

    def _search_one(question: UnansweredQuestion) -> MinimalSearchResults:
        try:
            top_sources = search(
                question.question,
                retriever,
                sources,
                k,
                corpus_documents=corpus_documents,
                dataset_type=dataset_type,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed for question_id=%s: %s",
                question.question_id,
                exc,
            )
            top_sources = []

        return MinimalSearchResults(
            question_id=question.question_id,
            question_str=question.question,
            retrieved_sources=top_sources,
        )

    max_workers = max(1, min(8, (os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        search_results = list(
            tqdm(
                executor.map(_search_one, questions),
                total=len(questions),
                desc="Searching",
                unit="q",
            )
        )

    return StudentSearchResults(search_results=search_results, k=k)


# ---------------------------------------------------------------------------
# Retriever Class
# ---------------------------------------------------------------------------


class Retriever:
    def __init__(
        self,
        retriever: bm25s.BM25,
        sources: list[MinimalSource],
    ) -> None:
        self._retriever = retriever
        self._sources = sources
        self._corpus_documents = [source.model_dump() for source in sources]

    @classmethod
    def from_disk(cls, index_dir: str = DEFAULT_INDEX_DIR) -> "Retriever":
        retriever, sources = load_index(index_dir)
        return cls(retriever, sources)

    def search(
        self, query: str, k: int = DEFAULT_K, dataset_type: str | None = None
    ) -> list[MinimalSource]:
        return search(
            query,
            self._retriever,
            self._sources,
            k,
            corpus_documents=self._corpus_documents,
            dataset_type=dataset_type,
        )

    def search_dataset(
        self,
        questions: list[UnansweredQuestion],
        k: int = DEFAULT_K,
        dataset_path: str | None = None,
        dataset_type: str | None = None,
    ) -> StudentSearchResults:
        return run_search_dataset(
            questions,
            self._retriever,
            self._sources,
            k,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
        )

    @property
    def corpus_size(self) -> int:
        return len(self._sources)

    @property
    def vocab_size(self) -> int:
        return len(self._retriever.vocab_dict)


# ---------------------------------------------------------------------------
# Grid search k1/b on a labeled evaluation set
# ---------------------------------------------------------------------------


@dataclass
class BM25EvalExample:
    query: str
    relevant_chunk_ids: list[str]


def chunk_id_for(source: MinimalSource) -> str:
    return (
        f"{source.file_path}:"
        f"{source.first_character_index}-{source.last_character_index}"
    )


def load_eval_set(path: str) -> list[BM25EvalExample]:
    raw = parse_json_file(path)
    return [
        BM25EvalExample(
            query=item["query"],
            relevant_chunk_ids=list(item["relevant_chunk_ids"]),
        )
        for item in raw
    ]


def recall_at_k(
    retrieved: list[MinimalSource],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    if not relevant_chunk_ids:
        return 0.0
    hit_ids = {chunk_id_for(src) for src in retrieved[:k]}
    return len(hit_ids & relevant_chunk_ids) / len(relevant_chunk_ids)


def _read_chunk_text(source: MinimalSource) -> str | None:
    full_path = PROJECT_ROOT / source.file_path
    try:
        text = full_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    return text[source.first_character_index:source.last_character_index]


def grid_search_k1_b(
    sources: list[MinimalSource],
    eval_set: list[BM25EvalExample],
    k1_values: tuple[float, ...] = (1.2, 1.5, 1.8, 2.0),
    b_values: tuple[float, ...] = (0.3, 0.5, 0.75, 0.9),
    k: int = DEFAULT_K,
    method: str = "lucene",
    use_stemmer: bool = True,
) -> list[dict[str, Any]]:
    logger.info(
        "Reconstructing text of %d chunks from disk…",
        len(sources),
    )
    texts: list[str] = []
    valid_sources: list[MinimalSource] = []
    for src in sources:
        text = _read_chunk_text(src)
        if text:
            texts.append(text)
            valid_sources.append(src)

    skipped = len(sources) - len(valid_sources)
    if skipped:
        logger.warning(
            "%d chunk(s) skipped — source file not found from "
            "%s (paths relative to PROJECT_ROOT).",
            skipped, PROJECT_ROOT,
        )
    if not texts:
        raise RuntimeError(
            "No chunk text could be reconstructed — check "
            f"that source files are accessible from "
            f"{PROJECT_ROOT} (data/raw/...)."
        )

    stemmer = None
    if use_stemmer:
        try:
            import Stemmer as _Stemmer

            stemmer = _Stemmer.Stemmer("english")
        except ImportError:
            logger.warning(
                "PyStemmer not installed — grid search without stemming "
                "(pip install PyStemmer to enable it)."
            )

    corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)
    query_tokens_by_example = [
        bm25s.tokenize(ex.query, stopwords="en", stemmer=stemmer)
        for ex in eval_set
    ]

    k_eff = min(k, len(valid_sources))
    if k_eff == 0:
        raise RuntimeError("Empty corpus after text reconstruction.")

    results: list[dict[str, Any]] = []
    for k1 in k1_values:
        for b in b_values:
            retriever = bm25s.BM25(k1=k1, b=b, method=method)
            retriever.index(corpus_tokens)

            recalls: list[float] = []
            for ex, q_tokens in zip(eval_set, query_tokens_by_example):
                if not ex.relevant_chunk_ids:
                    continue
                top_ids, _ = retriever.retrieve(q_tokens, k=k_eff)
                retrieved_sources = [
                    valid_sources[i] for i in top_ids[0].tolist()
                ]
                recalls.append(
                    recall_at_k(
                        retrieved_sources, set(ex.relevant_chunk_ids), k_eff
                    )
                )

            mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
            results.append({
                "k1": k1,
                "b": b,
                "method": method,
                f"recall@{k}": round(mean_recall, 4),
                "n_queries": len(recalls),
            })
            logger.info(
                "k1=%.2f b=%.2f → recall@%d=%.4f", k1, b, k, mean_recall,
            )

    results.sort(key=lambda r: r[f"recall@{k}"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# CLI (Fire)
# ---------------------------------------------------------------------------


class RetrievalCLI:
    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        retriever = Retriever.from_disk(index_dir)
        logger.info(
            "Index ready — %d chunks, %d terms",
            retriever.corpus_size,
            retriever.vocab_size,
        )

        t0 = time.perf_counter()
        results = retriever.search(query, k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\n🔍 Query: {query!r}")
        print(f"   Top-{k} results  ({elapsed_ms:.1f} ms)\n")

        if not results:
            print("   (no results)")
            return

        for rank, src in enumerate(results, start=1):
            print(
                f"   [{rank:>2}] {src.file_path}"
                f"  [{src.first_character_index}:{src.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str = DEFAULT_SEARCH_DATASET_OUTPUT_PATH,
        k: int = DEFAULT_K,
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        t_start = time.perf_counter()

        retriever = Retriever.from_disk(index_dir)
        questions = load_questions(dataset_path)

        if not questions:
            logger.error("No questions loaded — aborting.")
            return

        results = retriever.search_dataset(
            questions, k, dataset_path=dataset_path
        )
        save_results(results, save_directory, dataset_path)

        total = time.perf_counter() - t_start
        per_q_ms = (total / len(questions)) * 1000

        print(f"Saved student_search_results to {save_directory}")

        print("\n✅ Batch search complete")
        print(f"   Questions processed  : {len(questions)}")
        print(f"   k (results/question) : {k}")
        print(f"   Output file          : {save_directory}")
        print(f"   Total duration       : {total:.2f}s")
        print(f"   Average / question   : {per_q_ms:.1f} ms")

    def search_datasets(
        self,
        k: int = DEFAULT_K,
        index_dir: str = DEFAULT_INDEX_DIR,
        save_directory: str = DEFAULT_SEARCH_OUTPUT_DIR,
        docs_dataset_path: str = DEFAULT_DOCS_DATASET_PATH,
        code_dataset_path: str = DEFAULT_CODE_DATASET_PATH,
    ) -> None:
        t_start = time.perf_counter()
        retriever = Retriever.from_disk(index_dir)

        output_root = Path(save_directory)
        output_root.mkdir(parents=True, exist_ok=True)

        docs_output_path = output_root / "dataset_docs_public.json"
        code_output_path = output_root / "dataset_code_public.json"

        print("\n🔎 Generating docs and code results…")
        _search_dataset_to_path(
            docs_dataset_path,
            str(docs_output_path),
            retriever._retriever,
            retriever._sources,
            k,
        )
        _search_dataset_to_path(
            code_dataset_path,
            str(code_output_path),
            retriever._retriever,
            retriever._sources,
            k,
        )

        elapsed = time.perf_counter() - t_start
        print("\n✅ Results generated")
        print(f"   Docs : {docs_output_path}")
        print(f"   Code : {code_output_path}")
        print(f"   Total duration : {elapsed:.2f}s")

    def tune_bm25(
        self,
        eval_path: str,
        index_dir: str = DEFAULT_INDEX_DIR,
        k: int = DEFAULT_K,
        k1_values: str = "1.2,1.5,1.8,2.0",
        b_values: str = "0.3,0.5,0.75,0.9",
        method: str = "lucene",
    ) -> None:
        _, sources = load_index(index_dir)
        eval_set = load_eval_set(eval_path)
        k1_list = tuple(float(x) for x in k1_values.split(","))
        b_list = tuple(float(x) for x in b_values.split(","))

        print(
            f"\n🔬 Grid search k1/b — {len(eval_set)} labeled queries "
            f"{len(k1_list)}×{len(b_list)} combinations\n"
        )

        results = grid_search_k1_b(
            sources, eval_set, k1_values=k1_list, b_values=b_list,
            k=k, method=method,
        )

        print(f"📊 Top combinations (recall@{k}):\n")
        for row in results[:5]:
            print(
                f"   k1={row['k1']:.2f}  b={row['b']:.2f}  "
                f"recall@{k}={row[f'recall@{k}']:.4f}  "
                f"({row['n_queries']} queries evaluated)"
            )


@lru_cache(maxsize=1024)
def _expand_query_locally_cached(query: str) -> str:
    """Ultra-fast local expansion without network calls."""
    tokenized_query = tokenize_query(query)
    if not tokenized_query:
        return query

    query_tokens = [
        token for token_group in tokenized_query for token in token_group
    ]
    if not query_tokens:
        return query

    # OPTIMIZATION: Force extended terms to cast a wider net!
    active_terms = {
        **LOCAL_QUERY_EXPANSION_TERMS,
        **LOCAL_QUERY_EXPANSION_TERMS_EXTENDED,
    }

    expanded_terms: list[str] = []
    seen = set(query_tokens)
    for token in query_tokens:
        for synonym in active_terms.get(token, ()):
            if synonym not in seen:
                expanded_terms.append(synonym)
                seen.add(synonym)

    if not expanded_terms:
        return query
    return " ".join(query_tokens + expanded_terms)
