"""
retrieval.py — RAG against the machine
Recherche BM25 sur l'index produit par la phase d'ingestion.

L'index est chargé via l'API native de bm25s (BM25.load) depuis le
dossier `data/processed` — aucun pickle, aucun fichier fait main.

Usage CLI (via Fire) :
    python -m student search_dataset \\
        --dataset_path=datasets_public/public/UnansweredQuestions/\
dataset_docs_public.json \\
        --output_path=data/results_docs.json \\
        --k=5

    python -m student search_dataset \\
        --dataset_path=datasets_public/public/UnansweredQuestions/\
dataset_code_public.json \\
        --output_path=data/results_code.json \\
        --k=10

    python -m student search \\
        --query="how does vllm schedule requests" \\
        --k=5
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from typing import Any
import bm25s
from tqdm import tqdm

from .models import (
    MinimalSearchResults,
    MinimalSource,
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
# Constantes
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DEFAULT_INDEX_DIR: str = str(
    PROJECT_ROOT / "data" / "processed"
)
DEFAULT_K: int = 5
DEFAULT_SEARCH_OUTPUT_DIR: str = str(
    PROJECT_ROOT / "data" / "output" / "search_results"
)
DEFAULT_DOCS_DATASET_PATH: str = (
    "datasets_public/public/UnansweredQuestions/"
    "dataset_docs_public.json"
)
DEFAULT_CODE_DATASET_PATH: str = (
    "datasets_public/public/UnansweredQuestions/"
    "dataset_code_public.json"
)


# ---------------------------------------------------------------------------
# Path Normalization Helper
# ---------------------------------------------------------------------------


def normalize_file_path(raw_path: str) -> str:
    """
    Normalise le chemin de fichier pour s'assurer qu'il s'agit d'un
    chemin relatif propre commençant exactement par data/raw/...

    Supprime les préfixes de chemin absolu (ex: /home/opernod/...) avec
    .relative_to(PROJECT_ROOT) ou manipulation de chaîne.
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
# Chargement de l'index
# ---------------------------------------------------------------------------


def load_index(
    index_dir: str = DEFAULT_INDEX_DIR,
) -> tuple[bm25s.BM25, list[MinimalSource]]:
    """
    Charge l'index BM25 et les chunks depuis le dossier produit par
    bm25s.save().
    """
    t0 = time.perf_counter()
    idx_path = Path(index_dir)

    retriever = bm25s.BM25.load(str(idx_path), load_corpus=True)

    sources: list[MinimalSource] = []

    for doc in retriever.corpus:
        doc_dict = dict(doc)
        raw_path = str(doc_dict.get("file_path", ""))
        doc_dict["file_path"] = normalize_file_path(raw_path)
        sources.append(MinimalSource(**doc_dict))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Index chargé en %.2fs — %d chunks, %d termes dans le vocab",
        elapsed,
        len(sources),
        len(retriever.vocab_dict),
    )
    return retriever, sources


# ---------------------------------------------------------------------------
# Recherche unitaire
# ---------------------------------------------------------------------------


def search(
    query: str,
    retriever: bm25s.BM25,
    sources: list[MinimalSource],
    k: int = DEFAULT_K,
) -> list[MinimalSource]:
    query = query.strip()
    if not query:
        logger.warning("Requête vide — aucun résultat.")
        return []

    query_tokens = bm25s.tokenize(
        query, stopwords=None, return_ids=False, show_progress=False
    )

    k_eff = min(k, len(sources))
    if k_eff == 0:
        return []

    results, _scores = retriever.retrieve(
        query_tokens,
        corpus=[s.model_dump() for s in sources],
        k=k_eff,
    )

    top_docs: list[dict[str, Any]] = results[0].tolist()
    return [MinimalSource(**doc) for doc in top_docs]


# ---------------------------------------------------------------------------
# Chargement des questions
# ---------------------------------------------------------------------------


def load_questions(dataset_path: str) -> list[UnansweredQuestion]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : '{path.resolve()}'"
        )

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict):
        for key in ("rag_questions", "questions"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raise ValueError(
                f"Structure JSON non reconnue dans '{path}'. "
                "Clés trouvées : " + str(list(raw.keys()))
            )

    if not isinstance(raw, list):
        raise ValueError(
            f"Attendu une liste JSON dans '{path}', "
            f"obtenu : {type(raw).__name__}"
        )

    questions: list[UnansweredQuestion] = []
    for i, item in enumerate(raw):
        try:
            questions.append(UnansweredQuestion(**item))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Question ignorée à l'index %d : %s", i, exc)

    logger.info("%d question(s) chargée(s) depuis %s", len(questions), path)
    return questions


# ---------------------------------------------------------------------------
# Sauvegarde des résultats
# ---------------------------------------------------------------------------


def save_results(results: StudentSearchResults, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results.model_dump(by_alias=True), fh, indent=2,
                  ensure_ascii=False)

    logger.info("Résultats sauvegardés → %s", out)


def _search_dataset_to_path(
    dataset_path: str,
    output_path: str,
    retriever: bm25s.BM25,
    sources: list[MinimalSource],
    k: int,
) -> None:
    questions = load_questions(dataset_path)
    if not questions:
        logger.error("Aucune question chargée depuis %s", dataset_path)
        return

    results = run_search_dataset(questions, retriever, sources, k)
    save_results(results, output_path)


# ---------------------------------------------------------------------------
# Recherche batch
# ---------------------------------------------------------------------------


def run_search_dataset(
    questions: list[UnansweredQuestion],
    retriever: bm25s.BM25,
    sources: list[MinimalSource],
    k: int = DEFAULT_K,
) -> StudentSearchResults:
    search_results: list[MinimalSearchResults] = []

    for question in tqdm(questions, desc="Recherche", unit="q"):
        try:
            top_sources = search(question.question, retriever, sources, k)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Échec pour question_id=%s : %s", question.question_id, exc
            )
            top_sources = []

        search_results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question_str=question.question,
                retrieved_sources=top_sources,
            )
        )

    return StudentSearchResults(search_results=search_results, k=k)


# ---------------------------------------------------------------------------
# Classe Retriever
# ---------------------------------------------------------------------------


class Retriever:
    def __init__(
        self,
        retriever: bm25s.BM25,
        sources: list[MinimalSource],
    ) -> None:
        self._retriever = retriever
        self._sources = sources

    @classmethod
    def from_disk(cls, index_dir: str = DEFAULT_INDEX_DIR) -> "Retriever":
        retriever, sources = load_index(index_dir)
        return cls(retriever, sources)

    def search(self, query: str, k: int = DEFAULT_K) -> list[MinimalSource]:
        return search(query, self._retriever, self._sources, k)

    def search_dataset(
        self,
        questions: list[UnansweredQuestion],
        k: int = DEFAULT_K,
    ) -> StudentSearchResults:
        return run_search_dataset(questions, self._retriever, self._sources, k)

    @property
    def corpus_size(self) -> int:
        return len(self._sources)

    @property
    def vocab_size(self) -> int:
        return len(self._retriever.vocab_dict)


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
            "Index prêt — %d chunks, %d termes",
            retriever.corpus_size,
            retriever.vocab_size,
        )

        t0 = time.perf_counter()
        results = retriever.search(query, k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\n🔍 Requête : {query!r}")
        print(f"   Top-{k} résultats  ({elapsed_ms:.1f} ms)\n")

        if not results:
            print("   (aucun résultat)")
            return

        for rank, src in enumerate(results, start=1):
            print(
                f"   [{rank:>2}] {src.file_path}"
                f"  [{src.first_character_index}:{src.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        output_path: str,
        k: int = DEFAULT_K,
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        t_start = time.perf_counter()

        retriever = Retriever.from_disk(index_dir)
        questions = load_questions(dataset_path)

        if not questions:
            logger.error("Aucune question chargée — abandon.")
            return

        results = retriever.search_dataset(questions, k)
        save_results(results, output_path)

        total = time.perf_counter() - t_start
        per_q_ms = (total / len(questions)) * 1000

        print("\n✅ Recherche batch terminée")
        print(f"   Questions traitées   : {len(questions)}")
        print(f"   k (résultats/question): {k}")
        print(f"   Fichier de sortie    : {output_path}")
        print(f"   Durée totale         : {total:.2f}s")
        print(f"   Moyenne / question   : {per_q_ms:.1f} ms")

    def search_datasets(
        self,
        k: int = DEFAULT_K,
        index_dir: str = DEFAULT_INDEX_DIR,
        output_dir: str = DEFAULT_SEARCH_OUTPUT_DIR,
        docs_dataset_path: str = DEFAULT_DOCS_DATASET_PATH,
        code_dataset_path: str = DEFAULT_CODE_DATASET_PATH,
    ) -> None:
        t_start = time.perf_counter()
        retriever = Retriever.from_disk(index_dir)

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        docs_output_path = output_root / "dataset_docs_public.json"
        code_output_path = output_root / "dataset_code_public.json"

        print("\n🔎 Génération des résultats docs et code…")
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
        print("\n✅ Résultats générés")
        print(f"   Docs : {docs_output_path}")
        print(f"   Code : {code_output_path}")
        print(f"   Durée totale : {elapsed:.2f}s")
