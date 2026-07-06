"""
retrieval.py — RAG against the machine
Recherche BM25 sur l'index produit par la phase d'ingestion.

L'index est chargé via l'API native de bm25s (BM25.load) depuis le
dossier `data/processed` — aucun pickle, aucun fichier fait main.

Usage CLI (via Fire) :
    python -m student search_dataset \\
        --dataset_path=datasets_public/public/UnansweredQuestions/\
        dataset_docs_public.json \
        --output_path=data/results_docs.json \\
        --k=5

    python -m student search_dataset \\
        --dataset_path=datasets_public/public/UnansweredQuestions/\
        dataset_code_public.json \
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

# Chemin absolu du dossier racine du projet (rag/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DEFAULT_INDEX_DIR: str = str(PROJECT_ROOT / "data" / "processed")
DEFAULT_K: int = 5
DEFAULT_SEARCH_OUTPUT_DIR: str = str(PROJECT_ROOT / "data" / "output" / "search_results")
DEFAULT_DOCS_DATASET_PATH: str = "datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
DEFAULT_CODE_DATASET_PATH: str = "datasets_public/public/UnansweredQuestions/dataset_code_public.json"

# ---------------------------------------------------------------------------
# Chargement de l'index (API native bm25s — zéro code de désérialisation)
# ---------------------------------------------------------------------------


def load_index(
    index_dir: str = DEFAULT_INDEX_DIR,
) -> tuple[bm25s.BM25, list[MinimalSource]]:
    """
    Charge l'index BM25 et les chunks depuis le dossier produit par
    bm25s.save().
    """
    # 1. On initialise le chrono et le chemin d'accès
    t0 = time.perf_counter()
    idx_path = Path(index_dir)

    # 2. bm25s.BM25.load() recharge l'index ET le corpus en un seul appel
    retriever = bm25s.BM25.load(str(idx_path), load_corpus=True)

    sources: list[MinimalSource] = []
    
    # --- DÉBUT DE LA NOUVELLE LOGIQUE ---
    for doc in retriever.corpus:
        doc_dict = dict(doc)
        raw_path = str(doc_dict["file_path"])

        # 1. Si le chemin absolu contient "data", on coupe tout ce qu'il y a avant
        if "data" in Path(raw_path).parts:
            parts = Path(raw_path).parts
            data_index = parts.index("data")
            final_path = Path(*parts[data_index:]).as_posix()
            
        # 2. Si le chemin commence directement par "vllm", on préfixe le bon dossier
        else:
            final_path = f"data/raw/vllm-0.10.1/{raw_path}"

        # On écrase avec le format propre en string
        doc_dict["file_path"] = final_path
        sources.append(MinimalSource(**doc_dict))
    # --- FIN DE LA NOUVELLE LOGIQUE ---

    # 3. Calcul du temps écoulé
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
    """
    Retourne les k meilleures sources pour une requête.

    La requête est tokenisée avec bm25s.tokenize (return_ids=False) pour
    être compatible avec le vocabulaire de l'index chargé.

    Args:
        query:     Requête en langage naturel ou extrait de code.
        retriever: Index BM25 chargé via load_index().
        sources:   Liste de MinimalSource alignée avec l'index.
        k:         Nombre de résultats à retourner.

    Returns:
        Liste ordonnée de MinimalSource (meilleur score en premier).
        Liste vide si la requête est vide ou hors-vocabulaire.
    """
    query = query.strip()
    if not query:
        logger.warning("Requête vide — aucun résultat.")
        return []

    # return_ids=False : tokens texte bruts, compatibles avec l'index
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

    # results a la forme [[doc_0, doc_1, ...]] (1 ligne = 1 requête)
    top_docs: list[dict] = results[0].tolist()
    return [MinimalSource(**doc) for doc in top_docs]


# ---------------------------------------------------------------------------
# Chargement des questions
# ---------------------------------------------------------------------------


def load_questions(dataset_path: str) -> list[UnansweredQuestion]:
    """
    Charge un fichier JSON de questions vers une liste de UnansweredQuestion.

    Formats acceptés :
      - Liste bare :  [{question_id, question}, ...]
      - Objet wrappé : {\"questions\": [{question_id, question}, ...]}
      - Objet wrappé : {\"rag_questions\": [{question_id, question}, ...]}

    Args:
        dataset_path: Chemin vers le fichier JSON.

    Returns:
        Liste de UnansweredQuestion validés par Pydantic.

    Raises:
        FileNotFoundError: si le fichier est absent.
        ValueError:        si le format JSON est inconnu.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : '{path.resolve()}'\n"
            "Chemin attendu : "
            "datasets_public/public/"
            "UnansweredQuestions/dataset_docs_public.json"
        )

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # Normalise vers une liste brute
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
    """
    Sérialise un StudentSearchResults en JSON.

    Format produit (conforme au modèle Pydantic) ::

        {
          "search_results": [
            {
              "question_id": "...",
              "question": "...",
              "retrieved_sources": [
                {
                  "file_path": "vllm/engine/...",
                  "first_character_index": 0,
                  "last_character_index": 1842
                },
                ...
              ]
            },
            ...
          ],
          "k": 5
        }

    Args:
        results:     Résultats agrégés à sauvegarder.
        output_path: Chemin de sortie (répertoires créés si nécessaire).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results.model_dump(by_alias=True), fh, indent=2, ensure_ascii=False)

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
    """
    Lance la recherche pour toutes les questions et agrège les résultats.

    Args:
        questions: Questions à traiter.
        retriever: Index BM25 chargé.
        sources:   Sources alignées avec l'index.
        k:         Nombre de résultats par question.

    Returns:
        StudentSearchResults avec un MinimalSearchResults par question.
    """
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
                question=question.question,
                retrieved_sources=top_sources,
            )
        )

    return StudentSearchResults(search_results=search_results, k=k)


# ---------------------------------------------------------------------------
# Classe Retriever (wrapper stateful)
# ---------------------------------------------------------------------------


class Retriever:
    """
    Wrapper stateful qui détient l'index BM25 en mémoire.

    Préférable pour les appels répétés : l'index n'est chargé qu'une seule
    fois, ce qui amortit le temps de cold-start sur les gros corpus.

    Exemple ::

        r = Retriever.from_disk("data/processed")
        sources = r.search("PagedAttention KV cache", k=5)
    """

    def __init__(
        self,
        retriever: bm25s.BM25,
        sources: list[MinimalSource],
    ) -> None:
        self._retriever = retriever
        self._sources = sources

    @classmethod
    def from_disk(cls, index_dir: str = DEFAULT_INDEX_DIR) -> "Retriever":
        """Charge l'index depuis le disque et retourne un Retriever prêt."""
        retriever, sources = load_index(index_dir)
        return cls(retriever, sources)

    def search(self, query: str, k: int = DEFAULT_K) -> list[MinimalSource]:
        """Recherche unitaire."""
        return search(query, self._retriever, self._sources, k)

    def search_dataset(
        self,
        questions: list[UnansweredQuestion],
        k: int = DEFAULT_K,
    ) -> StudentSearchResults:
        """Recherche batch."""
        return run_search_dataset(questions, self._retriever, self._sources, k)

    @property
    def corpus_size(self) -> int:
        """Nombre de chunks indexés."""
        return len(self._sources)

    @property
    def vocab_size(self) -> int:
        """Taille du vocabulaire BM25."""
        return len(self._retriever.vocab_dict)


# ---------------------------------------------------------------------------
# CLI (Fire)
# ---------------------------------------------------------------------------


class RetrievalCLI:
    """
    CLI pour la phase de retrieval de RAG against the machine.

    Commandes
    ---------
    search          Recherche unitaire, affiche les résultats dans le
                    terminal.
    search_dataset  Batch : lit un fichier JSON de questions, écrit les
                    résultats.

    Exemples
    --------
    ::

        # Recherche unitaire
        python -m student search \
            --query="how does vllm schedule requests" \
            --k=5

        # Dataset docs
        python -m student search_dataset \
            --dataset_path=datasets_public/public/UnansweredQuestions/\
            dataset_docs_public.json \
            --output_path=data/results_docs.json \
            --k=5

        # Dataset code
        python -m student search_dataset \
            --dataset_path=datasets_public/public/UnansweredQuestions/\
            dataset_code_public.json \
            --output_path=data/results_code.json \
            --k=10

        # Index dans un autre dossier
        python -m student search_dataset \
            --dataset_path=... \
            --output_path=... \
            --k=5 \
            --index_dir=mon/autre/dossier
    """

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        """
        Recherche unitaire : affiche les k meilleures sources dans le terminal.

        Args:
            query:     La requête de recherche.
            k:         Nombre de résultats (défaut : 5).
            index_dir: Dossier de l'index BM25 (défaut : data/processed,
                       relatif à la racine du paquet).
        """
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
        """
        Batch : traite toutes les questions d'un fichier JSON et écrit les
        résultats.

        Le fichier de sortie contient un StudentSearchResults complet :
        k meilleures sources (retrieved_sources) pour chaque question.

        Args:
            dataset_path: Chemin vers le JSON de questions
                          (UnansweredQuestion).
                          Ex:
                          datasets_public/public/UnansweredQuestions/
                          dataset_docs_public.json
            output_path:  Chemin du JSON de sortie (StudentSearchResults).
                          Ex: data/results_docs.json
            k:            Nombre de résultats par question (défaut : 5).
            index_dir:    Dossier de l'index BM25 (défaut :
                          data/processed, relatif à la racine du paquet).
        """
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

        if len(questions) > 0:
            projected_1000 = per_q_ms * 1000 / 1000  # en secondes
            status = "✅" if projected_1000 < 90 else "⚠️ "
            print(f"   Projeté (1 000 q)    : {projected_1000:.1f}s  {status}")

    def search_datasets(
        self,
        k: int = DEFAULT_K,
        index_dir: str = DEFAULT_INDEX_DIR,
        output_dir: str = DEFAULT_SEARCH_OUTPUT_DIR,
        docs_dataset_path: str = DEFAULT_DOCS_DATASET_PATH,
        code_dataset_path: str = DEFAULT_CODE_DATASET_PATH,
    ) -> None:
        """
        Génère les résultats de recherche pour les jeux de questions docs
        et code.

        Les sorties sont écrites dans :
          - data/output/search_results/dataset_docs_public.json
          - data/output/search_results/dataset_code_public.json
        """
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
