"""
ingestion.py — RAG against the machine
Phase d'ingestion : parsing, chunking et indexation de fichiers Python/Markdown.

Usage CLI (via Fire) :
    python ingestion.py index --repo_path=<path> --max_chunk_size=2000
"""

from __future__ import annotations

import ast
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Generator, Iterator

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

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

DEFAULT_MAX_CHUNK_SIZE: int = 2000
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".py", ".md")
INDEX_OUTPUT_FILE: str = "index.pkl"


# ---------------------------------------------------------------------------
# Modèle de données
# ---------------------------------------------------------------------------


class MinimalSource(BaseModel):
    """Représente un chunk de fichier avec ses index de position."""

    file_path: str = Field(..., description="Chemin relatif du fichier source.")
    first_character_index: int = Field(
        ..., ge=0, description="Index du premier caractère dans le fichier original."
    )
    last_character_index: int = Field(
        ..., ge=0, description="Index du dernier caractère (exclu) dans le fichier original."
    )

    @property
    def length(self) -> int:
        """Longueur du chunk en caractères."""
        return self.last_character_index - self.first_character_index

    def __repr__(self) -> str:
        return (
            f"MinimalSource(file_path={self.file_path!r}, "
            f"range=[{self.first_character_index}:{self.last_character_index}])"
        )


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------


def _iter_markdown_chunks(
    content: str,
    file_path: str,
    max_chunk_size: int,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Découpe un fichier Markdown en chunks respectant les frontières de paragraphes.

    Stratégie :
      1. Découper sur les lignes vides (frontières de paragraphes).
      2. Accumuler les blocs jusqu'à atteindre max_chunk_size.
      3. Si un bloc seul dépasse max_chunk_size, le découper brutalement.

    Yields:
        (MinimalSource, texte_du_chunk)
    """
    paragraphs: list[tuple[int, str]] = []  # (start_index, text)
    current_pos: int = 0

    for line in content.splitlines(keepends=True):
        paragraphs.append((current_pos, line))
        current_pos += len(line)

    # Fusion des lignes en blocs séparés par des lignes vides
    blocks: list[tuple[int, int, str]] = []  # (start, end, text)
    block_lines: list[str] = []
    block_pos: int = 0

    for start_idx, line in paragraphs:
        if line.strip() == "" and block_lines:
            block_text = "".join(block_lines)
            blocks.append((block_pos, block_pos + len(block_text), block_text))
            block_lines = []
            block_pos = start_idx + len(line)
        else:
            if not block_lines:
                block_pos = start_idx
            block_lines.append(line)

    if block_lines:
        block_text = "".join(block_lines)
        blocks.append((block_pos, block_pos + len(block_text), block_text))

    # Accumulation des blocs en chunks
    chunk_start: int = 0
    chunk_text: str = ""

    for b_start, b_end, b_text in blocks:
        if not chunk_text:
            chunk_start = b_start

        if len(chunk_text) + len(b_text) <= max_chunk_size:
            chunk_text += b_text
        else:
            # Émettre le chunk accumulé s'il existe
            if chunk_text:
                yield from _split_if_oversized(
                    chunk_text, file_path, chunk_start, max_chunk_size
                )
            chunk_start = b_start
            chunk_text = b_text

    if chunk_text:
        yield from _split_if_oversized(
            chunk_text, file_path, chunk_start, max_chunk_size
        )


def _split_if_oversized(
    text: str,
    file_path: str,
    start: int,
    max_chunk_size: int,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Si le texte dépasse max_chunk_size, le découpe brutalement par tranches.
    Sinon, l'émet tel quel.
    """
    if len(text) <= max_chunk_size:
        source = MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=start + len(text),
        )
        yield source, text
        return

    pos: int = 0
    while pos < len(text):
        slice_text = text[pos: pos + max_chunk_size]
        source = MinimalSource(
            file_path=file_path,
            first_character_index=start + pos,
            last_character_index=start + pos + len(slice_text),
        )
        yield source, slice_text
        pos += max_chunk_size


def _iter_python_chunks(
    content: str,
    file_path: str,
    max_chunk_size: int,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Découpe un fichier Python en chunks en respectant les frontières de fonctions/classes.

    Stratégie :
      1. Parser l'AST pour identifier les nœuds top-level (fonctions, classes,
         imports, expressions).
      2. Associer chaque nœud à sa plage de caractères via les numéros de ligne.
      3. Accumuler les nœuds tant que max_chunk_size n'est pas dépassé.
      4. Fallback sur le découpage brut si le parsing AST échoue.

    Yields:
        (MinimalSource, texte_du_chunk)
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.warning("AST parse failed for %s — falling back to brute-force chunking.", file_path)
        yield from _split_if_oversized(content, file_path, 0, max_chunk_size)
        return

    lines: list[str] = content.splitlines(keepends=True)
    # line_offsets[i] = index du premier caractère de la ligne i (0-based)
    line_offsets: list[int] = []
    offset: int = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    # Sentinelle pour calculer la fin du dernier nœud
    line_offsets.append(offset)

    def node_char_range(node: ast.AST) -> tuple[int, int] | None:
        """Retourne (start_char, end_char) d'un nœud AST, ou None si introuvable."""
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        if start_line is None or end_line is None:
            return None
        start_char = line_offsets[start_line - 1]
        end_char = line_offsets[end_line]
        return start_char, end_char

    # Récupérer les nœuds top-level avec leurs plages
    top_nodes: list[tuple[int, int, str]] = []  # (start, end, text)
    for node in ast.iter_child_nodes(tree):
        rng = node_char_range(node)
        if rng is None:
            continue
        s, e = rng
        top_nodes.append((s, e, content[s:e]))

    if not top_nodes:
        # Fichier sans nœuds parsables (ex. fichier vide ou commentaires seuls)
        yield from _split_if_oversized(content, file_path, 0, max_chunk_size)
        return

    # Accumuler les nœuds en chunks
    chunk_start: int = top_nodes[0][0]
    chunk_text: str = ""

    for n_start, n_end, n_text in top_nodes:
        if not chunk_text:
            chunk_start = n_start

        if len(chunk_text) + len(n_text) <= max_chunk_size:
            chunk_text += n_text
        else:
            if chunk_text:
                yield from _split_if_oversized(
                    chunk_text, file_path, chunk_start, max_chunk_size
                )
            chunk_start = n_start
            # Un seul nœud peut dépasser max_chunk_size : on le découpe
            chunk_text = n_text

    if chunk_text:
        yield from _split_if_oversized(
            chunk_text, file_path, chunk_start, max_chunk_size
        )


# ---------------------------------------------------------------------------
# Collecte de fichiers
# ---------------------------------------------------------------------------


def collect_files(repo_path: str) -> Generator[Path, None, None]:
    """
    Parcourt récursivement repo_path et retourne les fichiers .py et .md.

    Args:
        repo_path: Chemin racine du dépôt à analyser.

    Yields:
        Chemins de fichiers supportés.
    """
    root = Path(repo_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Le chemin '{repo_path}' n'est pas un répertoire valide.")

    for dirpath, dirnames, filenames in os.walk(root):
        # Exclure les dossiers cachés et les environnements virtuels courants
        _excluded = {"__pycache__", ".git", "node_modules", ".venv", "venv"}
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in _excluded
        ]
        for filename in filenames:
            filepath = Path(dirpath) / filename
            if filepath.suffix in SUPPORTED_EXTENSIONS:
                yield filepath


# ---------------------------------------------------------------------------
# Parsing & chunking principal
# ---------------------------------------------------------------------------


def parse_file(
    filepath: Path,
    repo_root: Path,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[tuple[MinimalSource, str]]:
    """
    Lit un fichier et retourne la liste de ses chunks avec métadonnées.

    Args:
        filepath:      Chemin absolu (ou relatif) vers le fichier.
        repo_root:     Racine du dépôt, pour calculer le chemin relatif.
        max_chunk_size: Taille maximale d'un chunk en caractères.

    Returns:
        Liste de (MinimalSource, texte_du_chunk).
    """
    relative_path = str(filepath.relative_to(repo_root))

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    if not content.strip():
        logger.debug("Fichier vide ignoré : %s", relative_path)
        return []

    if filepath.suffix == ".py":
        chunks = list(_iter_python_chunks(content, relative_path, max_chunk_size))
    else:  # .md
        chunks = list(_iter_markdown_chunks(content, relative_path, max_chunk_size))

    logger.debug("%s → %d chunks", relative_path, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Indexation BM25 / TF-IDF
# ---------------------------------------------------------------------------


def build_index(
    chunks: list[tuple[MinimalSource, str]],
) -> tuple[TfidfVectorizer, object, list[MinimalSource]]:
    """
    Construit un index TF-IDF sur les chunks fournis.

    Choix de TF-IDF vs BM25 :
      - `scikit-learn` TF-IDF est natif, sans dépendance extra, très rapide sur
        un corpus de la taille de vLLM (< 1 minute en pratique).
      - BM25 (rank_bm25) est légèrement plus précis pour la recherche lexicale
        mais nécessite une dépendance externe et est plus lent sur de grands corpus.
      - TF-IDF avec sublinear_tf=True approche le comportement BM25 sur la saturation
        des termes fréquents, ce qui est suffisant pour cette phase.

    Args:
        chunks: Liste de (MinimalSource, texte).

    Returns:
        (vectorizer, matrice_tfidf, liste_de_sources)
    """
    sources: list[MinimalSource] = [src for src, _ in chunks]
    texts: list[str] = [text for _, text in chunks]

    logger.info("Construction de l'index TF-IDF sur %d chunks…", len(texts))
    t0 = time.perf_counter()

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,      # log(1 + tf) → approximation BM25 sur la saturation
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),     # unigrammes + bigrammes
        strip_accents="unicode",
        analyzer="word",
    )
    matrix = vectorizer.fit_transform(texts)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Index TF-IDF construit en %.2fs — shape: %s, vocab: %d termes",
        elapsed,
        matrix.shape,
        len(vectorizer.vocabulary_),
    )
    return vectorizer, matrix, sources


def save_index(
    vectorizer: TfidfVectorizer,
    matrix: object,
    sources: list[MinimalSource],
    output_path: str = INDEX_OUTPUT_FILE,
) -> None:
    """
    Sérialise l'index (vectoriseur + matrice + sources) sur disque.

    Args:
        vectorizer:  L'objet TfidfVectorizer fitté.
        matrix:      Matrice sparse (scipy).
        sources:     Liste des MinimalSource correspondantes.
        output_path: Chemin du fichier de sortie (.pkl).
    """
    payload = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "sources": [s.model_dump() for s in sources],
    }
    with open(output_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Index sauvegardé → %s", output_path)


def load_index(
    index_path: str = INDEX_OUTPUT_FILE,
) -> tuple[TfidfVectorizer, object, list[MinimalSource]]:
    """
    Charge un index préalablement sauvegardé.

    Args:
        index_path: Chemin du fichier .pkl.

    Returns:
        (vectorizer, matrice_tfidf, liste_de_sources)
    """
    with open(index_path, "rb") as fh:
        payload = pickle.load(fh)

    sources = [MinimalSource(**d) for d in payload["sources"]]
    logger.info("Index chargé depuis %s (%d chunks)", index_path, len(sources))
    return payload["vectorizer"], payload["matrix"], sources


# ---------------------------------------------------------------------------
# Interface CLI (Fire)
# ---------------------------------------------------------------------------


class IngestionCLI:
    """
    Commandes CLI pour la phase d'ingestion du projet RAG against the machine.

    Exemples :
        python ingestion.py index --repo_path=./vllm --max_chunk_size=2000
        python ingestion.py index --repo_path=./vllm --max_chunk_size=1500 --output=my_index.pkl
    """

    def index(
        self,
        repo_path: str,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        output: str = INDEX_OUTPUT_FILE,
    ) -> None:
        """
        Parcourt le dépôt, parse les fichiers, et construit l'index TF-IDF.

        Args:
            repo_path:      Chemin racine du dépôt à indexer (ex: ./vllm).
            max_chunk_size: Taille maximale d'un chunk en caractères (défaut: 2000).
            output:         Chemin du fichier d'index de sortie (défaut: index.pkl).
        """
        logger.info("=== Démarrage de l'ingestion ===")
        logger.info(
            "repo_path=%s | max_chunk_size=%d | output=%s",
            repo_path, max_chunk_size, output,
        )

        repo_root = Path(repo_path).resolve()
        all_chunks: list[tuple[MinimalSource, str]] = []
        file_count = 0

        t_start = time.perf_counter()

        for filepath in collect_files(str(repo_root)):
            try:
                file_chunks = parse_file(filepath, repo_root, max_chunk_size)
                all_chunks.extend(file_chunks)
                file_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Erreur lors du parsing de %s : %s", filepath, exc)

        logger.info(
            "Parsing terminé : %d fichiers, %d chunks (%.2fs)",
            file_count,
            len(all_chunks),
            time.perf_counter() - t_start,
        )

        if not all_chunks:
            logger.error("Aucun chunk produit. Vérifiez le chemin du dépôt.")
            return

        vectorizer, matrix, sources = build_index(all_chunks)
        save_index(vectorizer, matrix, sources, output)

        total_elapsed = time.perf_counter() - t_start
        logger.info("=== Ingestion complète en %.2fs ===", total_elapsed)

        # Résumé
        print("\n✅ Ingestion terminée")
        print(f"   Fichiers traités : {file_count}")
        print(f"   Chunks produits  : {len(all_chunks)}")
        print(f"   Index sauvegardé : {output}")
        print(f"   Durée totale     : {total_elapsed:.2f}s")
