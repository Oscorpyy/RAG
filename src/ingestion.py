"""
ingestion.py — RAG against the machine
Phase d'ingestion : parsing, chunking et indexation BM25 de fichiers
Python/Markdown.

Usage CLI (via Fire) :
    python -m student index --max_chunk_size=2000
    python -m student index --max_chunk_size=2000 \
        --overlap=200 --index_dir=data/processed
"""

from __future__ import annotations

import ast
import logging
import os
import re
import time
from pathlib import Path
from typing import Generator, Iterator, cast
import bm25s
import Stemmer

from .models import MinimalSource

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
DEFAULT_OVERLAP: int = 150
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".py", ".md")
DEFAULT_INDEX_DIR: str = "data/processed"

# BM25 tuning — k1 controls term-frequency saturation, b controls
# document-length normalisation.  For *code search* we keep k1 at the
# default (1.5) so repeated identifiers still boost relevance, and we
# lower b to 0.5 because source files vary wildly in length yet a long
# file is not inherently less relevant than a short one.
BM25_K1: float = 1.5
BM25_B: float = 0.5


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_minimal_source(
    file_path: str,
    start_index: int,
    chunk_content: str,
    total_file_length: int,
) -> MinimalSource:
    """
    Crée une instance de MinimalSource en garantissant que
    last_character_index est calculé strictement comme
    start_index + len(chunk_content) et ne dépasse jamais la longueur
    totale du fichier.
    """
    last_idx = min(start_index + len(chunk_content), total_file_length)
    return MinimalSource(
        file_path=file_path,
        first_character_index=start_index,
        last_character_index=last_idx,
    )


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------


def _split_if_oversized(
    text: str,
    file_path: str,
    start: int,
    max_chunk_size: int,
    overlap: int = 0,
    total_file_length: int = 0,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Si le texte dépasse max_chunk_size, le découpe en tranches qui se
    chevauchent de `overlap` caractères. Sinon, l'émet tel quel.
    """
    if total_file_length <= 0:
        total_file_length = start + len(text)

    if len(text) <= max_chunk_size:
        yield make_minimal_source(
            file_path, start, text, total_file_length
        ), text
        return

    step = max(1, max_chunk_size - overlap)
    pos: int = 0
    text_len = len(text)

    while pos < text_len:
        slice_end = min(pos + max_chunk_size, text_len)
        slice_text = text[pos:slice_end]
        yield make_minimal_source(
            file_path, start + pos, slice_text, total_file_length
        ), slice_text
        if slice_end >= text_len:
            break
        pos += step


def _accumulate_with_overlap(
    blocks: list[tuple[int, int, str]],
    content: str,
    file_path: str,
    max_chunk_size: int,
    overlap: int,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Accumule des blocs en chunks avec chevauchement contrôlé entre
    chunks consécutifs.
    """
    if not blocks:
        return

    file_len = len(content)
    chunk_start: int = blocks[0][0]
    chunk_text: str = ""

    def emit(start: int, end: int) -> Iterator[tuple[MinimalSource, str]]:
        start = max(0, min(start, file_len))
        end = max(start, min(end, file_len))
        text = content[start:end]
        yield from _split_if_oversized(
            text, file_path, start, max_chunk_size, overlap, file_len
        )

    for b_start, b_end, b_text in blocks:
        if not chunk_text:
            chunk_start = b_start

        if len(chunk_text) + len(b_text) <= max_chunk_size:
            chunk_text += b_text
        else:
            if chunk_text:
                end = chunk_start + len(chunk_text)
                yield from emit(chunk_start, end)

            new_start = max(0, b_start - overlap)
            chunk_start = new_start
            chunk_text = content[new_start:b_end]

    if chunk_text:
        end = min(chunk_start + len(chunk_text), file_len)
        yield from emit(chunk_start, end)


def _iter_markdown_chunks(
    content: str,
    file_path: str,
    max_chunk_size: int,
    overlap: int,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Découpe un fichier Markdown en chunks respectant 100% du contenu du fichier
    (y compris lignes vides et espaces) sans aucun trou d'index.
    """
    file_len = len(content)
    if file_len <= max_chunk_size:
        yield make_minimal_source(file_path, 0, content, file_len), content
        return

    blocks: list[tuple[int, int, str]] = []
    block_lines: list[str] = []
    block_pos: int = 0
    current_pos: int = 0

    for line in content.splitlines(keepends=True):
        if not block_lines:
            block_pos = current_pos
        block_lines.append(line)
        current_pos += len(line)

        if line.strip() == "":
            block_text = "".join(block_lines)
            blocks.append((block_pos, block_pos + len(block_text), block_text))
            block_lines = []

    if block_lines:
        block_text = "".join(block_lines)
        blocks.append((block_pos, block_pos + len(block_text), block_text))

    yield from _accumulate_with_overlap(
        blocks, content, file_path, max_chunk_size, overlap
    )


def _iter_python_chunks(
    content: str,
    file_path: str,
    max_chunk_size: int,
    overlap: int,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Découpe un fichier Python en chunks en s'appuyant sur l'AST top-level,
    tout en étendant les blocs pour couvrir 100% des caractères du fichier
    (commentaires, docstrings, lignes vides entre fonctions).
    """
    file_len = len(content)
    if file_len <= max_chunk_size:
        yield make_minimal_source(file_path, 0, content, file_len), content
        return

    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.warning(
            "AST parse failed for %s — falling back to brute-force chunking.",
            file_path,
        )
        yield from _split_if_oversized(
            content, file_path, 0, max_chunk_size, overlap, file_len
        )
        return

    lines: list[str] = content.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset: int = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    line_offsets.append(offset)

    def node_char_range(node: ast.AST) -> tuple[int, int] | None:
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        if start_line is None or end_line is None:
            return None

        # Inclure les lignes des décorateurs pour conserver le contexte
        # des fonctions/classes
        if hasattr(node, "decorator_list") and getattr(node, "decorator_list"):
            dec_lines = [
                d.lineno
                for d in getattr(node, "decorator_list")
                if hasattr(d, "lineno")
            ]
            if dec_lines:
                start_line = min(start_line, min(dec_lines))

        start_char = line_offsets[start_line - 1]
        end_char = line_offsets[end_line]
        return start_char, end_char

    raw_ranges: list[tuple[int, int]] = []
    for node in ast.iter_child_nodes(tree):
        rng = node_char_range(node)
        if rng is not None:
            raw_ranges.append(rng)

    if not raw_ranges:
        yield from _split_if_oversized(
            content, file_path, 0, max_chunk_size, overlap, file_len
        )
        return

    blocks: list[tuple[int, int, str]] = []
    prev_end = 0
    num_nodes = len(raw_ranges)

    for i, (s, e) in enumerate(raw_ranges):
        b_start = prev_end
        if i < num_nodes - 1:
            next_s = raw_ranges[i + 1][0]
            b_end = max(e, next_s)
        else:
            b_end = file_len

        b_end = min(max(b_start, b_end), file_len)
        block_text = content[b_start:b_end]
        if block_text:
            blocks.append((b_start, b_end, block_text))
            prev_end = b_end

    if prev_end < file_len:
        if blocks:
            last_s, _, _ = blocks[-1]
            blocks[-1] = (last_s, file_len, content[last_s:file_len])
        else:
            blocks.append((0, file_len, content))

    yield from _accumulate_with_overlap(
        blocks, content, file_path, max_chunk_size, overlap
    )


# ---------------------------------------------------------------------------
# Collecte de fichiers
# ---------------------------------------------------------------------------


def collect_files(repo_path: str) -> Generator[Path, None, None]:
    """
    Parcourt récursivement repo_path et retourne les fichiers .py et .md.
    """
    root = Path(repo_path)
    if not root.exists():
        raise FileNotFoundError(
            f"Le chemin '{root.resolve()}' n'existe pas."
        )
    if not root.is_dir():
        raise NotADirectoryError(
            f"Le chemin '{root.resolve()}' existe mais n'est pas un répertoire"
        )

    for dirpath, dirnames, filenames in os.walk(root):
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
    overlap: int = DEFAULT_OVERLAP,
) -> list[tuple[MinimalSource, str]]:
    """
    Lit un fichier et retourne la liste de ses chunks avec métadonnées.
    """
    relative_path = filepath.relative_to(repo_root).as_posix()

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    if not content.strip():
        logger.debug("Fichier vide ignoré : %s", relative_path)
        return []

    if filepath.suffix == ".py":
        chunks = list(
            _iter_python_chunks(content, relative_path,
                                max_chunk_size, overlap)
        )
    else:  # .md
        chunks = list(
            _iter_markdown_chunks(content, relative_path,
                                  max_chunk_size, overlap)
        )

    logger.debug("%s → %d chunks", relative_path, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Tokeniseur
# ---------------------------------------------------------------------------

STOP_PATH_WORDS: set[str] = {
    "data", "raw", "vllm", "vllm-0.10.1", "docs", "src", "py", "md",
    "txt", "json", "10", "0"
}


def preprocess_text_for_bm25(text: str, file_path: str = "") -> str:
    """
    Pré-traite le texte et le chemin de fichier pour BM25 en étendant
    les identifiants (snake_case, camelCase, mots avec tirets) et en
    intégrant les mots clés du chemin.
    """
    def expand_identifier(match: re.Match[str]) -> str:
        val = match.group(0)
        sub1 = val.replace("_", " ").replace("-", " ")
        sub2 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", val)
        return f"{val} {sub1} {sub2}"

    exp_text = re.sub(r"\b[a-zA-Z0-9_-]{3,}\b", expand_identifier, text)

    if file_path:
        raw_path_tokens = re.findall(r"[a-zA-Z0-9_-]+", file_path)
        path_words = [
            tok for tok in raw_path_tokens
            if tok.lower() not in STOP_PATH_WORDS
        ]
        if path_words:
            path_str = " ".join(path_words)
            exp_path = re.sub(
                r"\b[a-zA-Z0-9_-]{3,}\b", expand_identifier, path_str
            )
            # Ponderer les termes du chemin en les répétant 3 fois
            return f"{exp_path} {exp_path} {exp_path}\n{exp_text}"

    return exp_text


def _tokenize_corpus(texts: list[str]) -> bm25s.tokenization.Tokenized:
    stemmer = Stemmer.Stemmer("english")
    proc_texts = [preprocess_text_for_bm25(t) for t in texts]
    return bm25s.tokenize(
        proc_texts, stopwords="english", stemmer=stemmer, show_progress=False
    )


def tokenize_query(query: str | list[str]) -> list[list[str]]:
    stemmer = Stemmer.Stemmer("english")
    proc_query: str | list[str]
    if isinstance(query, str):
        proc_query = preprocess_text_for_bm25(query)
    else:
        proc_query = [preprocess_text_for_bm25(q) for q in query]
    return cast(
        list[list[str]],
        bm25s.tokenize(
            proc_query,
            stopwords="english",
            stemmer=stemmer,
            return_ids=False,
            show_progress=False,
        ),
    )


# ---------------------------------------------------------------------------
# Indexation BM25
# ---------------------------------------------------------------------------


def build_and_save_index(
    chunks: list[tuple[MinimalSource, str]],
    index_dir: str,
) -> None:
    sources: list[MinimalSource] = [src for src, _ in chunks]
    texts: list[str] = [
        preprocess_text_for_bm25(text, file_path=src.file_path)
        for src, text in chunks
    ]

    logger.info("Tokenisation de %d chunks pour BM25…", len(texts))
    t0 = time.perf_counter()

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(
        texts, stopwords="english", stemmer=stemmer, show_progress=False
    )

    retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B)
    retriever.index(corpus_tokens, show_progress=False)

    Path(index_dir).mkdir(parents=True, exist_ok=True)

    corpus_payload = [s.model_dump() for s in sources]
    retriever.save(index_dir, corpus=corpus_payload)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Index BM25 construit et sauvegardé en %.2fs → %s (%d documents)",
        elapsed,
        index_dir,
        len(texts),
    )


def load_index(
    index_dir: str,
) -> tuple[bm25s.BM25, list[MinimalSource]]:
    retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    sources = [MinimalSource(**doc) for doc in retriever.corpus]
    logger.info("Index chargé depuis %s (%d chunks)", index_dir, len(sources))
    return retriever, sources


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------


class IngestionCLI:
    def index(
        self,
        repo_path: str = "./data/raw/",
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        overlap: int | None = None,
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        # OPTIMISATION : On fixe l'overlap par défaut à 25% du chunk size.
        # Cela garantit un très grand chevauchement.
        if overlap is None:
            overlap = max_chunk_size // 4

        if max_chunk_size < 10:
            raise ValueError(
                f"max_chunk_size ne peut pas être inférieur à 10 "
                f"(valeur reçue : {max_chunk_size})."
            )
        if overlap < 0:
            raise ValueError(
                f"overlap ne peut pas être négatif (valeur reçue : {overlap})."
            )

        logger.info("=== Démarrage de l'ingestion ===")
        repo_root = Path(repo_path).resolve()
        all_chunks: list[tuple[MinimalSource, str]] = []
        file_count = 0
        chunks_empty = 0

        t_start = time.perf_counter()

        for filepath in collect_files(str(repo_root)):
            try:
                file_chunks = parse_file(
                    filepath, repo_root, max_chunk_size, overlap
                )
                all_chunks += file_chunks
                if not file_chunks:
                    chunks_empty += 1
                file_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Erreur lors du parsing de %s : %s",
                               filepath, exc)

        logger.info(
            "Parsing terminé : %d fichiers, %d chunks (%.2fs)",
            file_count,
            len(all_chunks),
            time.perf_counter() - t_start,
        )

        if not all_chunks:
            logger.error("Aucun chunk produit. Vérifiez le chemin du dépôt.")
            return

        build_and_save_index(all_chunks, index_dir)

        total_elapsed = time.perf_counter() - t_start
        logger.info("=== Ingestion complète en %.2fs ===", total_elapsed)

        print("\n✅ Ingestion terminée")
        print(f"   Fichiers traités : {file_count}")
        print(f"   Chunks produits  : {len(all_chunks)}")
        print(f"   Chunks vides     : {chunks_empty}")
        print(f"   Index BM25       : {index_dir}/")
        print(f"   Durée totale     : {total_elapsed:.2f}s")
