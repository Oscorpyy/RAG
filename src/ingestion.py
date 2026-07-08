"""
ingestion.py — RAG against the machine
Phase d'ingestion : parsing, chunking et indexation BM25 de fichiers
Python/Markdown.

Usage CLI (via Fire) :
    python -m student index --repo_path=./vllm --max_chunk_size=2000
    python -m student index --repo_path=./vllm --max_chunk_size=2000 \\
        --overlap=200 --index_dir=data/processed
"""

from __future__ import annotations

import ast
import logging
import os
import time
from pathlib import Path
from typing import Generator, Iterator, cast
import bm25s

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
DEFAULT_OVERLAP: int = 200
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".py", ".md")
DEFAULT_INDEX_DIR: str = "data/processed"


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------


def _iter_markdown_chunks(
    content: str,
    file_path: str,
    max_chunk_size: int,
    overlap: int,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Découpe un fichier Markdown en chunks respectant les frontières de
    paragraphes, avec chevauchement entre chunks consécutifs.

    Stratégie :
      1. Découper sur les lignes vides (frontières de paragraphes).
      2. Accumuler les blocs jusqu'à max_chunk_size.
      3. Reculer le début du chunk suivant de `overlap` caractères pour
         créer un chevauchement avec le chunk précédent (jamais avant 0,
         jamais hors du fichier courant).
      4. Si un bloc seul dépasse max_chunk_size, le découper brutalement
         (toujours avec le même overlap entre tranches).

    Garanties :
      - Si len(content) <= max_chunk_size, un seul chunk = le fichier entier.
      - Un chunk ne dépasse jamais [0, len(content)] (clamp strict).
      - Un chunk ne contient jamais de texte d'un autre fichier.

    Yields:
        (MinimalSource, texte_du_chunk)
    """
    # Cas trivial : le fichier entier tient dans un seul chunk
    if len(content) <= max_chunk_size:
        yield MinimalSource(
            file_path=file_path,
            first_character_index=0,
            last_character_index=max(0, len(content) - 1),
        ), content
        return

    paragraphs: list[tuple[int, str]] = []
    current_pos: int = 0
    for line in content.splitlines(keepends=True):
        paragraphs.append((current_pos, line))
        current_pos += len(line)

    # Fusion des lignes en blocs séparés par des lignes vides
    blocks: list[tuple[int, int, str]] = []
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

    yield from _accumulate_with_overlap(
        blocks, content, file_path, max_chunk_size, overlap
    )


def _accumulate_with_overlap(
    blocks: list[tuple[int, int, str]],
    content: str,
    file_path: str,
    max_chunk_size: int,
    overlap: int,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Accumule des blocs (paragraphes ou nœuds AST) en chunks avec
    chevauchement contrôlé entre chunks consécutifs.

    Quand l'accumulation dépasse max_chunk_size, le chunk courant est émis,
    puis le chunk suivant reprend `overlap` caractères avant la fin du
    chunk précédent — toujours borné à l'intérieur du fichier courant
    (jamais avant 0).

    Args:
        blocks:         Liste de (start, end, text) triés par position.
        content:        Contenu complet du fichier (pour les relectures
                        lors du calcul de l'overlap).
        file_path:      Chemin relatif du fichier.
        max_chunk_size: Taille maximale d'un chunk.
        overlap:        Nombre de caractères de chevauchement souhaité.

    Yields:
        (MinimalSource, texte_du_chunk)
    """
    if not blocks:
        return

    file_len = len(content)
    chunk_start: int = blocks[0][0]
    chunk_text: str = ""

    def emit(start: int, end: int) -> Iterator[tuple[MinimalSource, str]]:
        """Émet un chunk [start:end], en le découpant si > max_chunk_size."""
        start = max(0, min(start, file_len))
        end = max(start, min(end, file_len))
        text = content[start:end]
        yield from _split_if_oversized(text, file_path, start,
                                       max_chunk_size, overlap)

    for b_start, b_end, b_text in blocks:
        if not chunk_text:
            chunk_start = b_start

        if len(chunk_text) + len(b_text) <= max_chunk_size:
            chunk_text += b_text
        else:
            if chunk_text:
                end = chunk_start + len(chunk_text)
                yield from emit(chunk_start, end)

            # Chevauchement : reculer le début du prochain chunk de `overlap`
            # caractères, sans jamais repasser avant le début du fichier ni
            # avant le début du chunk précédemment émis (clamp local).
            new_start = max(0, b_start - overlap)
            new_start = max(new_start, 0)
            chunk_start = new_start
            chunk_text = content[new_start:b_end]

    if chunk_text:
        end = min(chunk_start + len(chunk_text), file_len)
        yield from emit(chunk_start, end)


def _split_if_oversized(
    text: str,
    file_path: str,
    start: int,
    max_chunk_size: int,
    overlap: int = 0,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Si le texte dépasse max_chunk_size, le découpe en tranches qui se
    chevauchent de `overlap` caractères. Sinon, l'émet tel quel.

    Le clamp est strict : aucune tranche ne dépasse [start, start+len(text)],
    donc jamais au-delà de la fin réelle du contenu fourni.
    """
    if len(text) <= max_chunk_size:
        yield MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=start + len(text),
        ), text
        return

    step = max(1, max_chunk_size - overlap)
    pos: int = 0
    text_len = len(text)

    while pos < text_len:
        slice_end = min(pos + max_chunk_size, text_len)
        slice_text = text[pos:slice_end]
        yield MinimalSource(
            file_path=file_path,
            first_character_index=start + pos,
            last_character_index=start + slice_end - 1,
        ), slice_text
        if slice_end >= text_len:
            break
        pos += step


def _iter_python_chunks(
    content: str,
    file_path: str,
    max_chunk_size: int,
    overlap: int,
) -> Iterator[tuple[MinimalSource, str]]:
    """
    Découpe un fichier Python en chunks en respectant les frontières de
    fonctions/classes, avec chevauchement entre chunks consécutifs.

    Stratégie :
      1. Parser l'AST pour identifier les nœuds top-level.
      2. Associer chaque nœud à sa plage de caractères via les numéros
         de ligne.
      3. Accumuler les nœuds avec le même mécanisme de chevauchement que
         pour le Markdown (voir `_accumulate_with_overlap`).
      4. Fallback sur le découpage brut si le parsing AST échoue.

    Garanties identiques à `_iter_markdown_chunks` : fichier entier si
    plus petit que max_chunk_size, jamais de dépassement hors fichier,
    jamais de mélange entre fichiers.

    Yields:
        (MinimalSource, texte_du_chunk)
    """
    # Cas trivial : le fichier entier tient dans un seul chunk
    if len(content) <= max_chunk_size:
        yield MinimalSource(
            file_path=file_path,
            first_character_index=0,
            last_character_index=max(0, len(content) - 1),
        ), content
        return

    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.warning(
            "AST parse failed for %s — falling back to brute-force chunking.",
            file_path,
        )
        yield from _split_if_oversized(content, file_path, 0,
                                       max_chunk_size, overlap)
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
        start_char = line_offsets[start_line - 1]
        end_char = line_offsets[end_line]
        return start_char, end_char

    top_nodes: list[tuple[int, int, str]] = []
    for node in ast.iter_child_nodes(tree):
        rng = node_char_range(node)
        if rng is None:
            continue
        s, e = rng
        top_nodes.append((s, e, content[s:e]))

    if not top_nodes:
        yield from _split_if_oversized(content, file_path, 0,
                                       max_chunk_size, overlap)
        return

    yield from _accumulate_with_overlap(
        top_nodes, content, file_path, max_chunk_size, overlap
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
    if not root.exists():
        raise FileNotFoundError(
            f"Le chemin '{root.resolve()}' n'existe pas. "
            "As-tu bien cloné le dépôt ? "
            "(git clone https://github.com/vllm-project/vllm.git)"
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

    Args:
        filepath:       Chemin absolu (ou relatif) vers le fichier.
        repo_root:      Racine du dépôt, pour calculer le chemin relatif.
        max_chunk_size: Taille maximale d'un chunk en caractères.
        overlap:        Chevauchement souhaité entre chunks consécutifs
                        du même fichier, en caractères.

    Returns:
        Liste de (MinimalSource, texte_du_chunk).
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
# Tokeniseur (réutilisé pour l'indexation et la recherche)
# ---------------------------------------------------------------------------


def _tokenize_corpus(texts: list[str]) -> bm25s.tokenization.Tokenized:
    """
    Tokenise un corpus pour la construction de l'index (`BM25.index`).

    Utilise `return_ids=True` (comportement par défaut de `bm25s.tokenize`) :
    le résultat est un objet `Tokenized` (ids + vocab) optimisé pour
    l'indexation, PAS pour interroger un index déjà construit.

    Pas de retrait de stopwords : le corpus est majoritairement du code
    source où les mots courts comme « if », « in », « is » sont significatifs.
    """
    return bm25s.tokenize(texts, stopwords=None, show_progress=False)


def tokenize_query(query: str | list[str]) -> list[list[str]]:
    """
    Tokenise une requête pour l'interroger contre un index déjà chargé.

    IMPORTANT : contrairement à `_tokenize_corpus`, on utilise ici
    `return_ids=False`. `bm25s.BM25.retrieve()` accepte des tokens texte
    bruts et les mappe lui-même sur le vocabulaire de l'index chargé ;
    passer des IDs construits sur un vocabulaire différent (celui d'un
    tokenizer recréé à la volée) produirait des résultats incorrects.

    Args:
        query: Une requête (str) ou une liste de requêtes.

    Returns:
        Liste de listes de tokens, une par requête.
    """
    return cast(
        list[list[str]],
        bm25s.tokenize(
            query,
            stopwords=None,
            return_ids=False,
            show_progress=False,
        ),
    )


# ---------------------------------------------------------------------------
# Indexation BM25 — sérialisation 100% gérée par bm25s
# ---------------------------------------------------------------------------


def build_and_save_index(
    chunks: list[tuple[MinimalSource, str]],
    index_dir: str,
) -> None:
    """
    Construit un index BM25 et le sauvegarde sur disque via l'API native
    de `bm25s` (`BM25.save`).

    Important : c'est `bm25s` qui pilote le format des fichiers produits
    (params.index.json, vocab.index.json, corpus.jsonl, ...) — aucun
    formatage JSON n'est écrit à la main ici. Les `MinimalSource` sont
    passées en tant que `corpus` ; `bm25s` les sérialise lui-même dans
    `corpus.jsonl`, une entrée JSON par ligne, dans le même ordre que
    l'index.

    Args:
        chunks:    Liste de (MinimalSource, texte).
        index_dir: Répertoire de sortie (créé si nécessaire).
    """
    sources: list[MinimalSource] = [src for src, _ in chunks]
    texts: list[str] = [text for _, text in chunks]

    logger.info("Tokenisation de %d chunks pour BM25…", len(texts))
    t0 = time.perf_counter()

    corpus_tokens = _tokenize_corpus(texts)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # corpus= : objets sérialisés tels quels par bm25s dans corpus.jsonl
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
    """
    Charge un index BM25 préalablement sauvegardé par `bm25s.BM25.save`.

    Args:
        index_dir: Répertoire contenant les fichiers produits par `.save()`.

    Returns:
        (retriever_bm25s, liste_de_sources)
    """
    retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    sources = [MinimalSource(**doc) for doc in retriever.corpus]
    logger.info("Index chargé depuis %s (%d chunks)", index_dir, len(sources))
    return retriever, sources


# ---------------------------------------------------------------------------
# Interface CLI (Fire)
# ---------------------------------------------------------------------------


class IngestionCLI:
    """
    Commandes CLI pour la phase d'ingestion du projet RAG against the machine.

    Exemples :
        python -m student index --repo_path=./vllm --max_chunk_size=2000
        python -m student index --repo_path=./vllm --max_chunk_size=2000 \\
            --overlap=200 --index_dir=data/processed
    """

    def index(
        self,
        repo_path: str,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        overlap: int | None = None,
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        """
        Parcourt le dépôt, parse les fichiers, construit l'index BM25
        et le sauvegarde via l'API native bm25s.

        Args:
            repo_path:      Chemin racine du dépôt à indexer (ex: ./vllm).
            max_chunk_size: Taille maximale d'un chunk en caractères
                            (défaut: 2000).
            overlap:        Chevauchement en caractères entre chunks
                            consécutifs d'un même fichier (défaut: 200).
            index_dir:      Répertoire de sortie de l'index bm25s
                            (défaut: data/processed).
        """
        if overlap is None:
            overlap = max_chunk_size // 10

        if max_chunk_size < 10:
            raise ValueError(
                f"max_chunk_size ne peut pas être inférieur à 10 "
                f"(valeur reçue : {max_chunk_size})."
            )
        if overlap < 0:
            raise ValueError(
                f"overlap ne peut pas être négatif (valeur reçue : {overlap})."
            )
        if overlap >= max_chunk_size:
            logger.warning(
                "overlap (%d) >= max_chunk_size (%d) — réduit automatiquement "
                "le pas d'avancement, le chunking peut être inefficace.",
                overlap, max_chunk_size,
            )

        logger.info("=== Démarrage de l'ingestion ===")
        logger.info(
            "repo_path=%s | max_chunk_size=%d | overlap=%d | index_dir=%s",
            repo_path, max_chunk_size, overlap, index_dir,
        )

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
        print(f"   Index BM25       : {index_dir}/ (params.index.json, "
              f"vocab.index.json, corpus.jsonl, ...)")
        print(f"   Durée totale     : {total_elapsed:.2f}s")
