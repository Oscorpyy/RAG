"""
generation.py — RAG against the machine
Génération de réponses avec Ollama (Version Optimisée & Nettoyée).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import fire
import ollama
from tqdm.asyncio import tqdm as async_tqdm

from .evaluation import extract_segment
from .models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .retrieval import Retriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes et Prompts
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "qwen3:0.6b"
DEFAULT_CONCURRENCY_LIMIT: int = 1
DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MAX_SOURCES_FOR_CONTEXT: int = 3
MAX_CHARS_PER_SOURCE: int = 800
DATASET_NUM_CTX: int = 2048

SYSTEM_PROMPT = (
    "You are a precise technical assistant for the vLLM codebase.\n"
    "\n"
    "## STRICT RULES\n"
    "1. Use ONLY the provided [Source N] context to answer. "
    "NEVER use outside knowledge, prior training data, or assumptions.\n"
    "2. CITE every claim with the exact tag [Source N] inline "
    "(e.g. 'The scheduler uses async iteration [Source 2].'). "
    "A sentence without a citation is FORBIDDEN.\n"
    "3. When MULTIPLE sources support a statement, cite ALL of them "
    "(e.g. [Source 1][Source 3]).\n"
    "4. QUOTE short relevant identifiers or phrases from the source "
    "when they strengthen the answer (e.g. function names, class names, "
    "config keys).\n"
    "5. If the context does NOT contain enough information, respond "
    "EXACTLY with: 'The provided context does not contain sufficient "
    "information to answer the question.' — nothing else.\n"
    "6. Do NOT speculate, do NOT add information not in the sources, "
    "do NOT say 'based on my knowledge'.\n"
    "7. Be concise and direct. Do NOT output <think> tags, reasoning "
    "traces, or preamble — final answer only.\n"
)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def clean_llm_response(text: str) -> str:
    """
    Supprime les blocs de pensée/raisonnement (<think>...</think>)
    générés par certains modèles (DeepSeek-R1, Qwen reasoning, etc.).
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def truncate_content(text: str, max_chars: int = MAX_CHARS_PER_SOURCE) -> str:
    """
    Tronque le contenu d'une source pour garder un prompt court (donc rapide).
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _detect_file_type(file_path: str) -> str:
    """
    Retourne un label lisible ("Python Code" / "Markdown Documentation")
    à partir de l'extension du fichier.
    """
    if file_path.endswith(".py"):
        return "Python Code"
    if file_path.endswith((".md", ".mdx", ".rst")):
        return "Markdown Documentation"
    return "Text"


def format_source_block(
    idx: int,
    file_path: str,
    content: str,
) -> str:
    """
    Formate un snippet de source avec une structure claire pour le LLM :
    - Tag [Source N] bien visible
    - Type de fichier (code vs doc)
    - Contenu dans un bloc fenced approprié
    """
    file_type = _detect_file_type(file_path)
    # Pour le code Python, on utilise un bloc ``` pour que le LLM
    # distingue clairement le code du texte naturel.
    if file_type == "Python Code":
        formatted_content = f"```python\n{content}\n```"
    else:
        formatted_content = content

    return (
        f"--- [Source {idx}] ---\n"
        f"File: {file_path}\n"
        f"Type: {file_type}\n"
        f"{formatted_content}\n"
        f"--- end [Source {idx}] ---\n"
    )


def build_context_from_question(
    question: str,
    index_dir: str = str(PROJECT_ROOT / "data" / "processed"),
    repo_path: str = str(PROJECT_ROOT),
    k: int = MAX_SOURCES_FOR_CONTEXT,
) -> str:
    """
    Retrieve and format source snippets for a single question.
    """
    retriever = Retriever.from_disk(index_dir)
    retrieved_sources = retriever.search(question, k=k)

    context_blocks: list[str] = []
    for idx, src in enumerate(retrieved_sources[:MAX_SOURCES_FOR_CONTEXT], 1):
        try:
            content = extract_segment(
                file_path=src.file_path,
                start_idx=src.first_character_index,
                end_idx=src.last_character_index,
                repo_path=repo_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping source %d ('%s') for question '%s': %s",
                idx,
                src.file_path,
                question,
                exc,
            )
            continue

        if not content:
            continue

        context_blocks.append(
            format_source_block(
                idx,
                src.file_path,
                truncate_content(content.strip()),
            )
        )

    return "\n".join(context_blocks)


# ---------------------------------------------------------------------------
# Ollama Health Check & Model Resolver
# ---------------------------------------------------------------------------


def is_ollama_running(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    """
    Vérifie si le service Ollama est actif et joignable.
    """
    host_url = host or DEFAULT_OLLAMA_HOST
    try:
        client = ollama.Client(host=host_url)
        client.list()
        return True
    except Exception:
        logger.error(
            "Ollama service is not reachable at %s. "
            "Please ensure Ollama is running (e.g., via 'make ollama-start').",
            host_url,
        )
        return False


def resolve_model(
    requested_model: str, host: str = DEFAULT_OLLAMA_HOST
) -> str:
    """
    Retourne requested_model si présent dans Ollama, sinon un fallback.
    """
    try:
        client = ollama.Client(host=host)
        models_resp = client.list()
        models_list = getattr(models_resp, "models", [])
        available_names: list[str] = [
            str(getattr(m, "model", ""))
            for m in models_list
            if hasattr(m, "model")
        ]
        if not available_names:
            return requested_model
        if requested_model in available_names:
            return requested_model
        for name in available_names:
            if requested_model.split(":")[0] in name:
                return str(name)
        return str(available_names[0])
    except Exception:
        return requested_model


# ---------------------------------------------------------------------------
# API / Core Functions
# ---------------------------------------------------------------------------


def answer(
    question: str,
    context: str,
    model: str = DEFAULT_MODEL,
    host: Optional[str] = None,
) -> str:
    """
    Answer a single question using the provided context string.
    """
    host_url = host or DEFAULT_OLLAMA_HOST
    if not is_ollama_running(host_url):
        error_msg = f"Error: Failed to connect to Ollama at {host_url}."
        logger.error(error_msg)
        return error_msg

    if not context.strip():
        return (
            "The provided context does not contain sufficient "
            "information to answer the question."
        )

    client = ollama.Client(host=host_url)
    model_to_use = resolve_model(model, host_url)

    stripped_context = context.strip()
    if stripped_context.startswith("--- [Source "):
        source_block = stripped_context
        source_count = max(
            1, source_block.count("--- [Source ")
        )
    else:
        source_block = format_source_block(1, "provided_context", context)
        source_count = 1

    source_tags = ", ".join(
        f"[Source {idx}]" for idx in range(1, source_count + 1)
    )
    prompt = (
        f"Context:\n{source_block}\n"
        f"Question: {question}\n"
        f"Answer (cite {source_tags} for every claim):"
    )

    try:
        response = client.generate(
            model=model_to_use,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            think=False,
            options={
                "temperature": 0.0,
                "num_predict": 256,
                "num_ctx": 4096,
                "stop": ["</think>"],
            },
        )
        raw_ans = str(response["response"]).strip()
        if not raw_ans:
            logger.warning(
                "LLM returned empty response for question: %s", question
            )
            return "No answer could be generated from the LLM."
        return clean_llm_response(raw_ans)
    except Exception as e:
        logger.error(
            "Error generating answer for question '%s': %s", question, e
        )
        return "Error: Failed to generate answer."


async def answer_question_async(
    client: ollama.AsyncClient,
    model: str,
    question: str,
    retrieved_sources: list[MinimalSource],
    repo_path: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    Asynchronously answer a single question using retrieved sources.
    Optimized context window & token limits for speed.
    """
    # Optimisation : On ne garde que les Top N meilleures sources pour le LLM
    top_sources = retrieved_sources[:MAX_SOURCES_FOR_CONTEXT]

    context_snippets: list[tuple[int, str, str]] = []
    for idx, src in enumerate(top_sources, 1):
        try:
            content = extract_segment(
                file_path=src.file_path,
                start_idx=src.first_character_index,
                end_idx=src.last_character_index,
                repo_path=repo_path,
            )
        except Exception as exc:
            logger.warning(
                "Skipping source %d/%d ('%s') for question '%s': %s",
                idx,
                len(top_sources),
                src.file_path,
                question,
                exc,
            )
            continue
        if content:
            context_snippets.append(
                (idx, src.file_path, truncate_content(content.strip()))
            )

    if not context_snippets:
        return (
            "The provided context does not contain sufficient information "
            "to answer the question."
        )

    # Build structured context — each source is clearly delimited with
    # its file type so the LLM can distinguish code from documentation.
    context_blocks = []
    for idx, file_path, content in context_snippets:
        context_blocks.append(format_source_block(idx, file_path, content))
    context_str = "\n".join(context_blocks)

    # Build available source tags for the citation reminder
    source_tags = ", ".join(
        f"[Source {idx}]" for idx, _, _ in context_snippets
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_str}\n"
                f"Question: {question}\n"
                f"Answer (cite {source_tags} for every claim):"
            ),
        },
    ]

    async with semaphore:
        try:
            response = await client.chat(
                model=model,
                messages=messages,
                think=False,
                options={
                    "temperature": 0.0,
                    "num_predict": 1024,  # Augmenté massivement
                    "num_ctx": DATASET_NUM_CTX,
                },
            )
            raw_ans = str(response["message"]["content"])
            return clean_llm_response(raw_ans)
        except Exception as e:
            logger.error(
                "Error generating answer for question '%s': %s", question, e
            )
            return (
                "Error: Failed to generate answer due to an internal "
                "error or connection failure."
            )


async def answer_dataset_async(
    student_search_results_path: str,
    save_directory: str,
    model: str = DEFAULT_MODEL,
    repo_path: str = ".",
    concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
    host: Optional[str] = None,
) -> None:
    """
    Core asynchronous logic for processing a dataset of search results.
    """
    host_url = host or DEFAULT_OLLAMA_HOST
    logger.info("Verifying Ollama service status on %s...", host_url)

    if not is_ollama_running(host_url):
        logger.error("Failed to connect to Ollama at %s.", host_url)
        raise RuntimeError(f"Ollama service unavailable at {host_url}.")

    model_to_use = resolve_model(model, host_url)
    logger.info("Loading search results from %s", student_search_results_path)

    with open(student_search_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    search_results_obj = StudentSearchResults.model_validate(data)

    client = ollama.AsyncClient(host=host_url)
    semaphore = asyncio.Semaphore(concurrency_limit)

    tasks: list[asyncio.Task[str]] = []
    for item in search_results_obj.search_results:
        task = asyncio.create_task(
            answer_question_async(
                client=client,
                model=model_to_use,
                question=item.question,
                retrieved_sources=item.retrieved_sources,
                repo_path=repo_path,
                semaphore=semaphore,
            )
        )
        tasks.append(task)

    logger.info(
        "Generating answers for %d questions using model '%s'...",
        len(tasks),
        model_to_use,
    )
    start_time = time.perf_counter()

    answers = await async_tqdm.gather(*tasks, desc="Generating answers")

    elapsed = time.perf_counter() - start_time
    logger.info("Completed answer generation in %.2f seconds.", elapsed)

    answered_results: list[MinimalAnswer] = []
    for item, answer_str in zip(search_results_obj.search_results, answers):
        minimal_answer = MinimalAnswer(
            question_id=item.question_id,
            question_str=item.question,
            retrieved_sources=item.retrieved_sources,
            answer=answer_str,
        )
        answered_results.append(minimal_answer)

    output_obj = StudentSearchResultsAndAnswer(
        search_results=answered_results,
        k=search_results_obj.k,
    )

    save_dir_path = Path(save_directory)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    output_file_path = save_dir_path / Path(student_search_results_path).name

    logger.info("Saving generated answers to %s", output_file_path)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(
            output_obj.model_dump(by_alias=True),
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Processing complete.")


# ---------------------------------------------------------------------------
# CLI Wrapper
# ---------------------------------------------------------------------------


class GenerationCLI:
    """
    CLI commands for generating answers using local LLM via Ollama.
    """

    def answer(
        self,
        question: str,
        context: str = "",
        model: str = DEFAULT_MODEL,
        host: Optional[str] = None,
        index_dir: str = str(PROJECT_ROOT / "data" / "processed"),
        repo_path: str = str(PROJECT_ROOT),
    ) -> None:
        if not context.strip():
            context = build_context_from_question(
                question=question,
                index_dir=index_dir,
                repo_path=repo_path,
            )

        ans = answer(
            question=question, context=context, model=model, host=host
        )
        print(ans)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        model: str = DEFAULT_MODEL,
        repo_path: str = ".",
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
        host: Optional[str] = None,
    ) -> None:
        asyncio.run(
            answer_dataset_async(
                student_search_results_path=student_search_results_path,
                save_directory=save_directory,
                model=model,
                repo_path=repo_path,
                concurrency_limit=concurrency_limit,
                host=host,
            )
        )


if __name__ == "__main__":
    fire.Fire(GenerationCLI)
