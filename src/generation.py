"""
generation.py — RAG against the machine
Génération de réponses avec Ollama (Version Optimisée & Nettoyée).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
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

DEFAULT_MODEL: str = "qwen:0.3b"
DEFAULT_CONCURRENCY_LIMIT: int = 1
DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"
MAX_SOURCES_FOR_CONTEXT: int = 3
MAX_CHARS_PER_SOURCE: int = 800
DATASET_NUM_CTX: int = 2048

SYSTEM_PROMPT = (
    "Answer the question using ONLY the given context — no outside "
    "knowledge, no guessing. Cite sources like [Source 1], [Source 2]. "
    "Be direct and self-contained. If the context is insufficient, say: "
    "'The provided context does not contain sufficient information to "
    "answer the question.' "
    "Do NOT output <think> tags or reasoning — final answer only."
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
        logger.info(
            "Ollama not reachable at %s — attempting to launch "
            "'ollama serve'...",
            host_url,
        )
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            logger.warning(
                "Could not launch 'ollama serve': %s",
                exc,
            )
            return False

        for _ in range(10):
            time.sleep(0.5)
            try:
                client = ollama.Client(host=host_url)
                client.list()
                return True
            except Exception:
                continue

        logger.warning(
            "Ollama service check failed on %s after attempting to start it.",
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

    client = ollama.Client(host=host_url)
    model_to_use = resolve_model(model, host_url)

    prompt = f"Context:\n[Source 1]\n{context}\n\nQuestion: {question}\nAnswer:"

    try:
        response = client.generate(
            model=model_to_use,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "num_predict": 128,
                "num_ctx": 4096,
                "stop": ["</think>"],
            },
        )
        raw_ans = str(response["response"])
        return clean_llm_response(raw_ans)
    except Exception as e:
        logger.error("Error generating answer for question '%s': %s", question, e)
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

    context_str = ""
    for idx, file_path, content in context_snippets:
        context_str += f"[Source {idx}] (File: {file_path}):\n{content}\n\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_str}Question: {question}"
        }
    ]

    async with semaphore:
        try:
            response = await client.chat(
                model=model,
                messages=messages,
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
    ) -> None:
        ans = answer(question=question, context=context, model=model, host=host)
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
