"""
generation.py — RAG against the machine
Génération de réponses avec Ollama.
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

DEFAULT_MODEL: str = "qwen:0.5b"
DEFAULT_CONCURRENCY_LIMIT: int = 10

SYSTEM_PROMPT = (
    "You are an expert AI assistant specialized in answering questions "
    "about codebases and technical documentation.\n"
    "Your task is to answer the user's question using ONLY the provided "
    "context snippets.\n\n"
    "You MUST strictly follow these rules:\n"
    "1. FAITHFULNESS: Base your answer ONLY on the provided context "
    "snippets. Do NOT use any pre-existing knowledge or external facts. "
    "If the context does not contain enough information to answer the "
    "question, clearly state: 'The provided context does not contain "
    "sufficient information to answer the question.' "
    "Do NOT hallucinate or extrapolate.\n"
    "2. SOURCE-GROUNDED: You must cite the specific source(s) you used to "
    "construct your answer. "
    "Use citations like [Source 1], [Source 2], etc., corresponding to "
    "the Source index provided in the context. "
    "Every fact or claim in your response must be backed by one or more "
    "citations.\n"
    "3. SELF-CONTAINED: Your answer must be fully self-contained and "
    "understandable on its own without needing to read the original "
    "question. "
    "For example, instead of saying: 'Yes, it is /v1/load_lora_adapter', "
    "say: 'The HTTP endpoint used to dynamically load a LoRA adapter in "
    "vLLM is /v1/load_lora_adapter [Source 1].'\n"
    "4. RELEVANCE: Directly address the user's query and avoid irrelevant "
    "details.\n"
    "5. CONCISENESS: Keep your answer clear and concise (typically 1-3 "
    "sentences)."
)


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

    Args:
        question: The user's query.
        context: Context text string.
        model: Ollama model name to use.
        host: Optional Ollama host URL.

    Returns:
        The generated answer string.
    """
    client = ollama.Client(host=host) if host else ollama.Client()

    prompt = (
        f"Context:\n[Source 1]\n{context}\n---\n\n"
        f"Question: {question}\n\n"
        f"Provide your self-contained, faithful, and source-grounded "
        f"answer below. Remember to cite sources (e.g. [Source 1]):"
    )

    try:
        response = client.generate(
            model=model,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "num_predict": 256,
            },
        )
        return str(response["response"]).strip()
    except Exception as e:
        logger.error(
            "Error generating answer for question '%s': %s", question, e
        )
        return (
            "Error: Failed to generate answer due to an internal error "
            "or connection failure."
        )


async def answer_question_async(
    client: ollama.AsyncClient,
    model: str,
    question: str,
    retrieved_sources: list[MinimalSource],
    repo_path: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    Asynchronously answer a single question using the retrieved sources.

    Args:
        client: The Ollama AsyncClient.
        model: Ollama model name to use.
        question: The user's query.
        retrieved_sources: List of retrieved sources for context.
        repo_path: Root path of the repository to extract segments.
        semaphore: Async semaphore to limit concurrent requests.

    Returns:
        The generated answer string.
    """
    # 1. Extract context snippets from sources
    context_snippets = []
    for idx, src in enumerate(retrieved_sources, 1):
        content = extract_segment(
            file_path=src.file_path,
            start_idx=src.first_character_index,
            end_idx=src.last_character_index,
            repo_path=repo_path,
        )
        if content:
            context_snippets.append((idx, src.file_path, content.strip()))

    if not context_snippets:
        return (
            "The provided context does not contain sufficient information "
            "to answer the question."
        )

    # 2. Format context for prompt
    context_str = ""
    for idx, file_path, content in context_snippets:
        context_str += (
            f"[Source {idx}] (File: {file_path}):\n"
            f"{content}\n"
            f"---\n\n"
        )

    prompt = (
        f"Context:\n{context_str}"
        f"Question: {question}\n\n"
        f"Provide your self-contained, faithful, and source-grounded "
        f"answer below. Remember to cite sources (e.g. [Source 1]):"
    )

    async with semaphore:
        try:
            response = await client.generate(
                model=model,
                system=SYSTEM_PROMPT,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "num_predict": 256,
                    "num_ctx": 8192,
                },
            )
            return str(response["response"]).strip()
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
    logger.info("Loading search results from %s", student_search_results_path)

    # 1. Load and validate the search results
    with open(student_search_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    search_results_obj = StudentSearchResults.model_validate(data)

    # 2. Setup AsyncClient and Semaphore
    client = ollama.AsyncClient(host=host) if host else ollama.AsyncClient()
    semaphore = asyncio.Semaphore(concurrency_limit)

    # 3. Create tasks for all search results
    tasks = []
    for item in search_results_obj.search_results:
        task = answer_question_async(
            client=client,
            model=model,
            question=item.question,
            retrieved_sources=item.retrieved_sources,
            repo_path=repo_path,
            semaphore=semaphore,
        )
        tasks.append(task)

    # 4. Execute concurrently with tqdm progress bar
    logger.info(
        "Generating answers for %d questions using model '%s'...",
        len(tasks),
        model,
    )
    start_time = time.perf_counter()

    answers = await async_tqdm.gather(*tasks, desc="Generating answers")

    elapsed = time.perf_counter() - start_time
    logger.info("Completed answer generation in %.2f seconds.", elapsed)

    # 5. Build the output objects
    answered_results = []
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

    # 6. Save the output
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
        context: str,
        model: str = DEFAULT_MODEL,
        host: Optional[str] = None,
    ) -> None:
        """
        Answer a single question using provided context.
        """
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
        """
        Process a StudentSearchResults JSON file to generate answers and save.
        """
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
