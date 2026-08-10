"""
parsing.py — RAG against the machine
Robust JSON parsing module capable of parsing malformed or dirty JSON.
Handles code fences, trailing commas, comments, single quotes, Python
literals, surrounding text, truncated JSON, BOM headers, and schema mismatches.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from .models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)

logger = logging.getLogger(__name__)

# Regex pattern for markdown codeblocks: ```json ... ``` or ``` ... ```
CODEBLOCK_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
)

# Trailing commas before } or ]
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_json_payload(text: str) -> str:
    """
    Extract candidate JSON substring (from outermost { ... } or [ ... ])
    if text contains surrounding commentary or preambles.
    """
    text = text.strip()
    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace == -1 and first_bracket == -1:
        return text

    if (
        first_brace != -1
        and (first_bracket == -1 or first_brace < first_bracket)
    ):
        start = first_brace
        end = text.rfind("}")
    else:
        start = first_bracket
        end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    if start != -1 and end == -1:
        return text[start:]

    return text


def fix_json_syntax(text: str) -> str:
    """
    Cleans dirty JSON formatting:
    - Strips BOM markers
    - Handles line comments (//), block comments (/* */), and # comments
    - Normalizes single-quoted strings and keys to double-quoted JSON strings
    - Escapes unescaped newlines inside string literals
    - Replaces Python boolean/None literals (True, False, None) with JSON
    - Strips trailing commas in objects and arrays
    """
    text = text.lstrip("\ufeff")

    # Extract codeblocks if present
    match = CODEBLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    result: list[str] = []
    i = 0
    length = len(text)
    in_string = False
    string_quote: str | None = None
    escaped = False

    while i < length:
        ch = text[i]

        if escaped:
            result.append(ch)
            escaped = False
            i += 1
            continue

        if ch == "\\" and in_string:
            result.append(ch)
            escaped = True
            i += 1
            continue

        if in_string:
            if ch == string_quote:
                in_string = False
                string_quote = None
                result.append('"')
            elif string_quote == "'" and ch == '"':
                result.append('\\"')
            elif string_quote == "'" and ch == "\n":
                result.append("\\n")
            elif string_quote == '"' and ch == "\n":
                result.append("\\n")
            else:
                result.append(ch)
            i += 1
            continue

        # Outside string literal
        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            result.append('"')
            i += 1
            continue

        # Comments outside strings
        if ch == "/" and i + 1 < length:
            if text[i + 1] == "/":
                end_line = text.find("\n", i + 2)
                if end_line == -1:
                    break
                i = end_line
                continue
            elif text[i + 1] == "*":
                end_block = text.find("*/", i + 2)
                if end_block == -1:
                    break
                i = end_block + 2
                continue

        if ch == "#" and (
            i == 0 or text[i - 1] in ("\n", " ", "\t", ",", "{", "[")
        ):
            end_line = text.find("\n", i + 1)
            if end_line == -1:
                break
            i = end_line
            continue

        result.append(ch)
        i += 1

    cleaned = "".join(result)

    # Convert Python literals outside strings (True, False, None)
    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)
    cleaned = re.sub(r"\bNone\b", "null", cleaned)

    # Remove trailing commas before } or ]
    cleaned = TRAILING_COMMA_RE.sub(r"\1", cleaned)
    cleaned = TRAILING_COMMA_RE.sub(r"\1", cleaned)

    return cleaned


def auto_close_json(text: str) -> str:
    """
    Attempts to auto-close unclosed brackets or braces in truncated JSON.
    """
    text = text.strip()
    stack: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()

    if in_string:
        text += '"'

    while stack:
        top = stack.pop()
        if top == "{":
            text += "}"
        elif top == "[":
            text += "]"

    return text


def parse_json_robust(raw_input: Any) -> Any:
    """
    Robustly parses JSON from str, bytes, dict, or list.
    Handles malformed JSON, code blocks, single quotes, trailing commas,
    comments, Python literals, surrounding text, and truncated JSON.
    """
    if isinstance(raw_input, (dict, list)):
        return raw_input

    if isinstance(raw_input, (bytes, bytearray)):
        try:
            text = raw_input.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = raw_input.decode("utf-8", errors="replace")
            except Exception:
                text = raw_input.decode("latin-1", errors="replace")
    elif isinstance(raw_input, str):
        text = raw_input
    else:
        raise ValueError(
            f"Unsupported input type: {type(raw_input).__name__}"
        )

    text = text.strip()
    if not text:
        raise ValueError("Empty JSON input")

    # Step 1: Standard json.loads
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 2: Extract codeblock if present
    cleaned = text
    match = CODEBLOCK_RE.search(cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3: Extract JSON payload from surrounding text
    payload = extract_json_payload(cleaned)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    # Step 4: Fix dirty syntax (quotes, trailing commas, comments)
    fixed = fix_json_syntax(payload)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Step 5: Auto-close missing brackets/braces
    closed = auto_close_json(fixed)
    try:
        return json.loads(closed)
    except json.JSONDecodeError:
        pass

    # Step 6: Fallback to ast.literal_eval for Python structures
    try:
        res = ast.literal_eval(text)
        if isinstance(res, (dict, list)):
            return res
    except Exception:
        pass

    try:
        res = ast.literal_eval(closed)
        if isinstance(res, (dict, list)):
            return res
    except Exception:
        pass

    excerpt = repr(text[:100])
    raise ValueError(
        f"Could not parse malformed JSON input. Excerpt: {excerpt}"
    )


def parse_json_file(file_path: str | Path) -> Any:
    """
    Reads a file and robustly parses its JSON contents.
    Raises FileNotFoundError if file does not exist.
    Raises ValueError if JSON is invalid and cannot be repaired.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: '{path.resolve()}'")

    try:
        content = path.read_bytes()
    except Exception as e:
        raise ValueError(f"Failed to read file '{path}': {e}") from e

    try:
        return parse_json_robust(content)
    except ValueError as e:
        raise ValueError(f"Failed to parse JSON file '{path}': {e}") from e


# ---------------------------------------------------------------------------
# Model / Schema Specific Parsers
# ---------------------------------------------------------------------------


def parse_minimal_source(data: Any) -> MinimalSource:
    """
    Parses MinimalSource object from dict, allowing field aliases.
    """
    if isinstance(data, MinimalSource):
        return data

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    file_path = (
        data.get("file_path")
        or data.get("filepath")
        or data.get("path")
        or data.get("file")
        or data.get("document_path")
        or ""
    )

    first_idx = (
        data.get("first_character_index")
        if "first_character_index" in data
        else data.get("first_character")
        if "first_character" in data
        else data.get("start_character_index")
        if "start_character_index" in data
        else data.get("start_idx")
        if "start_idx" in data
        else data.get("start_char")
        if "start_char" in data
        else data.get("start")
        if "start" in data
        else 0
    )

    last_idx = (
        data.get("last_character_index")
        if "last_character_index" in data
        else data.get("last_character")
        if "last_character" in data
        else data.get("end_character_index")
        if "end_character_index" in data
        else data.get("end_idx")
        if "end_idx" in data
        else data.get("end_char")
        if "end_char" in data
        else data.get("end")
        if "end" in data
        else 0
    )

    try:
        first_idx_val = int(first_idx)
    except (ValueError, TypeError):
        first_idx_val = 0

    try:
        last_idx_val = int(last_idx)
    except (ValueError, TypeError):
        last_idx_val = 0

    return MinimalSource(
        file_path=str(file_path),
        first_character_index=first_idx_val,
        last_character_index=last_idx_val,
    )


def parse_unanswered_question(data: Any) -> UnansweredQuestion:
    """
    Parses UnansweredQuestion from dict, allowing field aliases.
    """
    if isinstance(data, UnansweredQuestion):
        return data

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    question_id = (
        data.get("question_id")
        or data.get("id")
        or data.get("qid")
        or data.get("uuid")
        or str(uuid.uuid4())
    )

    question = (
        data.get("question")
        or data.get("question_str")
        or data.get("query")
        or data.get("text")
        or data.get("prompt")
        or ""
    )

    return UnansweredQuestion(
        question_id=str(question_id),
        question=str(question),
    )


def parse_answered_question(data: Any) -> AnsweredQuestion:
    """
    Parses AnsweredQuestion from dict, allowing field aliases.
    """
    if isinstance(data, AnsweredQuestion):
        return data

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    unanswered = parse_unanswered_question(data)

    raw_sources = (
        data.get("sources")
        or data.get("retrieved_sources")
        or data.get("ground_truth_sources")
        or []
    )
    if not isinstance(raw_sources, list):
        raw_sources = [raw_sources] if raw_sources else []

    sources: list[MinimalSource] = []
    for s in raw_sources:
        try:
            sources.append(parse_minimal_source(s))
        except Exception as exc:
            logger.warning("Skipping invalid source %r: %s", s, exc)

    answer = (
        data.get("answer")
        or data.get("response")
        or data.get("generated_answer")
        or ""
    )

    return AnsweredQuestion(
        question_id=unanswered.question_id,
        question=unanswered.question,
        sources=sources,
        answer=str(answer),
    )


def _is_file_path(val: Any) -> bool:
    if isinstance(val, Path):
        return True
    if isinstance(val, str):
        if "\n" in val or "{" in val or "[" in val:
            return False
        try:
            p = Path(val)
            return p.exists() and p.is_file()
        except Exception:
            return False
    return False


def parse_questions(raw_data_or_path: Any) -> list[UnansweredQuestion]:
    """
    Parses list of UnansweredQuestion from file path, string, dict, or list.
    Normalizes keys (e.g. question_str -> question) and unwraps containers.
    """
    if _is_file_path(raw_data_or_path):
        raw = parse_json_file(raw_data_or_path)
    else:
        raw = parse_json_robust(raw_data_or_path)

    if isinstance(raw, dict):
        for key in (
            "rag_questions", "questions", "data", "items", "search_results"
        ):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            raw = [raw]

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected list of questions, got {type(raw).__name__}"
        )

    questions: list[UnansweredQuestion] = []
    for i, item in enumerate(raw):
        try:
            questions.append(parse_unanswered_question(item))
        except Exception as exc:
            logger.warning("Question skipped at index %d: %s", i, exc)

    return questions


def parse_student_results(raw_data_or_path: Any) -> StudentSearchResults:
    """
    Parses StudentSearchResults from file path, JSON string, dict, or list.
    Supports alias 'question_str', missing 'k', and robust source parsing.
    """
    if _is_file_path(raw_data_or_path):
        raw = parse_json_file(raw_data_or_path)
    else:
        raw = parse_json_robust(raw_data_or_path)

    if isinstance(raw, StudentSearchResults):
        return raw

    search_results_list: list[MinimalSearchResults] = []
    k_val = 0

    if isinstance(raw, dict):
        k_val = raw.get("k", 0)
        raw_results = (
            raw.get("search_results")
            or raw.get("results")
            or raw.get("data")
            or raw.get("items")
        )
        if raw_results is not None and isinstance(raw_results, list):
            items_list = raw_results
        else:
            items_list = [raw]
    elif isinstance(raw, list):
        items_list = raw
    else:
        raise ValueError(
            f"Expected dict or list, got {type(raw).__name__}"
        )

    for item in items_list:
        if not isinstance(item, dict):
            continue

        q_id = (
            item.get("question_id")
            or item.get("id")
            or item.get("qid")
            or str(uuid.uuid4())
        )
        q_str = (
            item.get("question")
            or item.get("question_str")
            or item.get("query")
            or item.get("text")
            or ""
        )
        raw_sources = (
            item.get("retrieved_sources")
            or item.get("sources")
            or item.get("documents")
            or []
        )
        if not isinstance(raw_sources, list):
            raw_sources = [raw_sources] if raw_sources else []

        parsed_sources: list[MinimalSource] = []
        for s in raw_sources:
            try:
                parsed_sources.append(parse_minimal_source(s))
            except Exception as exc:
                logger.warning("Skipping bad retrieved source %r: %s", s, exc)

        if "answer" in item or "response" in item:
            ans = item.get("answer") or item.get("response") or ""
            search_results_list.append(
                MinimalAnswer(
                    question_id=str(q_id),
                    question_str=str(q_str),
                    retrieved_sources=parsed_sources,
                    answer=str(ans),
                )
            )
        else:
            search_results_list.append(
                MinimalSearchResults(
                    question_id=str(q_id),
                    question=str(q_str),
                    retrieved_sources=parsed_sources,
                )
            )

    try:
        k_val = int(k_val)
    except (ValueError, TypeError):
        k_val = 0

    if k_val <= 0 and search_results_list:
        max_sources = max(
            (len(r.retrieved_sources) for r in search_results_list), default=5
        )
        k_val = max(max_sources, 1)

    return StudentSearchResults(
        search_results=search_results_list,
        k=k_val,
    )


def parse_ground_truth(raw_data_or_path: Any) -> dict[str, AnsweredQuestion]:
    """
    Parses AnsweredQuestion dictionary from file path, string, dict, or list.
    Returns question_id -> AnsweredQuestion.
    """
    if _is_file_path(raw_data_or_path):
        raw = parse_json_file(raw_data_or_path)
    else:
        raw = parse_json_robust(raw_data_or_path)

    if isinstance(raw, dict):
        if "rag_questions" in raw and isinstance(raw["rag_questions"], list):
            raw = raw["rag_questions"]
        elif "questions" in raw and isinstance(raw["questions"], list):
            raw = raw["questions"]
        elif "data" in raw and isinstance(raw["data"], list):
            raw = raw["data"]

    if isinstance(raw, dict):
        raw = [raw]

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected list or dict for ground truth, got {type(raw).__name__}"
        )

    result: dict[str, AnsweredQuestion] = {}
    for item in raw:
        try:
            aq = parse_answered_question(item)
            result[aq.question_id] = aq
        except Exception as exc:
            logger.warning("Skipping ground truth item %r: %s", item, exc)

    return result
