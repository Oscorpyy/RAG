"""
Tests for src/parsing.py — Malformed JSON handling & Schema normalisation
"""

import pytest
from pathlib import Path
from src.parsing import (
    parse_json_robust,
    parse_questions,
    parse_student_results,
    parse_ground_truth,
    parse_minimal_source,
    fix_json_syntax,
    auto_close_json,
    parse_json_file,
)
from src.models import UnansweredQuestion, MinimalSource, StudentSearchResults


def test_standard_valid_json():
    data = '{"name": "test", "value": 123}'
    parsed = parse_json_robust(data)
    assert parsed == {"name": "test", "value": 123}


def test_markdown_codeblock():
    data = """
    Here is your JSON response:
    ```json
    {
        "question": "What is vLLM?",
        "id": "q1"
    }
    ```
    Hope this helps!
    """
    parsed = parse_json_robust(data)
    assert parsed["question"] == "What is vLLM?"
    assert parsed["id"] == "q1"


def test_trailing_commas():
    data = '{"items": [1, 2, 3,], "key": "val",}'
    parsed = parse_json_robust(data)
    assert parsed == {"items": [1, 2, 3], "key": "val"}


def test_single_quotes_and_python_literals():
    data = "{'is_active': True, 'data': None, 'items': ['a', 'b'], 'flag': False}"
    parsed = parse_json_robust(data)
    assert parsed == {"is_active": True, "data": None, "items": ["a", "b"], "flag": False}


def test_comments_in_json():
    data = """
    {
        // This is a line comment
        "question": "How to index?", # Python comment
        /* Block
           comment */
        "k": 10
    }
    """
    parsed = parse_json_robust(data)
    assert parsed["question"] == "How to index?"
    assert parsed["k"] == 10


def test_truncated_json_auto_close():
    data = '{"question": "What is RAG?", "sources": [{"file_path": "doc.md", "start_char": 0'
    parsed = parse_json_robust(data)
    assert parsed["question"] == "What is RAG?"
    assert len(parsed["sources"]) == 1


def test_parse_minimal_source_aliases():
    raw = {
        "filepath": "src/main.py",
        "start": "100",
        "end": "250",
    }
    source = parse_minimal_source(raw)
    assert isinstance(source, MinimalSource)
    assert source.file_path == "src/main.py"
    assert source.first_character_index == 100
    assert source.last_character_index == 250


def test_parse_questions_field_normalization():
    data = """
    [
        {"question_str": "What is vLLM?", "qid": "101"},
        {"query": "How to tune attention?", "id": "102"},
        {"text": "Explain PagedAttention"}
    ]
    """
    questions = parse_questions(data)
    assert len(questions) == 3
    assert questions[0].question == "What is vLLM?"
    assert questions[0].question_id == "101"
    assert questions[1].question == "How to tune attention?"
    assert questions[2].question == "Explain PagedAttention"


def test_parse_student_results_robust():
    data = """
    ```json
    {
        "k": "5",
        "search_results": [
            {
                "question_id": "q1",
                "question_str": "Where is config defined?",
                "retrieved_sources": [
                    {
                        "file": "vllm/config.py",
                        "first_character_index": 0,
                        "last_character_index": 500,
                    }
                ],
            }
        ]
    }
    ```
    """
    res = parse_student_results(data)
    assert isinstance(res, StudentSearchResults)
    assert res.k == 5
    assert len(res.search_results) == 1
    assert res.search_results[0].question_id == "q1"
    assert res.search_results[0].question == "Where is config defined?"
    assert res.search_results[0].retrieved_sources[0].file_path == "vllm/config.py"


def test_parse_nonexistent_file_raises_filenotfounderror(tmp_path):
    non_existent = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        parse_json_file(non_existent)
