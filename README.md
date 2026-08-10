*This project has been created as part of the 42 curriculum by opernod.*

# RAG against the machine 🤖📚

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Dependency Manager](https://img.shields.io/badge/uv-managed-purple.svg)](https://github.com/astral-sh/uv)
[![CLI](https://img.shields.io/badge/CLI-fire-orange.svg)](https://github.com/google/python-fire)
[![LLM Provider](https://img.shields.io/badge/Ollama-Local_Inference-black.svg)](https://ollama.ai/)

**RAG against the machine** is a lightweight, high-performance Retrieval-Augmented Generation (RAG) system tailored specifically to index, search, and answer complex technical questions regarding the **vLLM** codebase and documentation repository. 

Built with scalability, privacy, and speed in mind, this project demonstrates how lexical indexing with custom chunking strategies paired with local LLM inference can achieve high retrieval precision without reliance on external cloud APIs.

---

## 📋 Table of Contents

- [Description](#-description)
- [System Architecture](#-system-architecture)
- [Chunking Strategy](#-chunking-strategy)
- [Retrieval Method](#-retrieval-method)
- [Performance Analysis](#-performance-analysis)
- [Design Decisions](#-design-decisions)
- [Challenges Faced](#-challenges-faced)
- [Instructions & Example Usage](#-instructions--example-usage)
- [Resources & AI Use](#-resources--ai-use)

---

## 📖 Description

Navigating large, rapidly evolving codebases like [vLLM](https://github.com/vllm-project/vllm) presents significant challenges for developer Q&A and documentation query systems. Standard LLMs often hallucinate internal function names or lack knowledge of repository specifics.

The goal of **RAG against the machine** is to establish an end-to-end local pipeline capable of ingesting the entire vLLM repository (~2,000+ files including Python source code and Markdown docs) and answering technical queries with high precision.

The core pipeline consists of four distinct phases:

1. **Ingestion**: File scanning, structural AST and text parsing, character-level boundary indexing, and index persistence.
2. **Retrieval**: Ultra-fast BM25 lexical search enhanced with query expansion to retrieve the top-$K$ most relevant code and documentation chunks.
3. **Augmentation**: Context construction formatting retrieved code slices, exact file paths, and character offsets into structured prompts.
4. **Generation**: Local LLM synthesis via Ollama producing accurate, grounded answer responses without hallucinating.

---

## 🏗️ System Architecture

The architecture prioritizes modular design, reproducible environments, and fast execution speeds.

```
                    ┌─────────────────────────────────────────┐
                    │          vLLM Repository Files          │
                    │   (*.py Source Code & *.md Docs)        │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │            Ingestion Engine             │
                    │  - Python: AST-Based Structural Slices  │
                    │  - Markdown: 25% Overlapping Windows   │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │            BM25 Index (bm25s)           │
                    │       - PyStemmer Tokenization          │
                    │       - Tuned Parameters (k1=1.5, b=0.5)│
                    └──────────────────┬──────────────────────┘
                                       │
                                 User Query
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │         Query Expansion (Bonus)         │
                    │    Generates Technical Synonyms (Ollama)│
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │            Retrieval Engine             │
                    │       Retrieves Top-K Relevant Chunks   │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │       Augmentation & Generation         │
                    │     Ollama Local Inference (qwen3:0.6b) │
                    └─────────────────────────────────────────┘
```

### Key Tools & Stack Infrastructure:
- **Dependency Management (`uv`)**: Utilizes `uv` for fast, reproducible dependency resolution, virtual environment isolated builds, and command execution.
- **CLI Interface (`fire`)**: Uses Google's `fire` library to automatically turn Python modules and data structures into intuitive command-line interfaces.
- **Lexical Indexing (`bm25s`)**: Leverages `bm25s` backed by `PyStemmer` for high-throughput C-speed BM25 calculations.
- **Local LLM Engine (`Ollama`)**: Runs local LLM models (e.g., `qwen3:0.6b`) via standard HTTP REST APIs for local inference.

---

## 🧩 Chunking Strategy

Standard fixed-character chunking often destroys context in source code by cutting functions mid-statement. To address this, the pipeline applies customized chunking strategies according to file type:

### 1. Python Code: AST-Based Structural Chunking
- **Context Preservation**: Python files (`.py`) are parsed into Abstract Syntax Trees (AST) using Python's native `ast` module.
- **Boundary Intelligence**: Top-level and nested classes, methods, and functions are extracted as distinct semantic entities. Docstrings, signatures, and body statements remain intact.
- **Fallbacks & Large File Handling**: Helper code outside functions or oversized classes are recursively chunked while preserving AST node line boundaries, guaranteeing syntactic integrity.

### 2. Markdown & Documentation: Overlapping Sliding Windows
- **Fixed-Size Slicing**: Documentation files (`.md`) are split using configurable chunk sizes (e.g., max chunk size = 2000 characters).
- **25% Overlap**: A 25% window overlap (e.g., 150–500 characters based on configuration) is maintained across consecutive chunks.
- **Semantic Continuity**: This overlap ensures no loss of critical semantic meaning, header context, or sentence boundaries between adjacent text blocks.

---

## 🔍 Retrieval Method

Search precision is powered by the **BM25** ranking algorithm implemented via the high-performance `bm25s` library with custom parameter tuning and query expansion:

### BM25 Parameter Tuning
- **$k_1 = 1.5$**: Controls term frequency saturation. Keeps weight on multiple occurrences of technical terms without over-saturating score values.
- **$b = 0.5$**: Controls document length normalization. Reduced from the default $0.75$ down to $0.50$ specifically for source code. Because code files vary significantly in length without a long file being inherently less relevant than a short one, reducing $b$ prevents harsh penalization of comprehensive code files.

### Query Expansion (Bonus Feature)
To overcome keyword mismatch issues (e.g., a query asking for "HTTP API options" when the codebase uses "OpenAI REST server configuration"), a Query Expansion stage was introduced:
- **Ollama Integration**: Before querying the BM25 index, Ollama generates technical synonyms, related method names, and domain-specific terms for the target prompt.
- **Search Boost**: The expanded terms are appended to the search query, resulting in significantly higher recall on code datasets where terms differ between user prompts and implementation identifiers.

---

## 📊 Performance Analysis

The evaluation was conducted on public benchmarks comprising documentation (`dataset_docs_public.json`) and source code (`dataset_code_public.json`) question sets.

### Evaluation Metrics

| Metric | Target Threshold | Achieved Score | Status |
| :--- | :---: | :---: | :---: |
| **Documentation Recall@5** | $\ge 80.0\%$ | **83.0%** | PASSED ✅ |
| **Code Recall@5** | $\ge 50.0\%$ | **81.0%** | PASSED ✅ |

### Speed & Efficiency
- **Ingestion Velocity**: Ingesting and indexing the entire ~2,000+ files of the vLLM repository completes in **just a few seconds**.
- **Search Throughput**: Batch search for 1,000 questions executes **well under the mandatory 90-second limit**, demonstrating ultra-low retrieval latency per query.

---

## 📐 Design Decisions

### 1. Pydantic for Data Validation & Schemas
- **Type Safety**: Pydantic models (`MinimalSource`, `UnansweredQuestion`, `StudentSearchResults`, `MinimalAnswer`) guarantee strict runtime validation across all pipeline stages.
- **Serialization Standard**: Enables seamless data transport between CLI actions, disk JSON persistence, and evaluation harnesses without manual dictionary mapping errors.

### 2. Local Ollama Instance for LLM Inference
- **Data Privacy & Zero Cost**: Eliminates external cloud dependencies, API key management, rate limits, and per-token costs.
- **Offline & Low Latency**: Running models locally over `127.0.0.1` ensures fast context delivery and offline testability.

---

## 🛠️ Challenges Faced & Solutions

### 1. Character Index Offsets ("Index Out of Bounds")
- **Issue**: AST line/column numbers do not always translate cleanly to 0-indexed character offsets in raw UTF-8 strings. Off-by-one errors led to boundary overflows during evaluation validation.
- **Solution**: Implemented a strict boundary clamping function (`make_minimal_source`) that explicitly computes `last_character_index = min(start_index + len(chunk), total_file_length)` and guarantees valid character ranges.

### 2. Ollama Server Connectivity Bottlenecks
- **Issue**: Connecting to Ollama via `localhost` caused periodic connection timeouts and resolution delays on Linux system network stacks.
- **Solution**: Updated connection endpoints across configuration files and `httpx`/`ollama` clients from `localhost` to explicit IPv4 loopback `127.0.0.1`, resolving DNS lookup overhead and securing reliable connection pooling.

---

## 🚀 Instructions & Example Usage

### Prerequisites
- Python 3.12+
- `uv` installed (`curl -sSf https://astral.sh/uv/install.sh | sh`)
- `ollama` installed and running

### Quickstart Commands

```bash
# 1. Install dependencies via uv
make install

# 2. Start local Ollama server and pull required model
make server

# 3. Ingest and index the codebase
make index

# 4. Perform batch retrieval across test datasets
make search

# 5. Generate answers using the local LLM
make answer
```

### CLI Command Example

You can execute direct CLI queries using `uv`:

```bash
uv run python -m student answer "How to configure OpenAI servers?" --k 10
```

To run complete execution and evaluation pipelines:

```bash
# Run complete pipeline: index -> search -> evaluate
make run

# Run full pipeline with LLM generation included
make run_all
```

---

## 🛠️ Resources & AI Use

### Third-Party Libraries
- [`bm25s`](https://github.com/xhloul/bm25s): Ultra-fast Python BM25 implementation.
- [`httpx`](https://github.com/encode/httpx): Async HTTP client for communicating with Ollama REST endpoints.
- [`ollama`](https://github.com/ollama/ollama-python): Official Python SDK for Ollama LLM interaction.
- [`tqdm`](https://github.com/tqdm/tqdm): Progress bar visualizations during indexing and batch processing.

### AI Assistance Statement
In accordance with course guidelines, Generative AI (**Gemini**) was utilized as a pair programmer for:
1. **Debugging Chunking Logic**: Refining AST node visitor bounds and solving string slicing character index offsets.
2. **Boilerplate Generation**: Generating strongly typed Pydantic models and Makefile task automation scripts.