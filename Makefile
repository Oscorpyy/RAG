
# Colors
COLOR_RESET = \033[0m
COLOR_CYAN = \033[36m
COLOR_GREEN = \033[32m
COLOR_RED = \033[31m
COLOR_YELLOW = \033[33m
COLOR_MAGENTA = \033[35m

install:
	@printf "$(COLOR_CYAN)Installing dependencies...$(COLOR_RESET)\n"
	@uv sync
	@printf "$(COLOR_GREEN)Installation completed$(COLOR_RESET)\n"

index:
	@printf "$(COLOR_CYAN)Starting indexing $(COLOR_RESET)\n"
	@uv run python -m src index --repo_path=./vllm --max_chunk_size=2000

search:
	@printf "$(COLOR_CYAN)Starting searching $(COLOR_RESET)\n"
	@uv run python -m src search_dataset
	--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
	--k 10
	--save_directory data/output/search_results

answer:
	@printf "$(COLOR_CYAN)Starting answering $(COLOR_RESET)\n"
	@uv run python -m src answer_dataset
	--student_search_results_path data/output/search_results/dataset_docs_public.json
	--save_directory data/output/search_results_and_answer

evaluate:
	@printf "$(COLOR_CYAN)Starting evaluating $(COLOR_RESET)\n"
	@uv run python -m src evaluate_dataset
	--student_answer_path data/output/search_results/dataset_docs_public.json
	--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
	--k 10
	--max_context_length 2000

debug:
	@printf "$(COLOR_YELLOW)========================================================$(COLOR_RESET)\n"
	@printf "$(COLOR_YELLOW)Debug mode enabled (pdb)$(COLOR_RESET)\n"
	@printf "  → $(COLOR_YELLOW)s$(COLOR_RESET) : Step (enter functions)\n"
	@printf "  → $(COLOR_YELLOW)n$(COLOR_RESET) : Next (without entering)\n"
	@printf "  → $(COLOR_YELLOW)c$(COLOR_RESET) : Continue execution\n"
	@printf "  → $(COLOR_YELLOW)l$(COLOR_RESET) : Display current code\n"
	@printf "  → $(COLOR_YELLOW)q$(COLOR_RESET) : Quit debugger\n"
	@printf "$(COLOR_YELLOW)========================================================$(COLOR_RESET)\n"
	@uv run python -m pdb $(SRC_DIR)/main.py

clean:
	@printf "$(COLOR_RED)Cleaning project...$(COLOR_RESET)\n"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.log" -delete
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@rm -rf data/processed
	@rm -rf .venv
	@printf "$(COLOR_GREEN)✓ Cleanup completed$(COLOR_RESET)\n"

lint:
	@printf "$(COLOR_CYAN)Running Flake8...$(COLOR_RESET)\n"
	@uv run python -m flake8 $(SRC_DIR)/ && \
		printf "$(COLOR_GREEN)✓ $(COLOR_CYAN)Flake8$(COLOR_GREEN) [OK]$(COLOR_RESET)\n"

	@printf "$(COLOR_CYAN)Running Mypy...$(COLOR_RESET)\n"
	@uv run python -m mypy $(SRC_DIR)/ --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs && \
		printf "$(COLOR_GREEN)✓ $(COLOR_CYAN)Mypy$(COLOR_GREEN) [OK]$(COLOR_RESET)\n"

	@printf "$(COLOR_GREEN)✓ Lint completed$(COLOR_RESET)\n"

lint-strict:
	@printf "$(COLOR_MAGENTA)⚠ Strict linting$(COLOR_RESET)\n"

	@printf "$(COLOR_CYAN)Running Flake8...$(COLOR_RESET)\n"
	@uv run python -m flake8 $(SRC_DIR)/ && \
		printf "$(COLOR_GREEN)✓ $(COLOR_CYAN)Flake8$(COLOR_GREEN) [OK]$(COLOR_RESET)\n"

	@printf "$(COLOR_CYAN)Running Mypy...$(COLOR_RESET)\n"
	@uv run python -m mypy $(SRC_DIR) --strict && \
		printf "$(COLOR_GREEN)✓ $(COLOR_CYAN)Mypy strict$(COLOR_GREEN) [OK]$(COLOR_RESET)\n"

	@printf "$(COLOR_GREEN)✓ Strict verification completed$(COLOR_RESET)\n"

test:
	@printf "$(COLOR_CYAN)Running tests...$(COLOR_RESET)\n"


.PHONY: all install run debug clean re lint lint-strict test index search answer evaluate

