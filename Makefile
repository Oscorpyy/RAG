SRC_DIR = src

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

run:
	@printf "$(COLOR_MAGENTA)========================================$(COLOR_RESET)\n"
	@printf "$(COLOR_MAGENTA)Starting complete pipeline$(COLOR_RESET)\n"
	@printf "$(COLOR_MAGENTA)========================================$(COLOR_RESET)\n"

	@printf "\n$(COLOR_CYAN)▶ Launching indexing...$(COLOR_RESET)\n"
	@$(MAKE) index

	@printf "\n$(COLOR_CYAN)▶ Launching search...$(COLOR_RESET)\n"
	@$(MAKE) search

	@printf "\n$(COLOR_CYAN)▶ Launching answer generation...$(COLOR_RESET)\n"
	@$(MAKE) answer

	@printf "\n$(COLOR_CYAN)▶ Launching evaluation...$(COLOR_RESET)\n"
	@$(MAKE) evaluate

	@printf "\n$(COLOR_GREEN)✓ Complete pipeline finished successfully.$(COLOR_RESET)\n"

index:
	@printf "$(COLOR_CYAN)Starting indexing $(COLOR_RESET)\n"
	@uv run python -m src index --repo_path=./data/raw/vllm-0.10.1/vllm --max_chunk_size=2000

search:
	@printf "$(COLOR_CYAN)Starting searching $(COLOR_RESET)\n"
	@uv run python -m src search_datasets \
		--k 10 \
		--index_dir data/processed \
		--output_dir data/output/search_results \
		--docs_dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
		--code_dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json

answer:
	@printf "$(COLOR_CYAN)Starting answering $(COLOR_RESET)\n"
	@uv run python -m src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer

evaluate:
	@printf "$(COLOR_CYAN)Starting evaluating $(COLOR_RESET)\n"
	@uv run python -m src.evaluation evaluate_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--ground_truth_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
		--dataset_type docs \
		--repo_path . \
		--output_path data/eval_results_docs.json
	@uv run python -m src.evaluation evaluate_dataset \
		--student_search_results_path data/output/search_results/dataset_code_public.json \
		--ground_truth_path datasets_public/public/AnsweredQuestions/dataset_code_public.json \
		--dataset_type code \
		--repo_path . \
		--output_path data/eval_results_code.json
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
	@./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/dataset_docs_public.json datasets_public/public/AnsweredQuestions/dataset_docs_public.json --k 5 --max_context_length 1000 --threshold 0.80 

.PHONY: all install run debug clean re lint lint-strict test index search answer evaluate

