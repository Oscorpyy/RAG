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

ollama-pull:
	@printf "$(COLOR_CYAN)Checking/Downloading model qwen3:0.6b...$(COLOR_RESET)\n"
	@OLLAMA_MODELS=~/ollama_models ollama pull qwen3:0.6b
	@printf "$(COLOR_GREEN)✓ Model ready$(COLOR_RESET)\n"

ollama-start:
	@mkdir -p ~/ollama_models
	@printf "$(COLOR_CYAN)Starting Ollama server...$(COLOR_RESET)\n"
	@nohup env OLLAMA_MODELS=~/ollama_models ollama serve > /tmp/ollama.log 2>&1 < /dev/null &
	@printf "$(COLOR_CYAN)Waiting for Ollama to be ready...$(COLOR_RESET)\n"
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do \
		if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then \
			printf "$(COLOR_GREEN)✓ Ollama is ready!$(COLOR_RESET)\n"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	printf "$(COLOR_RED)✗ Ollama did not become ready. Check /tmp/ollama.log$(COLOR_RESET)\n"; \
	exit 1

server: ollama-start ollama-pull
	@printf "$(COLOR_CYAN)Starting Ollama server...$(COLOR_RESET)\n"


run:
	@start=$$(date +%s); \
	printf "$(COLOR_MAGENTA)========================================$(COLOR_RESET)\n"; \
	printf "$(COLOR_MAGENTA)Starting complete pipeline$(COLOR_RESET)\n"; \
	printf "$(COLOR_MAGENTA)========================================$(COLOR_RESET)\n"; \
	printf "\n$(COLOR_CYAN)▶ Launching indexing...$(COLOR_RESET)\n"; \
	$(MAKE) index; \
	printf "\n$(COLOR_CYAN)▶ Launching search...$(COLOR_RESET)\n"; \
	$(MAKE) search; \
	printf "\n$(COLOR_CYAN)▶ Launching evaluation...$(COLOR_RESET)\n"; \
	$(MAKE) evaluate; \
	end=$$(date +%s); \
	elapsed=$$((end - start)); \
	printf "\n$(COLOR_GREEN)✓ Complete pipeline finished successfully.$(COLOR_RESET)\n"; \
	printf "$(COLOR_YELLOW)⏱  Execution time: %02d:%02d$(COLOR_RESET)\n" \
		$$((elapsed / 60)) $$((elapsed % 60))

run_all: ollama-start ollama-pull
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
	@uv run python -m src index --max_chunk_size=2000

search:
	@printf "$(COLOR_CYAN)Starting searching $(COLOR_RESET)\n"
	@echo "$(COLOR_MAGENTA)Searching Docs dataset...$(COLOR_RESET)"
	@uv run python -m src search_dataset \
		--dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
		--save_directory data/output/search_results/dataset_docs_public.json \
		--k 10 \
		--index_dir data/processed
	@echo "$(COLOR_MAGENTA)Searching Code dataset...$(COLOR_RESET)"
	@uv run python -m src search_dataset \
		--dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json \
		--save_directory data/output/search_results/dataset_code_public.json \
		--k 10 \
		--index_dir data/processed

answer:
	@printf "$(COLOR_CYAN)Starting answering $(COLOR_RESET)\n"
	@echo "$(COLOR_MAGENTA)Answering Docs json...$(COLOR_RESET)"
	@uv run python -m src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer
	@echo "$(COLOR_MAGENTA)Answering Code json...$(COLOR_RESET)"
	@uv run python -m src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_code_public.json \
		--save_directory data/output/search_results_and_answer

evaluate:
	@printf "$(COLOR_CYAN)Starting evaluating $(COLOR_RESET)\n"
	@echo "$(COLOR_MAGENTA)Evaluating Docs json...$(COLOR_RESET)"
	@uv run python -m src.evaluation evaluate_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--ground_truth_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
		--dataset_type docs \
		--repo_path . \
		--output_path data/eval_results_docs.json
	@echo "$(COLOR_MAGENTA)Evaluating Code json...$(COLOR_RESET)"
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
	@rm -rf data/output
	@rm -rf data/*.json
	@rm -rf .venv
	@rm -rf evaluations
	@rm -rf \~
	@rm -rf out.txt
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

exam :
	@printf "$(COLOR_CYAN)Running exam...$(COLOR_RESET)\n"
	@./exams/scripts/exam_retrieval.sh  --student-path . --moulinette-path ./moulinette/moulinette

.PHONY: all install run debug clean re lint  index search answer evaluate ollama-start ollama-pull server exam

