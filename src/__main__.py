import fire
from .ingestion import IngestionCLI
from .retrieval import RetrievalCLI
from .generation import GenerationCLI


def main() -> None:
    fire.Fire({
        "index": IngestionCLI().index,
        "search": RetrievalCLI().search,
        "search_dataset": RetrievalCLI().search_dataset,
        "search_datasets": RetrievalCLI().search_datasets,
        "answer_dataset": GenerationCLI().answer_dataset,
        "answer": GenerationCLI().answer,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
