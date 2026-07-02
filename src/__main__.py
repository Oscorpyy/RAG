import fire
from .ingestion import IngestionCLI
from .retrieval import RetrievalCLI


def answer(*args, **kwargs):
    print("Answer step skipped for now.")


def main():
    fire.Fire({
        "index": IngestionCLI().index,
        "search": RetrievalCLI().search,
        "search_dataset": RetrievalCLI().search_dataset,
        "search_datasets": RetrievalCLI().search_datasets,
        "answer_dataset": answer,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
