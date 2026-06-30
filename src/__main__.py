from .ingestion import IngestionCLI
# from .retrieval import RetrievalCLI
# from .evaluation import EvaluationCLI
import fire


def main() -> None:
    """Point d'entrée Fire."""
    fire.Fire(IngestionCLI)
    # fire.Fire(RetrievalCLI)
    # fire.Fire(EvaluationCLI)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
