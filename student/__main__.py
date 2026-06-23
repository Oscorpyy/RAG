from .ingestion import IngestionCLI
import fire

def main() -> None:
    """Point d'entrée Fire."""
    fire.Fire(IngestionCLI)


if __name__ == "__main__":
    main()