from .ingestion import IngestionCLI
import fire


def main() -> None:
    """Point d'entrée Fire."""
    fire.Fire(IngestionCLI)


if __name__ == "__main__":
    try :
        main()
    except Exception as e:
        print(e)
