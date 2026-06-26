from rag.src.retrieval import RetrievalCLI

from .ingestion import IngestionCLI
import fire
import pickle
from typing import Any

def charger_index_pkl(chemin_fichier: str) -> Any:
    """
    Lit et charge le contenu d'un fichier binaire .pkl (Pickle).
    """
    with open(chemin_fichier, "rb") as fh:
        donnees: Any = pickle.load(fh)
    return donnees


def main() -> None:
    """Point d'entrée Fire."""
    fire.Fire(IngestionCLI)
    print("affichage de l'index")
    # with open("index.txt", "w") as fh:
    #     fh.write(str(charger_index_pkl("index.pkl")))
    print(charger_index_pkl("index.pkl"))
    fire.Fire(RetrievalCLI)
    

if __name__ == "__main__":
    main()