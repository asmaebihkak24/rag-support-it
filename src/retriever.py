import os
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIRECTORY = Path("indexes/chroma_support_it")
COLLECTION_NAME = "support_it"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

# GPU si disponible, sinon CPU (modifiable avec la variable EMBEDDING_DEVICE)
DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

print(f"Chargement du modèle d'embeddings sur {DEVICE}...", flush=True)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

print("Modèle chargé !", flush=True)

# Chargement de l'index Chroma
print("Connexion à l'index Chroma...", flush=True)

vectorstore = Chroma(
    persist_directory=str(PERSIST_DIRECTORY),
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

print("Index Chroma chargé !", flush=True)


def search_documents(question, k=TOP_K):
    """Recherche les k chunks les plus proches.

    Retourne une liste de tuples (document, distance).
    Plus la distance est petite, plus le passage est proche de la question.
    """
    return vectorstore.similarity_search_with_score(question, k=k)


def main():
    questions = [
        "Comment vider le cache DNS sous Windows ?",
        "Comment afficher l'adresse IP de mon ordinateur sous Windows ?",
        "Comment configurer le réseau avec Netplan sur Ubuntu ?",
        "Comment installer OpenVPN ?",
        "Comment vérifier que ma connexion VPN fonctionne ?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print("QUESTION :", question)

        results = search_documents(question, k=TOP_K)

        for i, (doc, score) in enumerate(results, start=1):
            print(f"\nRésultat {i}")
            print("Document :", doc.metadata.get("doc_id"))
            print("Titre :", doc.metadata.get("title"))
            print("Distance :", round(float(score), 4))
            print("Extrait :", doc.page_content[:250])


if __name__ == "__main__":
    main()
