

import os
import re
import unicodedata
from pathlib import Path
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

PERSIST_DIRECTORY = Path("indexes/chroma_support_it")
COLLECTION_NAME = "support_it"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 5
RRF_K = 60

DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")


# ============================================================
# RESSOURCES : CHARGEMENT DIFFÉRÉ (LAZY LOADING)
# ============================================================

embeddings = None
vectorstore = None
documents = []
bm25_retriever = None


def initialize_resources():
    """
    Initialise le modèle d'embeddings, Chroma et BM25.

    Les ressources sont chargées uniquement au premier appel
    d'une fonction de recherche.
    """

    global embeddings, vectorstore, documents
    global bm25_retriever, DEVICE

    if vectorstore is not None and bm25_retriever is not None:
        return

    import torch
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_community.retrievers import BM25Retriever

    DEVICE = os.getenv(
        "EMBEDDING_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"Chargement du modèle d'embeddings sur {DEVICE}...",
        flush=True
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Modèle chargé !", flush=True)

    # --------------------------------------------------------
    # CONNEXION À CHROMA
    # --------------------------------------------------------

    print("Connexion à l'index Chroma...", flush=True)

    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    print("Index Chroma chargé !", flush=True)

    # --------------------------------------------------------
    # CHARGEMENT DES DOCUMENTS POUR BM25
    # --------------------------------------------------------

    print("Chargement des documents pour BM25...", flush=True)

    data = vectorstore.get(
        include=["documents", "metadatas"]
    )

    documents = [
        Document(
            page_content=content,
            metadata=metadata or {}
        )
        for content, metadata in zip(
            data["documents"],
            data["metadatas"]
        )
    ]

    print(
        f"{len(documents)} chunks chargés pour BM25.",
        flush=True
    )

    # --------------------------------------------------------
    # INITIALISATION DE BM25
    # --------------------------------------------------------

    bm25_retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=tokenize,
        k=10
    )

    print("BM25 chargé !", flush=True)


# ============================================================
# TOKENISATION POUR BM25
# ============================================================

def tokenize(text):
    """
    Normalise et tokenise le texte.

    - Conversion en minuscules.
    - Suppression des accents.
    - Conservation des termes techniques.
    """

    text = text.lower()

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    return re.findall(r"[a-z0-9_./-]+", text)


# ============================================================
# RECHERCHE VECTORIELLE
# ============================================================

def search_documents(question, k=TOP_K):
    """
    Recherche vectorielle avec Chroma.

    Retourne une liste de tuples :
    (document, distance).

    Une distance plus faible indique une meilleure proximité.
    """

    initialize_resources()

    return vectorstore.similarity_search_with_score(
        question,
        k=k
    )


# ============================================================
# EXPANSION DE REQUÊTE
# ============================================================

def expand_query(question):
    """
    Enrichit certaines requêtes techniques pour améliorer
    la récupération des documents BIND9.
    """

    question_lower = question.lower()

    if "bind9" in question_lower or "named.conf.local" in question_lower:

        return (
            "BIND9 Ubuntu zone master slave "
            "primary secondary configuration"
        )

    return question


# ============================================================
# RECHERCHE HYBRIDE : VECTORIELLE + BM25 + RRF
# ============================================================

def search_documents_hybrid(question, k=TOP_K):
    """
    Combine la recherche vectorielle et BM25 avec
    Reciprocal Rank Fusion (RRF).

    Retourne une liste de tuples :
    (document, score_RRF).
    """

    initialize_resources()

    expanded_question = expand_query(question)

    # --------------------------------------------------------
    # RECHERCHE VECTORIELLE
    # --------------------------------------------------------

    vector_results = vectorstore.similarity_search(
        expanded_question,
        k=10
    )

    # --------------------------------------------------------
    # RECHERCHE BM25
    # --------------------------------------------------------

    bm25_results = bm25_retriever.invoke(
        expanded_question
    )

    # --------------------------------------------------------
    # FUSION RRF
    # --------------------------------------------------------

    scores = defaultdict(float)
    documents_by_id = {}

    for result_list in (vector_results, bm25_results):

        for rank, doc in enumerate(result_list, start=1):

            chunk_id = doc.metadata.get("chunk_id")

            if chunk_id is None:
                raise ValueError(
                    "Un document ne possède pas de chunk_id."
                )

            scores[chunk_id] += 1 / (RRF_K + rank)

            documents_by_id[chunk_id] = doc

    # --------------------------------------------------------
    # CLASSEMENT FINAL
    # --------------------------------------------------------

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        (documents_by_id[chunk_id], scores[chunk_id])
        for chunk_id in ranked_ids[:k]
    ]


# ============================================================
# DIAGNOSTIC BM25 POUR BIND9
# ============================================================

def diagnostic_bm25_bind9():

    initialize_resources()

    print("\n" + "=" * 70)
    print("COMPARAISON BM25 - BIND9")

    queries = {
        "Question originale": (
            "Comment configurer les serveurs DNS primaire "
            "et secondaire sur Ubuntu (BIND9) ?"
        ),

        "Reformulation technique 1": (
            "Ubuntu BIND9 named.conf.local "
            "serveur DNS primaire secondaire"
        ),

        "Reformulation technique 2": (
            "BIND9 Ubuntu zone master slave "
            "primary secondary configuration"
        )
    }

    for label, question in queries.items():

        print("\n" + "-" * 70)
        print(label)
        print("Requête :", question)

        results = bm25_retriever.invoke(question)

        doc005_found = False

        for i, doc in enumerate(results, start=1):

            doc_id = doc.metadata.get("doc_id")

            if doc_id == "DOC005":
                doc005_found = True

            print(f"\nRésultat {i}")
            print("Document :", doc_id)
            print("Chunk ID :", doc.metadata.get("chunk_id"))
            print("Extrait :", doc.page_content[:250])

        if doc005_found:
            print("\nDOC005 retrouvé dans le Top 10.")
        else:
            print("\nDOC005 absent du Top 10.")


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    questions = [
        "Comment vider le cache DNS sous Windows ?",

        "Comment configurer le réseau avec Netplan sur Ubuntu ?",

        "Comment configurer les serveurs DNS primaire "
        "et secondaire sur Ubuntu (BIND9) ?",

        "Comment installer OpenVPN ?",
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUESTION :", question)

        # ----------------------------------------------------
        # RECHERCHE VECTORIELLE
        # ----------------------------------------------------

        print("\n--- RECHERCHE VECTORIELLE ---")

        results = search_documents(
            question,
            k=TOP_K
        )

        for i, (doc, score) in enumerate(results, start=1):

            print(f"\nRésultat {i}")
            print("Document :", doc.metadata.get("doc_id"))
            print("Chunk ID :", doc.metadata.get("chunk_id"))
            print("Distance :", round(float(score), 4))
            print("Extrait :", doc.page_content[:250])

        # ----------------------------------------------------
        # RECHERCHE HYBRIDE
        # ----------------------------------------------------

        print("\n--- RECHERCHE HYBRIDE ---")

        results = search_documents_hybrid(
            question,
            k=TOP_K
        )

        for i, (doc, score) in enumerate(results, start=1):

            print(f"\nRésultat {i}")
            print("Document :", doc.metadata.get("doc_id"))
            print("Chunk ID :", doc.metadata.get("chunk_id"))
            print("Score RRF :", round(float(score), 5))
            print("Extrait :", doc.page_content[:250])

    # Diagnostic ciblé après les quatre questions
    diagnostic_bm25_bind9()


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    main()
