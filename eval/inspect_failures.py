
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PERSIST_DIRECTORY = Path("indexes/chroma_support_it")
COLLECTION_NAME = "support_it"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

QUESTIONS = [
    "Comment configurer les serveurs DNS primaire et secondaire sur Ubuntu avec BIND9 ?",
    "Comment configurer le réseau et le DNS sur Ubuntu ?",
    "Comment afficher l'adresse IP de mon ordinateur sous Windows ?",
    "Comment installer OpenVPN puis vérifier la connectivité VPN ?",
]

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory=str(PERSIST_DIRECTORY),
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

for question in QUESTIONS:
    print("\n" + "=" * 90)
    print("QUESTION :", question)
    print("=" * 90)

    results = vectorstore.similarity_search_with_score(
        question,
        k=10
    )

    for rank, (doc, distance) in enumerate(results, start=1):
        print(f"\nRang {rank} | Distance : {distance:.4f}")
        print("Document :", doc.metadata.get("doc_id"))
        print("Titre :", doc.metadata.get("title"))
        print("Section :", doc.metadata.get("section"))
        print("Contenu :")
        print(doc.page_content[:500])
        print("-" * 70)
