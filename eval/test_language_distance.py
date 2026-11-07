import os
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIRECTORY = Path("indexes/chroma_support_it")
COLLECTION_NAME = "support_it"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Même question en français puis en anglais (documents concernés : en anglais)
PAIRS = [
    ("Comment redémarrer le service BIND9 sous Ubuntu ?",
     "How do I restart the BIND9 service on Ubuntu?"),
    ("Quelle est la différence entre un routage dev tun et un pont dev tap pour OpenVPN ?",
     "What is the difference between routing with dev tun and bridging with dev tap in OpenVPN?"),
    ("Comment configurer le réseau avec Netplan sur Ubuntu ?",
     "How do I configure networking with Netplan on Ubuntu?"),
    ("Comment installer OpenVPN ?",
     "How do I install OpenVPN?"),
]

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = Chroma(
    persist_directory=str(PERSIST_DIRECTORY),
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)


def best(question):
    doc, dist = vectorstore.similarity_search_with_score(question, k=1)[0]
    return round(float(dist), 4), doc.metadata.get("doc_id")


print(f"{'FR':>8} {'EN':>8}   documents (FR / EN)")
for fr, en in PAIRS:
    d_fr, doc_fr = best(fr)
    d_en, doc_en = best(en)
    print(f"{d_fr:>8} {d_en:>8}   {doc_fr} / {doc_en}   | {fr[:50]}")