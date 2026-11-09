

import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import mlflow
import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_SET = Path("eval/golden_set.jsonl")
PERSIST_DIRECTORY = Path("indexes/chroma_support_it")
RESULTS_DIRECTORY = Path("eval/results")

OUTPUT_FILE = RESULTS_DIRECTORY / "retrieval_metrics_hybrid.json"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAME = "support_it"

K_VALUES = [3, 5]

CANDIDATE_K = 10
RRF_K = 60

DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# CONFIGURATION MLFLOW
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("rag_support_it_retrieval")


# ============================================================
# CHARGEMENT DU GOLDEN SET
# ============================================================

with open(GOLDEN_SET, "r", encoding="utf-8-sig") as file:
    questions = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

evaluated_questions = [
    q for q in questions
    if q["category"] in ["nominal", "multi_document"]
    and q["relevant_doc_ids"]
]

if not evaluated_questions:
    raise ValueError("Aucune question évaluable dans le Golden Set.")

print("Nombre total de questions :", len(questions))
print("Questions évaluées :", len(evaluated_questions))
print("Périphérique :", DEVICE)


# ============================================================
# CHARGEMENT DES EMBEDDINGS ET DE CHROMA
# ============================================================

print("\nChargement du modèle d'embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

print("Chargement de l'index Chroma...")

vectorstore = Chroma(
    persist_directory=str(PERSIST_DIRECTORY),
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)


# ============================================================
# CHARGEMENT DES CHUNKS POUR BM25
# ============================================================

print("Chargement des chunks pour BM25...")

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

if not documents:
    raise ValueError("L'index Chroma ne contient aucun chunk.")

print("Nombre de chunks :", len(documents))


# ============================================================
# TOKENISATION BM25
# ============================================================

def tokenize(text):
    text = text.lower()

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    return re.findall(r"[a-z0-9_./-]+", text)


# ============================================================
# INITIALISATION BM25
# ============================================================

bm25_retriever = BM25Retriever.from_documents(
    documents,
    preprocess_func=tokenize,
    k=CANDIDATE_K
)

print("BM25 initialisé !")


# ============================================================
# FUSION RRF
# ============================================================

def reciprocal_rank_fusion(vector_results, bm25_results):
    """
    Fusionne les résultats vectoriels et BM25
    avec Reciprocal Rank Fusion.
    """

    scores = defaultdict(float)
    documents_by_id = {}

    for rank, doc in enumerate(vector_results, start=1):
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id is None:
            raise ValueError("Chunk sans chunk_id.")

        scores[chunk_id] += 1 / (RRF_K + rank)
        documents_by_id[chunk_id] = doc

    for rank, doc in enumerate(bm25_results, start=1):
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id is None:
            raise ValueError("Chunk sans chunk_id.")

        scores[chunk_id] += 1 / (RRF_K + rank)
        documents_by_id[chunk_id] = doc

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        (documents_by_id[chunk_id], scores[chunk_id])
        for chunk_id in ranked_ids
    ]


# ============================================================
# ÉVALUATION POUR UNE VALEUR DE K
# ============================================================

def evaluate_k(k):

    hits = 0
    recalls = []

    category_results = defaultdict(list)
    details = []

    for index, item in enumerate(evaluated_questions, start=1):

        question = item["question"]
        relevant_ids = set(item["relevant_doc_ids"])

        vector_results = vectorstore.similarity_search(
            question,
            k=CANDIDATE_K
        )

        bm25_results = bm25_retriever.invoke(question)

        hybrid_results = reciprocal_rank_fusion(
            vector_results,
            bm25_results
        )

        top_results = hybrid_results[:k]

        retrieved_ids = list(dict.fromkeys(
            doc.metadata.get("doc_id")
            for doc, _ in top_results
            if doc.metadata.get("doc_id")
        ))

        retrieved_chunk_ids = [
            doc.metadata.get("chunk_id")
            for doc, _ in top_results
        ]

        intersection = set(retrieved_ids) & relevant_ids

        hit = int(bool(intersection))
        recall = len(intersection) / len(relevant_ids)

        hits += hit
        recalls.append(recall)

        category_results[item["category"]].append(recall)

        details.append({
            "question": question,
            "category": item["category"],
            "expected_doc_ids": sorted(relevant_ids),
            "retrieved_doc_ids": retrieved_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "rrf_scores": [
                round(float(score), 6)
                for _, score in top_results
            ],
            "hit": hit,
            "recall": recall,
        })

        print(
            f"[{index}/{len(evaluated_questions)}] "
            f"Question évaluée : {question[:70]}"
        )

    return {
        "k": k,
        "hit_rate": hits / len(evaluated_questions),
        "mean_recall": sum(recalls) / len(recalls),
        "by_category": {
            category: sum(values) / len(values)
            for category, values in category_results.items()
        },
        "details": details,
    }


# ============================================================
# LANCEMENT DE L'ÉVALUATION AVEC MLFLOW
# ============================================================

all_results = []

for k in K_VALUES:

    print("\n" + "=" * 65)
    print(f"ÉVALUATION HYBRIDE AVEC K={k}")
    print("=" * 65)

    metrics = evaluate_k(k)
    all_results.append(metrics)

    with mlflow.start_run(
        run_name=f"hybrid_retrieval_k_{k}"
    ):

        # Paramètres de l'expérience
        mlflow.log_params({
            "retrieval_method": "hybrid_bm25_vector_rrf",
            "embedding_model": EMBEDDING_MODEL,
            "collection_name": COLLECTION_NAME,
            "embedding_device": DEVICE,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "k": k,
            "number_of_questions": len(evaluated_questions),
            "normalize_embeddings": True,
            "query_expansion": False,
        })

        # Métriques principales
        mlflow.log_metrics({
            "hit_rate": metrics["hit_rate"],
            "mean_recall": metrics["mean_recall"],
        })

        # Métriques par catégorie
        for category, value in metrics["by_category"].items():
            mlflow.log_metric(
                f"recall_{category}",
                value
            )

        # Détails de l'évaluation
        mlflow.log_dict(
            metrics,
            artifact_file=f"retrieval_details_k_{k}.json"
        )

        print("Run MLflow :", mlflow.active_run().info.run_id)

    print(f"\nHit Rate@{k} : {metrics['hit_rate']:.2%}")
    print(f"Mean Recall@{k} : {metrics['mean_recall']:.2%}")

    print("Résultats par catégorie :")

    for category, value in metrics["by_category"].items():
        print(f"  {category} : {value:.2%}")


# ============================================================
# SAUVEGARDE DES RÉSULTATS HABITUELS
# ============================================================

RESULTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

output = {
    "experiment": "hybrid_vector_bm25_rrf",
    "embedding_model": EMBEDDING_MODEL,
    "device": DEVICE,
    "candidate_k": CANDIDATE_K,
    "rrf_k": RRF_K,
    "query_expansion": False,
    "golden_set": str(GOLDEN_SET),
    "total_questions": len(questions),
    "evaluated_questions": len(evaluated_questions),
    "results": all_results,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        output,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\n" + "=" * 65)
print("ÉVALUATION TERMINÉE")
print("=" * 65)

print("Résultats sauvegardés dans :", OUTPUT_FILE)
print("Expérience MLflow : rag_support_it_retrieval")
