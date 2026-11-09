
import json
import os
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
GOLDEN_SET = Path("eval/golden_set.jsonl")
PERSIST_DIRECTORY = Path("indexes/chroma_support_it")
RESULTS_DIRECTORY = Path("eval/results")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "support_it"
K_VALUES = [3, 5]

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("rag_support_it_retrieval")

# GPU si disponible, sinon CPU
DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Charger les questions
with open(GOLDEN_SET, "r", encoding="utf-8-sig") as file:
    questions = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

# Garder uniquement les questions évaluables en retrieval
evaluated_questions = [
    q for q in questions
    if q["category"] in ["nominal", "multi_document"]
    and q["relevant_doc_ids"]
]

if not evaluated_questions:
    raise ValueError("Aucune question évaluable trouvée dans le golden set.")

print("Questions évaluées :", len(evaluated_questions))
print("Périphérique d'embeddings :", DEVICE)

# Charger les embeddings et l'index une seule fois
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


def evaluate_k(k):
    hits = 0
    recalls = []
    category_results = defaultdict(list)
    details = []

    for item in evaluated_questions:
        question = item["question"]
        relevant_ids = set(item["relevant_doc_ids"])

        results = vectorstore.similarity_search_with_score(
            question,
            k=k
        )

        # Documents uniques, dans l'ordre de classement
        retrieved_ids = list(dict.fromkeys(
            doc.metadata.get("doc_id")
            for doc, _ in results
            if doc.metadata.get("doc_id")
        ))

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
            "distances": [
                round(float(score), 4)
                for _, score in results
            ],
            "hit": hit,
            "recall": recall,
        })

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


# Évaluation et suivi MLflow
all_results = []

for k in K_VALUES:
    print(f"\nÉvaluation avec k={k}...")

    metrics = evaluate_k(k)
    all_results.append(metrics)

    # Un run MLflow indépendant pour chaque valeur de k
    with mlflow.start_run(run_name=f"vector_retrieval_k_{k}"):

        # Paramètres
        mlflow.log_params({
            "retrieval_method": "vector_similarity",
            "embedding_model": EMBEDDING_MODEL,
            "collection_name": COLLECTION_NAME,
            "embedding_device": DEVICE,
            "k": k,
            "number_of_questions": len(evaluated_questions),
            "normalize_embeddings": True,
        })

        # Métriques principales
        mlflow.log_metrics({
            "hit_rate": metrics["hit_rate"],
            "mean_recall": metrics["mean_recall"],
        })

        # Métriques par catégorie
        for category, value in metrics["by_category"].items():
            mlflow.log_metric(f"recall_{category}", value)

        # Enregistrer les détails comme artefact
        mlflow.log_dict(
            metrics,
            artifact_file=f"retrieval_details_k_{k}.json"
        )

        print("Run MLflow :", mlflow.active_run().info.run_id)

    print(f"Hit Rate@{k} : {metrics['hit_rate']:.2%}")
    print(f"Recall@{k} : {metrics['mean_recall']:.2%}")
    print("Résultats par catégorie :", metrics["by_category"])


# Sauvegarder les résultats habituels
RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

output_file = RESULTS_DIRECTORY / "retrieval_metrics.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(
        all_results,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\nRésultats sauvegardés dans :", output_file)
print("Expérience MLflow :", "rag_support_it_retrieval")
