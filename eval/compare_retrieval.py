
import sys
import json
import re
import csv
import unicodedata
from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.retriever import vectorstore

GOLDEN_SET = ROOT_DIR / "eval" / "golden_set.jsonl"

K = 5
RRF_K = 60


# --------------------------------------------------
# 1. PRETRAITEMENT BM25
# --------------------------------------------------

def tokenize(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    return re.findall(r"[a-z0-9_./-]+", text)


# --------------------------------------------------
# 2. EXPANSION CONDITIONNELLE
# --------------------------------------------------

def expand_query(question):
    question_lower = question.lower()

    if "bind9" in question_lower or "named.conf.local" in question_lower:
        return (
            "BIND9 Ubuntu zone master slave "
            "primary secondary configuration"
        )

    return question


# --------------------------------------------------
# 3. CHARGEMENT DES DOCUMENTS
# --------------------------------------------------

def load_documents():
    data = vectorstore.get(
        include=["documents", "metadatas"]
    )

    return [
        Document(
            page_content=content,
            metadata=metadata or {}
        )
        for content, metadata in zip(
            data["documents"],
            data["metadatas"]
        )
    ]


# --------------------------------------------------
# 4. CHARGEMENT DU GOLDEN SET
# --------------------------------------------------

def load_questions():
    questions = []

    with open(GOLDEN_SET, "r", encoding="utf-8-sig") as f:
        for line in f:
            item = json.loads(line)

            if (
                item.get("category") in ("nominal", "multi_document")
                and item.get("relevant_doc_ids")
            ):
                questions.append(item)

    return questions


def get_expected_docs(item):
    return set(item.get("relevant_doc_ids", []))


# --------------------------------------------------
# 5. RECHERCHE VECTORIELLE
# --------------------------------------------------

def vector_search(question, k=K):
    results = vectorstore.similarity_search_with_score(
        question,
        k=k
    )

    return [doc for doc, score in results]


# --------------------------------------------------
# 6. RECHERCHE HYBRIDE RRF
# --------------------------------------------------

def hybrid_search(question, bm25, k=K, conditional=False):

    query = (
        expand_query(question)
        if conditional
        else question
    )

    vector_results = vector_search(query, k=10)
    bm25_results = bm25.invoke(query)

    scores = defaultdict(float)
    documents = {}

    for result_list in (vector_results, bm25_results):
        for rank, doc in enumerate(result_list, start=1):

            chunk_id = doc.metadata.get("chunk_id")

            if chunk_id is None:
                raise ValueError(
                    "A document does not have a chunk_id."
                )

            scores[chunk_id] += 1 / (RRF_K + rank)
            documents[chunk_id] = doc

    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [documents[cid] for cid in ranked[:k]]


# --------------------------------------------------
# 7. EXECUTION D'UNE CONFIGURATION
# --------------------------------------------------

def retrieve(question, bm25, mode):

    if mode == "vector":
        return vector_search(question, k=K)

    if mode == "hybrid_classic":
        return hybrid_search(
            question,
            bm25,
            k=K,
            conditional=False
        )

    if mode == "hybrid_conditional":
        return hybrid_search(
            question,
            bm25,
            k=K,
            conditional=True
        )

    raise ValueError(f"Unknown mode: {mode}")


# --------------------------------------------------
# 8. ANALYSE INDIVIDUELLE
# --------------------------------------------------

def analyze_questions(questions, bm25, configurations):

    detailed_results = []
    summary_results = {}

    for config_name, mode in configurations:

        print("\n" + "=" * 65)
        print(f"ANALYSE : {config_name}")
        print("=" * 65)

        hits = {3: 0, 5: 0}
        recalls = {3: 0.0, 5: 0.0}

        for index, item in enumerate(questions, start=1):

            question = item["question"]
            expected = get_expected_docs(item)

            results = retrieve(question, bm25, mode)

            retrieved = [
                doc.metadata.get("doc_id")
                for doc in results
            ]

            top3 = retrieved[:3]
            top5 = retrieved[:5]

            found3 = set(top3) & expected
            found5 = set(top5) & expected

            missing3 = expected - set(top3)
            missing5 = expected - set(top5)

            recall3 = len(found3) / len(expected)
            recall5 = len(found5) / len(expected)

            hits[3] += bool(found3)
            hits[5] += bool(found5)

            recalls[3] += recall3
            recalls[5] += recall5

            # Details des chunks retrouves
            chunk_details = []

            for rank, doc in enumerate(results, start=1):
                metadata = doc.metadata

                chunk_details.append(
                    f"{rank}:"
                    f"{metadata.get('doc_id', 'UNKNOWN')}:"
                    f"{metadata.get('chunk_id', 'UNKNOWN')}:"
                    f"{metadata.get('section', 'UNKNOWN')}"
                )

            detail = {
                "configuration": config_name,
                "question_id": item.get("id", index),
                "category": item.get("category", ""),
                "question": question,
                "expected_docs": sorted(expected),
                "top3_docs": top3,
                "top5_docs": top5,
                "relevant_found_top3": sorted(found3),
                "relevant_found_top5": sorted(found5),
                "missing_top3": sorted(missing3),
                "missing_top5": sorted(missing5),
                "recall_at_3": round(recall3, 4),
                "recall_at_5": round(recall5, 4),
                "hit_at_3": bool(found3),
                "hit_at_5": bool(found5),
                "top5_chunk_details": chunk_details
            }

            detailed_results.append(detail)

            status = "OK" if not missing5 else "INCOMPLETE"

            print(f"\nQuestion {index}/{len(questions)} - {status}")
            print("Question :", question)
            print("Expected :", sorted(expected))
            print("Top 3    :", top3)
            print("Top 5    :", top5)
            print("Missing Top 5 :", sorted(missing5))
            print(f"Recall@3 : {recall3:.2%}")
            print(f"Recall@5 : {recall5:.2%}")

        total = len(questions)

        summary_results[config_name] = {
            "Hit Rate@3": hits[3] / total,
            "Hit Rate@5": hits[5] / total,
            "Recall@3": recalls[3] / total,
            "Recall@5": recalls[5] / total
        }

    return detailed_results, summary_results


# --------------------------------------------------
# 9. SAUVEGARDE CSV ET JSON
# --------------------------------------------------

def save_results(detailed_results, summary_results):

    output_dir = ROOT_DIR / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "retrieval_analysis_current21.csv"

    fieldnames = [
        "configuration",
        "question_id",
        "category",
        "question",
        "expected_docs",
        "top3_docs",
        "top5_docs",
        "relevant_found_top3",
        "relevant_found_top5",
        "missing_top3",
        "missing_top5",
        "recall_at_3",
        "recall_at_5",
        "hit_at_3",
        "hit_at_5",
        "top5_chunk_details"
    ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in detailed_results:

            row = result.copy()

            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = " | ".join(map(str, value))

            writer.writerow(row)

    json_path = output_dir / "retrieval_analysis_current21.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            detailed_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    summary_path = output_dir / "comparison_retrieval_current21.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            summary_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("ANALYSE TERMINEE")
    print("=" * 65)

    print(f"CSV : {csv_path}")
    print(f"JSON detaille : {json_path}")
    print(f"Comparaison : {summary_path}")


# --------------------------------------------------
# 10. PROGRAMME PRINCIPAL
# --------------------------------------------------

def main():

    print("Chargement des documents...")

    documents = load_documents()

    print(f"{len(documents)} chunks charges.")

    bm25 = BM25Retriever.from_documents(
        documents,
        preprocess_func=tokenize,
        k=10
    )

    questions = load_questions()

    print(f"{len(questions)} questions chargees.")

    if not questions:
        raise ValueError("Aucune question valide dans le Golden Set.")

    configurations = [
        ("Vectoriel classique", "vector"),
        ("Hybride classique", "hybrid_classic"),
        ("Hybride avec expansion conditionnelle", "hybrid_conditional")
    ]

    detailed_results, summary_results = analyze_questions(
        questions,
        bm25,
        configurations
    )

    print("\nCOMPARAISON GLOBALE")

    for name, metrics in summary_results.items():

        print(f"\n{name}")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.2%}")

    save_results(
        detailed_results,
        summary_results
    )


if __name__ == "__main__":
    main()
