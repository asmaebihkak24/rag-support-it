import re
import json
from pathlib import Path
from datetime import datetime

import mlflow
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import run_rag, INSUFFICIENT_CONTEXT_MESSAGE


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = PROJECT_ROOT / "eval" / "golden_set.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"
OUTPUT_PATH = OUTPUT_DIR / "end_to_end_results.json"

TOP_K = 5
EXPERIMENT_NAME = "rag_support_it_end_to_end"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# CHARGEMENT DU GOLDEN SET
# ============================================================

def load_golden_set():
    questions = []

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                questions.append(json.loads(line))

    return questions
def detect_abstention(answer):
    if not answer:
        return True

    normalized = answer.lower().strip()

    # Supprimer les références documentaires entre parenthèses :
    # (DOC008), (DOC008 et DOC009), etc.
    normalized = re.sub(
        r"\(doc\d+(?:\s*(?:,|et)\s*doc\d+)*\)",
        "",
        normalized
    )

    abstention_patterns = [
        "je ne dispose pas d'informations suffisantes",
        "je ne dispose pas des informations nécessaires",
        "les documents fournis ne contiennent pas",
        "les documents ne contiennent aucune information",
        "ne contiennent aucune information",
        "ne contiennent pas les étapes détaillées",
        "ne contiennent pas d'instructions détaillées",
        "le contexte fourni ne contient pas",
        "les informations nécessaires ne sont pas présentes",
        "ne sont pas présentes dans le contexte documentaire fourni",
        "je ne peux pas répondre à cette question",
        "je ne peux pas répondre à votre demande",
        "je ne peux pas fournir une procédure complète",
        "les documents ne détaillent pas",
        "aucune information précise",
        "les sources disponibles ne permettent pas",
        "en l'absence de données pertinentes"
    ]

    return any(
        pattern in normalized
        for pattern in abstention_patterns
    )
# ============================================================
# ÉVALUATION D'UNE QUESTION
# ============================================================

def evaluate_question(item):
    question = item["question"]
    category = item["category"]
    expected_doc_ids = set(item.get("relevant_doc_ids", []))

    result = run_rag(question, k=TOP_K)

    answer = result["answer"]
    sources = result["sources"]

    retrieved_doc_ids = {
        source["doc_id"] for source in sources
    }

    is_abstention = detect_abstention(answer)
    source_overlap = (
        bool(expected_doc_ids.intersection(retrieved_doc_ids))
        if expected_doc_ids else None
    )

    # Évaluation automatique selon la catégorie
    if category == "nominal":
        assessment = (
            "answered_with_relevant_source"
            if not is_abstention and source_overlap
            else "abstention"
            if is_abstention
            else "answered_without_expected_source"
        )

    elif category == "insuffisant_contexte":
        assessment = (
            "correct_abstention"
            if is_abstention
            else "incorrect_answer"
        )

    elif category == "hors_perimetre":
        assessment = (
            "correct_abstention"
            if is_abstention
            else "possible_out_of_scope_answer"
        )

    elif category == "securite":
        # Le refus de sécurité nécessite une vérification humaine.
        assessment = "manual_security_review"

    else:
        assessment = "unknown_category"

    return {
        "question": question,
        "category": category,
        "expected_doc_ids": sorted(expected_doc_ids),
        "retrieved_doc_ids": sorted(retrieved_doc_ids),
        "source_overlap": source_overlap,
        "is_abstention": is_abstention,
        "answer": answer,
        "assessment": assessment,
        "sources": sources,
        "retrieved_chunks": result["retrieved_chunks"],
        "manual_review": {
            "grounded": None,
            "answers_question": None,
            "correct_abstention": None,
            "safe_response": None,
            "comments": ""
        }
    }


# ============================================================
# CALCUL DES MÉTRIQUES
# ============================================================

def calculate_metrics(results):

    successful = [
        r for r in results
        if r.get("assessment") != "evaluation_error"
        and isinstance(r.get("is_abstention"), bool)
    ]

    nominal = [
        r for r in successful if r["category"] == "nominal"
    ]

    insufficient = [
        r for r in successful
        if r["category"] == "insuffisant_contexte"
    ]

    out_of_scope = [
        r for r in successful if r["category"] == "hors_perimetre"
    ]

    security = [
        r for r in successful if r["category"] == "securite"
    ]

    metrics = {
        "total_questions": len(results),
        "evaluated_questions": len(successful),
        "evaluation_errors": len(results) - len(successful),
        "security_cases_to_review": len(security)
    }

    if nominal:
        metrics["nominal_answer_rate"] = (
            sum(not r["is_abstention"] for r in nominal)
            / len(nominal)
        )

        metrics["nominal_relevant_source_rate"] = (
            sum(r.get("source_overlap") is True for r in nominal)
            / len(nominal)
        )

        metrics["nominal_abstention_rate"] = (
            sum(r["is_abstention"] for r in nominal)
            / len(nominal)
        )

    if insufficient:
        metrics["insufficient_abstention_rate"] = (
            sum(r["is_abstention"] for r in insufficient)
            / len(insufficient)
        )

    if out_of_scope:
        metrics["out_of_scope_abstention_rate"] = (
            sum(r["is_abstention"] for r in out_of_scope)
            / len(out_of_scope)
        )

    return metrics


# ============================================================
# EXÉCUTION
# ============================================================

def main():

    print("=" * 65)
    print("ÉVALUATION END-TO-END DU RAG")
    print("=" * 65)

    questions = load_golden_set()

    print(f"Questions chargées : {len(questions)}")
    print(f"Top K : {TOP_K}")

    results = []

    with mlflow.start_run(run_name="end_to_end_top_5"):

        mlflow.log_param("evaluation_type", "end_to_end")
        mlflow.log_param("retrieval_method", "hybrid_rrf")
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("number_of_questions", len(questions))
        mlflow.log_param("golden_set", "golden_set.jsonl")

        for index, item in enumerate(questions, start=1):

            print("\n" + "-" * 65)
            print(f"Question {index}/{len(questions)}")
            print("Catégorie :", item["category"])
            print("Question :", item["question"])

            try:
                result = evaluate_question(item)

                results.append(result)

                print("Évaluation :", result["assessment"])
                print("Sources :", result["retrieved_doc_ids"])
                print("Réponse :", result["answer"][:300])

            except Exception as error:

                print("ERREUR :", str(error))

                results.append({
                       "question": item["question"],
                       "category": item["category"],
                       "assessment": "evaluation_error",
                       "is_abstention": None,
                       "source_overlap": None,
                       "answer": None,
                       "sources": [],
                       "error": str(error)
                    })

        metrics = calculate_metrics(results)

        # Enregistrement des métriques dans MLflow
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)

        # Sauvegarde locale des résultats
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output = {
            "evaluation_date": datetime.now().isoformat(),
            "top_k": TOP_K,
            "total_questions": len(questions),
            "metrics": metrics,
            "results": results
        }

        with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=4)

        # Sauvegarde comme artefact MLflow
        mlflow.log_artifact(
            str(OUTPUT_PATH),
            artifact_path="end_to_end"
        )

        print("\n" + "=" * 65)
        print("RÉSULTATS GLOBAUX")
        print("=" * 65)

        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"{name}: {value:.2%}")
            else:
                print(f"{name}: {value}")

        print("\nRésultats sauvegardés :", OUTPUT_PATH)
        print("Évaluation enregistrée dans MLflow.")
def recalculate_existing_results():

    with open(OUTPUT_PATH, "r", encoding="utf-8-sig") as file:
        output = json.load(file)

    results = output["results"]

    for result in results:

        if result.get("assessment") == "evaluation_error":
            continue

        result["is_abstention"] = detect_abstention(
            result.get("answer")
        )

        category = result["category"]
        is_abstention = result["is_abstention"]

        if category == "nominal":
            if is_abstention:
                result["assessment"] = "abstention"
            elif result.get("source_overlap"):
                result["assessment"] = "answered_with_relevant_source"
            else:
                result["assessment"] = "answered_without_expected_source"

        elif category == "insuffisant_contexte":
            result["assessment"] = (
                "correct_abstention"
                if is_abstention
                else "incorrect_answer"
            )

        elif category == "hors_perimetre":
            result["assessment"] = (
                "correct_abstention"
                if is_abstention
                else "possible_out_of_scope_answer"
            )

        elif category == "securite":
            result["assessment"] = "manual_security_review"

    metrics = calculate_metrics(results)

    output["metrics"] = metrics
    output["abstention_detection_version"] = "v2"

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)

    print("\nNOUVELLES MÉTRIQUES")
    print("=" * 50)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2%}")
        else:
            print(f"{name}: {value}")

    print("\nRésultats mis à jour :", OUTPUT_PATH)
def retry_failed_questions():

    with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
        output = json.load(file)

    results = output["results"]
    questions = load_golden_set()

    failed = [
        r for r in results
        if r.get("assessment") == "evaluation_error"
    ]

    print(f"Questions à réessayer : {len(failed)}")

    if not failed:
        print("Aucune erreur à corriger.")
        return

    for failed_result in failed:

        question_text = failed_result["question"]

        item = next(
            q for q in questions
            if q["question"] == question_text
        )

        print("\nNouvelle tentative :", question_text)

        try:
            result = evaluate_question(item)

            results[results.index(failed_result)] = result

            print("Évaluation :", result["assessment"])

        except Exception as error:
            print("Nouvel échec :", error)

    metrics = calculate_metrics(results)

    output["results"] = results
    output["metrics"] = metrics
    output["evaluation_date"] = datetime.now().isoformat()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)

    with mlflow.start_run(run_name="end_to_end_retry_failed"):

        mlflow.log_param("evaluation_type", "retry_failed_questions")
        mlflow.log_param("top_k", TOP_K)

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)

        mlflow.log_artifact(
            str(OUTPUT_PATH),
            artifact_path="end_to_end"
        )

    print("\nMÉTRIQUES FINALES")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2%}")
        else:
            print(f"{name}: {value}")

if __name__ == "__main__":
    retry_failed_questions()
