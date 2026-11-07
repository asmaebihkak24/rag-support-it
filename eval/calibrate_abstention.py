import json
import os
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

GOLDEN_SET = Path("eval/golden_set.jsonl")
PERSIST_DIRECTORY = Path("indexes/chroma_support_it")
RESULTS_DIRECTORY = Path("eval/results")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "support_it"

IN_SCOPE = {"nominal", "multi_document"}
OUT_SCOPE = {"hors_perimetre", "securite"}
MAX_ABUSIVE_ABSTENTION = 0.05  # règle métier : au plus 5 % d'abstentions abusives

DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

with open(GOLDEN_SET, "r", encoding="utf-8-sig") as f:
    questions = [json.loads(line) for line in f if line.strip()]

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

# 1. Distance du meilleur résultat pour chaque question
rows = []
for q in questions:
    if q["category"] not in IN_SCOPE | OUT_SCOPE:
        continue
    doc, dist = vectorstore.similarity_search_with_score(q["question"], k=1)[0]
    rows.append({
        "question": q["question"],
        "category": q["category"],
        "in_scope": q["category"] in IN_SCOPE,
        "best_distance": round(float(dist), 4),
        "best_doc_id": doc.metadata.get("doc_id"),
    })

in_d = sorted(r["best_distance"] for r in rows if r["in_scope"])
out_d = sorted(r["best_distance"] for r in rows if not r["in_scope"])

print(f"\nDans le périmètre ({len(in_d)}) : min={in_d[0]}  max={in_d[-1]}")
print(f"Hors périmètre   ({len(out_d)}) : min={out_d[0]}  max={out_d[-1]}")

# 2. Test de seuils : on s'abstient si la meilleure distance dépasse le seuil
candidates = sorted(set(in_d + out_d))
table = []
for t in candidates:
    correct = sum(d > t for d in out_d) / len(out_d)
    abusive = sum(d > t for d in in_d) / len(in_d)
    table.append({
        "threshold": t,
        "correct_abstention": round(correct, 3),
        "abusive_abstention": round(abusive, 3),
    })

print("\nseuil   abstention correcte   abstention abusive")
for r in table:
    print(f"{r['threshold']:<7} {r['correct_abstention']:<21} {r['abusive_abstention']}")

# 3. Seuil recommandé : le plus bas qui respecte la règle métier
valid = [r for r in table if r["abusive_abstention"] <= MAX_ABUSIVE_ABSTENTION]
recommended = valid[0] if valid else None
print("\nSeuil recommandé :", recommended)

# 4. Questions hors périmètre, de la plus proche à la plus lointaine du corpus
print("\nHors périmètre, du plus proche au plus lointain :")
for r in sorted((r for r in rows if not r["in_scope"]), key=lambda r: r["best_distance"]):
    print(f"  {r['best_distance']}  [{r['category']}]  {r['question']}")

RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIRECTORY / "abstention_calibration.json", "w", encoding="utf-8") as f:
    json.dump({"rows": rows, "table": table, "recommended": recommended},
              f, ensure_ascii=False, indent=2)