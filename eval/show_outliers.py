import json

with open("eval/results/abstention_calibration.json", encoding="utf-8") as f:
    rows = json.load(f)["rows"]

in_scope = sorted(
    (r for r in rows if r["in_scope"]),
    key=lambda r: -r["best_distance"]
)

for r in in_scope[:5]:
    print(
        r["best_distance"],
        r["best_doc_id"],
        "|",
        r["question"]
    )