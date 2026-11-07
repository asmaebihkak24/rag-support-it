
import csv
import hashlib
from pathlib import Path

RAW_DIR = Path("data/raw")
SOURCES_FILE = Path("data/sources.csv")

REQUIRED_COLUMNS = [
    "doc_id",
    "filename",
    "title",
    "organization",
    "source_type",
    "os_scope",
    "url",
    "download_date",
    "sha256",
]

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)

    return sha256.hexdigest()


def validate_data():
    errors = []

    if not SOURCES_FILE.exists():
        raise FileNotFoundError("data/sources.csv introuvable.")

    with SOURCES_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in (reader.fieldnames or [])
        ]

        if missing_columns:
            errors.append(
                f"Colonnes manquantes : {missing_columns}"
            )

        rows = list(reader)

    if errors:
        raise ValueError("\n".join(errors))

    doc_ids = [row["doc_id"].strip() for row in rows]

    if len(doc_ids) != len(set(doc_ids)):
        errors.append("Des doc_id sont dupliqués.")

    filenames = [row["filename"].strip() for row in rows]

    if len(filenames) != len(set(filenames)):
        errors.append("Des noms de fichiers sont dupliqués.")

    for row in rows:
        doc_id = row["doc_id"].strip()
        filename = row["filename"].strip()
        file_path = RAW_DIR / filename

        if not doc_id or not filename:
            errors.append("doc_id ou filename vide.")
            continue

        if not file_path.exists():
            errors.append(f"{doc_id} : fichier introuvable ({filename}).")
            continue

        if file_path.stat().st_size == 0:
            errors.append(f"{doc_id} : fichier vide.")
            continue

        actual_hash = calculate_sha256(file_path)
        expected_hash = row["sha256"].strip()

        if actual_hash != expected_hash:
            errors.append(f"{doc_id} : SHA256 incorrect.")

    if errors:
        print("\nÉCHEC DE VALIDATION")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("VALIDATION RÉUSSIE")
    print(f"Documents validés : {len(rows)}")
    print("Identifiants uniques : OK")
    print("Fichiers présents et non vides : OK")
    print("Empreintes SHA256 : OK")


if __name__ == "__main__":
    validate_data()
