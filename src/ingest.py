import csv
import shutil
import hashlib
import re
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chemins du projet
RAW_DIR = Path("data/raw")
SOURCES_FILE = Path("data/sources.csv")
PERSIST_DIRECTORY = Path("indexes/chroma_support_it")

# Configuration fixe (on ne change pas les paramètres pour l'instant)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Modèle d'embeddings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_sources_metadata():
    """Charge les métadonnées depuis sources.csv."""
    metadata_map = {}
    if not SOURCES_FILE.exists():
        raise FileNotFoundError(f"Le fichier d'inventaire {SOURCES_FILE} est introuvable.")

    with SOURCES_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_map[row["filename"]] = {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "organization": row["organization"],
                "os_scope": row["os_scope"],
                "source_url": row["url"],
                "filename": row["filename"]
            }
    return metadata_map


def clean_front_matter(content):
    """Retire le bloc front-matter YAML du début du fichier Markdown pour éviter le bruit textuel."""
    # Supprime tout ce qui se trouve entre '---' au début du document
    pattern = r"^---\s*\n.*?\n---\s*\n"
    cleaned = re.sub(pattern, "", content, flags=re.DOTALL)
    return cleaned.strip()


def extract_markdown_section(content):
    """Extrait dynamiquement la dernière section (titre Markdown) rencontrée dans le texte."""
    lines = content.splitlines()
    current_section = "Introduction / Général"
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("#"):
            current_section = line_stripped.lstrip("#").strip()
    return current_section


def generate_chunk_id(doc_id, chunk_index, content):
    """Génère un identifiant déterministe et stable pour chaque chunk."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}_{chunk_index}_{content_hash}"


def run_ingestion():
    print("--- DÉBUT DE L'INGESTION RAG (AVEC NETTOYAGE YAML) ---")

    # 1. Charger les métadonnées de référence
    metadata_map = load_sources_metadata()
    print(f"Métadonnées chargées : {len(metadata_map)} documents.")

    documents = []

    # 2. Charger et nettoyer les documents Markdown
    for file_path in sorted(RAW_DIR.glob("*.md")):
        filename = file_path.name
        if filename not in metadata_map:
            raise ValueError(f"Document absent de sources.csv : {filename}")

        metadata = metadata_map[filename]

        loader = TextLoader(str(file_path), encoding="utf-8")
        raw_docs = loader.load()

        for doc in raw_docs:
            # Nettoyer le front-matter YAML du texte brut du document
            doc.page_content = clean_front_matter(doc.page_content)
            # Mettre à jour les métadonnées
            doc.metadata.update(metadata)
            documents.append(doc)

        print(f"Chargé et nettoyé : {filename} ({metadata['doc_id']})")

    if not documents:
        raise ValueError("Aucun document trouvé à ingérer dans data/raw/.")

    # 3. Découper les documents en chunks
    print("\n--- DÉCOUPAGE EN CHUNKS ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Nombre total de chunks générés : {len(chunks)}")

    # 4. Enrichissement (section) et génération d'identifiants stables
    chunk_ids = []
    for index, chunk in enumerate(chunks):
        doc_id = chunk.metadata["doc_id"]
        
        section_title = extract_markdown_section(chunk.page_content)
        chunk.metadata["section"] = section_title

        chunk_index = sum(
            1 for previous in chunks[:index]
            if previous.metadata["doc_id"] == doc_id
        )

        chunk_id = generate_chunk_id(doc_id, chunk_index, chunk.page_content)
        chunk.metadata["chunk_id"] = chunk_id
        chunk_ids.append(chunk_id)

    # 5. Initialiser les embeddings (sans changer le modèle)
    print("\n--- INITIALISATION DES EMBEDDINGS ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 6. Reconstruire l'index Chroma
    print("\n--- CRÉATION DE L'INDEX CHROMA ---")
    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)

    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=chunk_ids,
        collection_name="support_it",
        persist_directory=str(PERSIST_DIRECTORY)
    )

    print("\n--- INGESTION TERMINÉE AVEC SUCCÈS ---")
    print(f"Chunks indexés (sans bruit YAML) : {len(chunks)}")


if __name__ == "__main__":
    run_ingestion()