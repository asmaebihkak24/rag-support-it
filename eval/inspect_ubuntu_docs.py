
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PERSIST_DIRECTORY = Path("indexes/chroma_support_it")
COLLECTION_NAME = "support_it"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

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

for doc_id in ["DOC004", "DOC005", "DOC006"]:
    print("\n" + "=" * 90)
    print(f"DOCUMENT : {doc_id}")
    print("=" * 90)

    data = vectorstore.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"]
    )

    print("Nombre de chunks :", len(data["documents"]))

    for i, (content, metadata) in enumerate(
        zip(data["documents"], data["metadatas"]), start=1
    ):
        print(f"\n--- Chunk {i} ---")
        print("Titre :", metadata.get("title"))
        print("Section :", metadata.get("section"))
        print("Contenu :")
        print(content)
