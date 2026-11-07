
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory="indexes/chroma_support_it",
    embedding_function=embeddings,
    collection_name="support_it"
)

results = vectorstore.similarity_search(
    "Comment afficher l'adresse IP de mon ordinateur sous Windows ?",
    k=3
)

for i, doc in enumerate(results, start=1):
    print(f"\n--- CHUNK {i} ---")
    print("Métadonnées :", doc.metadata)
    print("Contenu complet :\n", doc.page_content)
