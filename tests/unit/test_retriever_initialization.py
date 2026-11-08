
import sys
from types import ModuleType, SimpleNamespace

from src import retriever


def test_initialize_resources_with_mocked_dependencies(monkeypatch):
    # Réinitialiser les ressources globales
    monkeypatch.setattr(retriever, "embeddings", None)
    monkeypatch.setattr(retriever, "vectorstore", None)
    monkeypatch.setattr(retriever, "documents", [])
    monkeypatch.setattr(retriever, "bm25_retriever", None)

    # Simuler PyTorch
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: False
    )

    # Simuler les documents LangChain
    class FakeDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    fake_core_documents = ModuleType("langchain_core.documents")
    fake_core_documents.Document = FakeDocument

    # Simuler les embeddings
    class FakeEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_huggingface = ModuleType("langchain_huggingface")
    fake_huggingface.HuggingFaceEmbeddings = FakeEmbeddings

    # Simuler Chroma
    class FakeChroma:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get(self, include):
            return {
                "documents": [
                    "Contenu du document DNS",
                    "Contenu du document OpenVPN"
                ],
                "metadatas": [
                    {"doc_id": "DOC001"},
                    {"doc_id": "DOC007"}
                ]
            }

    fake_chroma = ModuleType("langchain_chroma")
    fake_chroma.Chroma = FakeChroma

    # Simuler BM25
    class FakeBM25Retriever:
        @classmethod
        def from_documents(cls, documents, **kwargs):
            return SimpleNamespace(
                documents=documents,
                kwargs=kwargs
            )

    fake_community_retrievers = ModuleType(
        "langchain_community.retrievers"
    )
    fake_community_retrievers.BM25Retriever = FakeBM25Retriever

    # Injecter les modules simulés
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "langchain_chroma", fake_chroma)
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_huggingface)
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.documents",
        fake_core_documents
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_community.retrievers",
        fake_community_retrievers
    )

    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")

    # Initialiser les ressources
    retriever.initialize_resources()

    # Vérifier les ressources chargées
    assert retriever.embeddings is not None
    assert retriever.vectorstore is not None
    assert retriever.bm25_retriever is not None

    assert len(retriever.documents) == 2
    assert retriever.documents[0].metadata["doc_id"] == "DOC001"
    assert retriever.documents[1].metadata["doc_id"] == "DOC007"

    assert retriever.DEVICE == "cpu"

    # Vérifier la configuration de BM25
    assert retriever.bm25_retriever.kwargs["k"] == 10

    # Vérifier que le deuxième appel ne réinitialise pas les ressources
    original_embeddings = retriever.embeddings
    retriever.initialize_resources()

    assert retriever.embeddings is original_embeddings
