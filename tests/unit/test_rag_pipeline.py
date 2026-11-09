
from types import SimpleNamespace
from types import SimpleNamespace
import pytest

from src.rag_pipeline import (
    build_context,
    extract_sources,
    run_rag,
    INSUFFICIENT_CONTEXT_MESSAGE,
)


# ============================================================
# DONNÉES DE TEST
# ============================================================

def create_document(
    doc_id="DOC001",
    chunk_id="DOC001_1",
    content="ipconfig /flushdns permet de vider le cache DNS.",
):
    return SimpleNamespace(
        page_content=content,
        metadata={
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "title": "Commande ipconfig",
            "section": "Commandes DNS",
            "organization": "Microsoft",
            "os_scope": "Windows",
            "source_url": "https://example.com",
            "filename": "windows_ipconfig.md",
        },
    )


# ============================================================
# TESTS DE BUILD_CONTEXT
# ============================================================

def test_build_context_contains_document_information():
    doc = create_document()

    context = build_context([(doc, 0.03125)])

    assert "DOC001" in context
    assert "Commande ipconfig" in context
    assert "Commandes DNS" in context
    assert "ipconfig /flushdns" in context


def test_build_context_excludes_chunk_id_and_score():
    doc = create_document(chunk_id="DOC001_SECRET")

    context = build_context([(doc, 0.03125)])

    assert "DOC001_SECRET" not in context
    assert "0.03125" not in context
    assert "Score RRF" not in context


def test_build_context_empty_list():
    assert build_context([]) == ""


# ============================================================
# TESTS DE EXTRACT_SOURCES
# ============================================================

def test_extract_sources_returns_unique_documents():
    doc1 = create_document(
        doc_id="DOC001",
        chunk_id="DOC001_1",
    )

    doc2 = create_document(
        doc_id="DOC001",
        chunk_id="DOC001_2",
    )

    sources, chunks = extract_sources([
        (doc1, 0.03),
        (doc2, 0.02),
    ])

    assert len(sources) == 1
    assert sources[0]["doc_id"] == "DOC001"
    assert len(chunks) == 2


def test_extract_sources_preserves_chunk_details():
    doc = create_document()

    sources, chunks = extract_sources([(doc, 0.0312549)])

    assert chunks[0]["chunk_id"] == "DOC001_1"
    assert chunks[0]["doc_id"] == "DOC001"
    assert chunks[0]["score_rrf"] == 0.031255


def test_extract_sources_empty_list():
    sources, chunks = extract_sources([])

    assert sources == []
    assert chunks == []


# ============================================================
# TESTS DE RUN_RAG
# ============================================================

def test_run_rag_rejects_empty_question():
    with pytest.raises(ValueError):
        run_rag(
            "",
            retriever=lambda question, k: [],
            generator=lambda question, context: "",
        )


def test_run_rag_returns_abstention_when_no_chunks():
    generator_called = False

    def fake_generator(question, context):
        nonlocal generator_called
        generator_called = True
        return "Réponse fictive"

    result = run_rag(
        "Question sans réponse documentaire",
        retriever=lambda question, k: [],
        generator=fake_generator,
    )

    assert result["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert result["sources"] == []
    assert result["retrieved_chunks"] == []
    assert generator_called is False


def test_run_rag_executes_complete_pipeline():
    doc = create_document()

    def fake_retriever(question, k):
        assert question == "Comment vider le cache DNS ?"
        assert k == 3
        return [(doc, 0.03125)]

    def fake_generator(question, context):
        assert question == "Comment vider le cache DNS ?"
        assert "ipconfig /flushdns" in context
        assert "DOC001_1" not in context
        return "Exécutez ipconfig /flushdns."

    result = run_rag(
        "Comment vider le cache DNS ?",
        k=3,
        retriever=fake_retriever,
        generator=fake_generator,
    )

    assert result["question"] == "Comment vider le cache DNS ?"
    assert result["answer"] == "Exécutez ipconfig /flushdns."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["doc_id"] == "DOC001"
    assert len(result["retrieved_chunks"]) == 1

def test_run_rag_loads_default_retriever(monkeypatch):
    from src import retriever as retriever_module

    fake_doc = SimpleNamespace(
    page_content="Contenu DNS",
    metadata={
        "doc_id": "DOC001",
        "chunk_id": "chunk1"
    }
   )

    monkeypatch.setattr(
        retriever_module,
        "search_documents_hybrid",
        lambda question, k: [(fake_doc, 0.9)]
    )

    result = run_rag(
        "Question DNS",
        generator=lambda question, context: "Réponse simulée"
    )

    assert result["answer"] == "Réponse simulée"


def test_run_rag_loads_default_generator(monkeypatch):
    from src import generator as generator_module

    fake_doc = SimpleNamespace(
        page_content="Contenu DNS",
        metadata={
            "doc_id": "DOC001",
            "chunk_id": "chunk1"
        }
    )

    monkeypatch.setattr(
        "src.retriever.search_documents_hybrid",
        lambda question, k: [(fake_doc, 0.9)]
    )

    monkeypatch.setattr(
        generator_module,
        "generate_answer",
        lambda question, context: "Réponse simulée"
    )

    result = run_rag("Question DNS")

    assert result["answer"] == "Réponse simulée"

def test_run_rag_with_irrelevant_retrieved_document():
    doc = create_document(
        doc_id="DOC_OPENVPN",
        chunk_id="DOC_OPENVPN_1",
        content="OpenVPN permet de créer un tunnel VPN sécurisé."
    )

    generator_called = False

    def fake_generator(question, context):
        nonlocal generator_called
        generator_called = True
        return "Réponse simulée"

    result = run_rag(
        "Comment installer Photoshop ?",
        retriever=lambda question, k: [(doc, 0.03)],
        generator=fake_generator
    )

    assert generator_called is True
    assert result["sources"][0]["doc_id"] == "DOC_OPENVPN"
