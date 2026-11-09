
import pytest

from langchain_core.documents import Document

from src.rag_pipeline import (
    extract_sources,
    is_abstention,
    is_security_refusal,
    classify_response,
    run_rag,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_chunks():
    doc1 = Document(
        page_content="Informations sur la résolution DNS.",
        metadata={
            "doc_id": "DOC001",
            "title": "Résolution des problèmes DNS côté client",
            "organization": "Microsoft",
            "os_scope": "Windows",
            "source_url": "https://example.com/dns",
            "filename": "windows_dns.md",
            "chunk_id": "DOC001_1",
            "section": "Résolution DNS",
        },
    )

    doc2 = Document(
        page_content="La commande ipconfig /flushdns vide le cache DNS.",
        metadata={
            "doc_id": "DOC002",
            "title": "Commande ipconfig",
            "organization": "Microsoft",
            "os_scope": "Windows",
            "source_url": "https://example.com/ipconfig",
            "filename": "windows_ipconfig.md",
            "chunk_id": "DOC002_1",
            "section": "Commande ipconfig",
        },
    )

    return [
        (doc1, 0.95),
        (doc2, 0.85),
    ]


@pytest.fixture
def irrelevant_chunks():
    doc = Document(
        page_content="Installation du logiciel OpenVPN.",
        metadata={
            "doc_id": "DOC007",
            "title": "Installing OpenVPN",
            "organization": "OpenVPN",
            "os_scope": "Windows/Linux",
            "source_url": "https://example.com/openvpn",
            "filename": "openvpn_installation.md",
            "chunk_id": "DOC007_1",
            "section": "Installation",
        },
    )

    return [(doc, 0.40)]


# ============================================================
# TESTS : CLASSIFICATION
# ============================================================

def test_is_abstention():
    answer = (
        "Les documents fournis ne contiennent aucune information "
        "concernant l'installation de Photoshop sur Windows."
    )

    assert is_abstention(answer) is True


def test_is_not_abstention():
    answer = (
        "Pour vider le cache DNS, exécutez la commande "
        "ipconfig /flushdns. Source : DOC002."
    )

    assert is_abstention(answer) is False


def test_is_security_refusal():
    answer = (
        "Je suis désolé, mais je ne peux pas aider "
        "avec cette demande."
    )

    assert is_security_refusal(answer) is True


def test_classify_abstention():
    answer = (
        "Les documents fournis ne contiennent aucune information "
        "sur Photoshop."
    )

    assert classify_response(answer) == "abstention"


def test_classify_security_refusal():
    answer = "Je ne peux pas aider avec cette demande."

    assert classify_response(answer) == "security_refusal"


def test_classify_standard_response():
    answer = "Exécutez ipconfig /flushdns. Source : DOC002."

    assert classify_response(answer) == "standard"


# ============================================================
# TESTS : EXTRACTION DES SOURCES
# ============================================================

def test_extract_sources_returns_cited_sources(sample_chunks):
    answer = (
        "Pour vider le cache DNS, utilisez ipconfig /flushdns. "
        "Source : DOC002."
    )

    sources, chunks = extract_sources(sample_chunks, answer)

    assert len(sources) == 1
    assert sources[0]["doc_id"] == "DOC002"
    assert len(chunks) == 2


def test_extract_sources_clears_sources_on_abstention(
    irrelevant_chunks,
):
    answer = (
        "Les documents fournis ne contiennent aucune information "
        "concernant l'installation de Photoshop sur Windows."
    )

    sources, chunks = extract_sources(irrelevant_chunks, answer)

    assert sources == []
    assert len(chunks) == 1


def test_extract_sources_does_not_return_uncited_sources(sample_chunks):
    answer = "Pour vider le cache DNS, utilisez ipconfig /flushdns."

    sources, chunks = extract_sources(sample_chunks, answer)

    assert sources == []
    assert len(chunks) == 2


def test_extract_sources_keeps_only_matching_citations(sample_chunks):
    answer = (
        "Pour vider le cache DNS, utilisez ipconfig /flushdns. "
        "Source : DOC002."
    )

    sources, chunks = extract_sources(sample_chunks, answer)

    assert len(sources) == 1
    assert sources[0]["doc_id"] == "DOC002"


def test_extract_sources_returns_empty_for_security_refusal(
    sample_chunks,
):
    answer = "Je ne peux pas aider avec cette demande."

    sources, chunks = extract_sources(sample_chunks, answer)

    assert sources == []
    assert len(chunks) == 2


def test_extract_sources_without_answer_returns_retrieved_sources(
    sample_chunks,
):
    sources, chunks = extract_sources(sample_chunks)

    assert len(sources) == 2
    assert len(chunks) == 2


# ============================================================
# TESTS : PIPELINE RAG
# ============================================================

def test_run_rag_executes_complete_pipeline(sample_chunks):

    def fake_retriever(question, k=5):
        return sample_chunks

    def fake_generator(question, context):
        assert question == "Comment vider le cache DNS sous Windows ?"
        assert "Informations sur la résolution DNS." in context

        return (
            "Pour vider le cache DNS, utilisez ipconfig /flushdns. "
            "Source : DOC002."
        )

    result = run_rag(
        question="Comment vider le cache DNS sous Windows ?",
        retriever=fake_retriever,
        generator=fake_generator,
    )

    assert "answer" in result
    assert "sources" in result
    assert "retrieved_chunks" in result

    assert len(result["sources"]) == 1
    assert result["sources"][0]["doc_id"] == "DOC002"

    assert len(result["retrieved_chunks"]) == 2


def test_run_rag_with_irrelevant_retrieved_document(
    irrelevant_chunks,
):

    def fake_retriever(question, k=5):
        return irrelevant_chunks

    def fake_generator(question, context):
        return (
            "Les documents fournis ne contiennent aucune information "
            "concernant l'installation de Photoshop sur Windows."
        )

    result = run_rag(
        question="Comment installer Photoshop sur Windows ?",
        retriever=fake_retriever,
        generator=fake_generator,
    )

    assert is_abstention(result["answer"]) is True
    assert result["sources"] == []

    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0]["doc_id"] == "DOC007"


def test_run_rag_with_security_refusal(sample_chunks):

    def fake_retriever(question, k=5):
        return sample_chunks

    def fake_generator(question, context):
        return "Je ne peux pas aider avec cette demande."

    result = run_rag(
        question="Comment contourner les règles de sécurité du réseau ?",
        retriever=fake_retriever,
        generator=fake_generator,
    )

    assert is_security_refusal(result["answer"]) is True
    assert result["sources"] == []


def test_run_rag_with_no_retrieved_chunks():

    def fake_retriever(question, k=5):
        return []

    def fake_generator(question, context):
        pytest.fail("Le générateur ne doit pas être appelé sans documents.")

    result = run_rag(
        question="Question sans documents pertinents",
        retriever=fake_retriever,
        generator=fake_generator,
    )

    assert result["sources"] == []
    assert result["retrieved_chunks"] == []
    assert is_abstention(result["answer"]) is True


def test_run_rag_rejects_empty_question():

    with pytest.raises(ValueError):
        run_rag(
            question="   ",
            retriever=lambda question, k=5: [],
            generator=lambda question, context: "",
        )
