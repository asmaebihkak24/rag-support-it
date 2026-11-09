
from types import SimpleNamespace

import pytest

from src import retriever


# ============================================================
# DONNÉES DE TEST
# ============================================================

def create_document(
    chunk_id="DOC001_1",
    doc_id="DOC001",
    content="Configuration DNS sous Windows."
):
    return SimpleNamespace(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "doc_id": doc_id,
        },
    )


class FakeVectorStore:

    def __init__(self, vector_results=None, score_results=None):
        self.vector_results = vector_results or []
        self.score_results = score_results or []
        self.received_question = None
        self.received_k = None

    def similarity_search(self, question, k=10):
        self.received_question = question
        self.received_k = k
        return self.vector_results

    def similarity_search_with_score(self, question, k=5):
        self.received_question = question
        self.received_k = k
        return self.score_results


class FakeBM25:

    def __init__(self, results=None):
        self.results = results or []
        self.received_question = None

    def invoke(self, question):
        self.received_question = question
        return self.results


# ============================================================
# TESTS DE TOKENISATION
# ============================================================

def test_tokenize_lowercases_text():
    assert retriever.tokenize("WINDOWS DNS") == [
        "windows",
        "dns"
    ]


def test_tokenize_removes_accents():
    assert retriever.tokenize("Été Réseau") == [
        "ete",
        "reseau"
    ]


def test_tokenize_preserves_technical_terms():
    result = retriever.tokenize(
        "BIND9 named.conf.local ipconfig /flushdns"
    )

    assert "bind9" in result
    assert "named.conf.local" in result
    assert "ipconfig" in result
    assert "/flushdns" in result


def test_tokenize_empty_string():
    assert retriever.tokenize("") == []


# ============================================================
# TESTS D'EXPANSION DE REQUÊTE
# ============================================================

def test_expand_query_for_bind9():
    question = "Comment configurer BIND9 sur Ubuntu ?"

    result = retriever.expand_query(question)

    assert "BIND9" in result
    assert "Ubuntu" in result
    assert "master slave" in result
    assert "primary secondary" in result


def test_expand_query_for_named_conf_local():
    question = "Comment modifier named.conf.local ?"

    result = retriever.expand_query(question)

    assert "BIND9" in result
    assert "configuration" in result


def test_expand_query_preserves_other_questions():
    question = "Comment vider le cache DNS sous Windows ?"

    assert retriever.expand_query(question) == question


# ============================================================
# TESTS DE RECHERCHE VECTORIELLE
# ============================================================

def test_search_documents_calls_vectorstore(monkeypatch):

    doc = create_document()

    fake_vectorstore = FakeVectorStore(
        score_results=[(doc, 0.25)]
    )

    monkeypatch.setattr(
        retriever,
        "vectorstore",
        fake_vectorstore
    )

    monkeypatch.setattr(
        retriever,
        "bm25_retriever",
        FakeBM25()
    )

    results = retriever.search_documents(
        "Question DNS",
        k=3
    )

    assert len(results) == 1
    assert results[0][0] == doc
    assert results[0][1] == 0.25

    assert fake_vectorstore.received_question == "Question DNS"
    assert fake_vectorstore.received_k == 3


# ============================================================
# TESTS DE RECHERCHE HYBRIDE ET RRF
# ============================================================

def test_hybrid_search_merges_vector_and_bm25(monkeypatch):

    doc_a = create_document(
        chunk_id="A",
        doc_id="DOC001"
    )

    doc_b = create_document(
        chunk_id="B",
        doc_id="DOC002"
    )

    doc_c = create_document(
        chunk_id="C",
        doc_id="DOC003"
    )

    fake_vectorstore = FakeVectorStore(
        vector_results=[doc_a, doc_b]
    )

    fake_bm25 = FakeBM25(
        results=[doc_b, doc_c]
    )

    monkeypatch.setattr(
        retriever,
        "vectorstore",
        fake_vectorstore
    )

    monkeypatch.setattr(
        retriever,
        "bm25_retriever",
        fake_bm25
    )

    results = retriever.search_documents_hybrid(
        "Question DNS",
        k=3
    )

    assert len(results) == 3

    # Le document B apparaît dans les deux recherches.
    # Il doit donc obtenir le meilleur score RRF.
    assert results[0][0].metadata["chunk_id"] == "B"

    expected_b_score = (
        1 / (retriever.RRF_K + 2)
        + 1 / (retriever.RRF_K + 1)
    )

    assert results[0][1] == pytest.approx(expected_b_score)

    assert fake_vectorstore.received_question == "Question DNS"
    assert fake_bm25.received_question == "Question DNS"


def test_hybrid_search_applies_query_expansion(monkeypatch):

    fake_vectorstore = FakeVectorStore()
    fake_bm25 = FakeBM25()

    monkeypatch.setattr(
        retriever,
        "vectorstore",
        fake_vectorstore
    )

    monkeypatch.setattr(
        retriever,
        "bm25_retriever",
        fake_bm25
    )

    retriever.search_documents_hybrid(
        "Comment configurer BIND9 ?"
    )

    expected = (
        "BIND9 Ubuntu zone master slave "
        "primary secondary configuration"
    )

    assert fake_vectorstore.received_question == expected
    assert fake_bm25.received_question == expected


def test_hybrid_search_respects_top_k(monkeypatch):

    docs = [
        create_document(chunk_id=f"DOC001_{i}")
        for i in range(5)
    ]

    fake_vectorstore = FakeVectorStore(
        vector_results=docs
    )

    fake_bm25 = FakeBM25()

    monkeypatch.setattr(
        retriever,
        "vectorstore",
        fake_vectorstore
    )

    monkeypatch.setattr(
        retriever,
        "bm25_retriever",
        fake_bm25
    )

    results = retriever.search_documents_hybrid(
        "Question DNS",
        k=2
    )

    assert len(results) == 2


def test_hybrid_search_rejects_document_without_chunk_id(
    monkeypatch
):

    invalid_doc = create_document(
        chunk_id=None
    )

    fake_vectorstore = FakeVectorStore(
        vector_results=[invalid_doc]
    )

    fake_bm25 = FakeBM25()

    monkeypatch.setattr(
        retriever,
        "vectorstore",
        fake_vectorstore
    )

    monkeypatch.setattr(
        retriever,
        "bm25_retriever",
        fake_bm25
    )

    with pytest.raises(
        ValueError,
        match="chunk_id"
    ):
        retriever.search_documents_hybrid(
            "Question DNS"
        )

# ============================================================
# TESTS DU DIAGNOSTIC BM25 BIND9
# ============================================================

def test_diagnostic_bm25_bind9_finds_doc005(monkeypatch, capsys):
    doc = create_document(
        chunk_id="DOC005_1",
        doc_id="DOC005",
        content="Configuration DNS avec BIND9."
    )

    fake_bm25 = FakeBM25(results=[doc])

    monkeypatch.setattr(
        retriever,
        "initialize_resources",
        lambda: None
    )

    monkeypatch.setattr(
        retriever,
        "bm25_retriever",
        fake_bm25
    )

    retriever.diagnostic_bm25_bind9()

    output = capsys.readouterr().out

    assert "DOC005 retrouvé dans le Top 10." in output
    assert "DOC005" in output
    assert "Configuration DNS avec BIND9." in output


def test_diagnostic_bm25_bind9_does_not_find_doc005(
    monkeypatch,
    capsys
):
    doc = create_document(
        chunk_id="DOC001_1",
        doc_id="DOC001"
    )

    fake_bm25 = FakeBM25(results=[doc])

    monkeypatch.setattr(
        retriever,
        "initialize_resources",
        lambda: None
    )

    monkeypatch.setattr(
        retriever,
        "bm25_retriever",
        fake_bm25
    )

    retriever.diagnostic_bm25_bind9()

    output = capsys.readouterr().out

    assert "DOC005 absent du Top 10." in output


# ============================================================
# TEST DU PROGRAMME PRINCIPAL
# ============================================================

def test_main_executes_all_searches(monkeypatch, capsys):

    doc = create_document(
        chunk_id="DOC002_1",
        doc_id="DOC002",
        content="Commande ipconfig."
    )

    vector_calls = []
    hybrid_calls = []

    def fake_search_documents(question, k=5):
        vector_calls.append((question, k))
        return [(doc, 0.25)]

    def fake_search_documents_hybrid(question, k=5):
        hybrid_calls.append((question, k))
        return [(doc, 0.016)]

    monkeypatch.setattr(
        retriever,
        "search_documents",
        fake_search_documents
    )

    monkeypatch.setattr(
        retriever,
        "search_documents_hybrid",
        fake_search_documents_hybrid
    )

    diagnostic_calls = []

    monkeypatch.setattr(
        retriever,
        "diagnostic_bm25_bind9",
        lambda: diagnostic_calls.append(True)
    )

    retriever.main()

    output = capsys.readouterr().out

    assert len(vector_calls) == 4
    assert len(hybrid_calls) == 4

    assert all(k == retriever.TOP_K for _, k in vector_calls)
    assert all(k == retriever.TOP_K for _, k in hybrid_calls)

    assert vector_calls[0][0] == (
        "Comment vider le cache DNS sous Windows ?"
    )

    assert len(diagnostic_calls) == 1

    assert "--- RECHERCHE VECTORIELLE ---" in output
    assert "--- RECHERCHE HYBRIDE ---" in output
    assert "Score RRF" in output
