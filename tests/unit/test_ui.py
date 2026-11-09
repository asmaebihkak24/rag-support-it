"""
Tests unitaires pour l'interface utilisateur Streamlit (app.py).
Validation de la logique de classification, gestion des entrées,
extraction des sources et formats de messages sans dépendances externes.
"""

import pytest

from app import (
    CATEGORIES,
    EXAMPLE_QUERIES,
    INSUFFICIENT_CONTEXT_MESSAGE,
    classify_response,
    is_insufficient_context,
    is_security_refusal,
    process_question,
)

# ============================================================
# TESTS DE CLASSIFICATION : ABSTENTION
# ============================================================

def test_is_insufficient_context_with_exact_message():
    assert is_insufficient_context(INSUFFICIENT_CONTEXT_MESSAGE) is True


def test_is_insufficient_context_with_empty_or_none():
    assert is_insufficient_context("") is True
    assert is_insufficient_context(None) is True


def test_is_insufficient_context_with_standard_answer():
    answer = "Pour vider le cache DNS, ouvrez cmd et tapez ipconfig /flushdns [Source : DOC001]."
    assert is_insufficient_context(answer) is False


# ============================================================
# TESTS DE CLASSIFICATION : REFUS DE SÉCURITÉ
# ============================================================

def test_is_security_refusal_with_unauthorized_access_text():
    answer = (
        "Je ne peux pas vous aider à contourner les règles de sécurité. "
        "Toute demande d'accès non autorisé est refusée. Contactez votre administrateur système."
    )
    assert is_security_refusal(answer) is True


def test_is_security_refusal_with_standard_technical_answer():
    answer = "La commande ipconfig /release permet de libérer l'adresse IPv4 [Source : DOC002]."
    assert is_security_refusal(answer) is False


def test_is_security_refusal_ignores_abstention_message():
    # Même si l'abstention contient des termes neutres, elle ne doit pas être un refus de sécurité
    assert is_security_refusal(INSUFFICIENT_CONTEXT_MESSAGE) is False


# ============================================================
# TESTS DE CLASSIFICATION GLOBALE
# ============================================================

def test_classify_response_abstention():
    assert classify_response(INSUFFICIENT_CONTEXT_MESSAGE) == "abstention"


def test_classify_response_security_refusal():
    refusal = "Je refuse cette demande car elle vise un accès non autorisé. Veuillez contacter l'administrateur."
    assert classify_response(refusal) == "security_refusal"


def test_classify_response_standard():
    answer = "Pour configurer Netplan, éditez /etc/netplan/01-netcfg.yaml puis lancez netplan apply [DOC004]."
    assert classify_response(answer) == "standard"


# ============================================================
# TESTS DU TRAITEMENT DES QUESTIONS (process_question)
# ============================================================

def test_process_question_rejects_empty_input():
    with pytest.raises(ValueError, match="La question ne peut pas être vide"):
        process_question("")


def test_process_question_rejects_whitespace_input():
    with pytest.raises(ValueError, match="La question ne peut pas être vide"):
        process_question("   \n\t  ")


def test_process_question_calls_pipeline_and_structures_result():
    mock_pipeline_called = {}

    def mock_rag(question):
        mock_pipeline_called["question"] = question
        return {
            "question": question,
            "answer": "Pour vider le cache DNS, exécutez ipconfig /flushdns.",
            "sources": [
                {
                    "doc_id": "DOC001",
                    "title": "Résolution des problèmes DNS",
                    "organization": "Microsoft",
                    "os_scope": "Windows",
                    "source_url": "https://example.com/doc1",
                }
            ],
            "retrieved_chunks": [
                {"doc_id": "DOC001", "chunk_id": "DOC001_1", "score_rrf": 0.032}
            ],
        }

    result = process_question("  Comment vider le cache DNS ?  ", rag_pipeline_fn=mock_rag)

    assert mock_pipeline_called["question"] == "Comment vider le cache DNS ?"
    assert result["question"] == "Comment vider le cache DNS ?"
    assert result["response_type"] == "standard"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["doc_id"] == "DOC001"
    assert result["answer"] == "Pour vider le cache DNS, exécutez ipconfig /flushdns."


def test_process_question_identifies_abstention_pipeline_output():
    def mock_rag_abstention(question):
        return {
            "question": question,
            "answer": INSUFFICIENT_CONTEXT_MESSAGE,
            "sources": [],
            "retrieved_chunks": [],
        }

    result = process_question("Comment installer Mac OS sur un Dell ?", rag_pipeline_fn=mock_rag_abstention)

    assert result["response_type"] == "abstention"
    assert result["sources"] == []
    assert result["answer"] == INSUFFICIENT_CONTEXT_MESSAGE


def test_process_question_identifies_security_refusal_pipeline_output():
    def mock_rag_security(question):
        return {
            "question": question,
            "answer": (
                "Je ne peux pas vous fournir cette information car il s'agit d'un contournement "
                "de la sécurité du réseau. Veuillez vous rapprocher de l'administrateur système."
            ),
            "sources": [],
            "retrieved_chunks": [],
        }

    result = process_question("Comment cracker le mot de passe Wi-Fi ?", rag_pipeline_fn=mock_rag_security)

    assert result["response_type"] == "security_refusal"
    assert "administrateur" in result["answer"]


def test_render_sources_section_without_sources():
    from app import render_sources_section
    # S'il n'y a pas de sources, la fonction doit s'arrêter proprement sans erreur
    render_sources_section([])


def test_render_sources_section_with_sources(monkeypatch):
    from unittest.mock import MagicMock

    import streamlit as st

    from app import render_sources_section

    mock_expander = MagicMock()
    mock_expander.__enter__ = MagicMock(return_value=mock_expander)
    mock_expander.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(st, "expander", MagicMock(return_value=mock_expander))

    mock_container = MagicMock()
    mock_container.__enter__ = MagicMock(return_value=mock_container)
    mock_container.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(st, "container", MagicMock(return_value=mock_container))

    monkeypatch.setattr(st, "markdown", MagicMock())
    monkeypatch.setattr(st, "caption", MagicMock())

    sources = [
        {
            "doc_id": "DOC001",
            "title": "Résolution des problèmes DNS",
            "organization": "Microsoft",
            "os_scope": "Windows",
            "source_url": "https://example.com/doc1",
        },
        {
            "doc_id": "DOC002",
            "title": "Commande ipconfig",
            "organization": "Microsoft",
            "os_scope": "Windows",
            "source_url": None,
        },
    ]

    render_sources_section(sources)
    assert st.markdown.called
    assert st.caption.called


def test_render_assistant_message_all_types(monkeypatch):
    from unittest.mock import MagicMock

    import streamlit as st

    from app import render_assistant_message

    monkeypatch.setattr(st, "markdown", MagicMock())
    monkeypatch.setattr(st, "expander", MagicMock())
    monkeypatch.setattr(st, "info", MagicMock())
    monkeypatch.setattr(st, "error", MagicMock())

    # Standard
    render_assistant_message({
        "response_type": "standard",
        "content": "Réponse technique standard.",
        "sources": [],
    })

    # Abstention
    render_assistant_message({
        "response_type": "abstention",
        "content": INSUFFICIENT_CONTEXT_MESSAGE,
        "sources": [],
    })

    # Security refusal
    render_assistant_message({
        "response_type": "security_refusal",
        "content": "Refus pour raison de sécurité.",
        "sources": [],
    })

    assert st.markdown.call_count >= 3
    assert st.info.called
    assert st.error.called


def test_render_assistant_message_never_renders_sources_for_abstention_or_refusal(monkeypatch):
    """
    Vérifie que même si un objet message contenait des sources par erreur,
    celles-ci ne sont JAMAIS affichées pour une abstention ou un refus de sécurité.
    """
    from unittest.mock import MagicMock

    import streamlit as st

    from app import render_assistant_message

    mock_render_sources = MagicMock()
    monkeypatch.setattr("app.render_sources_section", mock_render_sources)
    monkeypatch.setattr(st, "markdown", MagicMock())
    monkeypatch.setattr(st, "info", MagicMock())
    monkeypatch.setattr(st, "error", MagicMock())

    dummy_sources = [{"doc_id": "DOC001", "title": "DNS"}]

    # Abstention avec sources résiduelles
    render_assistant_message({
        "response_type": "abstention",
        "content": INSUFFICIENT_CONTEXT_MESSAGE,
        "sources": dummy_sources,
    })
    assert mock_render_sources.called is False

    # Refus de sécurité avec sources résiduelles
    render_assistant_message({
        "response_type": "security_refusal",
        "content": "Cette demande est refusée.",
        "sources": dummy_sources,
    })
    assert mock_render_sources.called is False

    # Réponse normale avec sources
    render_assistant_message({
        "response_type": "standard",
        "content": "Procédure normale.",
        "sources": dummy_sources,
    })
    assert mock_render_sources.call_count == 1


# ============================================================
# TESTS DE CLASSIFICATION : CAS RÉEL DU CACHE DNS ET MOTS IT
# ============================================================

def test_dns_flush_cache_is_classified_as_standard():
    """
    Vérifie spécifiquement que la question sur le vidage du cache DNS,
    même si la réponse mentionne 'administrateur' ou 'sécurité',
    est classifiée en 'standard' et non en refus de sécurité.
    """
    dns_answer = (
        "Pour vider le cache DNS sous Windows :\n"
        "1. Ouvrez l'invite de commandes en tant qu'administrateur.\n"
        "2. Exécutez la commande suivante : `ipconfig /flushdns` [Source : DOC002].\n"
        "3. Le système confirme que le cache de résolution DNS a été vidé avec succès.\n"
        "Si le problème persiste avec le pare-feu ou le serveur DNS, contactez votre administrateur réseau."
    )

    # 1. Vérification unitaire du détecteur
    assert is_security_refusal(dns_answer) is False
    assert is_insufficient_context(dns_answer) is False

    # 2. Vérification de la classification globale
    assert classify_response(dns_answer) == "standard"

    # 3. Vérification du flux complet process_question
    mock_rag = lambda q: {
        "question": q,
        "answer": dns_answer,
        "sources": [
            {"doc_id": "DOC002", "title": "Commande ipconfig"},
            {"doc_id": "DOC003", "title": "Paramètres du client DNS"},
        ],
        "retrieved_chunks": [],
    }

    res = process_question("Comment vider le cache DNS sous Windows ?", rag_pipeline_fn=mock_rag)
    assert res["response_type"] == "standard"
    assert len(res["sources"]) == 2
    assert res["sources"][0]["doc_id"] == "DOC002"


def test_benign_it_support_terms_do_not_trigger_security_refusal():
    """
    Vérifie que les termes courants de support IT ('administrateur', 'pare-feu',
    'politique', 'droits') dans des contextes normaux ne déclenchent aucun faux positif.
    """
    benign_answers = [
        "Ouvrez un terminal avec les privilèges administrateur (sudo) pour éditer netplan.",
        "Le pare-feu Windows peut être vérifié dans les paramètres de sécurité avancés.",
        "La politique de groupe locale permet de configurer le client DNS via DOC003.",
        "Si vous rencontrez une erreur, demandez à un administrateur système de valider votre port.",
    ]

    for ans in benign_answers:
        assert is_security_refusal(ans) is False, f"Faux positif détecté pour : {ans}"
        assert classify_response(ans) == "standard"


def test_classification_states_are_strictly_mutually_exclusive():
    """
    Vérifie que les 3 états ('standard', 'abstention', 'security_refusal')
    sont rigoureusement étanches et mutuellement exclusifs.
    """
    # Cas 1 : Réponse normale
    normal_ans = "Exécutez ipconfig /flushdns en tant qu'administrateur [DOC002]."
    assert is_insufficient_context(normal_ans) is False
    assert is_security_refusal(normal_ans) is False
    assert classify_response(normal_ans) == "standard"

    # Cas 2 : Abstention
    abstention_ans = INSUFFICIENT_CONTEXT_MESSAGE
    assert is_insufficient_context(abstention_ans) is True
    assert is_security_refusal(abstention_ans) is False
    assert classify_response(abstention_ans) == "abstention"

    # Cas 3 : Refus de sécurité
    security_ans = (
        "Cette demande est refusée pour des raisons de sécurité. "
        "Je ne peux pas vous aider à contourner les règles de sécurité ni obtenir un accès non autorisé."
    )
    assert is_insufficient_context(security_ans) is False
    assert is_security_refusal(security_ans) is True
    assert classify_response(security_ans) == "security_refusal"


# ============================================================
# TESTS DES 3 SCÉNARIOS FONCTIONNELS SPÉCIFIQUES
# ============================================================

def test_functional_photoshop_abstention_has_no_sources():
    """
    Scénario fonctionnel 1 : Installation Photoshop.
    Même si le retriever remonte des chunks non pertinents (DOC007, DOC003, DOC001),
    la réponse d'abstention ne doit comporter AUCUNE source documentaire.
    """
    mock_rag_photoshop = lambda q: {
        "question": q,
        "answer": "La documentation ne contient pas d'informations suffisantes pour installer Photoshop sur Windows.",
        "sources": [
            {"doc_id": "DOC007", "title": "Installing OpenVPN"},
            {"doc_id": "DOC003", "title": "Paramètres du client DNS"},
            {"doc_id": "DOC001", "title": "Résolution des problèmes DNS"},
        ],
        "retrieved_chunks": [],
    }

    result = process_question("Comment installer Photoshop sur Windows ?", rag_pipeline_fn=mock_rag_photoshop)
    assert result["response_type"] == "abstention"
    assert result["sources"] == []


def test_functional_admin_password_refusal_has_no_sources():
    """
    Scénario fonctionnel 2 : Mot de passe administrateur / contournement.
    Même si le retriever remonte des documents (DOC001, DOC003),
    la réponse de refus de sécurité ne doit comporter AUCUNE source documentaire.
    """
    mock_rag_security = lambda q: {
        "question": q,
        "answer": (
            "Cette demande est refusée pour des raisons de sécurité. "
            "Je ne peux pas vous aider à récupérer un mot de passe administrateur ni à obtenir un accès non autorisé."
        ),
        "sources": [
            {"doc_id": "DOC001", "title": "Résolution des problèmes DNS"},
            {"doc_id": "DOC003", "title": "Paramètres du client DNS"},
        ],
        "retrieved_chunks": [],
    }

    result = process_question("Comment récupérer le mot de passe administrateur ?", rag_pipeline_fn=mock_rag_security)
    assert result["response_type"] == "security_refusal"
    assert result["sources"] == []


def test_functional_dns_flush_cache_preserves_doc002_and_doc003():
    """
    Scénario fonctionnel 3 : Vidage du cache DNS sous Windows.
    La réponse normale doit obligatoirement conserver et afficher les sources pertinentes (DOC002 et DOC003).
    """
    mock_rag_dns = lambda q: {
        "question": q,
        "answer": (
            "Pour vider le cache DNS sous Windows, exécutez `ipconfig /flushdns` "
            "dans une invite de commandes en tant qu'administrateur [DOC002]. "
            "Les paramètres du client DNS sont détaillés dans [DOC003]."
        ),
        "sources": [
            {"doc_id": "DOC002", "title": "Commande ipconfig"},
            {"doc_id": "DOC003", "title": "Paramètres du client DNS"},
        ],
        "retrieved_chunks": [],
    }

    result = process_question("Comment vider le cache DNS sous Windows ?", rag_pipeline_fn=mock_rag_dns)
    assert result["response_type"] == "standard"
    assert len(result["sources"]) == 2
    doc_ids = [s["doc_id"] for s in result["sources"]]
    assert "DOC002" in doc_ids
    assert "DOC003" in doc_ids


# ============================================================
# TESTS DES DONNÉES DE CONFIGURATION STATIQUE
# ============================================================

def test_categories_contain_required_scopes():
    category_names = [cat["name"] for cat in CATEGORIES]
    assert "Windows" in category_names
    assert "Ubuntu" in category_names
    assert "OpenVPN" in category_names


def test_example_queries_structure():
    assert len(EXAMPLE_QUERIES) >= 5
    for ex in EXAMPLE_QUERIES:
        assert "category" in ex
        assert "label" in ex
        assert "query" in ex
        assert len(ex["query"].strip()) > 0
