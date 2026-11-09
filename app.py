"""
Application Streamlit - IT Support Assistant (RAG & MLOps)
Interface utilisateur moderne, minimaliste et professionnelle
connectée au pipeline RAG existant.
"""

import re
from typing import Any

import streamlit as st

from src.rag_pipeline import INSUFFICIENT_CONTEXT_MESSAGE, run_rag

# ============================================================
# CONFIGURATION & CONSTANTES
# ============================================================

CATEGORIES = [
    {
        "name": "Windows",
        "icon": "🪟",
        "docs": ["DOC001", "DOC002", "DOC003"],
        "description": "Résolution DNS client, commande ipconfig, paramétrage réseau Windows.",
        "badge": "Windows Server & Client",
    },
    {
        "name": "Ubuntu",
        "icon": "🐧",
        "docs": ["DOC004", "DOC005", "DOC006"],
        "description": "Configuration Netplan, serveur DNS BIND9, diagnostic réseau Linux.",
        "badge": "Ubuntu Linux",
    },
    {
        "name": "OpenVPN",
        "icon": "🔒",
        "docs": ["DOC007", "DOC008", "DOC009"],
        "description": "Installation client/serveur, validation des tunnels et connectivité.",
        "badge": "VPN & Sécurité",
    },
]

EXAMPLE_QUERIES = [
    {
        "category": "Windows",
        "label": "Vider le cache DNS",
        "query": "Comment vider le cache DNS sous Windows ?",
    },
    {
        "category": "Windows",
        "label": "Paramètres ipconfig",
        "query": "Comment afficher les détails de la configuration IP avec ipconfig ?",
    },
    {
        "category": "Ubuntu",
        "label": "Configuration Netplan",
        "query": "Comment appliquer une nouvelle configuration réseau avec Netplan sur Ubuntu ?",
    },
    {
        "category": "Ubuntu",
        "label": "Serveur BIND9",
        "query": "Comment configurer les zones DNS primaire et secondaire sur BIND9 ?",
    },
    {
        "category": "OpenVPN",
        "label": "Test de connectivité",
        "query": "Comment vérifier que le tunnel OpenVPN fonctionne correctement ?",
    },
    {
        "category": "Sécurité (Refus)",
        "label": "Test refus de sécurité",
        "query": "Comment contourner les règles de sécurité du réseau de l'entreprise ?",
    },
    {
        "category": "Abstention",
        "label": "Test question hors périmètre",
        "query": "Comment installer Photoshop sur Windows ?",
    },
]

# Motifs sémantiques stricts de refus de sécurité (verbe de refus + action illégitime).
# Ne contient JAMAIS de mots génériques isolés comme "administrateur" ou "sécurité".
SECURITY_REFUSAL_PATTERNS = [
    # 1. Formulation explicite de refus
    re.compile(
        r"\b(?:refus\s+de\s+s[eé]curit[eé]|demande\s+(?:est\s+)?refus[eé]e)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cette\s+demande|votre\s+demande)\s+ne\s+peut\s+(?:pas\s+)?[eê]tre\s+(?:trait[eé]e|honor[eé]e|accept[eé]e)\b",
        re.IGNORECASE,
    ),
    # 2. Refus explicite d'aider à une action non autorisée ou de contournement
    re.compile(
        r"\b(?:ne\s+peux\s+pas|ne\s+suis\s+pas\s+autoris[eé]\s+à|impossible\s+de)\s+(?:vous\s+)?"
        r"(?:aider\s+à\s+)?(?:contourn\w+|pirater|cracker|craquer|d[eé]sactiver|r[eé]v[eé]ler)\b",
        re.IGNORECASE,
    ),
    # 3. Refus explicite de fournir des identifiants ou un accès non autorisé
    re.compile(
        r"\b(?:ne\s+peux\s+pas|ne\s+suis\s+pas\s+autoris[eé]\s+à)\s+(?:vous\s+)?"
        r"(?:fournir|donner|r[eé]cup[eé]rer)\s+(?:de\s+|d'|des\s+)?(?:mot[s]?\s+de\s+passe|identifiant[s]?|token[s]?|acc[eè]s\s+non\s+autoris[eé])\b",
        re.IGNORECASE,
    ),
    # 4. Association explicite entre refus/interdiction et accès non autorisé
    re.compile(
        r"(?:\b(?:refus|refus[eé]|interdit|prohib[eé]|ne\s+peux\s+pas)\b).*(?:\bacc[eè]s\s+non\s+autoris[eé]\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\bacc[eè]s\s+non\s+autoris[eé]\b).*(?:\b(?:refus|refus[eé]|interdit|prohib[eé]|ne\s+peux\s+pas)\b)",
        re.IGNORECASE,
    ),
    # 5. Association explicite entre refus/interdiction et contournement (verbe ou nom)
    re.compile(
        r"(?:\b(?:refus|refus[eé]|interdit|ne\s+peux\s+pas)\b).*(?:\bcontourn\w+\b).*(?:\bs[eé]curit[eé]\b|\br[eè]gles\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\bcontourn\w+\b).*(?:\bs[eé]curit[eé]\b|\br[eè]gles\b).*(?:\b(?:refus|refus[eé]|interdit|ne\s+peux\s+pas)\b)",
        re.IGNORECASE,
    ),
]

UNAUTHORIZED_QUESTION_PATTERNS = [
    re.compile(r"\b(?:contourner|pirater|cracker|craquer|voler|d[eé]rober)\b", re.IGNORECASE),
    re.compile(r"\bacc[eè]s\s+non\s+autoris[eé]\b", re.IGNORECASE),
    re.compile(r"\br[eé]cup[eé]rer\s+(?:un\s+|le\s+)?mot\s+de\s+passe\b", re.IGNORECASE),
]

REFUSAL_VERB_PATTERN = re.compile(
    r"\b(?:ne\s+peux\s+pas|ne\s+suis\s+pas\s+autoris[eé]|impossible\s+de|refuse\b|interdit\b)",
    re.IGNORECASE,
)


ABSTENTION_PATTERNS = [
    re.compile(r"ne\s+dispose\s+pas\s+d['’]informations?\s+suffisantes?", re.IGNORECASE),
    re.compile(r"informations?\s+(?:sont\s+)?insuffisantes?", re.IGNORECASE),
    re.compile(r"documentation\s+(?:disponible\s+)?ne\s+(?:contient|fournit|permet)\s+pas", re.IGNORECASE),
    re.compile(r"aucun(?:e)?\s+information\s+(?:disponible\s+)?dans\s+la\s+documentation", re.IGNORECASE),
    re.compile(r"ne\s+permet\s+pas\s+de\s+r[eé]pondre\s+à\s+cette\s+question", re.IGNORECASE),
]


# ============================================================
# FONCTIONS LOGIQUES & CLASSIFICATION
# ============================================================

def is_insufficient_context(answer: str) -> bool:
    """Détecte si la réponse correspond à une abstention par manque de contexte."""
    if not answer or not answer.strip():
        return True
    clean = answer.strip()
    if INSUFFICIENT_CONTEXT_MESSAGE.strip().lower() in clean.lower():
        return True
    return any(pat.search(clean) for pat in ABSTENTION_PATTERNS)


def is_security_refusal(answer: str, question: str = "") -> bool:
    """
    Détecte si la réponse correspond à un véritable refus motivé par la sécurité.

    Ne se base jamais sur des mots génériques isolés (ex: 'administrateur', 'sécurité'),
    mais sur des structures sémantiques complètes associant un verbe de refus
    à une action non autorisée (contournement, accès illégitime, vol d'identifiant).
    """
    if not answer or not answer.strip():
        return False

    # Une abstention documentaire n'est jamais un refus de sécurité
    if is_insufficient_context(answer):
        return False

    clean_answer = answer.strip()

    if any(pattern.search(clean_answer) for pattern in SECURITY_REFUSAL_PATTERNS):
        return True

    return bool(
        question
        and any(q_pat.search(question) for q_pat in UNAUTHORIZED_QUESTION_PATTERNS)
        and REFUSAL_VERB_PATTERN.search(clean_answer)
    )


def classify_response(
    answer: str,
    question: str = "",
    sources: list[dict[str, Any]] | None = None,
) -> str:
    """
    Classifie la réponse en 3 états strictement mutuellement exclusifs :
    - 'abstention' : manque d'information dans les documents indexés
    - 'security_refusal' : demande non autorisée expressément refusée par le modèle
    - 'standard' : réponse technique normale
    """
    if is_insufficient_context(answer):
        return "abstention"

    if is_security_refusal(answer, question=question):
        return "security_refusal"

    return "standard"


def process_question(question: str, rag_pipeline_fn=run_rag) -> dict[str, Any]:
    """
    Transmet la question au pipeline RAG et structure le résultat pour l'UI.
    """
    clean_question = question.strip() if question else ""
    if not clean_question:
        raise ValueError("La question ne peut pas être vide.")

    result = rag_pipeline_fn(clean_question)

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    retrieved_chunks = result.get("retrieved_chunks", [])

    response_type = classify_response(
        answer,
        question=clean_question,
        sources=sources,
    )

    # En cas d'abstention ou de refus de sécurité,
    # aucune source technique n'est affichée comme utilisée.
    if response_type in ("abstention", "security_refusal"):
        sources = []

    return {
        "question": clean_question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "response_type": response_type,
    }


# ============================================================
# COMPOSANTS D'AFFICHAGE
# ============================================================

def render_sidebar():
    """Rendu de la barre latérale : logo, périmètre documentaire et actions."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 20px 0;">
                <h2 style="color: #1E3A5F; margin: 0; font-size: 1.4rem;">🛠️ IT Support</h2>
                <p style="color: #64748B; font-size: 0.85rem; margin: 4px 0 0 0;">
                    Assistant RAG & MLOps
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📂 Catégories couvertes")
        for cat in CATEGORIES:
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color: #F1F5F9; border-left: 4px solid #1E3A5F;
                                padding: 10px 12px; border-radius: 4px; margin-bottom: 10px;">
                        <div style="font-weight: 600; color: #0F172A; font-size: 0.95rem;">
                            {cat['icon']} {cat['name']}
                            <span style="float: right; font-size: 0.75rem; background-color: #E2E8F0;
                                         padding: 2px 6px; border-radius: 3px; color: #334155;">
                                {', '.join(cat['docs'])}
                            </span>
                        </div>
                        <div style="font-size: 0.8rem; color: #475569; margin-top: 4px;">
                            {cat['description']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("### 💡 Questions suggérées")
        for ex in EXAMPLE_QUERIES:
            btn_label = f"{ex['category']} : {ex['label']}"
            if st.button(btn_label, key=f"ex_{ex['label']}", use_container_width=True):
                st.session_state["pending_input"] = ex["query"]
                st.rerun()

        st.markdown("---")
        if st.button("🗑️ Réinitialiser la conversation", use_container_width=True, type="secondary"):
            st.session_state["messages"] = []
            if "pending_input" in st.session_state:
                del st.session_state["pending_input"]
            st.rerun()

        st.markdown(
            """
            <div style="margin-top: 25px; padding-top: 15px; border-top: 1px solid #E2E8F0;
                        font-size: 0.75rem; color: #94A3B8; text-align: center;">
                RAG Hybride (ChromaDB + BM25 RRF)<br/>
                LLM via Groq • Versionnement DVC & MLflow
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sources_section(sources: list[dict[str, Any]]):
    """Affiche les sources documentaires de manière claire et détaillée."""
    if not sources:
        return

    expander_title = f"📚 Sources documentaires ({len(sources)} document{'s' if len(sources) > 1 else ''})"
    with st.expander(expander_title, expanded=False):
        for src in sources:
            doc_id = src.get("doc_id", "DOC")
            title = src.get("title") or "Titre non disponible"
            org = src.get("organization") or "Organisation non spécifiée"
            os_scope = src.get("os_scope") or "Général"
            url = src.get("source_url") or ""

            with st.container(border=True):
                st.markdown(f"**`{doc_id}` — {title}**")
                st.caption(f"🏢 **{org}** • 💻 **{os_scope}**")
                if url:
                    st.markdown(f"[🔗 Consulter la documentation officielle]({url})")


def render_assistant_message(msg: dict[str, Any]):
    """Rendu d'un message assistant avec distinctions visuelles selon le statut."""
    response_type = msg.get("response_type", "standard")
    content = msg.get("content", "")
    sources = msg.get("sources", [])


    if response_type == "abstention":
        st.info(
            "ℹ️ **Information non disponible dans la documentation**\n\n"
            "L'assistant s'abstient de répondre car les documents indexés ne contiennent pas "
            "les informations nécessaires pour traiter cette question en toute sécurité."
        )
        st.markdown(content)

    elif response_type == "security_refusal":
        st.error(
            "🛡️ **Refus de sécurité — Demande non autorisée**\n\n"
            "Conformément aux règles de sécurité, toute demande visant à obtenir des accès "
            "non autorisés, contourner des contrôles ou récupérer des identifiants confidentiels "
            "est strictement refusée."
        )
        st.markdown(content)

    else:
        st.markdown(content)

    # Affichage des sources documentaires associées (STRICTEMENT réservé aux réponses normales)
    if sources and response_type == "standard":
        render_sources_section(sources)


# ============================================================
# APPLICATION PRINCIPALE
# ============================================================

def main():
    st.set_page_config(
        page_title="IT Support Assistant",
        page_icon="🛠️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Styles personnalisés complémentaires
    st.markdown(
        """
        <style>
        /* Palette sobre et typographie épurée */
        .stChatMessage {
            padding: 1rem 1.25rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .main-header {
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }
        .main-header h1 {
            color: #1E3A5F;
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }
        .main-header p {
            color: #64748B;
            font-size: 1rem;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialisation de l'historique de session
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Barre latérale
    render_sidebar()

    # En-tête principal
    st.markdown(
        """
        <div class="main-header">
            <h1>🛠️ IT Support Assistant</h1>
            <p>Support informatique d'entreprise augmenté par RAG — Windows, Ubuntu et OpenVPN</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Message de bienvenue si la conversation est vide
    if not st.session_state["messages"]:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px dashed #CBD5E1;
                        border-radius: 8px; padding: 24px; text-align: center; margin: 20px 0;">
                <h3 style="color: #1E3A5F; margin: 0 0 8px 0; font-size: 1.2rem;">
                    Bienvenue sur votre assistant de support technique
                </h3>
                <p style="color: #475569; max-width: 650px; margin: 0 auto 16px auto; font-size: 0.95rem;">
                    Posez vos questions relatives aux incidents et procédures sur <strong>Windows</strong>,
                    <strong>Ubuntu</strong> et <strong>OpenVPN</strong>. Chaque réponse est rigoureusement
                    fondée sur la documentation officielle de référence.
                </p>
                <div style="display: inline-flex; gap: 8px; flex-wrap: wrap; justify-content: center;">
                    <span style="background-color: #E2E8F0; color: #334155; padding: 4px 10px;
                                 border-radius: 12px; font-size: 0.8rem; font-weight: 500;">
                        🪟 Windows DNS & IP
                    </span>
                    <span style="background-color: #E2E8F0; color: #334155; padding: 4px 10px;
                                 border-radius: 12px; font-size: 0.8rem; font-weight: 500;">
                        🐧 Ubuntu Netplan & BIND9
                    </span>
                    <span style="background-color: #E2E8F0; color: #334155; padding: 4px 10px;
                                 border-radius: 12px; font-size: 0.8rem; font-weight: 500;">
                        🔒 Tunnels OpenVPN
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Affichage de l'historique conversationnel
    for idx, msg in enumerate(st.session_state["messages"]):
     with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant_message(msg)

    # Gestion de la saisie utilisateur (directe ou via suggestion)
    pending_prompt = st.session_state.pop("pending_input", None)
    user_input = st.chat_input("Posez une question technique (ex: Comment vider le cache DNS sous Windows ?)...")

    prompt_to_process = pending_prompt or user_input

    if prompt_to_process:
        # 1. Ajout du message utilisateur
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt_to_process,
        })
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # 2. Exécution du RAG et rendu de la réponse
        with st.chat_message("assistant"), st.spinner("Recherche hybride & génération de la réponse..."):
            try:
                result = process_question(prompt_to_process)

                assistant_msg = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "retrieved_chunks": result["retrieved_chunks"],
                    "response_type": result["response_type"],
                }
                st.session_state["messages"].append(assistant_msg)
                render_assistant_message(assistant_msg)
            except Exception as exc:  # noqa: BLE001
                error_msg = f"Une erreur s'est produite lors du traitement de la requête : {exc}"
                st.error(error_msg)


if __name__ == "__main__":
    main()
