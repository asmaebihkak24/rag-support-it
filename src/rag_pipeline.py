
import re


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5

INSUFFICIENT_CONTEXT_MESSAGE = (
    "Je ne dispose pas d'informations suffisantes "
    "dans la documentation pour répondre à cette question."
)


# ============================================================
# MOTIFS D'ABSTENTION
# ============================================================

ABSTENTION_PATTERNS = [
    re.compile(
        r"ne\s+dispose\s+pas\s+d['’]informations?\s+suffisantes",
        re.IGNORECASE,
    ),
    re.compile(
        r"informations?\s+(?:sont\s+)?insuffisantes",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:documents?|documentation)\s+"
        r"(?:fournis?|disponibles?)?\s*"
        r"(?:\([^)]*\))?\s*"
        r"ne\s+(?:contiennent?|contient|fournissent?|fournit)\s+"
        r"(?:aucune?|pas\s+d['’])\s*information",
        re.IGNORECASE,
    ),
    re.compile(
        r"aucun(?:e)?\s+information\s+"
        r"(?:disponible\s+)?dans\s+la\s+documentation",
        re.IGNORECASE,
    ),
    re.compile(
        r"absence\s+de\s+(?:références?|informations?)\s+documentaires?",
        re.IGNORECASE,
    ),
    re.compile(
        r"ne\s+peux\s+pas\s+fournir\s+de\s+procédure\s+fiable",
        re.IGNORECASE,
    ),
    re.compile(
        r"ne\s+dispose\s+donc\s+pas\s+de\s+données\s+suffisantes",
        re.IGNORECASE,
    ),
    re.compile(
        r"je\s+ne\s+peux\s+donc\s+pas\s+vous\s+fournir",
        re.IGNORECASE,
    ),
]


def is_abstention(answer: str) -> bool:
    """Détecte une réponse indiquant que le corpus est insuffisant."""

    if not answer:
        return False

    clean_answer = answer.strip()

    if INSUFFICIENT_CONTEXT_MESSAGE.lower() in clean_answer.lower():
        return True

    return any(
        pattern.search(clean_answer)
        for pattern in ABSTENTION_PATTERNS
    )


# ============================================================
# MOTIFS DE REFUS DE SECURITE
# ============================================================

SECURITY_REFUSAL_PATTERNS = [
    re.compile(
        r"\b(?:refus\s+de\s+s[eé]curit[eé]|"
        r"demande\s+(?:est\s+)?refus[eé]e)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cette\s+demande|votre\s+demande)\s+ne\s+peut\s+"
        r"(?:pas\s+)?[eê]tre\s+"
        r"(?:trait[eé]e|honor[eé]e|accept[eé]e)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ne\s+peux\s+pas|ne\s+suis\s+pas\s+autoris[eé]\s+à|"
        r"impossible\s+de)\s+(?:vous\s+)?(?:aider\s+à\s+)?"
        r"(?:contourn\w+|pirater|cracker|craquer|"
        r"d[eé]sactiver|r[eé]v[eé]ler)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ne\s+peux\s+pas|ne\s+suis\s+pas\s+autoris[eé]\s+à)\s+"
        r"(?:vous\s+)?(?:fournir|donner|r[eé]cup[eé]rer)\s+"
        r"(?:de\s+|d'|des\s+|cette\s+)?"
        r"(?:mot[s]?\s+de\s+passe|identifiant[s]?|token[s]?|"
        r"acc[eè]s\s+non\s+autoris[eé]|information)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:refus|refus[eé]|interdit|prohib[eé]|ne\s+peux\s+pas)\b"
        r".*\bacc[eè]s\s+non\s+autoris[eé]\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bacc[eè]s\s+non\s+autoris[eé]\b"
        r".*\b(?:refus|refus[eé]|interdit|prohib[eé]|ne\s+peux\s+pas)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bje\s+(?:suis\s+d[eé]sol[eé],?\s+)?"
        r"ne\s+peux\s+pas\s+(?:vous\s+)?aider"
        r"(?:\s+avec\s+cette\s+demande)?\b",
        re.IGNORECASE,
    ),
]


def is_security_refusal(answer: str) -> bool:
    """Détecte un refus lié à une demande non autorisée ou risquée."""

    if not answer:
        return False

    return any(
        pattern.search(answer)
        for pattern in SECURITY_REFUSAL_PATTERNS
    )


# ============================================================
# CLASSIFICATION DES REPONSES
# ============================================================

def classify_response(answer: str) -> str:
    """
    Catégories :
    - abstention
    - security_refusal
    - standard
    """

    if is_security_refusal(answer):
        return "security_refusal"

    if is_abstention(answer):
        return "abstention"

    return "standard"


# ============================================================
# CONSTRUCTION DU CONTEXTE
# ============================================================

def build_context(retrieved_chunks):
    """
    Construit le contexte documentaire envoyé au LLM.

    Format attendu :
    [(Document, score), ...]
    """

    context_parts = []

    for doc, score in retrieved_chunks:
        metadata = doc.metadata

        doc_id = metadata.get("doc_id", "UNKNOWN")
        title = metadata.get("title", "Titre inconnu")
        section = metadata.get("section", "Section inconnue")

        context_parts.append(
            f"""
[Source : {doc_id}]
Titre : {title}
Section : {section}

Contenu :
{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# EXTRACTION DES SOURCES ET DES CHUNKS
# ============================================================

def extract_sources(retrieved_chunks, answer=None):
    """
    Retourne :
        sources, retrieved_chunk_details

    Les sources affichées sont limitées aux documents cités
    dans une réponse standard.
    """

    sources = {}
    retrieved_chunk_details = []

    for doc, score in retrieved_chunks:
        metadata = doc.metadata
        doc_id = metadata.get("doc_id", "UNKNOWN")

        if doc_id not in sources:
            sources[doc_id] = {
                "doc_id": doc_id,
                "title": metadata.get("title"),
                "organization": metadata.get("organization"),
                "os_scope": metadata.get("os_scope"),
                "source_url": metadata.get("source_url"),
                "filename": metadata.get("filename"),
            }

        retrieved_chunk_details.append({
            "doc_id": doc_id,
            "chunk_id": metadata.get("chunk_id"),
            "section": metadata.get("section"),
            "score_rrf": round(float(score), 6),
        })

    # Sans réponse générée : retourner les documents récupérés.
    if answer is None:
        return list(sources.values()), retrieved_chunk_details

    response_type = classify_response(answer)

    # Aucune source pour une abstention ou un refus.
    if response_type in ("abstention", "security_refusal"):
        return [], retrieved_chunk_details

    # Extraction des identifiants DOCxxx.
    cited_doc_ids = {
        doc_id.upper()
        for doc_id in re.findall(
            r"\bDOC\d{3}\b",
            answer,
            re.IGNORECASE,
        )
    }

    cited_sources = [
        source
        for doc_id, source in sources.items()
        if doc_id.upper() in cited_doc_ids
    ]

    return cited_sources, retrieved_chunk_details


# ============================================================
# PIPELINE RAG COMPLET
# ============================================================

def run_rag(question, k=TOP_K, retriever=None, generator=None):
    """
    Exécute le pipeline RAG :

    1. Recherche hybride.
    2. Construction du contexte.
    3. Génération.
    4. Classification.
    5. Extraction des sources et chunks.
    """

    if not question or not question.strip():
        raise ValueError("La question ne peut pas être vide.")

    # Chargement du retriever.
    if retriever is None:
        from src.retriever import search_documents_hybrid
        retriever = search_documents_hybrid

    # Chargement du générateur.
    if generator is None:
        from src.generator import generate_answer
        generator = generate_answer

    # 1. Recherche hybride.
    retrieved_chunks = retriever(question, k=k)

    # 2. Aucun résultat.
    if not retrieved_chunks:
        return {
            "question": question,
            "answer": INSUFFICIENT_CONTEXT_MESSAGE,
            "sources": [],
            "retrieved_chunks": [],
        }

    # 3. Construction du contexte.
    context = build_context(retrieved_chunks)

    # 4. Génération.
    answer = generator(
        question=question,
        context=context,
    )

    # 5. Classification et extraction.
    response_type = classify_response(answer)

    sources, chunk_details = extract_sources(
        retrieved_chunks,
        answer=answer,
    )

    # 6. Diagnostics.
    print(
        "[DIAGNOSTIC] Chunks récupérés :",
        [doc.metadata.get("doc_id") for doc, score in retrieved_chunks],
        flush=True,
    )

    print(
        "[DIAGNOSTIC] Type de réponse :",
        response_type,
        flush=True,
    )

    print(
        "[DIAGNOSTIC] Sources retournées :",
        [source["doc_id"] for source in sources],
        flush=True,
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunk_details,
    }


# ============================================================
# TEST MANUEL
# ============================================================

if __name__ == "__main__":

    questions = [
        "Comment vider le cache DNS sous Windows ?",
        "Comment installer Photoshop sur Windows ?",
        "Comment récupérer un mot de passe Wi-Fi oublié ?",
        "Comment désactiver le pare-feu Windows pour résoudre un problème de connexion ?",
        "Comment contourner les règles de sécurité du réseau de l'entreprise ?",
    ]

    for question in questions:

        result = run_rag(question)

        print("\n" + "=" * 70)
        print("QUESTION :", question)
        print("=" * 70)

        print("RÉPONSE :", result["answer"])
        print("SOURCES :", result["sources"])
