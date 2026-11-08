
# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5

INSUFFICIENT_CONTEXT_MESSAGE = (
    "Je ne dispose pas d'informations suffisantes "
    "dans la documentation pour répondre à cette question."
)


# ============================================================
# CONSTRUCTION DU CONTEXTE
# ============================================================

def build_context(retrieved_chunks):
    """
    Construit le contexte documentaire envoyé au LLM.
    Les identifiants de chunks et les scores ne sont pas transmis.
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
# EXTRACTION DES SOURCES ET CHUNKS
# ============================================================

def extract_sources(retrieved_chunks):
    """
    Extrait les documents uniques récupérés
    et les détails des chunks correspondants.
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

    return list(sources.values()), retrieved_chunk_details


# ============================================================
# PIPELINE RAG COMPLET
# ============================================================

def run_rag(question, k=TOP_K, retriever=None, generator=None):
    """
    Exécute le pipeline RAG complet.

    1. Recherche hybride.
    2. Construction du contexte.
    3. Génération de la réponse.
    4. Extraction des documents et chunks récupérés.

    Les dépendances peuvent être injectées pour faciliter les tests.
    """

    if not question or not question.strip():
        raise ValueError("La question ne peut pas être vide.")

    # Chargement des dépendances uniquement si nécessaire
    if retriever is None:
        from src.retriever import search_documents_hybrid
        retriever = search_documents_hybrid

    if generator is None:
        from src.generator import generate_answer
        generator = generate_answer

    # 1. Recherche hybride
    retrieved_chunks = retriever(question, k=k)

    # 2. Aucun résultat
    if not retrieved_chunks:
        return {
            "question": question,
            "answer": INSUFFICIENT_CONTEXT_MESSAGE,
            "sources": [],
            "retrieved_chunks": [],
        }

    # 3. Construction du contexte
    context = build_context(retrieved_chunks)

    # 4. Génération de la réponse
    answer = generator(
        question=question,
        context=context
    )

    # 5. Extraction des documents et chunks récupérés
    sources, chunk_details = extract_sources(retrieved_chunks)

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
        "Comment contourner les règles de sécurité du réseau de l'entreprise ?"
    ]

    for question in questions:
        result = run_rag(question)

        print("\n" + "=" * 70)
        print("QUESTION :", question)
        print("=" * 70)
        print("RÉPONSE :", result["answer"])
        print("SOURCES :", result["sources"])
