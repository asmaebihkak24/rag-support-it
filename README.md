# rag-support-it
# Assistant de Support Informatique — RAG & MLOps

Un assistant intelligent de support informatique basé sur le **Retrieval-Augmented Generation (RAG)**. Il répond aux questions techniques en s'appuyant sur une collection de documentations publiques et fournit les sources utilisées pour construire ses réponses.

Ce projet vise à concevoir et industrialiser un pipeline RAG fiable, en intégrant la gestion des données, la recherche documentaire, l'évaluation, le suivi des expériences et les tests automatisés.

## 1. Présentation du projet

L'assistant permet d'accompagner les utilisateurs dans la résolution de problèmes informatiques liés à :

- La configuration réseau et DNS sous Windows.
- La configuration réseau et DNS sous Ubuntu.
- L'installation et le dépannage d'OpenVPN.

Contrairement à un chatbot classique qui génère des réponses uniquement à partir de ses connaissances internes, cet assistant recherche d'abord des informations pertinentes dans une base documentaire, puis utilise les passages récupérés comme contexte pour générer sa réponse.

Le système est conçu pour :

- Fournir des réponses fondées sur la documentation disponible.
- Afficher les sources documentaires utilisées.
- S'abstenir de répondre lorsque le contexte est insuffisant.
- Reconnaître les questions hors périmètre.
- Refuser les demandes impliquant un accès non autorisé ou le contournement des règles de sécurité.

## 2. Architecture du système

```text
                  Question utilisateur
                          |
                          v
                   Pipeline RAG
                          |
                          v
                 Traitement de la requête
                          |
                 +--------+--------+
                 |                 |
                 v                 v
          Recherche vectorielle   Recherche BM25
               Chroma                BM25
                 |                 |
                 +--------+--------+
                          |
                          v
                   Recherche hybride
                          |
                          v
                  Construction du contexte
                          |
                          v
                     Modèle Groq
                          |
                          v
                  Génération de réponse
                          |
                          v
                Réponse + Sources utilisées
```

## 3. Technologies utilisées

| Composant | Technologie |
|---|---|
| Langage | Python 3.11 |
| Modèle d'embeddings | Sentence Transformers |
| Base vectorielle | ChromaDB |
| Recherche lexicale | BM25 |
| Stratégie de recherche | Recherche hybride |
| Expansion des requêtes | Expansion ciblée BIND9 |
| Modèle de langage | Groq API |
| Versionnement des données | DVC |
| Suivi des expériences | MLflow |
| Tests automatisés | Pytest |
| Formats documentaires | Markdown, JSONL, CSV |

## 4. Collecte et préparation des données

La base de connaissances est constituée de **9 documents techniques publics**, provenant notamment de :

- Microsoft Learn.
- La documentation officielle Ubuntu.
- La documentation Netplan.
- La documentation OpenVPN.

Les documents sont téléchargés et extraits, puis préparés avant leur découpage en segments (*chunks*) et leur indexation.

Les métadonnées des documents et leurs informations de provenance sont conservées dans `data/sources.csv`.

## 5. Stratégie de recherche documentaire

Le système utilise une approche de recherche hybride qui combine plusieurs méthodes.

### Recherche vectorielle

La recherche vectorielle repose sur les embeddings pour retrouver les passages sémantiquement proches de la question de l'utilisateur. Les vecteurs sont stockés dans ChromaDB.

### Recherche lexicale BM25

BM25 permet de retrouver les passages en fonction de la correspondance entre les termes de la question et ceux des documents.

### Recherche hybride et expansion des requêtes

Les résultats des recherches vectorielle et lexicale sont combinés afin d'améliorer la pertinence des passages récupérés.

Une expansion ciblée des requêtes liées à BIND9 est également intégrée pour améliorer la recherche sur certaines questions techniques.

## 6. Industrialisation du pipeline MLOps

### 6.1. Gestion des données et du pipeline avec DVC

DVC est utilisé pour suivre les données et définir les différentes étapes du pipeline.

Le pipeline comprend notamment :

- La validation des données.
- L'ingestion et l'indexation des documents.
- L'évaluation de la recherche documentaire.

Vérifier l'état du pipeline :

```bash
dvc status
```

Reproduire le pipeline lorsque cela est nécessaire :

```bash
dvc repro
```

### 6.2. Suivi des expériences avec MLflow

MLflow permet de suivre les expériences d'évaluation de la recherche documentaire, notamment les paramètres et les métriques obtenues avec les différentes stratégies de recherche.

L'interface MLflow peut être lancée avec :

```bash
mlflow ui --host 127.0.0.1 --port 5000 --workers 1
```

Accéder ensuite à :

http://127.0.0.1:5000

### 6.3. Tests automatisés

Des tests unitaires sont développés pour vérifier le fonctionnement des principaux composants :

- Générateur de réponses.
- Moteur de recherche documentaire.
- Pipeline RAG.

Exécuter les tests :

```bash
python -m pytest tests/unit/ -v
```

**Résultat obtenu : 33 tests réussis.**

## 7. Évaluation du système

Le jeu d'évaluation contient **30 questions**, réparties en quatre catégories :

| Catégorie | Nombre de questions |
|---|---:|
| Questions nominales | 21 |
| Contexte insuffisant | 1 |
| Hors périmètre | 6 |
| Sécurité | 2 |
| **Total** | **30** |

Plusieurs stratégies de recherche sont évaluées et comparées :

- Recherche vectorielle.
- Recherche hybride.
- Recherche hybride avec expansion des requêtes.

L'évaluation de bout en bout examine également le comportement de génération, la pertinence des sources et la capacité du système à s'abstenir lorsque cela est nécessaire.

Les métriques de recherche évaluent la capacité à retrouver les documents pertinents. Elles ne constituent pas, à elles seules, une preuve de l'exactitude factuelle des réponses générées.

## 8. Structure du projet

```text
rag-support-it/
│
├── configs/                      # Fichiers de configuration
│
├── data/
│   ├── raw/                      # Documents sources
│   ├── raw.dvc                   # Suivi du corpus avec DVC
│   └── sources.csv               # Métadonnées des documents
│
├── eval/
│   ├── golden_set.jsonl          # Jeu de questions de référence
│   ├── eval_retrieval.py         # Évaluation de la recherche
│   ├── eval_end_to_end.py        # Évaluation de bout en bout
│   ├── compare_retrieval.py      # Comparaison des méthodes
│   └── results/                  # Résultats des évaluations
│
├── indexes/                      # Index vectoriel local
│
├── reports/                      # Rapports de validation
│
├── src/
│   ├── download_docs.py          # Téléchargement des documents
│   ├── validate_data.py           # Validation du corpus
│   ├── retriever.py              # Recherche documentaire
│   ├── generator.py              # Génération des réponses
│   └── rag_pipeline.py           # Pipeline principal
│
├── tests/
│   └── unit/                     # Tests unitaires
│
├── dvc.yaml                      # Définition du pipeline DVC
├── dvc.lock                      # État des étapes DVC
├── .env.example                  # Exemple de configuration
└── README.md
```

## 9. Installation et exécution

### Prérequis

- Python 3.11.
- Git.
- Une clé API Groq.

### Configuration de l'environnement

Créer un fichier `.env` à partir du modèle fourni :

```powershell
copy .env.example .env
```

Configurer ensuite la clé API Groq et le modèle utilisé dans ce fichier.

**Important :** le fichier `.env` contenant les informations confidentielles ne doit jamais être publié sur GitHub.

### Création de l'environnement virtuel

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Installer les dépendances à partir du fichier de dépendances du projet.

### Lancement de l'assistant

Depuis la racine du projet :

```bash
python -m src.rag_pipeline
```

## 10. Limites et perspectives

### Limites actuelles

- Les connaissances de l'assistant sont limitées aux documents indexés.
- La qualité des réponses dépend de la pertinence des passages récupérés.
- Le jeu d'évaluation reste de taille limitée.
- Le stockage DVC est actuellement local : aucun remote DVC n'est configuré.

### Améliorations possibles

- Enrichir la base documentaire avec de nouvelles sources techniques.
- Étendre le jeu d'évaluation.
- Renforcer l'évaluation de la fidélité des réponses aux documents.
- Ajouter un mécanisme de monitoring continu de la qualité.
- Déployer l'assistant avec une interface utilisateur.
- Configurer un stockage distant pour DVC afin de faciliter la reproductibilité.



---

*Projet académique — Industrialisation d'un assistant de support informatique basé sur le Retrieval-Augmented Generation (RAG) et les pratiques MLOps.*
