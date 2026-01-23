import chromadb
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
CHROMA_DIR = "chroma"
COLLECTION_NAME = "docs"


class RAGService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------------------------------------------------------
    # SEMANTIC RETRIEVAL (PRIMARY)
    # ---------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[List[str], List[str]]:
        """
        Semantic retrieval using embeddings.
        No hard distance threshold (top_k controls relevance).
        """

        logger.info(f"Semantic retrieval for query: {query}")

        # Light query expansion (important for cases like "orange" → "food")
        queries = [
            query,
            f"policy about {query}",
            f"workplace rules regarding {query}",
        ]

        query_embeddings = self.model.encode(queries).tolist()

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        documents = []
        citations = set()

        # Merge results from all expanded queries
        for docs, metas in zip(
            results.get("documents", []),
            results.get("metadatas", [])
        ):
            for doc, meta in zip(docs, metas):
                documents.append(doc)
                citations.add(meta.get("source", "unknown"))

        logger.info(f"Retrieved {len(documents)} documents")
        return documents, list(citations)

    # ---------------------------------------------------------
    # KEYWORD FALLBACK (SECONDARY SAFETY NET)
    # ---------------------------------------------------------
    def keyword_fallback(
        self,
        keywords: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Simple keyword-based fallback to avoid empty context.
        Used ONLY if semantic retrieval returns nothing.
        """

        logger.info(f"Keyword fallback triggered: {keywords}")

        results = self.collection.get(
            include=["documents", "metadatas"],
            limit=10_000
        )

        documents = []
        citations = set()

        for doc, meta in zip(
            results.get("documents", []),
            results.get("metadatas", [])
        ):
            doc_lower = doc.lower()
            if any(keyword in doc_lower for keyword in keywords):
                documents.append(doc)
                citations.add(meta.get("source", "unknown"))

        logger.info(f"Keyword fallback found {len(documents)} documents")
        return documents, list(citations)

    # ---------------------------------------------------------
    # FULL CORPUS (EVALUATION / DEBUG ONLY)
    # ---------------------------------------------------------
    def retrieve_all_documents(self) -> Tuple[List[str], List[str]]:
        """
        Returns the entire corpus.
        Use ONLY for debugging or evaluation.
        """

        logger.warning("FULL CORPUS RETRIEVAL — NOT FOR PRODUCTION")

        results = self.collection.get(
            include=["documents", "metadatas"],
            limit=10_000
        )

        documents = results.get("documents", [])
        citations = {
            meta.get("source", "unknown")
            for meta in results.get("metadatas", [])
        }

        return documents, list(citations)

    # ---------------------------------------------------------
    # ORCHESTRATOR (RECOMMENDED ENTRY POINT)
    # ---------------------------------------------------------
    def get_context(
        self,
        query: str
    ) -> Tuple[List[str], List[str]]:
        """
        Production-safe context retrieval:
        1. Semantic retrieval
        2. Keyword fallback (if empty)
        """

        documents, citations = self.retrieve(query)

        if documents:
            return documents, citations

        # Very small, controlled fallback keyword set
        fallback_keywords = [
            "food",
            "snack",
            "eat",
            "consume",
            "beverage",
        ]

        return self.keyword_fallback(fallback_keywords)

    # ---------------------------------------------------------
    # PROMPT FORMATTING
    # ---------------------------------------------------------
    def format_prompt(self, query: str, context: List[str]) -> str:
        context_str = "\n\n---\n\n".join(context)

        prompt = f"""
You are an HR assistant bot named Bakarwadi.
You answer employee questions using ONLY official HR policy text.

Rules:
- Use ONLY the information from the Context.
- Do NOT infer intent or give advice.
- If the Context does not contain sufficient policy information, respond EXACTLY with:
"Sorry, I can not answer that. Please reach out to your assigned HR employee!"

Context:
{context_str}

Question:
{query}

Answer (grounded strictly in context):
"""
        return prompt


# ---------------------------------------------------------
# USAGE
# ---------------------------------------------------------
rag_service = RAGService()
