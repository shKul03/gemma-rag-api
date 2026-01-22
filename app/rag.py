import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
CHROMA_DIR = "chroma"
COLLECTION_NAME = "docs"

class RAGService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, query: str, top_k: int = 5,threshold: int = 0.75):
        # Query expansion for robustness
        queries = [
            query,
            query.lower(),
            f"Information about {query}"
        ]

        query_embeddings = self.model.encode(queries).tolist()

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents","distances","metadatas"]
        )

        # documents = results["documents"][0] if results.get("documents") else []
        documents = []
        distances = results.get("distances", [[]])[0]
        docs = results.get("documents", [[]])[0]

        for doc, dist in zip(docs, distances):
            if dist <= threshold:
                documents.append(doc)

        citations = []
        for meta in results["metadatas"][0]:
            citations.append(f'{meta["source"]}')

        return documents, citations

    def format_prompt(self, query: str, context: list[str]) -> str:
        context_str = "\n".join(context)

        prompt = f"""
        You are an HR assistant bot named Bakarwadi.
        You answer employee questions using ONLY official HR policy text.

        Rules:
        - Use ONLY the information from the Context.
        - HR policies may describe rules, eligibility, limits, and procedures.
        - You may paraphrase or summarize policy statements.
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


rag_service = RAGService()
