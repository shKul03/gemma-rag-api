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

    def retrieve(self, query: str, top_k: int = 5):
        # Query expansion for robustness
        queries = [
            query,
            query.lower(),
            f"Information about {query}"
        ]

        query_embeddings = self.model.encode(queries).tolist()

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )

        documents = results["documents"][0] if results.get("documents") else []
        return documents

    def format_prompt(self, query: str, context: list[str]) -> str:
        context_str = "\n".join(context)

        prompt = f"""
        You are a document-grounded AI assistant.
    
        STRICT RULES (you must follow ALL):
        - You are an HR Assistant bot.
        - Your name is Bakarwadi.
        - Use ONLY the information explicitly stated in the Context.
        - Do NOT use prior knowledge, assumptions, or reasoning outside the Context.
        - If the answer is NOT explicitly present in the Context, respond EXACTLY with:
        "Sorry, I can not answer that. Please reach out to your assigned HR employee!"
        - Do NOT add explanations, suggestions, or extra text.
        - Do NOT partially answer.
    
        Context:
        {context_str}
    
        Question:
        {query}
    
        Answer (grounded strictly in context):
        """
        return prompt


rag_service = RAGService()
