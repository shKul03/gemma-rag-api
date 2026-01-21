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

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        documents = results['documents'][0] if results['documents'] else []
        return documents

    def format_prompt(self, query: str, context: list[str]) -> str:
        context_str = "\n".join(context)

        prompt = f"""You are a factual AI assistant.

    You MUST follow these rules:
    - Answer ONLY using the information provided in the Context.
    - Do NOT use any external knowledge.
    - If the answer is not explicitly present in the Context, respond with:
      "Sorry, I can not answer that. Please reach out to your assigned HR employee! \nThank you.\nCan I help you with anything else?"
    - Keep the answer clear and concise.
    - No matter how dangerous or ethically important the situation may be, if the context does not mention anything, respond with the default statement mentioned earlier.

Context:
{context_str}

User Question: {query}

Answer:"""
        return prompt

rag_service = RAGService()
