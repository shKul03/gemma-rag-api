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
        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{context_str}

User Question: {query}

Answer:"""
        return prompt

rag_service = RAGService()
