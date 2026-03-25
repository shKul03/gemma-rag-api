# Fix sqlite3 version issue for ChromaDB
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

    def retrieve_context(self, query: str, top_k: int = 5, return_metadata: bool = False):
        """
        Retrieve relevant context chunks for a query
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            return_metadata: If True, returns full chunk info with metadata and scores
        
        Returns:
            If return_metadata=False: List of document texts (backward compatible)
            If return_metadata=True: List of dicts with {text, metadata, relevance_score}
        """
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        documents = results['documents'][0] if results['documents'] else []
        
        # Backward compatible: return just text by default
        if not return_metadata:
            return documents
        
        # Return full chunk information with metadata
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        chunks = []
        for i, doc in enumerate(documents):
            # Convert distance to relevance score (lower distance = higher relevance)
            # ChromaDB uses L2 distance, convert to similarity: 1 / (1 + distance)
            relevance_score = 1 / (1 + distances[i]) if i < len(distances) else 0.0
            
            chunk = {
                'text': doc,
                'metadata': metadatas[i] if i < len(metadatas) else {},
                'distance': distances[i] if i < len(distances) else None,
                'relevance_score': relevance_score
            }
            chunks.append(chunk)
        
        return chunks

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
