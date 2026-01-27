import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma"
COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL_NAME)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve_context(query: str, k: int = 5) -> str:
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs)

# if __name__ == "__main__":
#     q = "maternity leave policy"
#     ctx = retrieve_context(q)
#
#     print("\n--- RETRIEVED CONTEXT ---\n")
#     print(ctx if ctx else "❌ No context returned")
