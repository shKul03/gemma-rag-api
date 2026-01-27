from fastapi import FastAPI
from rag import retrieve_context

app = FastAPI()


@app.post("/ask")
def ask(question: str):
    context = retrieve_context(question)

    if not context.strip():
        return {"answer": "NOT FOUND IN CONTEXT"}

    prompt = f"""
You are an HR policy assistant.
Answer ONLY using the context below.
If the answer is not present, say "NOT FOUND IN CONTEXT".

Context:
{context}

Question:
{question}
"""

    # 🔁 Call Phi-3 here (wherever you do it)
    answer = call_phi3(prompt)

    return {"answer": answer}
