import logging

import requests

from app import config

logger = logging.getLogger("bakarwadi")


class LLMNotConfigured(RuntimeError):
    pass


def chat(system: str, user: str) -> str:
    """One chat-completion call against any OpenAI-compatible endpoint (Groq by default)."""
    key = config.llm_api_key()
    if not key:
        raise LLMNotConfigured("GROQ_API_KEY (or LLM_API_KEY) is not set")

    logger.info("Calling %s", config.LLM_MODEL)
    resp = requests.post(
        f"{config.LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 1024,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").strip()
