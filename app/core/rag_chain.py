from google import genai

from app.config import settings
from app.core.vector_store import VectorStore

SYSTEM_INSTRUCTION = """You are a helpful assistant that answers questions \
using ONLY the provided context from the user's documents.

Rules:
- Answer only from the context below. Do not use outside knowledge.
- If the context does not contain the answer, say: "I couldn't find that in \
your documents."
- Be concise and direct.
- Quote specific numbers, dates, and terms exactly as they appear."""


class RagChain:
    def __init__(self, store: VectorStore | None = None):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.LLM_MODEL
        self.store = store or VectorStore()

    def ask(self, question: str, top_k: int | None = None) -> dict:
        top_k = top_k or settings.TOP_K

        hits = self.store.search(question, top_k)

        if not hits:
            return {
                "question": question,
                "answer": "No documents have been uploaded yet.",
                "sources": [],
            }

        context = self._build_context(hits)
        prompt = f"CONTEXT FROM DOCUMENTS:\n{context}\n\nQUESTION: {question}"

        interaction = self.client.interactions.create(
            model=self.model,
            system_instruction=SYSTEM_INSTRUCTION,
            input=prompt,
        )

        return {
            "question": question,
            "answer": interaction.output_text,
            "sources": [
                {
                    "source": h["source"],
                    "chunk_index": h["chunk_index"],
                    "score": h["score"],
                    "preview": h["text"][:150],
                }
                for h in hits
            ],
        }

    def _build_context(self, hits: list[dict]) -> str:
        parts = []
        for i, hit in enumerate(hits, start=1):
            parts.append(
                f"[Source {i} — {hit['source']}, chunk {hit['chunk_index']}]\n"
                f"{hit['text']}"
            )
        return "\n\n".join(parts)
