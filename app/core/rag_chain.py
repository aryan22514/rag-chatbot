from collections import OrderedDict

from app.config import settings
from app.core.llm import LLM
from app.core.vector_store import VectorStore

SYSTEM_INSTRUCTION = """You are a helpful assistant that answers questions \
using ONLY the provided context from the user's documents.

Rules:
- Answer only from the context below. Do not use outside knowledge.
- If the context does not contain the answer, say: "I couldn't find that in \
your documents."
- Be concise and direct.
- Quote specific numbers, dates, and terms exactly as they appear."""

CACHE_SIZE = 128


def page_label(start: int | None, end: int | None) -> str | None:
    """'page 2' or 'pages 2–3'; None when the format has no pages."""
    if start is None:
        return None
    return f"page {start}" if start == end or end is None else f"pages {start}–{end}"


class RagChain:
    def __init__(self, store: VectorStore | None = None, llm: LLM | None = None):
        self.store = store or VectorStore()
        self.llm = llm or LLM()
        self.models = settings.llm_models
        # Same question + same library = same answer, so don't spend quota twice
        self._cache: OrderedDict[tuple, dict] = OrderedDict()

    def ask(self, question: str, top_k: int | None = None) -> dict:
        top_k = top_k or settings.TOP_K

        key = (" ".join(question.lower().split()), top_k, self.store.fingerprint())
        if key in self._cache:
            self._cache.move_to_end(key)
            return {**self._cache[key], "question": question, "cached": True}

        hits = self.store.search(question, top_k)

        if not hits:
            return {
                "question": question,
                "answer": "No documents have been uploaded yet.",
                "sources": [],
                "model": None,
                "cached": False,
            }

        context = self._build_context(hits)
        prompt = f"CONTEXT FROM DOCUMENTS:\n{context}\n\nQUESTION: {question}"

        generation = self.llm.generate(SYSTEM_INSTRUCTION, prompt, self.models)

        result = {
            "question": question,
            "answer": generation.text,
            "sources": [
                {
                    "source": h["source"],
                    "chunk_index": h["chunk_index"],
                    "page_start": h.get("page_start"),
                    "page_end": h.get("page_end"),
                    "score": h["score"],
                    "preview": h["text"][:150],
                }
                for h in hits
            ],
            "model": generation.model,
            "cached": False,
        }

        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def _build_context(self, hits: list[dict]) -> str:
        parts = []
        for i, hit in enumerate(hits, start=1):
            where = page_label(hit.get("page_start"), hit.get("page_end")) or (
                f"chunk {hit['chunk_index']}"
            )
            parts.append(f"[Source {i} — {hit['source']}, {where}]\n{hit['text']}")
        return "\n\n".join(parts)
