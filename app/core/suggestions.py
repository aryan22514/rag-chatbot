"""Suggested questions — generated from each document's own text.

When a PDF is uploaded we ask Gemini for a few short questions that the
document clearly answers. The UI shows them as clickable chips, so users see
what they can ask instead of guessing.

Generation is best-effort: if Gemini fails or returns something unusable,
the upload still succeeds and the document simply has no suggestions yet.
"""

import json
import logging
import random
import re

from app.config import settings
from app.core.llm import LLM

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """You write example questions a reader could ask about a document.

Rules:
- Only write questions whose answers are clearly stated in the provided text.
- Make each question specific: use concrete terms from the text.
- Keep each question under 14 words, phrased the way a person would ask it.
- Cover different parts of the text rather than repeating one topic.
- Return ONLY a JSON array of strings. No numbering, no commentary."""

MAX_CONTEXT_CHARS = 9000
MAX_AVOID = 40  # most recent questions to list as "don't repeat"


class QuestionSuggester:
    def __init__(self, llm: LLM | None = None, count: int | None = None):
        self.llm = llm or LLM()
        self.models = settings.suggestion_models
        self.count = count or settings.SUGGESTION_COUNT

    def generate(
        self, texts: list[str], filename: str, avoid: list[str] | None = None
    ) -> list[str]:
        """Ask Gemini for questions about these chunks. May raise on API errors.

        `avoid` lists questions already suggested or asked. When given, we ask
        for different questions and read a different random part of the
        document, so each new set covers new ground.
        """
        avoid = [q for q in (avoid or []) if q.strip()][-MAX_AVOID:]
        sample = self._sample(texts, shuffle=bool(avoid))
        if not sample:
            return []

        prompt = f"Write {self.count} questions about this document.\n\nDOCUMENT: {filename}\n\n"
        if avoid:
            prompt += (
                "Do not repeat or reword any of these questions; "
                "ask about other details instead:\n"
                + "\n".join(f"- {q}" for q in avoid)
                + "\n\n"
            )
        prompt += f"TEXT:\n{sample}"

        generation = self.llm.generate(SYSTEM_INSTRUCTION, prompt, self.models)
        questions = parse_questions(generation.text, self.count)
        seen = {normalize(q) for q in avoid}
        return [q for q in questions if normalize(q) not in seen]

    def safe_generate(self, texts: list[str], filename: str) -> list[str]:
        """Like generate(), but never raises — suggestions are optional."""
        try:
            return self.generate(texts, filename)
        except Exception:
            logger.warning("Could not generate suggestions for %s", filename, exc_info=True)
            return []

    @staticmethod
    def _sample(texts: list[str], shuffle: bool = False) -> str:
        """Pick passages spread across the document, within a size budget.

        With shuffle=True the passages are picked at random, so repeated
        calls see different parts of a long document.
        """
        if not texts:
            return ""
        if len(texts) <= 4:
            picked = list(texts)
            if shuffle:
                random.shuffle(picked)
        elif shuffle:
            picked = [texts[i] for i in sorted(random.sample(range(len(texts)), 4))]
        else:
            step = (len(texts) - 1) / 3
            picked = [texts[round(i * step)] for i in range(4)]
        return "\n\n---\n\n".join(picked)[:MAX_CONTEXT_CHARS]


def normalize(question: str) -> str:
    """Compare questions ignoring case, spacing and the trailing '?'."""
    return " ".join(question.lower().split()).rstrip("?").strip()


def parse_questions(raw: str, limit: int) -> list[str]:
    """Turn the model's reply into a clean list of questions.

    Prefers a JSON array; falls back to one-question-per-line if the model
    ignored the format.
    """
    raw = raw or ""
    items: list[str] = []

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            items = [x for x in data if isinstance(x, str)]
        except json.JSONDecodeError:
            items = []

    if not items:
        items = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line) for line in raw.splitlines()]

    questions: list[str] = []
    seen: set[str] = set()
    for item in items:
        q = " ".join(item.split()).strip().strip('"').strip()
        if len(q) < 8 or len(q) > 140:
            continue
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        questions.append(q)

    return questions[:limit]
