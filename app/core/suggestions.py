"""Suggested questions — generated from each document's own text.

When a PDF is uploaded we ask Gemini for a few short questions that the
document clearly answers. The UI shows them as clickable chips, so users see
what they can ask instead of guessing.

Generation is best-effort: if Gemini fails or returns something unusable,
the upload still succeeds and the document simply has no suggestions yet.
"""

import json
import logging
import re

from google import genai

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """You write example questions a reader could ask about a document.

Rules:
- Only write questions whose answers are clearly stated in the provided text.
- Make each question specific: use concrete terms from the text.
- Keep each question under 14 words, phrased the way a person would ask it.
- Cover different parts of the text rather than repeating one topic.
- Return ONLY a JSON array of strings. No numbering, no commentary."""

MAX_CONTEXT_CHARS = 9000


class QuestionSuggester:
    def __init__(self, count: int | None = None):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.LLM_MODEL
        self.count = count or settings.SUGGESTION_COUNT

    def generate(self, texts: list[str], filename: str) -> list[str]:
        """Ask Gemini for questions about these chunks. May raise on API errors."""
        sample = self._sample(texts)
        if not sample:
            return []

        interaction = self.client.interactions.create(
            model=self.model,
            system_instruction=SYSTEM_INSTRUCTION,
            input=(
                f"Write {self.count} questions about this document.\n\n"
                f"DOCUMENT: {filename}\n\nTEXT:\n{sample}"
            ),
        )
        return parse_questions(interaction.output_text, self.count)

    def safe_generate(self, texts: list[str], filename: str) -> list[str]:
        """Like generate(), but never raises — suggestions are optional."""
        try:
            return self.generate(texts, filename)
        except Exception:
            logger.warning("Could not generate suggestions for %s", filename, exc_info=True)
            return []

    @staticmethod
    def _sample(texts: list[str]) -> str:
        """Pick passages spread across the document, within a size budget."""
        if not texts:
            return ""
        if len(texts) <= 4:
            picked = texts
        else:
            step = (len(texts) - 1) / 3
            picked = [texts[round(i * step)] for i in range(4)]
        return "\n\n---\n\n".join(picked)[:MAX_CONTEXT_CHARS]


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
