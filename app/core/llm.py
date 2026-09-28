"""One place that talks to Gemini for text generation — with model fallback.

Free-tier API keys get a small daily quota per model (e.g. 20 requests/day).
Quotas are tracked separately for each model, so when one model is rate
limited we can often still answer with the next model in the list.
"""

import logging
import re
from dataclasses import dataclass

from google import genai

from app.config import settings

logger = logging.getLogger(__name__)

_RATE_LIMIT = re.compile(r"\b429\b|rate.?limit|resource.?exhausted|too.?many.?requests|quota", re.I)
_NOT_FOUND = re.compile(r"\b404\b|not.?found|is not supported|unsupported model", re.I)
_RETRY_IN = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.I)


class RateLimitedError(Exception):
    """Every model we tried is out of quota right now."""

    def __init__(self, models: list[str], retry_after: int | None):
        self.models = models
        self.retry_after = retry_after
        wait = f" Try again in about {retry_after} seconds." if retry_after else ""
        super().__init__(
            "Gemini's free usage limit is used up for "
            + ", ".join(models)
            + "."
            + wait
            + " Limits reset daily; a paid API key removes them."
        )


@dataclass
class Generation:
    text: str
    model: str


def _status(error: Exception) -> int | None:
    for attr in ("code", "status_code", "status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_rate_limited(error: Exception) -> bool:
    return _status(error) == 429 or bool(_RATE_LIMIT.search(str(error)))


def is_model_unavailable(error: Exception) -> bool:
    return _status(error) == 404 or bool(_NOT_FOUND.search(str(error)))


def retry_after_seconds(error: Exception) -> int | None:
    match = _RETRY_IN.search(str(error))
    return max(1, round(float(match.group(1)))) if match else None


class LLM:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def generate(self, system_instruction: str, prompt: str, models: list[str]) -> Generation:
        """Try each model in order; move on only when a model is rate limited or missing."""
        limited: list[str] = []
        retry_after: int | None = None

        for model in models:
            try:
                interaction = self.client.interactions.create(
                    model=model,
                    system_instruction=system_instruction,
                    input=prompt,
                )
                return Generation(text=interaction.output_text or "", model=model)
            except Exception as error:
                if is_rate_limited(error):
                    logger.warning("%s is rate limited, trying the next model", model)
                    limited.append(model)
                    wait = retry_after_seconds(error)
                    if wait is not None:
                        retry_after = wait if retry_after is None else min(retry_after, wait)
                    continue
                if is_model_unavailable(error):
                    logger.warning("%s is unavailable for this key, trying the next model", model)
                    continue
                raise

        if limited:
            raise RateLimitedError(limited, retry_after)
        raise RuntimeError("No configured Gemini model is available: " + ", ".join(models))
