"""Shared test setup.

The real app talks to Gemini for embeddings and answers. Tests replace the
Gemini client with a small fake so they run offline, for free, and give the
same result every time.
"""

import hashlib
import json
import os
import types
from unittest.mock import patch

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")


def fake_vector(text: str, dim: int = 768) -> list[float]:
    """Bag-of-words vector: texts sharing words point in similar directions."""
    vec = [0.0] * dim
    for word in {w.strip(".,?!:;").lower() for w in text.split()}:
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    return vec


class FakeModels:
    def embed_content(self, model, contents, config):
        return types.SimpleNamespace(
            embeddings=[types.SimpleNamespace(values=fake_vector(c)) for c in contents]
        )


FAKE_QUESTIONS = [
    "How many days of paid sick leave do employees get?",
    "Can unused annual leave be carried forward?",
    "How long is paternity leave?",
    "What happens to leave during the notice period?",
]


class FakeInteractions:
    fail = False  # every call fails with a server error
    suggest_fail = False  # only question generation fails
    limited: set = set()  # models that answer with a 429 rate-limit error
    calls: list = []  # every model called, in order
    batches = 0  # "new questions" sets handed out so far

    def create(self, model, system_instruction, input):
        FakeInteractions.calls.append(model)
        if model in FakeInteractions.limited:
            raise RuntimeError(
                f"Error code: 429 - Rate limit exceeded for model {model} "
                "(limit: 20 requests per day on Free Tier). Please retry in 59s."
            )
        if FakeInteractions.fail:
            raise RuntimeError("500 internal server error")
        if "example questions" in system_instruction:
            if FakeInteractions.suggest_fail:
                raise RuntimeError("503 model overloaded")
            if "Do not repeat" in input:
                # A refresh: hand out a new, numbered set each time
                FakeInteractions.batches += 1
                n = FakeInteractions.batches
                fresh = [f"What does rule {n}.{i} of the policy say?" for i in range(1, 5)]
                return types.SimpleNamespace(output_text=json.dumps(fresh))
            return types.SimpleNamespace(output_text=json.dumps(FAKE_QUESTIONS))
        return types.SimpleNamespace(output_text="Employees get **twelve days** of sick leave.")


class FakeGeminiClient:
    def __init__(self, api_key=None):
        self.models = FakeModels()
        self.interactions = FakeInteractions()


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    os.environ["CHROMA_DIR"] = str(tmp_path_factory.mktemp("chroma"))
    with patch("google.genai.Client", FakeGeminiClient):
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as test_client:
            test_client.delete("/api/reset")
            yield test_client


@pytest.fixture
def llm_down():
    FakeInteractions.fail = True
    yield
    FakeInteractions.fail = False


@pytest.fixture
def suggestions_down():
    FakeInteractions.suggest_fail = True
    yield
    FakeInteractions.suggest_fail = False


class RateLimiter:
    """rate_limit("model-a", ...) makes those models return 429s; .calls logs every model used."""

    calls = FakeInteractions.calls

    def __call__(self, *models):
        FakeInteractions.limited = set(models)


@pytest.fixture
def rate_limit():
    FakeInteractions.calls.clear()
    yield RateLimiter()
    FakeInteractions.limited = set()
