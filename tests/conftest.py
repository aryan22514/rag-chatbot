"""Shared test setup.

The real app talks to Gemini for embeddings and answers. Tests replace the
Gemini client with a small fake so they run offline, for free, and give the
same result every time.
"""

import hashlib
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


class FakeInteractions:
    fail = False

    def create(self, model, system_instruction, input):
        if FakeInteractions.fail:
            raise RuntimeError("429 quota exceeded")
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
