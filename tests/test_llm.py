"""Rate limits: model fallback, the answer cache, and clear 429 errors."""

from pathlib import Path

from app.config import settings
from app.core.llm import is_rate_limited, retry_after_seconds

SAMPLE = Path(__file__).parent / "fixtures" / "sample-leave-policy.pdf"
REAL_ERROR = (
    "Error code: 429 - {'error': {'message': 'Rate limit exceeded for model gemini-3.8-flash "
    "(limit: 20 requests per day on Free Tier). Please retry in 59s or upgrade your tier.', "
    "'code': 'too_many_requests'}}"
)
MAIN, FALLBACK, LITE = settings.llm_models


def upload(client):
    return client.post(
        "/api/upload",
        files={"file": ("sample-leave-policy.pdf", SAMPLE.read_bytes(), "application/pdf")},
    )


def test_recognises_the_real_free_tier_error():
    error = RuntimeError(REAL_ERROR)
    assert is_rate_limited(error)
    assert retry_after_seconds(error) == 59
    assert not is_rate_limited(RuntimeError("500 internal server error"))


def test_model_order():
    assert settings.llm_models == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
    assert settings.suggestion_models[0] == "gemini-3.5-flash-lite"


def test_suggestions_use_the_cheap_model_first(client, rate_limit):
    client.delete("/api/reset")
    upload(client)
    assert rate_limit.calls == [LITE]


def test_falls_back_when_main_model_is_rate_limited(client, rate_limit):
    rate_limit(MAIN)
    body = client.get("/api/ask", params={"q": "How many sick days?"}).json()
    assert body["model"] == FALLBACK
    assert "twelve days" in body["answer"]


def test_falls_through_to_the_lite_model(client, rate_limit):
    rate_limit(MAIN, FALLBACK)
    body = client.get("/api/ask", params={"q": "Is sick leave paid?"}).json()
    assert body["model"] == LITE


def test_clear_429_when_every_model_is_rate_limited(client, rate_limit):
    rate_limit(MAIN, FALLBACK, LITE)
    res = client.get("/api/ask", params={"q": "What is the notice period?"})
    assert res.status_code == 429
    assert res.headers["retry-after"] == "59"
    detail = res.json()["detail"]
    assert detail == "The AI service is busy right now. Please try again in a minute."
    assert "gemini" not in detail.lower() and "429" not in detail


def test_repeat_questions_are_answered_from_cache(client, rate_limit):
    first = client.get("/api/ask", params={"q": "How long is paternity leave?"}).json()
    assert first["cached"] is False
    calls_before = len(rate_limit.calls)

    # Even with every model rate limited, a repeat question still works
    rate_limit(MAIN, FALLBACK, LITE)
    again = client.get("/api/ask", params={"q": "  how long is PATERNITY leave? "}).json()
    assert again["cached"] is True
    assert again["answer"] == first["answer"]
    assert again["question"] == "  how long is PATERNITY leave? "
    assert len(rate_limit.calls) == calls_before


def test_cache_resets_when_the_library_changes(client, rate_limit):
    client.get("/api/ask", params={"q": "Who approves leave?"})
    doc_id = client.get("/api/documents").json()["documents"][0]["document_id"]
    client.delete(f"/api/documents/{doc_id}")
    upload(client)
    again = client.get("/api/ask", params={"q": "Who approves leave?"}).json()
    assert again["cached"] is False
    client.delete("/api/reset")
