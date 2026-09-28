"""Suggested questions: parsing the model's reply, and the API around it."""

from pathlib import Path

from app.core.suggestions import parse_questions

SAMPLE = Path(__file__).parent / "fixtures" / "sample-leave-policy.pdf"


def upload(client):
    return client.post(
        "/api/upload",
        files={"file": ("sample-leave-policy.pdf", SAMPLE.read_bytes(), "application/pdf")},
    )


# ── parsing ────────────────────────────────────────────────────────────


def test_parses_json_array():
    reply = '["What is X?", "How long is Y?"]'
    assert parse_questions(reply, 4) == ["What is X?", "How long is Y?"]


def test_parses_json_inside_code_fence():
    raw = 'Sure!\n```json\n["What is the notice period?"]\n```'
    assert parse_questions(raw, 4) == ["What is the notice period?"]


def test_falls_back_to_lines_and_adds_question_marks():
    raw = "1. What is the sick leave policy\n- How many holidays are there?\n"
    assert parse_questions(raw, 4) == [
        "What is the sick leave policy?",
        "How many holidays are there?",
    ]


def test_drops_duplicates_junk_and_respects_limit():
    raw = (
        '["What is the notice period?", "what is the notice period?", "ok", "'
        + "Very long " * 20
        + '?", "Who approves leave?", "When do holidays reset?"]'
    )
    assert parse_questions(raw, 2) == ["What is the notice period?", "Who approves leave?"]


def test_empty_reply_gives_no_questions():
    assert parse_questions("", 4) == []
    assert parse_questions(None, 4) == []


# ── API ────────────────────────────────────────────────────────────────


def test_no_suggestions_for_empty_library(client):
    client.delete("/api/reset")
    assert client.get("/api/suggestions").json() == {"suggestions": []}


def test_upload_returns_and_stores_questions(client):
    body = upload(client).json()
    assert len(body["suggested_questions"]) == 4

    suggestions = client.get("/api/suggestions").json()["suggestions"]
    assert [s["question"] for s in suggestions] == body["suggested_questions"]
    assert all(s["source"] == "sample-leave-policy.pdf" for s in suggestions)
    limited = client.get("/api/suggestions", params={"limit": 2}).json()["suggestions"]
    assert limited == suggestions[:2]


def test_questions_do_not_leak_into_answers(client):
    body = client.get("/api/ask", params={"q": "sick leave"}).json()
    assert "twelve days" in body["answer"]
    keys = {"source", "chunk_index", "page_start", "page_end", "score", "preview"}
    assert all(set(s) == keys for s in body["sources"])


def test_deleting_a_document_removes_its_questions(client):
    doc_id = client.get("/api/documents").json()["documents"][0]["document_id"]
    client.delete(f"/api/documents/{doc_id}")
    assert client.get("/api/suggestions").json() == {"suggestions": []}


def test_upload_succeeds_even_if_suggestions_fail(client, suggestions_down):
    res = upload(client)
    assert res.status_code == 200
    assert res.json()["suggested_questions"] == []
    assert res.json()["chunks_stored"] == 3


def test_missing_questions_are_generated_on_request_and_cached(client):
    # The document above was stored without questions; the endpoint fills the gap
    first = client.get("/api/suggestions").json()["suggestions"]
    assert len(first) == 4
    doc_id = first[0]["document_id"]

    from app.api.routes import store

    assert store.get_questions(doc_id) == [s["question"] for s in first]
    client.delete("/api/reset")


# ── a new set once everything has been asked ──────────────────────────


def test_more_questions_for_empty_library(client):
    client.delete("/api/reset")
    res = client.post("/api/suggestions/more", json={"asked": []})
    assert res.json() == {"suggestions": []}


def test_more_questions_are_new_and_saved(client):
    first = upload(client).json()["suggested_questions"]

    res = client.post("/api/suggestions/more", json={"asked": first})
    assert res.status_code == 200
    fresh = [s["question"] for s in res.json()["suggestions"]]
    assert len(fresh) == 4
    assert not {q.lower() for q in fresh} & {q.lower() for q in first}

    # Saved newest first, so a reload shows the new set
    stored = [s["question"] for s in client.get("/api/suggestions?limit=20").json()["suggestions"]]
    assert stored[:4] == fresh
    assert stored[4:] == first


def test_every_refresh_gives_a_different_set(client):
    seen = [s["question"] for s in client.get("/api/suggestions?limit=20").json()["suggestions"]]
    for _ in range(2):
        batch = client.post("/api/suggestions/more", json={"asked": seen}).json()["suggestions"]
        questions = [s["question"] for s in batch]
        assert questions and not set(questions) & set(seen)
        seen += questions


def test_more_questions_when_ai_is_busy(client, rate_limit):
    from app.config import settings

    rate_limit(*settings.suggestion_models)
    res = client.post("/api/suggestions/more", json={"asked": ["anything?"]})
    assert res.status_code == 429
    assert "busy" in res.json()["detail"]
    client.delete("/api/reset")


def test_more_questions_validates_input(client):
    res = client.post("/api/suggestions/more", json={"asked": [], "limit": 0})
    assert res.status_code == 422
