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
