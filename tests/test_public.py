"""Public demo mode: private libraries per visitor and daily limits."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings

SAMPLE = Path(__file__).parent / "fixtures" / "sample-leave-policy.pdf"
NOTES = ("notes.txt", b"The office is closed on the first Monday of every month.", "text/plain")


@pytest.fixture
def public(client, monkeypatch):
    """Turn public mode on, with two visitors in separate browsers."""
    from app.api import routes

    monkeypatch.setattr(settings, "PUBLIC_MODE", True)
    routes.question_quota.clear()
    routes.upload_quota.clear()
    alice = TestClient(client.app)
    bob = TestClient(client.app)
    yield alice, bob
    for visitor in (alice, bob):
        visitor.delete("/api/reset")
    routes.question_quota.clear()
    routes.upload_quota.clear()


def upload(visitor, file=None):
    file = file or ("sample-leave-policy.pdf", SAMPLE.read_bytes(), "application/pdf")
    return visitor.post("/api/upload", files={"file": file})


def test_visitors_get_an_anonymous_cookie(public):
    alice, _ = public
    res = alice.get("/api/documents")
    assert len(res.cookies["gw_visitor"]) == 32
    # Once set, it isn't reissued
    assert "gw_visitor" not in alice.get("/api/documents").cookies


def test_each_visitor_has_a_private_library(public, client):
    alice, bob = public
    assert upload(alice).status_code == 200

    assert len(alice.get("/api/documents").json()["documents"]) == 1
    assert bob.get("/api/documents").json()["documents"] == []
    assert bob.get("/api/stats").json()["total_documents"] == 0

    # Bob can't delete Alice's document, and clearing his library leaves hers alone
    doc_id = alice.get("/api/documents").json()["documents"][0]["document_id"]
    assert bob.delete(f"/api/documents/{doc_id}").status_code == 404
    bob.delete("/api/reset")
    assert len(alice.get("/api/documents").json()["documents"]) == 1

    # The same file can be uploaded by both: duplicates are checked per library
    assert upload(bob).status_code == 200


def test_local_library_is_untouched_by_visitors(public, client):
    alice, _ = public
    upload(alice)
    settings.PUBLIC_MODE = False  # restored by monkeypatch
    assert client.get("/api/documents").json()["documents"] == []
    settings.PUBLIC_MODE = True


def test_document_limit(public, monkeypatch):
    alice, _ = public
    monkeypatch.setattr(settings, "PUBLIC_MAX_DOCUMENTS", 1)
    assert upload(alice).status_code == 200
    res = upload(alice, NOTES)
    assert res.status_code == 400
    assert "up to 1 files" in res.json()["detail"]


def test_daily_question_limit(public, monkeypatch):
    alice, bob = public
    monkeypatch.setattr(settings, "PUBLIC_DAILY_QUESTIONS", 2)
    upload(alice)
    assert alice.get("/api/ask", params={"q": "sick leave"}).status_code == 200
    # A repeat is answered from the cache and doesn't count
    assert alice.get("/api/ask", params={"q": "sick leave"}).json()["cached"] is True
    assert alice.get("/api/ask", params={"q": "paternity leave"}).status_code == 200

    res = alice.get("/api/ask", params={"q": "notice period"})
    assert res.status_code == 429
    assert "2 free questions" in res.json()["detail"]
    # The allowance is per IP address, so a new cookie doesn't reset it
    assert bob.get("/api/ask", params={"q": "holidays"}).status_code == 429


def test_daily_upload_limit(public, monkeypatch):
    alice, _ = public
    monkeypatch.setattr(settings, "PUBLIC_DAILY_UPLOADS", 1)
    assert upload(alice).status_code == 200
    res = upload(alice, NOTES)
    assert res.status_code == 429
    assert "free uploads" in res.json()["detail"]


def test_config_reports_public_limits(public):
    alice, _ = public
    body = alice.get("/api/config").json()
    assert body["public"] is True
    assert body["max_documents"] == settings.PUBLIC_MAX_DOCUMENTS
    assert body["daily_questions"] == settings.PUBLIC_DAILY_QUESTIONS


def test_config_locally(client):
    body = client.get("/api/config").json()
    assert body["public"] is False
    assert body["max_documents"] is None
    assert "gw_visitor" not in client.get("/api/documents").cookies
