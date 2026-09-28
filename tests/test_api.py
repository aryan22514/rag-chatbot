"""End-to-end API tests: upload → ask → delete, plus every error path."""

from pathlib import Path

SAMPLE = Path(__file__).parent / "fixtures" / "sample-leave-policy.pdf"


def upload(client, name="sample-leave-policy.pdf", data=None, mime="application/pdf"):
    body = data if data is not None else SAMPLE.read_bytes()
    return client.post("/api/upload", files={"file": (name, body, mime)})


def test_health(client):
    assert client.get("/api/health").json() == {"status": "Ok"}


def test_serves_landing_page_and_app(client):
    assert "Groundwork" in client.get("/").text
    assert client.get("/app.html").status_code == 200


def test_ask_with_empty_library(client):
    body = client.get("/api/ask", params={"q": "anything"}).json()
    assert body["answer"] == "No documents have been uploaded yet."
    assert body["sources"] == []


def test_rejects_non_pdf(client):
    res = upload(client, name="notes.txt", data=b"hello", mime="text/plain")
    assert res.status_code == 400
    assert "Only PDF" in res.json()["detail"]


def test_rejects_corrupt_pdf(client):
    res = upload(client, name="broken.pdf", data=b"not really a pdf")
    assert res.status_code == 400
    assert "Could not read" in res.json()["detail"]


def test_upload_chunks_the_sample(client):
    res = upload(client)
    assert res.status_code == 200
    body = res.json()
    assert body["total_words"] == 1089
    assert body["chunks_stored"] == 3
    assert client.get("/api/stats").json() == {"total_chunks": 3, "total_documents": 1}


def test_ask_returns_grounded_answer_with_sources(client):
    body = client.get("/api/ask", params={"q": "How many sick leaves do I get?"}).json()
    assert "twelve days" in body["answer"]
    assert 1 <= len(body["sources"]) <= 5
    scores = [s["score"] for s in body["sources"]]
    assert scores == sorted(scores, reverse=True)
    assert all(s["source"] == "sample-leave-policy.pdf" for s in body["sources"])


def test_top_k_is_validated(client):
    assert client.get("/api/ask", params={"q": "x", "top_k": 0}).status_code == 422
    assert client.get("/api/ask", params={"q": "x", "top_k": 99}).status_code == 422


def test_llm_failure_is_reported_not_crashed(client, llm_down):
    res = client.get("/api/ask", params={"q": "sick leave"})
    assert res.status_code == 502
    assert "internal server error" in res.json()["detail"]


def test_delete_document(client):
    assert client.delete("/api/documents/does-not-exist").status_code == 404
    doc_id = client.get("/api/documents").json()["documents"][0]["document_id"]
    assert client.delete(f"/api/documents/{doc_id}").json() == {"deleted_chunks": 3}
    assert client.get("/api/stats").json()["total_chunks"] == 0


def test_reset_clears_everything(client):
    upload(client)
    assert client.delete("/api/reset").json() == {"deleted_chunks": 3}
    assert client.get("/api/stats").json() == {"total_chunks": 0, "total_documents": 0}
