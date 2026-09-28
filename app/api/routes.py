import hashlib
import logging
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config import settings
from app.core.document_processor import SUPPORTED_TYPES, DocumentProcessor
from app.core.llm import LLM, RateLimitedError, is_rate_limited
from app.core.rag_chain import RagChain
from app.core.suggestions import QuestionSuggester
from app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()
processor = DocumentProcessor(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)
store = VectorStore()
llm = LLM()
rag = RagChain(store, llm)
suggester = QuestionSuggester(llm)


BUSY = "The AI service is busy right now. Please try again in a minute."
UNSUPPORTED = "Unsupported file type. Upload a PDF, Word (.docx), or text (.txt, .md) file."


def ai_error(error: Exception, doing: str) -> HTTPException:
    """Log the real Gemini error; give the user a short, plain message."""
    logger.error("AI call failed while %s: %s", doing, error)
    if isinstance(error, RateLimitedError) or is_rate_limited(error):
        retry = getattr(error, "retry_after", None)
        headers = {"Retry-After": str(retry)} if retry else None
        return HTTPException(status_code=429, detail=BUSY, headers=headers)
    return HTTPException(
        status_code=502,
        detail=f"Something went wrong while {doing}. Please try again.",
    )


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    name = file.filename or ""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail=UNSUPPORTED)

    # Stream to a temp file, hashing as we go and stopping at the size limit
    limit = int(settings.MAX_UPLOAD_MB * 1024 * 1024)
    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        while block := await file.read(1024 * 1024):
            size += len(block)
            if size > limit:
                break
            digest.update(block)
            tmp.write(block)

    try:
        if size > limit:
            raise HTTPException(
                status_code=413,
                detail=f"This file is larger than {settings.MAX_UPLOAD_MB:g} MB. "
                "Please upload a smaller file.",
            )
        content_hash = digest.hexdigest()
        existing = store.find_by_hash(content_hash)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"This document is already in your library as \u201c{existing}\u201d.",
            )
        try:
            pages, paged = processor.extract(tmp_path)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Could not read this file. Is it damaged or password-protected?",
            ) from None
        chunks = processor.chunk_pages(pages, name, paged)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text could be found in this file. "
            "If it's a scanned PDF, it has no text layer to read.",
        )

    document_id = str(uuid4())
    try:
        stored = store.add_chunks(chunks, document_id, content_hash)
    except Exception as e:
        raise ai_error(e, "processing your document") from e

    questions = suggester.safe_generate([c.text for c in chunks], name)
    if questions:
        store.set_questions(document_id, questions)

    return {
        "document_id": document_id,
        "filename": name,
        "total_words": sum(len(page.split()) for page in pages),
        "pages": len(pages) if paged else None,
        "chunks_stored": stored,
        "suggested_questions": questions,
    }


@router.get("/search")
def search(q: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=20)):
    hits = store.search(q, top_k)
    return {"question": q, "results": hits}


@router.get("/documents")
def list_documents():
    return {"documents": store.list_documents()}


@router.get("/stats")
def stats():
    return store.stats()

@router.get("/ask")
def ask(q: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=20)):
    try:
        return rag.ask(q, top_k)
    except Exception as e:
        raise ai_error(e, "getting your answer") from e


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    deleted = store.delete_document(document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted_chunks": deleted}


@router.delete("/reset")
def reset():
    return {"deleted_chunks": store.reset()}


@router.get("/suggestions")
def suggestions(limit: int = Query(6, ge=1, le=20)):
    """Questions generated from the uploaded documents, mixed across documents.

    Documents uploaded before this feature existed get their questions
    generated on first request and cached.
    """
    per_doc: list[list[dict]] = []
    for doc in store.list_documents():
        doc_id = doc["document_id"]
        questions = store.get_questions(doc_id)
        if questions is None:
            questions = suggester.safe_generate(store.document_texts(doc_id), doc["source"])
            if questions:
                store.set_questions(doc_id, questions)
        per_doc.append(
            [{"question": q, "source": doc["source"], "document_id": doc_id} for q in questions]
        )

    # Round-robin so every document is represented near the top
    mixed = []
    for i in range(max((len(qs) for qs in per_doc), default=0)):
        mixed.extend(qs[i] for qs in per_doc if i < len(qs))
    return {"suggestions": mixed[:limit]}
