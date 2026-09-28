import hashlib
import logging
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.core.document_processor import SUPPORTED_TYPES, DocumentProcessor
from app.core.library import Libraries, Library
from app.core.llm import LLM, RateLimitedError, is_rate_limited
from app.core.quota import DailyQuota
from app.core.suggestions import QuestionSuggester, normalize

logger = logging.getLogger(__name__)
router = APIRouter()
processor = DocumentProcessor(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)
llm = LLM()
libraries = Libraries(llm)
local = libraries.get(settings.COLLECTION_NAME)
store, rag = local.store, local.rag  # the single library used when running locally
suggester = QuestionSuggester(llm)
question_quota = DailyQuota()
upload_quota = DailyQuota()


def current_library(request: Request) -> Library:
    """The visitor's own library on the public demo; the shared one locally."""
    if settings.PUBLIC_MODE:
        return libraries.for_visitor(request.state.visitor)
    return local


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def check_quota(request: Request, quota: DailyQuota, limit: int, what: str) -> None:
    """On the public demo, stop a visitor who has used up today's allowance."""
    if settings.PUBLIC_MODE and quota.remaining(client_ip(request), limit) <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"You've used today's {limit} free {what} on this demo. "
            "They reset at midnight UTC, or run Groundwork yourself for unlimited use.",
        )


def spend_quota(request: Request, quota: DailyQuota) -> None:
    if settings.PUBLIC_MODE:
        quota.spend(client_ip(request))


BUSY = "The AI service is busy right now. Please try again in a minute."
MAX_STORED_QUESTIONS = 60  # per document, newest kept
MAX_REFRESH_DOCS = 3  # documents asked for new questions per refresh (saves quota)
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
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    lib: Library = Depends(current_library),
):
    name = file.filename or ""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail=UNSUPPORTED)
    if settings.PUBLIC_MODE:
        limit = settings.PUBLIC_MAX_DOCUMENTS
        if len(lib.store.list_documents()) >= limit:
            raise HTTPException(
                status_code=400,
                detail=f"Demo libraries hold up to {limit} files. "
                "Remove one to add another.",
            )
        check_quota(request, upload_quota, settings.PUBLIC_DAILY_UPLOADS, "uploads")

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
        existing = lib.store.find_by_hash(content_hash)
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
    spend_quota(request, upload_quota)
    try:
        stored = lib.store.add_chunks(chunks, document_id, content_hash)
    except Exception as e:
        raise ai_error(e, "processing your document") from e

    questions = suggester.safe_generate([c.text for c in chunks], name)
    if questions:
        lib.store.set_questions(document_id, questions)

    return {
        "document_id": document_id,
        "filename": name,
        "total_words": sum(len(page.split()) for page in pages),
        "pages": len(pages) if paged else None,
        "chunks_stored": stored,
        "suggested_questions": questions,
    }


@router.get("/search")
def search(
    request: Request,
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    lib: Library = Depends(current_library),
):
    check_quota(request, question_quota, settings.PUBLIC_DAILY_QUESTIONS, "questions")
    spend_quota(request, question_quota)
    hits = lib.store.search(q, top_k)
    return {"question": q, "results": hits}


@router.get("/documents")
def list_documents(lib: Library = Depends(current_library)):
    return {"documents": lib.store.list_documents()}


@router.get("/stats")
def stats(lib: Library = Depends(current_library)):
    return lib.store.stats()

@router.get("/ask")
def ask(
    request: Request,
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    lib: Library = Depends(current_library),
):
    check_quota(request, question_quota, settings.PUBLIC_DAILY_QUESTIONS, "questions")
    try:
        result = lib.rag.ask(q, top_k)
    except Exception as e:
        raise ai_error(e, "getting your answer") from e
    if not result.get("cached"):
        spend_quota(request, question_quota)  # repeat questions are free
    return result


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, lib: Library = Depends(current_library)):
    deleted = lib.store.delete_document(document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted_chunks": deleted}


@router.delete("/reset")
def reset(lib: Library = Depends(current_library)):
    return {"deleted_chunks": lib.store.reset()}


@router.get("/suggestions")
def suggestions(
    limit: int = Query(6, ge=1, le=20), lib: Library = Depends(current_library)
):
    """Questions generated from the uploaded documents, mixed across documents.

    Documents uploaded before this feature existed get their questions
    generated on first request and cached.
    """
    per_doc: list[list[dict]] = []
    for doc in lib.store.list_documents():
        doc_id = doc["document_id"]
        questions = lib.store.get_questions(doc_id)
        if questions is None:
            questions = suggester.safe_generate(lib.store.document_texts(doc_id), doc["source"])
            if questions:
                lib.store.set_questions(doc_id, questions)
        per_doc.append(
            [{"question": q, "source": doc["source"], "document_id": doc_id} for q in questions]
        )

    return {"suggestions": round_robin(per_doc)[:limit]}


def round_robin(per_doc: list[list[dict]]) -> list[dict]:
    """Interleave documents so every one is represented near the top."""
    mixed = []
    for i in range(max((len(qs) for qs in per_doc), default=0)):
        mixed.extend(qs[i] for qs in per_doc if i < len(qs))
    return mixed


class MoreSuggestionsRequest(BaseModel):
    asked: list[str] = Field(default_factory=list, max_length=500)
    limit: int = Field(6, ge=1, le=20)


@router.post("/suggestions/more")
def more_suggestions(
    request: Request,
    body: MoreSuggestionsRequest,
    lib: Library = Depends(current_library),
):
    """A fresh set of questions, different from everything suggested or asked so far.

    The UI calls this once every suggested question has been asked. New
    questions are saved (newest first), so they survive a page reload.
    """
    docs = lib.store.list_documents()
    if not docs:
        return {"suggestions": []}
    check_quota(request, question_quota, settings.PUBLIC_DAILY_QUESTIONS, "questions")
    spend_quota(request, question_quota)

    asked = {normalize(q) for q in body.asked}
    # Documents the user has been asking about come first
    stored = {d["document_id"]: lib.store.get_questions(d["document_id"]) or [] for d in docs}
    docs.sort(key=lambda d: -sum(normalize(q) in asked for q in stored[d["document_id"]]))

    per_doc: list[list[dict]] = []
    failure: Exception | None = None
    for doc in docs[:MAX_REFRESH_DOCS]:
        doc_id = doc["document_id"]
        existing = stored[doc_id]
        known = {normalize(q) for q in existing} | asked
        try:
            fresh = suggester.generate(
                lib.store.document_texts(doc_id), doc["source"], avoid=existing + body.asked
            )
        except Exception as error:
            failure = error
            continue
        fresh = [q for q in fresh if normalize(q) not in known]
        if fresh:
            lib.store.set_questions(doc_id, (fresh + existing)[:MAX_STORED_QUESTIONS])
            per_doc.append(
                [{"question": q, "source": doc["source"], "document_id": doc_id} for q in fresh]
            )

    if not per_doc and failure is not None:
        raise ai_error(failure, "writing new questions")
    return {"suggestions": round_robin(per_doc)[: body.limit]}


@router.get("/config")
def public_config():
    """What the UI needs to know about limits (the public demo shows them)."""
    return {
        "public": settings.PUBLIC_MODE,
        "max_upload_mb": settings.MAX_UPLOAD_MB,
        "max_documents": settings.PUBLIC_MAX_DOCUMENTS if settings.PUBLIC_MODE else None,
        "daily_questions": settings.PUBLIC_DAILY_QUESTIONS if settings.PUBLIC_MODE else None,
    }
