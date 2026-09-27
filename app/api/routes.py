import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config import settings
from app.core.document_processor import DocumentProcessor
from app.core.rag_chain import RagChain
from app.core.vector_store import VectorStore

router = APIRouter()
processor = DocumentProcessor(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)
store = VectorStore()
rag = RagChain(store)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = processor.extract_pdf(tmp_path)
        chunks = processor.chunk_text(text, file.filename)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF. Is it damaged or password-protected?",
        ) from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted. Is this a scanned PDF?",
        )

    document_id = str(uuid4())
    try:
        stored = store.add_chunks(chunks, document_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}") from e

    return {
        "document_id": document_id,
        "filename": file.filename,
        "total_words": len(text.split()),
        "chunks_stored": stored,
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
        raise HTTPException(
            status_code=502, detail=f"The AI service returned an error: {e}"
        ) from e


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    deleted = store.delete_document(document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted_chunks": deleted}


@router.delete("/reset")
def reset():
    return {"deleted_chunks": store.reset()}
