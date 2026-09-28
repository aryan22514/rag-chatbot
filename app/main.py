import logging

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(title="RAG Chatbot")

app.include_router(router, prefix="/api")

logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
    """Last line of defence: log the details, show the user a plain message."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on our side. Please try again."},
    )


@app.get("/api/health")
def health():
    return {"status": "Ok"}


@app.get("/")
def home():
    return FileResponse("public/index.html")


app.mount("/", StaticFiles(directory="public"), name="static")
