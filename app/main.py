import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.core.library import VISITOR_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("httpx", "httpcore", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = FastAPI(title="RAG Chatbot")

app.include_router(router, prefix="/api")

logger = logging.getLogger(__name__)
VISITOR_COOKIE = "gw_visitor"


@app.middleware("http")
async def visitor_id(request: Request, call_next):
    """Public demo: give each browser an anonymous id that keys its private library."""
    if not settings.PUBLIC_MODE:
        return await call_next(request)

    visitor = request.cookies.get(VISITOR_COOKIE, "")
    is_new = not VISITOR_ID.match(visitor)
    if is_new:
        visitor = uuid4().hex
    request.state.visitor = visitor

    response = await call_next(request)
    if is_new:
        response.set_cookie(
            VISITOR_COOKIE,
            visitor,
            max_age=60 * 60 * 24 * 30,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
        )
    return response


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
