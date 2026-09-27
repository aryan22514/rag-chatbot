from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(title="RAG Chatbot")

app.include_router(router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "Ok"}


@app.get("/")
def home():
    return FileResponse("public/index.html")


app.mount("/", StaticFiles(directory="public"), name="static")
