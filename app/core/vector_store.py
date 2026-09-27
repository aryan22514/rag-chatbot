import chromadb

from app.config import settings
from app.core.document_processor import DocumentChunk
from app.core.embeddings import EmbeddingService


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self.embeddings = EmbeddingService()
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk], document_id: str) -> int:
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self.embeddings.embed_documents(texts)

        self.collection.add(
            ids=[f"{document_id}_{c.chunk_index}" for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "document_id": document_id,
                }
                for c in chunks
            ],
        )

        return len(chunks)

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        if self.collection.count() == 0:
            return []

        query_vector = self.embeddings.embed_query(question)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            hits.append({
                "text": text,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "score": round(1 - distance, 4),
            })

        return hits

    def list_documents(self) -> list[dict]:
        data = self.collection.get(include=["metadatas"])

        docs = {}
        for meta in data["metadatas"]:
            doc_id = meta["document_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "source": meta["source"],
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def stats(self) -> dict:
        return {
            "total_chunks": self.collection.count(),
            "total_documents": len(self.list_documents()),
        }

    def delete_document(self, document_id: str) -> int:
        found = self.collection.get(where={"document_id": document_id}, include=[])
        if not found["ids"]:
            return 0
        self.collection.delete(ids=found["ids"])
        return len(found["ids"])

    def reset(self) -> int:
        count = self.collection.count()
        self.client.delete_collection(settings.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return count
