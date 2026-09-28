import json

import chromadb

from app.config import settings
from app.core.document_processor import DocumentChunk
from app.core.embeddings import EmbeddingService


class VectorStore:
    def __init__(
        self,
        name: str | None = None,
        client: chromadb.ClientAPI | None = None,
        embeddings: EmbeddingService | None = None,
    ):
        # One collection = one library. Locally there is a single shared library;
        # the public demo gives every visitor a collection of their own.
        self.name = name or settings.COLLECTION_NAME
        self.client = client or chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self.embeddings = embeddings or EmbeddingService()
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self, chunks: list[DocumentChunk], document_id: str, content_hash: str | None = None
    ) -> int:
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self.embeddings.embed_documents(texts)

        self.collection.add(
            ids=[f"{document_id}_{c.chunk_index}" for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[self._metadata(c, document_id, content_hash) for c in chunks],
        )

        return len(chunks)

    @staticmethod
    def _metadata(chunk: DocumentChunk, document_id: str, content_hash: str | None) -> dict:
        # Chroma metadata can't hold None, so optional fields are only added when set
        meta = {
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "document_id": document_id,
        }
        if chunk.page_start is not None:
            meta["page_start"] = chunk.page_start
            meta["page_end"] = chunk.page_end
        if content_hash:
            meta["content_hash"] = content_hash
        return meta

    def find_by_hash(self, content_hash: str) -> str | None:
        """Name of an already-stored document with identical content, if any."""
        found = self.collection.get(
            where={"content_hash": content_hash}, limit=1, include=["metadatas"]
        )
        return found["metadatas"][0]["source"] if found["ids"] else None

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
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
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

    def fingerprint(self) -> int:
        """Changes whenever documents are added or removed (used to key the answer cache)."""
        return hash(tuple(sorted(self.collection.get(include=[])["ids"])))

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
        self.client.delete_collection(self.name)
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine"},
        )
        return count

    # ── suggested questions ─────────────────────────────────────────
    # Stored as JSON on each document's first chunk, so deleting the
    # document (or resetting the library) removes them automatically.

    def _first_chunk(self, document_id: str) -> dict:
        return self.collection.get(
            where={"$and": [{"document_id": document_id}, {"chunk_index": 0}]},
            include=["metadatas"],
        )

    def set_questions(self, document_id: str, questions: list[str]) -> None:
        first = self._first_chunk(document_id)
        if first["ids"]:
            self.collection.update(
                ids=first["ids"][:1],
                metadatas=[{"questions": json.dumps(questions)}],
            )

    def get_questions(self, document_id: str) -> list[str] | None:
        """Stored questions, or None if they were never generated."""
        first = self._first_chunk(document_id)
        if not first["ids"]:
            return None
        raw = first["metadatas"][0].get("questions")
        return json.loads(raw) if raw else None

    def document_texts(self, document_id: str) -> list[str]:
        """All chunk texts for one document, in reading order."""
        data = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        pairs = sorted(
            zip(data["metadatas"], data["documents"], strict=True),
            key=lambda pair: pair[0]["chunk_index"],
        )
        return [text for _, text in pairs]
