"""Document libraries: one shared library locally, one per visitor on the public demo.

Each library is its own ChromaDB collection, so visitors can never see,
search or delete each other's files. All libraries share one database
client, one embedding client and one Gemini client.
"""

import re
from collections import OrderedDict
from dataclasses import dataclass

import chromadb

from app.config import settings
from app.core.embeddings import EmbeddingService
from app.core.llm import LLM
from app.core.rag_chain import RagChain
from app.core.vector_store import VectorStore

VISITOR_ID = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class Library:
    store: VectorStore
    rag: RagChain


class Libraries:
    def __init__(self, llm: LLM, keep_open: int = 256):
        self.llm = llm
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self.embeddings = EmbeddingService()
        self.keep_open = keep_open
        self._open: OrderedDict[str, Library] = OrderedDict()

    def get(self, name: str) -> Library:
        """The library with this collection name, opened (or created) on first use."""
        if name in self._open:
            self._open.move_to_end(name)
            return self._open[name]
        store = VectorStore(name, client=self.client, embeddings=self.embeddings)
        library = Library(store=store, rag=RagChain(store, self.llm))
        self._open[name] = library
        if len(self._open) > self.keep_open:
            self._open.popitem(last=False)  # just closes it; the data stays on disk
        return library

    def for_visitor(self, visitor_id: str) -> Library:
        if not VISITOR_ID.match(visitor_id):
            raise ValueError("invalid visitor id")
        return self.get(f"v_{visitor_id}")
