from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_index: int


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_pdf(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)

    def chunk_text(self, text: str, source: str) -> list[DocumentChunk]:
        words = text.split()

        if not words:
            return []

        chunks = []
        stride = self.chunk_size - self.chunk_overlap

        for i, start in enumerate(range(0, len(words), stride)):
            chunk_words = words[start : start + self.chunk_size]
            piece = " ".join(chunk_words).strip()

            if piece:
                chunks.append(
                    DocumentChunk(
                        text=piece,
                        source=source,
                        chunk_index=i,
                    )
                )

        return chunks
