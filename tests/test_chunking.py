import pytest

from app.core.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor(chunk_size=5, chunk_overlap=2)


def test_empty_text_returns_no_chunks(processor):
    assert processor.chunk_text("", "empty.pdf") == []


def test_whitespace_only_returns_no_chunks(processor):
    assert processor.chunk_text("   \n\t  ", "blank.pdf") == []


def test_short_text_makes_one_chunk(processor):
    chunks = processor.chunk_text("one two three", "short.pdf")
    assert len(chunks) == 1
    assert chunks[0].text == "one two three"
    assert chunks[0].chunk_index == 0


def test_chunks_overlap(processor):
    text = " ".join(str(n) for n in range(10))
    chunks = processor.chunk_text(text, "nums.pdf")

    first = chunks[0].text.split()
    second = chunks[1].text.split()

    assert first[-2:] == second[:2]


def test_chunk_indexes_are_sequential(processor):
    text = " ".join(str(n) for n in range(30))
    chunks = processor.chunk_text(text, "nums.pdf")

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_source_is_recorded(processor):
    chunks = processor.chunk_text("a b c d e f g", "policy.pdf")
    assert all(c.source == "policy.pdf" for c in chunks)


def test_no_overlap_when_configured():
    processor = DocumentProcessor(chunk_size=3, chunk_overlap=0)
    chunks = processor.chunk_text("a b c d e f", "x.pdf")

    assert chunks[0].text == "a b c"
    assert chunks[1].text == "d e f"


@pytest.mark.parametrize(
    "word_count,size,overlap,expected",
    [
        (10, 5, 0, 2),
        (10, 5, 2, 4),
        (1089, 500, 50, 3),
    ],
)
def test_chunk_count_math(word_count, size, overlap, expected):
    processor = DocumentProcessor(chunk_size=size, chunk_overlap=overlap)
    text = " ".join(["w"] * word_count)
    assert len(processor.chunk_text(text, "x.pdf")) == expected
