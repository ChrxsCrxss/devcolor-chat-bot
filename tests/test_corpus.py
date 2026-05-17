from pathlib import Path

from devcolor_rag.corpus import build_chunks, load_faq

CORPUS = Path(__file__).resolve().parents[1] / "data" / "devcolorfaq.txt"


def test_load_faq_count():
    entries = load_faq(CORPUS)
    assert len(entries) == 10


def test_chunks_have_parent():
    entries = load_faq(CORPUS)
    chunks = build_chunks(entries)
    assert len(chunks) > len(entries)
    assert all(c.parent_id >= 1 for c in chunks)
