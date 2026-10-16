from pathlib import Path

from forenote.bench_embed import bench_pdf, chunks_per_page, run_bench, sample_chunks
from forenote.embed.fake import FakeEmbedder
from forenote.kb.chunker import CHUNK_TOKENS
from forenote.kb.parse_process import ParserProcess
from forenote.kb.tokens import estimate

FIX = Path(__file__).resolve().parent / "fixtures" / "kb"


def test_sample_chunks_are_full_size_chunks():
    chunks = sample_chunks(10)
    assert len(chunks) == 10
    assert all(CHUNK_TOKENS * 0.75 <= estimate(c) <= CHUNK_TOKENS + 1 for c in chunks)


def test_run_bench_with_a_fake_embedder():
    r = run_bench(FakeEmbedder())
    assert r.chunks == 48 and r.chunks_per_s > 0
    assert r.pages_per_min == round(60 * r.chunks_per_s / chunks_per_page(), 1)
    assert r.pdf_pages is None


def test_bench_pdf_parses_chunks_and_embeds():
    parser = ParserProcess()
    try:
        pages, rate = bench_pdf(FakeEmbedder(), parser, FIX / "sample.pdf")
    finally:
        parser.close()
    assert pages == 3 and rate > 0
