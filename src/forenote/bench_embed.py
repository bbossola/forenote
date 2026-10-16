"""Embedding benchmark: chunks per second and PDF pages per minute (spec C 6.4)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from forenote.kb.chunker import CHUNK_TOKENS, OVERLAP_TOKENS, chunk_sections, split_text

PAGE_TOKENS = 500  # decision 0001: one page has about 500 tokens
SAMPLE = (
    "The team reviews the budget for the next quarter. The costs for hardware grow, and the plan moves two "
    "engineers to the new project in May. The meeting agrees to check the numbers again next week. "
)


@dataclass(frozen=True)
class EmbedBench:
    chunks: int
    chunks_per_s: float
    pages_per_min: float  # an estimate from the chunk speed
    pdf_pages: int | None = None
    pdf_pages_per_min: float | None = None  # measured: parse, chunk, and embed


def sample_chunks(n: int = 48) -> list[str]:
    return split_text(SAMPLE * (n * 8))[:n]  # the sample has about 60 tokens


def chunks_per_page() -> float:
    return PAGE_TOKENS / (CHUNK_TOKENS - OVERLAP_TOKENS)


def bench_embedder(embedder, chunks: list[str], batch: int = 8) -> float:
    embedder.embed_text(chunks[:1], "document")  # warm up
    start = time.perf_counter()
    for i in range(0, len(chunks), batch):
        embedder.embed_text(chunks[i : i + batch], "document")
    return len(chunks) / (time.perf_counter() - start)


def bench_pdf(embedder, parser, path: Path, max_pages: int = 5000) -> tuple[int, float]:
    """Parse, chunk, and embed one PDF. Return the last page with text, and the pages per minute."""
    start = time.perf_counter()
    result = parser.parse(path, max_pages)
    if result.error is not None:
        raise RuntimeError(f"The PDF cannot be parsed ({result.error}).")
    chunks = chunk_sections(result.sections or [])
    for i in range(0, len(chunks), 8):
        embedder.embed_text([c.text for c in chunks[i : i + 8]], "document")
    seconds = time.perf_counter() - start
    pages = max((s.locator.get("page", 0) for s in result.sections or []), default=0)
    return pages, 60 * pages / seconds if seconds > 0 else 0.0


def run_bench(embedder, parser=None, pdf: Path | None = None) -> EmbedBench:
    chunks = sample_chunks()
    cps = round(bench_embedder(embedder, chunks), 2)
    pdf_pages = pdf_rate = None
    if pdf is not None:
        pdf_pages, pdf_rate = bench_pdf(embedder, parser, pdf)
        pdf_rate = round(pdf_rate, 1)
    return EmbedBench(len(chunks), cps, round(60 * cps / chunks_per_page(), 1), pdf_pages, pdf_rate)
