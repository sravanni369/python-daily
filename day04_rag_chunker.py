"""Day 04 - Mini RAG text chunker.

A dependency-free chunker for Retrieval-Augmented Generation (RAG) pipelines.
It splits long text into overlapping chunks of a target size while respecting
natural boundaries (paragraphs, then sentences) so an idea is rarely cut in the
middle. Sizing is measured in words here to stay dependency-free; swap in a real
tokenizer (e.g. tiktoken) for production token counts.

Why overlap? A sentence that lands on a chunk boundary would otherwise be split
across two chunks and lose its meaning. A small overlap keeps it whole in at
least one chunk, which improves retrieval quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single retrievable piece of text plus its position metadata."""

    index: int
    text: str
    word_count: int
    start_word: int
    end_word: int


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on ., ! or ? followed by whitespace.

    This is a deliberately simple, dependency-free splitter. It keeps the
    terminating punctuation attached to each sentence.
    """
    text = text.strip()
    if not text:
        return []
    # Split after sentence-ending punctuation followed by a space.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def split_units(text: str) -> list[str]:
    """Break text into the smallest natural units: sentences within paragraphs.

    Paragraphs (blank-line separated) are honored first, then each paragraph is
    split into sentences. This ordering means we never merge across a paragraph
    break unless a chunk still has room.
    """
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if para:
            units.extend(split_sentences(para))
    return units


def chunk_text(text: str, chunk_size: int = 60, overlap: int = 10) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of ~``chunk_size`` words.

    Args:
        text: The document to chunk.
        chunk_size: Target maximum words per chunk.
        overlap: How many trailing words to repeat at the start of the next
            chunk so ideas on a boundary stay whole. Must be < ``chunk_size``.

    Returns:
        A list of :class:`Chunk` objects in reading order.

    Raises:
        ValueError: If ``overlap`` is not smaller than ``chunk_size``.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Flatten to a word stream but remember sentence boundaries so we can end a
    # chunk on a boundary instead of mid-sentence when possible.
    units = split_units(text)
    words: list[str] = []
    boundary_after: set[int] = set()  # word indices that end a sentence
    for unit in units:
        unit_words = unit.split()
        if not unit_words:
            continue
        words.extend(unit_words)
        boundary_after.add(len(words) - 1)

    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    n = len(words)
    step = chunk_size - overlap

    while start < n:
        hard_end = min(start + chunk_size, n)
        # Prefer to end on a sentence boundary within the last third of the
        # window, so we don't cut a sentence in half unless we have to.
        end = hard_end
        floor = start + max(1, int(chunk_size * 0.67))
        for i in range(hard_end - 1, floor - 1, -1):
            if i in boundary_after:
                end = i + 1
                break

        chunk_words = words[start:end]
        chunks.append(
            Chunk(
                index=len(chunks),
                text=" ".join(chunk_words),
                word_count=len(chunk_words),
                start_word=start,
                end_word=end,
            )
        )

        if end >= n:
            break
        # Advance, keeping ``overlap`` words of context. Use the actual end so a
        # boundary-shortened chunk still overlaps correctly.
        start = max(end - overlap, start + step)

    return chunks


def _demo() -> None:
    sample = (
        "Retrieval-augmented generation grounds a language model in your own "
        "data. First the documents are split into chunks. Each chunk is embedded "
        "into a vector and stored in a vector database. At query time the most "
        "similar chunks are retrieved and placed in the prompt.\n\n"
        "Chunk size is the quiet decision that shapes answer quality. If chunks "
        "are too large the answer is diluted by noise. If they are too small the "
        "context is severed and retrieval misses the meaning. A small overlap "
        "keeps a sentence on the boundary whole in at least one chunk."
    )

    chunks = chunk_text(sample, chunk_size=30, overlap=6)

    print(f"Input words : {len(sample.split())}")
    print(f"Chunks made : {len(chunks)} (size=30, overlap=6)\n")
    for c in chunks:
        print(f"[chunk {c.index}] words {c.start_word}-{c.end_word} "
              f"({c.word_count} words)")
        print(f"    {c.text}\n")

    # Prove the overlap: the tail of chunk 0 reappears at the head of chunk 1.
    if len(chunks) >= 2:
        tail = chunks[0].text.split()[-6:]
        head = chunks[1].text.split()[:6]
        print(f"Overlap check -> tail of chunk 0: {tail}")
        print(f"              -> head of chunk 1: {head}")


if __name__ == "__main__":
    _demo()
