from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    # Cat SAU dau cau bang lookbehind, nho vay dau ". " / "! " / "? " / ".\n"
    # van nam lai o cuoi cau thay vi bi nuot mat nhu khi split bang r"[.!?]\s+".
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in self._SENTENCE_BOUNDARY.split(text.strip())]
        sentences = [s for s in sentences if s]
        if not sentences:
            return []

        size = self.max_sentences_per_chunk
        chunks: list[str] = []
        for start in range(0, len(sentences), size):
            group = sentences[start : start + size]
            chunks.append(" ".join(group).strip())
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self._split(text, list(self.separators))

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []

        # Base case 1: manh da du nho -> giu nguyen.
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Base case 2: het separator (hoac separator rong "") -> cat cung theo chunk_size.
        if not remaining_separators or remaining_separators[0] == "":
            return self._hard_split(current_text)

        separator = remaining_separators[0]
        rest = remaining_separators[1:]

        # Base case 3: separator khong xuat hien -> ha xuong separator nho hon.
        if separator not in current_text:
            return self._split(current_text, rest)

        pieces: list[str] = []
        for piece in current_text.split(separator):
            if not piece:
                continue
            if len(piece) > self.chunk_size:
                # Chieu 1 - de quy xuong sau: manh van qua dai thi dung separator nho hon.
                pieces.extend(self._split(piece, rest))
            else:
                pieces.append(piece)

        # Chieu 2 - gom len: noi cac manh nho lien ke toi sat chunk_size. Thieu buoc
        # nay, mot file nhieu dong ngan se sinh ra hang tram chunk vun 5-10 ky tu.
        return self._merge(pieces, separator)

    def _hard_split(self, text: str) -> list[str]:
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _merge(self, pieces: list[str], separator: str) -> list[str]:
        merged: list[str] = []
        buffer = ""
        for piece in pieces:
            if not buffer:
                buffer = piece
                continue
            candidate = buffer + separator + piece
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                merged.append(buffer)
                buffer = piece
        if buffer:
            merged.append(buffer)
        return merged


class HeadingChunker:
    """
    Chunk theo tieu de/muc cua van ban quy dinh — chien luoc rieng cho K4-L3A.

    Ly do thiet ke: van ban hoc vu duoc bien soan theo muc ("## Dieu 4 - ..."),
    moi muc da la mot don vi ngu nghia tron ven do nguoi soan chia san. Cat theo
    heading giu nguyen ranh gioi do, thay vi cat giua cau nhu FixedSizeChunker.

    Chi tiet de bo sot: khi mot section dai hon max_chunk_size, no duoc ha xuong
    RecursiveChunker va tieu de duoc GAN LAI vao tung manh con — khong co no,
    manh thu hai tro di mat ngu canh "day la muc noi ve cai gi".
    """

    _HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 150, keep_heading: bool = True) -> None:
        self.max_chunk_size = max_chunk_size
        # Dong "# Tieu de tai lieu" dung mot minh la mot section dai ~20 ky tu.
        # Chunk vun kieu do khong tra loi duoc gi ma van chiem slot top-k, nen
        # section ngan hon min_chunk_size duoc gop vao section ke tiep.
        self.min_chunk_size = min_chunk_size
        self.keep_heading = keep_heading

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        matches = list(self._HEADING.finditer(text))
        if not matches:
            # Khong co heading nao -> quay ve recursive, dung tra ve nguyen file.
            return RecursiveChunker(chunk_size=self.max_chunk_size).chunk(text)

        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            heading = match.group(2).strip()
            body = text[match.start() : end].strip()
            sections.append((heading, body))

        sections = self._merge_short_sections(sections)

        chunks: list[str] = []
        for heading, body in sections:
            if len(body) <= self.max_chunk_size:
                chunks.append(body)
                continue

            prefix = f"{heading}\n" if (self.keep_heading and heading) else ""
            budget = max(50, self.max_chunk_size - len(prefix))
            for sub_index, sub in enumerate(RecursiveChunker(chunk_size=budget).chunk(body)):
                chunks.append(sub if sub_index == 0 else f"{prefix}{sub}")

        return [c for c in chunks if c.strip()]

    def _merge_short_sections(self, sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
        merged: list[tuple[str, str]] = []
        for heading, body in sections:
            if merged:
                previous_heading, previous_body = merged[-1]
                # Gop ca hai chieu: section truoc qua ngan (dong "## 2. Hoc phi
                # doi voi giao duc dai hoc" dung mot minh vi ngay duoi no la "###
                # 2.1") HOAC section nay qua ngan (muc chi co mot dong so lieu).
                candidate = f"{previous_body}\n\n{body}"
                if len(previous_body) < self.min_chunk_size:
                    # Manh truoc chi la tieu de cut ("## 2. Hoc phi doi voi giao
                    # duc dai hoc" vi ngay duoi no la "### 2.1"): khong mang noi
                    # dung nao ca. Gop VO DIEU KIEN — neu ket qua vuot nguong thi
                    # no se duoc cat tiep o buoc sau va tieu de van duoc gan lai
                    # vao tung manh con. Van hon la de mot chunk rong nghia
                    # chiem mot slot top-k.
                    merged[-1] = (previous_heading or heading, candidate)
                    continue
                if len(body) < self.min_chunk_size and len(candidate) <= self.max_chunk_size:
                    merged[-1] = (previous_heading or heading, candidate)
                    continue
            merged.append((heading, body))
        return merged


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, chunk_size // 10)),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison: dict[str, dict] = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            # Chan chia cho 0 khi text rong / chunker tra ve list rong.
            avg_length = round(sum(len(c) for c in chunks) / count, 2) if count else 0.0
            comparison[name] = {
                "count": count,
                "avg_length": avg_length,
                "min_length": min((len(c) for c in chunks), default=0),
                "max_length": max((len(c) for c in chunks), default=0),
                "chunks": chunks,
            }
        return comparison
