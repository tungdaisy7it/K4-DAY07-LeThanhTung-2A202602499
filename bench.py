"""
bench.py — công cụ đo retrieval cho Lab 07 (K4-L3A).

Chạy:
    python bench.py                      # chiến lược mặc định (heading) + ghi ket_qua_benchmark.txt
    python bench.py --strategy recursive # đổi ĐÚNG MỘT THỨ: chiến lược chunking
    python bench.py --all                # chạy cả 4 chiến lược để so sánh trong nhóm
    python bench.py --baseline           # bảng ChunkingStrategyComparator cho REPORT_NHOM mục 2

Bốn việc nó làm (theo mục 6 của codelab):
    1. Đọc từng file .md, tách frontmatter thành metadata và phần thân thành content
    2. Chunk phần thân NGOÀI store, mỗi chunk thành một Document
    3. Nạp vào EmbeddingStore, chạy 5 query qua search_with_filter()
    4. In top-3 kèm score và doc_id để đối chiếu với gold answer

Chấm ở HAI MỨC (codelab mục 7): mức doc_id (ngây thơ, thổi phồng kết quả) và mức nội dung
(ngữ cảnh truy xuất được có thật sự chứa chuỗi trả lời được câu hỏi không).
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src import (
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    HeadingChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)

CORPUS_DIR = Path("data/hoc-phi")
OUTPUT_FILE = Path("ket_qua_benchmark.txt")

# Chiến lược của tôi (Lê Thanh Tùng): chunk theo heading/section của văn bản quy định.
# Đây là vai R3 bắt buộc của K4-L3A. Đổi chiến lược = đổi đúng dòng này (hoặc --strategy).
DEFAULT_STRATEGY = "heading"

STRATEGIES = {
    "fixed": lambda: FixedSizeChunker(chunk_size=600, overlap=100),
    "sentence": lambda: SentenceChunker(max_sentences_per_chunk=3),
    "recursive": lambda: RecursiveChunker(chunk_size=600),
    "heading": lambda: HeadingChunker(max_chunk_size=800),
}

# ---------------------------------------------------------------------------
# 5 benchmark query của nhóm — dùng chung cho mọi thành viên, mọi chiến lược.
# must_contain = chuỗi đặc trưng PHẢI xuất hiện trong ngữ cảnh truy xuất được.
# Đây là phần chấm ở mức nội dung; thiếu nó thì "top-3 đúng tài liệu" vẫn có thể
# là chunk không chứa câu trả lời.
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Sinh viên các ngành khóa QH-2023 trở về trước đóng bao nhiêu tiền một tín chỉ?",
        "kind": "tra số liệu cụ thể",
        "gold_doc_id": "ussh-hoc-phi-dai-hoc-2026-2027",
        "gold_answer": "530.000đ/tín chỉ/hệ số, thu theo tín chỉ đăng ký học (theo Nghị định 238/2025/NĐ-CP).",
        "must_contain": ["530.000đ/tín chỉ"],
        "metadata_filter": None,
    },
    {
        "id": 2,
        "query": "Học phí một tín chỉ hệ vừa làm vừa học được tính như thế nào?",
        "kind": "MƠ HỒ VỀ ĐỐI TƯỢNG — cần metadata_filter audience=student",
        "gold_doc_id": "ussh-hoc-phi-dai-hoc-2026-2027",
        "gold_answer": "USSH thu 700.000đ/tín chỉ cho các ngành khóa QH-2023 trở về trước và các ngành 'còn lại' của những khóa sau; ngành đã kiểm định thu 880.000đ hoặc 1.050.000đ/tín chỉ.",
        # Không lọc thì tài liệu audience=all (Nghị định 238) chiếm top-k và chỉ
        # trả về QUY TẮC "không vượt quá 150% hệ chính quy", không có con số.
        "must_contain": ["700.000đ/tín chỉ"],
        "metadata_filter": {"audience": "student"},
    },
    {
        "id": 3,
        "query": "Thời gian nộp học phí học lại học kỳ phụ năm học 2025-2026 là khi nào?",
        "kind": "tra mốc thời gian",
        "gold_doc_id": "hvtc-hoc-phi-hoc-lai-ky-phu-2025-2026",
        "gold_answer": "Từ ngày 23/05/2026 đến hết ngày 01/06/2026 (Thông báo 731/TB-HVTC).",
        "must_contain": ["23/05/2026 đến hết ngày 01/06/2026"],
        "metadata_filter": None,
    },
    {
        "id": 4,
        "query": "Lớp học phần học lại có ít sinh viên đăng ký thì học phí tính thế nào?",
        "kind": "hỏi điều kiện / ngưỡng",
        "gold_doc_id": "utt-muc-thu-hoc-phi-2026-2027",
        "gold_answer": "Lớp có từ 4 sinh viên trở lên: bằng mức học phí hiện hành của học phần học kỳ chính. Lớp có dưới 4 sinh viên: bằng 2,5 lần mức học phí hiện hành.",
        "must_contain": ["dưới 4 sinh viên", "2,5 lần"],
        "metadata_filter": None,
    },
    {
        "id": 5,
        "query": "Mức thu học phí một tháng cho một người học là bao nhiêu?",
        "kind": "MƠ HỒ VỀ BẬC HỌC — cần metadata_filter level=postgraduate",
        "gold_doc_id": "ussh-hoc-phi-sau-dai-hoc-2026-2027",
        "gold_answer": "Thạc sĩ: 2.865.000đ/tháng/HV cho các ngành 'còn lại' (2.685.000đ với Quản trị Văn phòng, Khoa học Quản lý); tiến sĩ: 4.775.000đ/tháng/NCS. Bậc đại học là mức khác (1.910.000đ/tháng/SV).",
        "must_contain": ["2.865.000đ/tháng/HV"],
        "metadata_filter": {"level": "postgraduate"},
    },
]


# ---------------------------------------------------------------------------
# 1. Đọc corpus: tách frontmatter thành metadata, phần thân thành content
# ---------------------------------------------------------------------------
def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Tách YAML frontmatter đơn giản (key: value) khỏi phần thân Markdown.

    Dấu phân cách phải là một DÒNG chỉ có "---". Không dùng raw.split("---")
    được: một source_url thật trong corpus là
    ".../quy-dinh-moi-nhat-ve-muc-hoc-phi-tu-nam-hoc-2025---2026" — chuỗi "---"
    nằm ngay giữa URL, tách kiểu đó sẽ cắt nhầm frontmatter làm đôi và nuốt mất
    audience/level. (Script kiểm tra CP2 trong codelab dính đúng lỗi này.)
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw

    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if closing is None:
        return {}, raw

    metadata: dict[str, str] = {}
    for line in lines[1:closing]:
        match = re.match(r"^(\w+):\s*(.+?)\s*(?:\s#.*)?$", line)
        if match:
            metadata[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return metadata, "\n".join(lines[closing + 1 :]).strip()


def load_corpus(corpus_dir: Path) -> list[tuple[Path, dict, str]]:
    documents = []
    for path in sorted(corpus_dir.glob("*.md")):
        if path.stem.lower() == "readme":
            continue
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        if not body.strip():
            print(f"  ! bỏ qua {path.name}: phần thân rỗng")
            continue
        documents.append((path, metadata, body))
    return documents


# ---------------------------------------------------------------------------
# 2 + 3. Chunk NGOÀI store rồi nạp vào EmbeddingStore
# ---------------------------------------------------------------------------
def build_store(corpus, chunker, embedder, collection_name: str) -> tuple[EmbeddingStore, int]:
    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedder)
    total_chunks = 0

    for path, frontmatter, body in corpus:
        chunks = chunker.chunk(body)
        docs = [
            Document(
                # Document.id là id của CHUNK; metadata['doc_id'] mới trỏ về file gốc.
                id=f"{path.stem}#{index}",
                content=chunk,
                # Trải frontmatter vào MỌI chunk, nếu không search_with_filter
                # sẽ không có gì để lọc.
                metadata={
                    **frontmatter,
                    "doc_id": path.stem,
                    "chunk_index": index,
                    "chunk_total": len(chunks),
                    "source_path": str(path).replace("\\", "/"),
                },
            )
            for index, chunk in enumerate(chunks)
        ]
        store.add_documents(docs)
        total_chunks += len(docs)

    return store, total_chunks


# ---------------------------------------------------------------------------
# LLM thay thế: trích xuất câu trả lời TỪ ngữ cảnh, không bịa.
# Dùng khi không có API key — vẫn kiểm được grounding + source traceability.
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "là", "của", "gì", "khi", "nào", "thì", "bị", "được", "có", "và", "các", "những",
    "cho", "trong", "bao", "nhiêu", "thế", "nào?", "gồm", "một", "với", "theo", "ra",
}


def extractive_llm(prompt: str) -> str:
    """Trả lời bằng cách trích câu khớp nhất trong ngữ cảnh + trích dẫn [n]."""
    context_part = prompt.split("=== NGU CANH TRUY XUAT DUOC ===", 1)[-1]
    context_part, question_part = context_part.split("=== CAU HOI ===", 1)
    question = question_part.split("=== CAU TRA LOI", 1)[0].strip()

    keywords = {w.lower().strip(".,?:;") for w in question.split() if len(w) > 2}
    keywords -= _STOPWORDS

    best: list[tuple[int, str, str]] = []
    citation = "?"
    for line in context_part.splitlines():
        header = re.match(r"^\[(\d+)\] \(doc_id=", line)
        if header:
            citation = header.group(1)
            continue
        stripped = line.strip()
        # Bo dong heading: no khop keyword rat de nhung khong chua cau tra loi.
        if stripped.startswith("#"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            if len(sentence) < 20:
                continue
            hits = sum(1 for kw in keywords if kw in sentence.lower())
            if hits:
                best.append((hits, sentence.strip(), citation))

    if not best:
        return KnowledgeBaseAgent.NO_CONTEXT_ANSWER

    best.sort(key=lambda item: item[0], reverse=True)
    picked = best[:2]
    return " ".join(f"{sentence} [{cite}]" for _, sentence, cite in picked)


# ---------------------------------------------------------------------------
# Chấm điểm hai mức
# ---------------------------------------------------------------------------
def grade(query_spec: dict, results: list[dict]) -> dict:
    doc_ids = [r["metadata"].get("doc_id") for r in results]
    context = "\n".join(r["content"] for r in results)

    gold_rank = doc_ids.index(query_spec["gold_doc_id"]) + 1 if query_spec["gold_doc_id"] in doc_ids else 0
    missing = [needle for needle in query_spec["must_contain"] if needle not in context]
    context_has_answer = not missing

    # Mức 1 (ngây thơ): gold doc_id có nằm trong top-3 không.
    naive_score = 2 if gold_rank == 1 else (1 if gold_rank in (2, 3) else 0)

    # Mức 2 (SCORING.md): phải có chunk liên quan TRONG top-3 VÀ ngữ cảnh trả lời được.
    if not context_has_answer or gold_rank == 0:
        strict_score = 0
    elif gold_rank == 1:
        strict_score = 2
    else:
        strict_score = 1

    return {
        "gold_rank": gold_rank,
        "naive_score": naive_score,
        "strict_score": strict_score,
        "context_has_answer": context_has_answer,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Embedding backend (cùng quy tắc fallback như main.py)
# ---------------------------------------------------------------------------
CACHE_DIR = Path(".embedding_cache")


def select_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    try:
        if provider == "local":
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        elif provider == "openai":
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        elif provider == "gemini":
            embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        else:
            return _mock_embed, "mock"

        wrapped = cached(embedder)
        # Gọi thử MỘT lần trước khi chạy: hết credit / sai key / mất mạng thì rơi
        # về mock ngay tại đây, thay vì chết giữa chừng sau khi đã nạp 150 chunk.
        wrapped("kiểm tra kết nối embedding")
        return wrapped, provider
    except Exception as error:  # thiếu thư viện / thiếu key / hết credit -> mock, không crash
        print(f"  ! không dùng được backend '{provider}': {str(error)[:160]}")
        print("  ! quay về MockEmbedder — số liệu score sẽ là nhiễu, xem cảnh báo bên dưới.")
        return _mock_embed, f"{provider} -> mock (fallback)"


def cached(embedder):
    """Cache hai tầng theo nội dung: RAM cho lượt chạy này, đĩa cho lượt sau.

    Codelab yêu cầu cache theo hash nội dung để chạy lại không tốn thêm tiền.
    Cache RAM không đủ: mỗi lần `python bench.py` là một tiến trình mới, nên
    không có cache đĩa thì 4 chiến lược × ~150 chunk sẽ trả tiền lại từ đầu.
    """
    import hashlib
    import json

    model_name = getattr(embedder, "model_name", embedder.__class__.__name__)
    cache_path = CACHE_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", model_name) + ".json")
    try:
        memo: dict[str, list[float]] = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        memo = {}
    dirty = [False]

    def key_of(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def wrapped(text: str) -> list[float]:
        key = key_of(text)
        if key in memo:
            return memo[key]

        # Free tier cua Gemini gioi han 100 request/phut; mot luot --all can ~190
        # lan goi nen cham tran la chuyen chac chan xay ra. Doi dung so giay ma
        # server bao roi thu lai, thay vi de ca luot chay sap.
        for attempt in range(6):
            try:
                memo[key] = embedder(text)
                dirty[0] = True
                if len(memo) % 25 == 0:
                    flush()  # luu dan: sap giua chung van giu duoc phan da tra tien
                return memo[key]
            except Exception as error:
                message = str(error)
                # 429 = het quota phut; 503/500 = server ben kia chap chon, deu la
                # loi tam thoi. Sai key hay sai model thi raise ngay, dung thu lai.
                retryable = any(
                    token in message
                    for token in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL")
                )
                if not retryable or attempt == 5:
                    raise
                match = re.search(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'", message)
                delay = float(match.group(1) or match.group(2)) + 2 if match else 5 * 2**attempt
                flush()
                reason = "rate limit" if "429" in message else "loi tam thoi tu server"
                print(f"  . {reason}, cho {delay:.0f}s roi thu lai...", flush=True)
                time.sleep(delay)
        raise RuntimeError("het so lan thu lai")

    def flush() -> None:
        if not dirty[0]:
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(memo), encoding="utf-8")

    wrapped.flush = flush
    wrapped._backend_name = getattr(embedder, "_backend_name", "unknown") + f" (cache: {len(memo)} vector)"
    return wrapped


# ---------------------------------------------------------------------------
# Chạy benchmark cho MỘT chiến lược
# ---------------------------------------------------------------------------
def run_strategy(strategy_name: str, corpus, embedder, top_k: int = 3) -> dict:
    chunker = STRATEGIES[strategy_name]()
    store, total_chunks = build_store(corpus, chunker, embedder, f"bench_{strategy_name}")

    lengths = [len(record["content"]) for record in store._store]
    print(f"\n{'=' * 78}")
    print(f"CHIẾN LƯỢC: {strategy_name}  ({chunker.__class__.__name__})")
    print(f"{'=' * 78}")
    print(f"Đã nạp {total_chunks} chunk từ {len(corpus)} tài liệu "
          f"| độ dài TB {sum(lengths) / len(lengths):.0f} ký tự "
          f"| ngắn nhất {min(lengths)} | dài nhất {max(lengths)}")

    agent = KnowledgeBaseAgent(store=store, llm_fn=extractive_llm)
    rows = []

    for spec in BENCHMARK_QUERIES:
        results = store.search_with_filter(spec["query"], top_k=top_k, metadata_filter=spec["metadata_filter"])
        scoring = grade(spec, results)
        answer = agent.answer(spec["query"], top_k=top_k, metadata_filter=spec["metadata_filter"])

        print(f"\n--- Q{spec['id']} [{spec['kind']}] ---")
        print(f"Câu hỏi : {spec['query']}")
        print(f"Filter  : {spec['metadata_filter']}")
        print(f"Gold    : {spec['gold_doc_id']} — {spec['gold_answer']}")
        for rank, result in enumerate(results, start=1):
            marker = "<<< GOLD" if result["metadata"].get("doc_id") == spec["gold_doc_id"] else ""
            preview = result["content"].replace("\n", " ")[:110]
            print(f"  top{rank} score={result['score']:+.4f} doc_id={result['metadata'].get('doc_id'):<28}"
                  f" audience={result['metadata'].get('audience', '-'):<8} {marker}")
            print(f"        {preview}...")
        print(f"Agent   : {answer[:300]}")
        print(f"Chấm    : doc_id-level={scoring['naive_score']}/2 | nội dung-level={scoring['strict_score']}/2"
              f" | gold ở hạng {scoring['gold_rank'] or 'KHÔNG CÓ TRONG TOP-3'}")
        if scoring["missing"]:
            print(f"          thiếu trong ngữ cảnh: {scoring['missing']}")

        rows.append({**spec, "results": results, "answer": answer, **scoring})

    naive_total = sum(r["naive_score"] for r in rows)
    strict_total = sum(r["strict_score"] for r in rows)
    print(f"\nTỔNG {strategy_name}: doc_id-level {naive_total}/10 | nội dung-level {strict_total}/10")

    return {
        "strategy": strategy_name,
        "chunker": chunker.__class__.__name__,
        "total_chunks": total_chunks,
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "rows": rows,
        "naive_total": naive_total,
        "strict_total": strict_total,
        "store": store,
    }


# ---------------------------------------------------------------------------
# A/B bắt buộc: câu cần filter, chạy 2 lần có/không metadata_filter
# ---------------------------------------------------------------------------
def run_ab_filter_test(reports: list[dict]) -> None:
    for spec in [q for q in BENCHMARK_QUERIES if q["metadata_filter"]]:
        key = next(iter(spec["metadata_filter"]))
        print(f"\n{'=' * 78}")
        print(f"A/B METADATA FILTER — Q{spec['id']}: {spec['query']}")
        print(f"filter = {spec['metadata_filter']}")
        print(f"{'=' * 78}")

        for report in reports:
            store = report["store"]
            for label, metadata_filter in (("KHÔNG filter", None), ("CÓ filter", spec["metadata_filter"])):
                results = store.search_with_filter(spec["query"], top_k=3, metadata_filter=metadata_filter)
                scoring = grade(spec, results)
                tops = " | ".join(
                    f"{r['metadata'].get('doc_id')}[{key}={r['metadata'].get(key)}] {r['score']:+.3f}"
                    for r in results
                )
                print(f"{report['strategy']:<10} {label:<13} -> {tops}")
                print(f"{'':<24}    nội dung-level {scoring['strict_score']}/2, thiếu={scoring['missing'] or 'không'}")


# ---------------------------------------------------------------------------
# Baseline ChunkingStrategyComparator (REPORT_NHOM mục 2)
# ---------------------------------------------------------------------------
def run_baseline(corpus, sample: int = 3) -> None:
    print(f"\n{'=' * 78}")
    print("BASELINE — ChunkingStrategyComparator().compare(), chunk_size=600")
    print("(đã bỏ frontmatter, chỉ đo phần thân)")
    print(f"{'=' * 78}")
    print(f"{'Tài liệu':<40}{'Chiến lược':<14}{'count':>7}{'avg_len':>10}{'min':>7}{'max':>7}")

    for path, _metadata, body in corpus[:sample]:
        comparison = ChunkingStrategyComparator().compare(body, chunk_size=600)
        for name, stats in comparison.items():
            print(f"{path.stem:<40}{name:<14}{stats['count']:>7}{stats['avg_length']:>10.1f}"
                  f"{stats['min_length']:>7}{stats['max_length']:>7}")


# ---------------------------------------------------------------------------
# Xuat ket qua co cau truc cho giao dien demo (demo/index.html)
# ---------------------------------------------------------------------------
def dump_json(path: Path, reports: list[dict], corpus, provider: str, backend: str, top_k: int) -> None:
    import json

    ab: dict[str, dict] = {}
    for spec in [q for q in BENCHMARK_QUERIES if q["metadata_filter"]]:
        per_strategy = {}
        for report in reports:
            variants = {}
            for label, metadata_filter in (("off", None), ("on", spec["metadata_filter"])):
                results = report["store"].search_with_filter(
                    spec["query"], top_k=top_k, metadata_filter=metadata_filter
                )
                variants[label] = {
                    "hits": [
                        {
                            "doc_id": r["metadata"].get("doc_id"),
                            "audience": r["metadata"].get("audience"),
                            "level": r["metadata"].get("level"),
                            "score": round(r["score"], 4),
                        }
                        for r in results
                    ],
                    **{k: v for k, v in grade(spec, results).items() if k != "naive_score"},
                }
            per_strategy[report["strategy"]] = variants
        ab[str(spec["id"])] = {"filter": spec["metadata_filter"], "per_strategy": per_strategy}

    payload = {
        "generated_at": __import__("datetime").date.today().isoformat(),
        "provider": provider,
        "backend": backend,
        "top_k": top_k,
        "corpus": [
            {
                "doc_id": fm.get("doc_id", path_.stem),
                "title": fm.get("title", path_.stem),
                "audience": fm.get("audience"),
                "level": fm.get("level"),
                "university": fm.get("university"),
                "category": fm.get("category"),
                "source_url": fm.get("source_url"),
                "document_version": fm.get("document_version"),
                "chars": len(body),
            }
            for path_, fm, body in corpus
        ],
        "queries": [
            {k: spec[k] for k in ("id", "query", "kind", "gold_doc_id", "gold_answer", "must_contain", "metadata_filter")}
            for spec in BENCHMARK_QUERIES
        ],
        "strategies": [
            {
                "name": report["strategy"],
                "chunker": report["chunker"],
                "total_chunks": report["total_chunks"],
                "avg_length": round(report["avg_length"]),
                "min_length": report["min_length"],
                "max_length": report["max_length"],
                "naive_total": report["naive_total"],
                "strict_total": report["strict_total"],
                "rows": [
                    {
                        "id": row["id"],
                        "gold_rank": row["gold_rank"],
                        "naive_score": row["naive_score"],
                        "strict_score": row["strict_score"],
                        "missing": row["missing"],
                        "answer": row["answer"],
                        "hits": [
                            {
                                "doc_id": r["metadata"].get("doc_id"),
                                "audience": r["metadata"].get("audience"),
                                "level": r["metadata"].get("level"),
                                "university": r["metadata"].get("university"),
                                "chunk_index": r["metadata"].get("chunk_index"),
                                "score": round(r["score"], 4),
                                "content": r["content"],
                            }
                            for r in row["results"]
                        ],
                    }
                    for row in report["rows"]
                ],
            }
            for report in reports
        ],
        "ab": ab,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nDa ghi du lieu co cau truc vao {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval — Lab 07 K4-L3A")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default=DEFAULT_STRATEGY)
    parser.add_argument("--all", action="store_true", help="chạy cả 4 chiến lược để so sánh")
    parser.add_argument("--baseline", action="store_true", help="chỉ in bảng baseline comparator")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--corpus", default=str(CORPUS_DIR))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--json", dest="json_out", default=None,
                        help="ghi ket qua co cau truc ra file JSON (dung cho giao dien demo)")
    args = parser.parse_args()

    # Console Windows mặc định là cp1252, không in được tiếng Việt có dấu.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    # Ghi song song ra terminal và ra file kết quả.
    buffer = io.StringIO()

    class Tee:
        def write(self, text):
            sys.__stdout__.write(text)
            buffer.write(text)

        def flush(self):
            sys.__stdout__.flush()

    sys.stdout = Tee()

    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        print(f"Không tìm thấy corpus: {corpus_dir}")
        return 1

    corpus = load_corpus(corpus_dir)
    embedder, provider = select_embedder()
    backend_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)

    print("=" * 78)
    print("BENCHMARK RETRIEVAL — Lab 07 K4-L3A | Lê Thanh Tùng — 2A202602499")
    print("=" * 78)
    print(f"Corpus            : {corpus_dir} ({len(corpus)} tài liệu)")
    print(f"Embedding provider: {provider} -> {backend_name}")
    if "mock" in str(backend_name).lower():
        print("CẢNH BÁO: MockEmbedder băm MD5, KHÔNG mã hóa ngữ nghĩa. Mọi điểm score dưới đây")
        print("          là nhiễu; phân tích phải dựa vào count/avg_length/độ mạch lạc của chunk.")
    print(f"Top-k             : {args.top_k}")

    if args.baseline:
        run_baseline(corpus)
    else:
        names = sorted(STRATEGIES) if args.all else [args.strategy]
        reports = [run_strategy(name, corpus, embedder, top_k=args.top_k) for name in names]

        run_ab_filter_test(reports)

        if args.json_out:
            dump_json(Path(args.json_out), reports, corpus, provider, backend_name, args.top_k)

        if len(reports) > 1:
            print(f"\n{'=' * 78}")
            print("SO SÁNH GIỮA CÁC CHIẾN LƯỢC")
            print(f"{'=' * 78}")
            print(f"{'Chiến lược':<12}{'chunk':>7}{'avg_len':>10}{'min':>6}{'max':>6}"
                  f"{'doc_id-level':>15}{'nội dung-level':>17}")
            for report in reports:
                print(f"{report['strategy']:<12}{report['total_chunks']:>7}{report['avg_length']:>10.0f}"
                      f"{report['min_length']:>6}{report['max_length']:>6}"
                      f"{str(report['naive_total']) + '/10':>15}{str(report['strict_total']) + '/10':>17}")

        run_baseline(corpus)

    if hasattr(embedder, "flush"):
        embedder.flush()

    sys.stdout = sys.__stdout__
    Path(args.output).write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
