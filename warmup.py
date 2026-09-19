"""
warmup.py — Phần 1 (khởi động) và Bài tập 3.3 (dự đoán độ tương tự).

Chạy:
    python warmup.py

In ra:
    1. Bài toán chunking: công thức so với số chunk FixedSizeChunker thật sự sinh ra
    2. 5 cặp câu + dự đoán của tôi + điểm cosine thực tế + dự đoán đúng/sai
"""

from __future__ import annotations

import math
import sys

from bench import select_embedder
from src import FixedSizeChunker, compute_similarity

# Dự đoán được ghi TRƯỚC khi chạy (Bài tập 3.3).
# "cao"  = kỳ vọng > 0.5 với embedder thật
# "thấp" = kỳ vọng < 0.3 với embedder thật
SENTENCE_PAIRS = [
    {
        "a": "Mức thu học phí là 530.000đ mỗi tín chỉ.",
        "b": "Mỗi tín chỉ người học phải nộp năm trăm ba mươi nghìn đồng.",
        "predict": "cao",
        "why": "Cùng nghĩa, khác hoàn toàn cách viết (số vs chữ) — phép thử xem embedding hiểu nghĩa hay chỉ so khớp ký tự.",
    },
    {
        "a": "Học phí hệ vừa làm vừa học không vượt quá 150% mức thu của hệ chính quy.",
        "b": "Hệ vừa học vừa làm đóng cao hơn hệ chính quy nhưng tối đa chỉ gấp rưỡi.",
        "predict": "cao",
        "why": "Cùng một quy tắc, diễn đạt hành chính vs diễn đạt đời thường.",
    },
    {
        "a": "Thời gian nộp học phí từ ngày 23/05/2026 đến hết ngày 01/06/2026.",
        "b": "Nhà trường hiện có 15 chương trình đào tạo đại học đã đạt kiểm định chất lượng.",
        "predict": "thấp",
        "why": "Cùng nằm trong chủ đề học phí/đào tạo nhưng nói hai chuyện không liên quan (mốc thu tiền vs số chương trình kiểm định).",
    },
    {
        "a": "Học phí trình độ thạc sĩ bằng mức học phí đại học nhân hệ số 1,5.",
        "b": "Học phí trình độ tiến sĩ bằng mức học phí đại học nhân hệ số 2,5.",
        "predict": "cao",
        "why": "Bẫy cố ý: hai câu gần như giống hệt về hình thức nhưng KHÁC ĐÁP ÁN. Dự đoán điểm cao — và đó chính là chế độ hỏng của retrieval trên corpus học phí.",
    },
    {
        "a": "Lớp học phần có dưới 4 sinh viên thì học phí bằng 2,5 lần mức hiện hành.",
        "b": "Lớp học phần có dưới 4 sinh viên thì học phí bằng 2,5 lần mức hiện hành.",
        "predict": "cao",
        "why": "Hai câu giống hệt — mốc kiểm chứng, phải ra đúng 1.0.",
    },
]


def chunking_math() -> None:
    print("=" * 78)
    print("BÀI TẬP 1.2 — Bài toán chunking")
    print("=" * 78)

    doc_length, chunk_size = 10_000, 500
    for overlap in (50, 100):
        formula = math.ceil((doc_length - overlap) / (chunk_size - overlap))
        actual = len(FixedSizeChunker(chunk_size=chunk_size, overlap=overlap).chunk("a" * doc_length))
        print(f"  doc={doc_length}, chunk_size={chunk_size}, overlap={overlap:>3}"
              f" -> công thức ceil(({doc_length}-{overlap})/({chunk_size}-{overlap})) = {formula:>2}"
              f" | FixedSizeChunker thực tế = {actual:>2}"
              f" | {'khớp' if formula == actual else 'LỆCH'}")


def similarity_predictions() -> None:
    embedder, provider = select_embedder()
    backend = getattr(embedder, "_backend_name", embedder.__class__.__name__)

    print()
    print("=" * 78)
    print("BÀI TẬP 3.3 — Dự đoán độ tương tự cosine")
    print("=" * 78)
    print(f"Embedding backend: {provider} -> {backend}")
    if "mock" in str(backend).lower():
        print("CẢNH BÁO: mock embedder không mã hóa ngữ nghĩa — điểm dưới đây là nhiễu,")
        print("          chỉ cặp số 5 (hai câu giống hệt) là còn ý nghĩa.")
    print()

    for index, pair in enumerate(SENTENCE_PAIRS, start=1):
        score = compute_similarity(embedder(pair["a"]), embedder(pair["b"]))
        actual = "cao" if score > 0.5 else ("thấp" if score < 0.3 else "trung bình")
        verdict = "ĐÚNG" if actual == pair["predict"] else "SAI"
        print(f"Cặp {index} | dự đoán={pair['predict']:<5} | thực tế={score:+.4f} ({actual:<10}) | {verdict}")
        print(f"        A: {pair['a']}")
        print(f"        B: {pair['b']}")
        print(f"        Lý do dự đoán: {pair['why']}")
        print()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    chunking_math()
    similarity_predictions()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
