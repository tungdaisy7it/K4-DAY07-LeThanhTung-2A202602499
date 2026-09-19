"""Dựng demo/index.html từ demo/template.html + demo/benchmark.json.

Trang demo phải chạy được offline trong buổi thuyết trình, và khi publish lên
Artifact thì `fetch` bị CSP chặn — nên dữ liệu được nhúng thẳng vào HTML thay vì
tải runtime. Script này làm đúng một việc đó, nên chạy lại benchmark xong chỉ cần:

    python bench.py --all --json demo/benchmark.json
    python demo/build.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = HERE / "template.html"
DATA = HERE / "benchmark.json"
OUTPUT = HERE / "index.html"
MARKER = "__BENCHMARK_DATA__"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    for path in (TEMPLATE, DATA):
        if not path.is_file():
            print(f"Thiếu {path}. Chạy `python bench.py --all --json demo/benchmark.json` trước.")
            return 1

    template = TEMPLATE.read_text(encoding="utf-8")
    if MARKER not in template:
        print(f"Template không có mốc {MARKER}.")
        return 1

    payload = json.loads(DATA.read_text(encoding="utf-8"))

    # Nhúng vào <script type="application/json">: chỉ cần chặn chuỗi đóng thẻ.
    # Dùng dạng nén để trang nhẹ, JSON.parse() ở phía trình duyệt lo phần còn lại.
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    OUTPUT.write_text(template.replace(MARKER, blob), encoding="utf-8")

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Đã dựng {OUTPUT} ({size_kb:.0f} KB)")
    print(f"  backend  : {payload['backend']}")
    print(f"  corpus   : {len(payload['corpus'])} tài liệu")
    print(f"  chiến lược: {', '.join(s['name'] for s in payload['strategies'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
