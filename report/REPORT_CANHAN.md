# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lê Thanh Tùng — MSSV 2A202602499
**Nhóm:** Aura (K4-L3A)
**Ngày:** 2026-09-19

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất (10).

| Thiết lập đo | Giá trị |
| --- | --- |
| Corpus | `data/hoc-phi/` — 5 tài liệu học phí crawl từ website chính thức của 4 trường |
| Embedding backend | `gemini-embedding-001` (Google Gemini, free tier), 3072 chiều, vector chuẩn hóa `‖v‖ = 1` |
| Chiến lược của tôi | `HeadingChunker(max_chunk_size=800, min_chunk_size=150)` — chunk theo tiêu đề/mục |
| Lệnh tái lập | `python bench.py --all` · `python warmup.py` · `python -m pytest tests/ -v` |
| Kết quả đầy đủ | `ket_qua_benchmark.txt` |

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**
> Hai vector embedding chỉ về **cùng một hướng** trong không gian nhiều chiều, nghĩa là mô hình đặt hai đoạn text vào cùng một vùng ngữ nghĩa. Cosine chỉ đo **góc**, không đo độ dài — "gần nhau" ở đây là gần về *nội dung*, không phải gần về *số lượng chữ*.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Mức thu học phí là 530.000đ mỗi tín chỉ."
- Câu B: "Mỗi tín chỉ người học phải nộp năm trăm ba mươi nghìn đồng."
- Tại sao tương đồng: cùng một phát biểu, nhưng một câu viết số còn câu kia viết chữ, và dùng từ khác nhau (mức thu/phải nộp, học phí/—). Nếu điểm vẫn cao thì embedding đang hiểu nghĩa chứ không so khớp ký tự. **Đo thật: +0.8796.**

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Thời gian nộp học phí từ ngày 23/05/2026 đến hết ngày 01/06/2026."
- Câu B: "Nhà trường hiện có 15 chương trình đào tạo đại học đã đạt kiểm định chất lượng."
- Tại sao khác: cùng nằm trong bối cảnh học phí/đào tạo nhưng nói hai chuyện không liên quan — một mốc thu tiền và một con số kiểm định. **Đo thật: +0.5431** — thấp hơn hẳn các cặp đồng nghĩa nhưng **không thấp tuyệt đối**; xem phân tích ở mục 4.

**Tại sao cosine được ưu tiên hơn khoảng cách Euclid cho text embeddings?**
> Độ dài vector embedding bị chi phối bởi độ dài văn bản và tần suất từ, trong khi *hướng* mới mang nghĩa. Euclid phạt hai đoạn cùng chủ đề chỉ vì một đoạn dài hơn; cosine chuẩn hóa độ dài đi nên một câu 20 chữ và một đoạn 200 chữ nói cùng một điều vẫn được coi là gần nhau. Thêm nữa, khi vector đã chuẩn hóa thì cosine **bằng đúng** dot product — tôi đã kiểm chứng điều này với backend đang dùng (`‖v‖ = 1.0`, `dot = cosine = 0.8341` trên cùng một cặp), nên `EmbeddingStore.search()` chỉ cần một phép nhân vô hướng là xong.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10.000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: `ceil((10000 − 50) / (500 − 50))` = `ceil(9950 / 450)` = `ceil(22.11)` = **23 chunks**.
> Mỗi bước trượt `500 − 50 = 450` ký tự; chunk đầu phủ 500 ký tự, mỗi chunk sau phủ thêm 450.
> **Đã kiểm chứng bằng code thật, không tin công thức suông** (`python warmup.py`):
> ```
> doc=10000, chunk_size=500, overlap= 50 -> công thức = 23 | FixedSizeChunker thực tế = 23 | khớp
> doc=10000, chunk_size=500, overlap=100 -> công thức = 25 | FixedSizeChunker thực tế = 25 | khớp
> ```

**Nếu overlap tăng lên 100, số chunk thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Bước trượt giảm còn 400 nên số chunk **tăng từ 23 lên 25** (+8,7% chi phí embedding và lưu trữ). Đánh đổi lại là **bảo toàn ngữ cảnh ở ranh giới**: một câu bị cắt đôi ở mốc 500 sẽ xuất hiện trọn vẹn trong chunk kế tiếp.
> Với corpus học phí điều này không phải lý thuyết — Q1 của nhóm hỏi mức thu một tín chỉ, và chiến lược `fixed` (overlap 100) **vẫn trượt** vì chuỗi `530.000đ/tín chỉ` rơi đúng vào chỗ nối. Overlap lớn hơn là cách rẻ nhất để giảm rủi ro đó.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng `re.split(r"(?<=[.!?])\s+", text)` — **lookbehind**, cắt ở khoảng trắng *sau* dấu câu. Nếu split bằng `[.!?]\s+` như phản xạ đầu tiên thì dấu câu bị nuốt mất và mọi chunk thành câu cụt. Sau khi tách, strip từng câu, bỏ câu rỗng, rồi gom theo `max_sentences_per_chunk` bằng bước nhảy `range(0, n, size)`.
> Edge case tôi **chưa** xử lý được và chủ động nêu ra: chữ viết tắt (`TS.`, `v.v.`) và số thập phân bị coi là kết câu. Với corpus học phí đây là rủi ro thật — chuỗi `Số: 731/TB-HVTC ngày 18 tháng 5 năm 2026.` và các mốc `23/05/2026` nằm cạnh dấu chấm câu. Cách sửa nếu có thêm thời gian: negative lookbehind loại danh sách viết tắt, hoặc yêu cầu ký tự sau khoảng trắng phải là chữ hoa.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán có **hai chiều** và tôi viết cả hai. *Xuống sâu:* thử separator theo thứ tự `["\n\n", "\n", ". ", " ", ""]`; mảnh nào vẫn dài hơn `chunk_size` thì gọi lại `_split` với phần đuôi danh sách. *Gom lên:* `_merge()` nối các mảnh nhỏ liền kề (bằng đúng separator vừa cắt) tới sát `chunk_size` — thiếu bước này, một file nhiều dòng ngắn sinh ra hàng trăm chunk vụn và retrieval hỏng hoàn toàn.
> Ba base case: (1) `len(text) <= chunk_size` → trả nguyên mảnh; (2) hết separator **hoặc** separator đầu là `""` → cắt cứng theo `chunk_size` (nhánh mà `test_empty_separators_falls_back_gracefully` nhắm tới, truyền thẳng `separators=[]`); (3) separator không xuất hiện → bỏ qua, hạ xuống separator nhỏ hơn.
> Bằng chứng bước gom chạy đúng: trên corpus thật, `recursive` cho min = 197, max = 599 với `chunk_size=600` — **không chunk nào chạm trần**, tức nó cắt ở ranh giới đoạn thật chứ không cắt cứng.

**`HeadingChunker`** (chiến lược riêng của tôi — xem `REPORT_NHOM.md` mục 2):
> Regex `^(#{1,6})\s+(.+)$` với `re.MULTILINE` tìm mọi heading, cắt văn bản thành section theo vị trí heading. Ba chi tiết quyết định chất lượng:
> - Section dài hơn `max_chunk_size` được hạ xuống `RecursiveChunker` và **gắn lại tiêu đề vào từng mảnh con** — không có nó, mảnh thứ hai trở đi mất ngữ cảnh "đây là mục nói về cái gì".
> - Section ngắn hơn `min_chunk_size` được gộp với section liền kề. Trường hợp đặc biệt: một dòng heading đứng một mình (`## 2. Học phí đối với giáo dục đại học (Điều 10)`, vì ngay dưới nó là `### 2.1`) thì gộp **vô điều kiện** về phía sau, kể cả khi vượt ngưỡng — phần tràn tối đa chỉ bằng chính độ dài tiêu đề, vẫn hơn là để một chunk rỗng nghĩa chiếm một slot top-k.
> - File không có heading nào thì fallback về `RecursiveChunker` thay vì trả về nguyên file.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Làm **hai helper trước, bốn method công khai sau**, nếu không sẽ viết lặp cùng một logic bốn lần. `_make_record()` chuẩn hóa một `Document` thành record: **copy** metadata (`dict(doc.metadata or {})`) thay vì dùng trực tiếp object của caller, rồi `setdefault("doc_id", doc.id)` — `delete_document` phụ thuộc khóa này, và `setdefault` (không phải gán đè) giữ nguyên `doc_id` trỏ về **file gốc** mà `bench.py` đã gắn khi chunk một file thành `file#0`, `file#1`.
> `_search_records()` chạy similarity trên **một tập record bất kỳ**; `search()` chỉ là `_search_records(query, self._store, top_k)`. Vector chuẩn hóa nên dot product bằng đúng cosine, dùng lại `_dot` có sẵn. Kết quả **bỏ khóa `embedding`** đi cho sạch output — với `gemini-embedding-001` mỗi vector là 3072 số, in ra terminal thì không đọc được gì.
> `add_documents` **không tự chunk** — 1 `Document` = 1 record, đúng như test mong đợi. Việc chunk nằm ở tầng ngoài (`bench.py`).
> Tôi **bỏ hẳn nhánh ChromaDB**: không test nào cần nó, `requirements.txt` không cài nó, và code khởi tạo sẵn có bẫy — `self._use_chroma = True` được gán *trước* khi client được tạo, nên nếu máy chấm bài tình cờ có `chromadb` thì mọi method rẽ vào nhánh chưa cài đặt và cả 14 test store sập.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> **Lọc TRƯỚC rồi mới search.** Nếu lấy top-k xong mới bỏ cái không khớp thì k slot có thể đã bị tài liệu sai chiếm hết và trả về 0 kết quả *dù store vẫn còn tài liệu hợp lệ*. Đây không phải lo xa: ở Q2, chiến lược của tôi cho top-3 **toàn bộ** là tài liệu `audience=all` (bài về Nghị định 238). Lọc sau thì còn 0 kết quả; lọc trước thì được 3 tài liệu sinh viên và điểm nội dung tăng từ 0 lên 1.
> Filter là phép AND trên mọi cặp key-value. Cả `search()` và `search_with_filter()` đều đi qua `_search_records()`, khác nhau **duy nhất ở tập ứng viên** — nhờ vậy hai đường không thể lệch kết quả và `test_no_filter_returns_all_candidates` pass hiển nhiên.
> `delete_document` lọc ngược: giữ lại mọi record có `metadata['doc_id'] != doc_id`, so sánh kích thước trước/sau để trả `True`/`False`. Xóa **mọi chunk** của một file chỉ bằng một lần duyệt.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Ba nhịp: `search_with_filter(...)` (tôi cho agent nhận luôn `metadata_filter` để benchmark chạy được câu cần lọc) → dựng prompt → gọi `llm_fn`.
> Ngữ cảnh được **đánh số `[1] [2] [3]` kèm `doc_id`, `audience`, `source_url` và score**, và prompt yêu cầu model trích dẫn số đó. Nhờ vậy mỗi câu trả lời truy vết được về đúng chunk và đúng file — tiêu chí *Source Traceability* trong `docs/EVALUATION.md`. Với corpus học phí gom từ 4 trường khác nhau thì đây không phải tính năng phụ: cùng một câu hỏi có 4 đáp án đúng khác nhau, người đọc **bắt buộc** phải biết con số đến từ trường nào.
> Chống bịa bằng hai lớp: (a) prompt ràng buộc "chỉ dùng ngữ cảnh được cung cấp"; (b) store rỗng hoặc filter loại hết ứng viên thì `answer()` trả thẳng `NO_CONTEXT_ANSWER` — **không crash và không gọi LLM vô ích**.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```
$ python -m pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.13.2, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\AI thực chiến\Labs\K4-DAY07-LeThanhTung-2A202602499
plugins: anyio-4.14.1, asyncio-1.4.0, respx-0.23.1
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
... (38 test còn lại đều PASSED)
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.06s ==============================
```

**Số lượng bài test vượt qua:** **42 / 42**

`python main.py "Chunking là gì?"` chạy được từ đầu đến cuối.

### Hai sửa lỗi ngoài phạm vi TODO

Cả hai đều lộ ra khi chạy trên **dữ liệu thật**, dữ liệu mẫu không bao giờ chạm tới:

1. **`main.py` crash `UnicodeEncodeError`** trên Windows: `sys.stdout` mặc định là cp1252 nên in preview chunk tiếng Việt là chết. Thêm `stream.reconfigure(encoding="utf-8", errors="replace")` vào `main()`.
2. **Fallback embedder của `main.py` chỉ bọc constructor, không bọc lần gọi đầu.** README hứa "chọn `openai` mà thiếu thiết lập thì tự quay về mock", nhưng một API key **hợp lệ mà hết credit** chỉ báo lỗi ở lần gọi đầu tiên — lúc đó store đã bắt đầu chạy và cả script sập. Tôi gặp đúng tình huống này khi key OpenRouter hết credit giữa buổi. Sửa: gọi thử embedder một lần ngay sau khi khởi tạo, hỏng thì rơi về mock kèm thông báo. `bench.py` dùng cùng cách.

Không đụng gì tới logic được chấm bởi test.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Chạy: `python warmup.py`. Dự đoán được ghi **trong code trước khi chạy** (hằng `SENTENCE_PAIRS`), không điền ngược sau khi thấy kết quả. Ngưỡng phân loại đặt trước: `> 0.5` là "cao", `< 0.3` là "thấp".

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Mức thu học phí là 530.000đ mỗi tín chỉ. | Mỗi tín chỉ người học phải nộp năm trăm ba mươi nghìn đồng. | cao | **+0.8796** | ✓ |
| 2 | Học phí hệ VLVH không vượt quá 150% mức thu của hệ chính quy. | Hệ vừa học vừa làm đóng cao hơn hệ chính quy nhưng tối đa chỉ gấp rưỡi. | cao | **+0.8971** | ✓ |
| 3 | Thời gian nộp học phí từ 23/05/2026 đến hết 01/06/2026. | Nhà trường hiện có 15 chương trình đào tạo đại học đã đạt kiểm định. | thấp | **+0.5431** | ✗ |
| 4 | Học phí thạc sĩ bằng mức học phí đại học nhân **hệ số 1,5**. | Học phí tiến sĩ bằng mức học phí đại học nhân **hệ số 2,5**. | cao | **+0.8589** | ✓ |
| 5 | *(hai câu giống hệt nhau)* | — | cao | **+1.0000** | ✓ |

**4/5 dự đoán đúng.**

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ là **cặp 3 ra 0.54** — tôi chọn hai câu cố tình không liên quan mà vẫn vượt ngưỡng "cao". Nhưng sai lầm nằm ở **ngưỡng của tôi**, không phải ở mô hình: với `gemini-embedding-001`, hai câu tiếng Việt bất kỳ cùng nằm trong ngữ cảnh "trường đại học" đã có sàn tương đồng khoảng 0.5. Cặp đồng nghĩa thật thì lên 0.86–0.90. Nghĩa là **giá trị cosine tuyệt đối không so sánh được giữa các mô hình**; thứ có ý nghĩa là *thứ hạng* trong cùng một không gian. Đây chính là lý do `search()` sắp xếp rồi lấy top-k chứ không cắt theo một ngưỡng cứng.
>
> Đáng lo hơn là **cặp 4: 0.8589**. Hai câu này khác nhau đúng ở chỗ quan trọng nhất — 1,5 với thạc sĩ và 2,5 với tiến sĩ — nhưng embedding gần như không phân biệt được, vì chúng giống hệt nhau về cấu trúc và chỉ lệch một chữ số. Với corpus học phí, nơi câu trả lời **là** con số, đây là chế độ hỏng nguy hiểm nhất: retrieval sẽ vui vẻ trả về điều khoản đúng chủ đề nhưng sai con số. Nó biện minh trực tiếp cho việc tôi chấm benchmark ở mức nội dung (`must_contain`) thay vì chỉ kiểm `doc_id`, và cho việc tách corpus theo `level` để filter chặn trước nhầm lẫn này.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chiến lược: **`HeadingChunker(max_chunk_size=800, min_chunk_size=150)`** — vai R3 của nhóm (chunk theo tiêu đề/mục văn bản quy định).
Lệnh: `python bench.py --strategy heading` → **39 chunk** từ 5 tài liệu, độ dài TB 492 ký tự (min 125, max 798).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được | Score | Gold ở hạng | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên khóa QH-2023 trở về trước đóng bao nhiêu một tín chỉ? | `ussh-hoc-phi-dai-hoc` — mục 1.1 | +0.8630 | **1** ✓ | Trích đúng **530.000đ/tín chỉ/hệ số** [3] |
| 2 | Học phí một tín chỉ hệ vừa làm vừa học tính thế nào? *(filter `audience=student`)* | `utt-muc-thu-hoc-phi` — mục 2 | +0.7768 | **2** | Trích mức của UTT (590.000/650.000đ) thay vì USSH — **đúng loại thông tin, sai trường** |
| 3 | Thời gian nộp học phí học lại kỳ phụ 2025-2026? | `hvtc-hoc-phi-hoc-lai` — phần mở đầu | +0.7966 | **1** ✓ | Chunk chứa mốc **23/05/2026 – 01/06/2026**; agent trích câu căn cứ trước đó |
| 4 | Lớp học lại ít sinh viên thì học phí tính thế nào? | `utt-muc-thu-hoc-phi` — mục 7 | +0.8581 | **1** ✓ | Trích đủ cả hai vế: **từ 4 SV trở lên** = mức hiện hành, **dưới 4 SV** = **2,5 lần** [1] |
| 5 | Mức thu học phí một tháng cho một người học? *(filter `level=postgraduate`)* | `ussh-hoc-phi-sau-dai-hoc` — mục 1.1 | +0.7561 | **1** ✓ | Trích đúng **2.685.000đ** và **2.865.000đ/tháng/HV** [1] |

**Bao nhiêu câu hỏi trả về chunk liên quan trong top-3?** **5 / 5.**
**Điểm doc_id-level: 9/10 · Điểm nội dung-level: 9/10.**

Chỉ Q2 không đạt tối đa: gold xếp hạng 2 thay vì hạng 1. Lý do là câu hỏi không nêu tên trường, mà corpus có **4 trường cùng nói về học phí tín chỉ** — UTT nói thẳng cụm "đại học chính quy, VLVH" trong tiêu đề mục nên thắng về mặt ngữ nghĩa, dù con số gold nằm ở USSH. Đây là giới hạn của chính câu hỏi, không phải của chiến lược: với corpus đa nguồn, một câu hỏi không nêu trường thì **không có đáp án duy nhất đúng**.

**So với ba chiến lược còn lại** — số liệu dưới đây là **tôi tự chạy cả bốn chiến lược trên máy mình** (`python bench.py --all`, cùng corpus, cùng 5 query, cùng code store, chỉ đổi dòng chọn chunker) để có mốc so sánh trước buổi demo. Số liệu chính thức của từng thành viên do chính họ chạy và nộp trong báo cáo riêng:

| Chiến lược | Số chunk | Độ dài TB | doc_id-level | **nội dung-level** |
|---|---|---|---|---|
| `fixed` | 40 | 556 | 8/10 | 6/10 |
| `sentence` | 54 | 345 | **10/10** | 6/10 |
| `recursive` | 40 | 467 | 8/10 | 6/10 |
| **`heading` (chiến lược của tôi)** | 39 | 492 | 9/10 | **9/10** |

**Điều hay nhất tôi học được:**
> **Hai cách chấm cho hai người thắng khác nhau, và bảng trên là bằng chứng đắt nhất của buổi lab.** Chấm theo `doc_id`: `sentence` thắng tuyệt đối 10/10 — cả 5 câu đều lấy đúng tài liệu ở hạng 1. Chấm ở mức nội dung: `sentence` rơi xuống 6/10 còn chiến lược của tôi lên 9/10.
>
> Nguyên nhân rất cụ thể: `sentence` cắt thành 54 chunk ngắn (TB 345 ký tự), nên ở Q4 nó lấy đúng tài liệu UTT nhưng **chunk chứa điều kiện "từ 4 sinh viên trở lên" và chunk chứa "dưới 4 sinh viên → 2,5 lần" nằm ở hai chunk khác nhau** — top-3 có tài liệu đúng mà không trả lời được câu hỏi. Ở Q3 cũng vậy: mốc `23/05/2026 đến hết ngày 01/06/2026` bị tách khỏi phần ngữ cảnh.
>
> Chunk theo heading không dính lỗi này vì ranh giới chunk **trùng với ranh giới mà người soạn văn bản đã chia**: mục "7. Học phí lớp học lại, học cải thiện" tự chứa cả điều kiện lẫn con số. Nói cách khác, với văn bản quy định, **ranh giới ngữ nghĩa tốt nhất không phải thứ mình tính ra bằng ký tự hay số câu — nó đã có sẵn trong tài liệu, việc của mình là đừng phá nó.**

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — 42/42 tests) | 30 / 30 |
| Dự đoán độ tương tự (4/5 đúng + phân tích ngưỡng) | 5 / 5 |
| Kết quả truy xuất của tôi (9/10 nội dung-level) | 9 / 10 |
| **Tổng phần cá nhân** | **59 / 60** |

Tự trừ 1 điểm ở mục 5: Q2 chỉ đạt hạng 2 vì câu hỏi của nhóm không nêu tên trường trong khi corpus gom từ 4 trường — lỗi thiết kế câu hỏi mà tôi có phần trách nhiệm.
