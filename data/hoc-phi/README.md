# Corpus `data/hoc-phi/` — nguồn gốc và cách thu thập

Chủ đề: **học phí đại học** (K4-L3A — dịch vụ/quy định đại học). 5 tài liệu, toàn bộ là nội dung
**công khai crawl từ website chính thức của các trường**, không phải dữ liệu mô phỏng.

## 1. Cách thu thập

```bash
python scripts/fetch_public_pages.py data/urls.csv --output-dir data/hoc-phi --delay 1.5
```

Danh sách URL đầu vào: [`data/urls.csv`](../urls.csv). Script tự kiểm `robots.txt`, giãn tối thiểu
1,5 giây giữa các request, chỉ nhận `text/html` / `text/plain`.

Sau khi crawl, **mọi file đều được làm sạch thủ công**: bỏ menu, breadcrumb, khối "Tin liên quan",
footer, form bình luận; giữ lại đúng điều khoản, con số và mốc thời gian. Không chunk trên bản thô.

## 2. Kết quả crawl: 7 URL vào, 5 tài liệu ra

| URL | Kết quả | Lý do |
| --- | --- | --- |
| ussh.vnu.edu.vn — lộ trình thu học phí 2026-2027 | ✅ dùng, **tách làm 2 file** | Một trang gộp cả bậc đại học lẫn sau đại học → tách theo `level` để filter có việc thật |
| utt.edu.vn — mức thu học phí 2026-2027 | ✅ dùng | |
| hvtc.edu.vn — học phí học lại kỳ phụ 2025-2026 | ✅ dùng | |
| phongkhtc.ntu.edu.vn — khung học phí theo NĐ 238/2025 | ✅ dùng | |
| pkhtc.hcmuaf.edu.vn — học phí HK1 2026-2027 | ❌ loại | Server trả chứng chỉ SSL thiếu mắt xích trung gian → `CERTIFICATE_VERIFY_FAILED` khi đọc `robots.txt`. **Không tắt xác minh SSL để lách**; đổi hướng thay vì hạ mức bảo mật |
| pnt.edu.vn — học phí hệ đại học 2025-2026 | ❌ loại | Trang chỉ có link tải PDF phụ lục; phần text trích ra được không chứa một con số học phí nào |
| giaovu.ptit.edu.vn — học phí | ❌ loại | Trang **danh sách tin**, mỗi mục bị cắt cụt bằng `[…]`; không có nội dung đầy đủ để làm gold answer |

5 tài liệu nằm trong khoảng 5–10 mà lab yêu cầu. Rubric chấm chất lượng và tính minh bạch nguồn,
không chấm số lượng — nên ba nguồn bị loại được ghi lại ở đây thay vì che đi.

## 3. Hai chỗ mất dữ liệu đã biết (và không bịa để bù)

- `utt-muc-thu-hoc-phi-2026-2027`: bảng mức thu cho **các khóa từ 75 trở đi** nằm trong bảng HTML mà
  trình trích xuất text không đọc được. Tài liệu chỉ còn mức của khóa 74 trở về trước.
- `ntu-khung-hoc-phi-nd238-2025`: các **bảng mức sàn - mức trần** của Nghị định 238 là ảnh/bảng HTML,
  cũng mất. Tài liệu chỉ giữ phần quy tắc dạng văn bản (hệ số 1,5 / 2,5; trần 2x / 2,5x; VLVH ≤ 150%).

Cả hai chỗ đều được ghi chú ngay trong file bằng một block `>`. Benchmark query được thiết kế để
**không** hỏi vào phần dữ liệu đã mất.

## 4. Metadata schema

Bắt buộc theo `docs/DATA_COLLECTION.md`: `doc_id`, `title`, `source_url`, `retrieved_at`,
`document_version`, `audience`. Trường lọc thêm: `level`, `university`, `department`, `category`, `language`.

| Trường | Giá trị trong corpus |
| --- | --- |
| `audience` | `student` (4) · `all` (1 — bài về khung pháp lý NĐ 238, áp dụng cho mọi đối tượng) |
| `level` | `undergraduate` (3) · `postgraduate` (1) · `any` (1) |
| `university` | `ussh-vnu`, `utt`, `hvtc`, `ntu` |
| `category` | `tuition-rate`, `tuition-retake`, `tuition-policy` |

**Hai chiều lọc đều có việc thật:**

- `audience=student` loại tài liệu khung pháp lý (`all`) — tài liệu đó chỉ trả lời "VLVH không vượt quá
  150% hệ chính quy" chứ không có con số, nên với câu hỏi của sinh viên nó là câu trả lời sai loại.
- `level=postgraduate` tách mức thu theo tháng của học viên cao học (2.865.000đ) khỏi mức của sinh viên
  đại học (1.910.000đ) — cùng một câu hỏi, hai đáp án khác nhau.

## 5. Ghi chú về `document_version`

Chỉ ghi số hiệu khi trang nguồn thực sự nêu. `hvtc` có số hiệu thật (`731/TB-HVTC ngày 18/05/2026`);
bốn tài liệu còn lại không nêu số hiệu nên dùng **ngày đăng** của trang. Không bịa số hiệu văn bản.

Một số điện thoại cá nhân của cán bộ xuất hiện trong bản gốc HVTC đã được **gỡ bỏ** khi làm sạch,
theo quy tắc không đưa dữ liệu cá nhân vào repo.
