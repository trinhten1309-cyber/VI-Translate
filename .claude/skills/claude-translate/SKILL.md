---
name: claude-translate
description: Translate an ACCA/IFRS PDF into Vietnamese and a side-by-side bilingual PDF with Claude itself as the translator, no API key, using the fewest tokens. Use when the user asks Claude (not the Colab notebook's APIs) to translate a PDF in this repository.
---

# Dịch PDF với Claude là người dịch

Quy trình giống notebook Colab (OCR cho trang scan, dựng lại lớp chữ, tách
đoạn, kiểm tra thẻ/số/thuật ngữ, Cabin, song ngữ side-by-side), chỉ thay bước
gọi API bằng Claude. Không cần đọc thêm file nào khác trong repo.

## Các bước

```bash
.venv/bin/python colab/claude_translate.py prepare --base BASE --pdf FILE.pdf [--pages 5-6] [--mon FR]
```

`BASE` chứa `glossary/` (CSV; không có thì dùng `colab/glossary/ifrs-acca.csv`),
`cache/`, `output/`. Lệnh in ra các file lô `BASE/claude/<tên>/batches/lo-NNN.txt`
(tối đa 60 đoạn hoặc 12.000 ký tự). Đoạn đã có trong cache không vào lô nào.

Với mỗi lô: đọc `lo-NNN.txt`, rồi dùng Write ghi `answers/lo-NNN.txt`, mỗi đoạn
một dòng `id<TAB>bản dịch`. Xong hết các lô:

```bash
.venv/bin/python colab/claude_translate.py finish --job BASE/claude/<tên> --render 5,6
```

`finish` chỉ in các id lỗi kèm lý do ngắn, dựng `output/<tên>-vi.pdf` và
`output/<tên>-vi-dual.pdf`, in các đoạn còn tiếng Anh, và với `--render` xuất
PNG 60 dpi vào `png/`. Hãy xem các ảnh đó. `--check-only` chỉ kiểm tra, không
dựng PDF.

Sửa lỗi: lấy dòng gốc của các id bị báo bằng
`grep -P "^(12|44)\t" BASE/claude/<tên>/batches/*.txt`, ghi bản sửa vào
`answers/sua-1.txt` (cùng dạng `id<TAB>bản dịch`), rồi chạy lại `finish`. File
`sua-*` đứng sau `lo-*` nên thay câu trả lời cũ; các lô khác giữ nguyên.
Id thay đổi giữa các lần `prepare`: luôn trả lời theo id của lô vừa đọc,
không chép câu trả lời của job cũ (một lần lệch id đã đổi chỗ bốn dòng mục
lục IFRS 3 trong cache). `sua-*` chỉ để sửa câu dịch. Chữ còn tiếng Anh vì engine không tách được (tiêu
đề bảng, đầu trang, tiêu đề báo cáo) phải sửa trong `pdf2zh/`, không sửa tay.

Kiểm tra chữ còn sót sau khi dựng:

```bash
.venv/bin/python scripts/audit_untranslated.py BASE/output/<tên>-vi.pdf --min-words 1
```

Danh sách in từng trang các dòng còn chữ Latin không dấu. Tên người, email,
mã chứng khoán, chữ trong hình và trang chỉ mục (Index) là bình thường; tiêu đề
cột, ngày ở đầu cột, đầu trang, tiêu đề báo cáo, trang bìa hay mục lục còn
tiếng Anh là lỗi engine cần báo lại.

## Quy tắc dịch

1. Đúng thuật ngữ ở dòng `# Thuật ngữ:` đầu lô; "giữ nguyên" = để tiếng Anh.
2. Giữ nguyên tên chuẩn mực (IFRS 3, IAS 36), viết tắt (NCI, PUP), tên công ty,
   năm giả định (20X4), mọi con số và cách viết số ($5,200,000; $'000).
3. Thẻ `<b0></b0>`… giữ đủ, đúng thứ tự. `<s1>` đậm, `<s2>` nghiêng, `<s3>` đậm
   nghiêng: giữ đủ cặp, bao quanh cụm tiếng Việt tương ứng.
4. Giữ `\n` và `\t` (viết đúng hai ký tự) nếu chúng có trong đoạn gốc.
5. Đoạn bắt đầu giữa câu (nối tiếp từ trang trước) thì dịch tiếp đúng phần đó.
6. Văn phong giáo trình, không thêm, không bỏ ý. Giữ ký hiệu mục "(a)", "B27".

## Tiết kiệm token

- Không chép lại tiếng Anh, không in lại bản dịch ra chat, không đọc lại file
  trả lời vừa ghi.
- Không đọc `segments.jsonl`, `state.json`, file PDF hay output của engine.
- Chỉ render vài trang cần xem, ở 60 dpi.
- Lỗi thuật ngữ chỉ là cảnh báo (bản dịch vẫn được dùng); chỉ sửa khi sai thật.

## Prompt mẫu cho phiên mới

> Dùng skill claude-translate. Dịch `/đường/dẫn/file.pdf`, trang 1-10, base
> `/đường/dẫn/BASE`. Dịch hết các lô, chạy finish với `--render 1,2`, xem ảnh,
> sửa id lỗi nếu có, rồi báo tôi đường dẫn file song ngữ. Trả lời bằng tiếng Việt.

Lần đầu trên máy mới: `python3 -m venv .venv && .venv/bin/pip install -r requirements-ocr.txt`
(engine tự tải mô hình bố cục lần chạy đầu). Trang scan cần thêm `tesseract`
nếu muốn kiểm tra số bằng bộ đọc thứ hai (`--ocr-check off` để bỏ qua).
