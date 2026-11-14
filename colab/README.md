# Dịch tài liệu ACCA trên Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trinhten1309-cyber/VI-Translate/blob/main/colab/ACCA_Translate.ipynb)

Quy trình miễn phí cho cả team: Google Colab (chỉ cần CPU) + Google Drive dùng
chung + API LLM miễn phí (Gemini, Groq, OpenRouter). Không cần cài gì trên máy.

## Bắt đầu

1. Bấm nút **Open In Colab** ở trên → **File → Save a copy in Drive**.
2. Thêm shortcut thư mục dùng chung (mặc định `DOCUMENTS TRANSLATION`) vào My Drive.
3. Lưu API key vào 🔑 Secrets của Colab: `GEMINI_API_KEY`, `GROQ_API_KEY`,
   `OPENROUTER_API_KEY` (bật **Notebook access**). Không nhập thẻ thanh toán ở đâu cả.
4. Upload PDF vào `input/` → **Runtime → Run all**.

Kết quả nằm trong `output/`: `<tên>-vi.pdf` (bản dịch) và `<tên>-vi-dual.pdf`
(song ngữ), chỉ gồm các trang đã chọn ở `PAGES` (để trống = cả file). Mỗi lần
chạy, file cũ cùng tên được xoá (vào Thùng rác của Drive) và file mới được tạo,
để bản xem trước trên Drive luôn là bản mới nhất. Với PDF scan, mọi dòng chữ
nhận dạng được đều được dịch, kể cả tiêu đề trong khung và chân trang; số trong
bảng dùng cùng font với bản dịch. Notebook luôn chạy code mới nhất của repo,
nên khi repo được sửa, chỉ cần chạy lại.

## Các file

| File | Vai trò |
|---|---|
| `ACCA_Translate.ipynb` | Notebook: kết nối Drive, cài đặt, nạp key, form dịch |
| `acca_translate.py` | Quy trình chính: tách đoạn → dịch theo lô kèm thuật ngữ → kiểm tra → dựng PDF |
| `router.py` | Tự chuyển nguồn API khi hết lượt ngày, chờ khi bị giới hạn theo phút |
| `handoff_tools.py` | Kiểm tra bản dịch: thẻ `<bN>`/`<sN>`, số liệu, thuật ngữ |
| `ocr_layer.py` | Trang chỉ có ảnh (scan) → nhận dạng chữ bằng RapidOCR thành lớp chữ ẩn; dòng không chắc chắn giữ nguyên ảnh |
| `rebuild_text_layer.py` | PDF dạng ảnh có lớp chữ OCR ẩn → xoá chữ trong ảnh, dựng lại chữ thật |
| `fonts/` | Font Cabin (SIL OFL 1.1, xem `fonts/OFL.txt`), thường và SemiCondensed |
| `make_cabin.py` | Dựng lại `fonts/` từ bản variable font của Google Fonts |

## Chạy không cần notebook

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-ocr.txt openai   # Python 3.10–3.12
export GEMINI_API_KEY=...  GROQ_API_KEY=...
.venv/bin/python colab/acca_translate.py test
.venv/bin/python colab/acca_translate.py run --base "/đường/dẫn/thư mục" --pdf "/đường/dẫn/file.pdf" \
    --pages 1-5 --style Cabin --bilingual side-by-side
```

`--base` là thư mục chứa `glossary/`, `cache/`, `output/`.

## Bảng thuật ngữ

Một hoặc nhiều file CSV UTF-8 trong `glossary/` với các cột `en`, `vi`, `mon`
(F5, F7…; nhiều môn cách nhau bằng `;`; `chung` = mọi môn), `keep_en` (`1` = giữ
nguyên tiếng Anh), `ghi_chu` (không gửi cho API).

## Giới hạn

- Trang mục lục/chỉ mục (bảng có số trang cuối dòng) được engine giữ nguyên.
- Bảng không có đường kẻ hoặc cột căn thẳng có thể chỉ được dịch một phần.
- PDF scan không có lớp chữ: `--ocr auto` (mặc định) nhận dạng chữ rồi dịch, kể cả bảng;
  dòng nhận dạng không chắc chắn được giữ nguyên ảnh. `--ocr engine` dùng OCR thận trọng
  của repo (chỉ đoạn văn an toàn).
- Gói miễn phí của Gemini/Mistral có thể dùng dữ liệu gửi lên để cải thiện mô
  hình: chỉ dịch tài liệu học tập, không gửi tài liệu nội bộ hay của khách hàng.
