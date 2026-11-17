# Bàn giao phiên làm việc: sửa engine dịch PDF (VI-Translate)

Ngày 17/11/2026. Tài liệu này để mở một phiên Claude mới và làm tiếp đúng chỗ
đang dừng. Đọc hết mục 1–4 trước khi động vào code.

---

## 1. Bối cảnh và yêu cầu của người dùng

- Người dùng (nói tiếng Việt, **luôn trả lời bằng tiếng Việt**) dùng repo
  `trinhten1309-cyber/VI-Translate` để dịch PDF tài chính và học thuật sang
  tiếng Việt, xuất bản dịch và bản song ngữ đặt cạnh nhau. Cấp trên của họ
  sẽ dùng repo này.
- Người dịch là chính Claude, qua skill `claude-translate`
  (`.claude/skills/claude-translate/SKILL.md`). Engine dựng PDF nằm trong
  `pdf2zh/`.
- **Yêu cầu quan trọng nhất:** mọi lỗi bố cục phải sửa **trong engine**
  (`pdf2zh/`) và commit vào repo. **Không** sửa tay bằng token cho từng file.
  File `answers/sua-*.txt` chỉ để sửa câu chữ của bản dịch (thuật ngữ, câu
  dịch sai), không dùng để che lỗi engine.
- Cách người dùng làm việc: họ soát PDF song ngữ bằng mắt, gửi ảnh chụp hoặc
  danh sách lỗi theo trang. Claude chẩn đoán, sửa engine, soát hồi quy trên
  cả bộ tài liệu, commit, push, gửi lại file và báo cáo bằng tiếng Việt.

## 2. Trạng thái repo

- Nhánh làm việc: **`claude/determined-lovelace-39ctjx`**. Chỉ push lên nhánh
  này; người dùng đã cho phép push. **Chưa có PR**, không tạo PR khi chưa được
  yêu cầu.
- Nền: `main` tại `d863c18` (Merge PR #4).
- Các commit của phiên này, cũ đến mới:

| Commit | Nội dung |
|---|---|
| `4d1a6fc` | Đầu trang căn giữa dịch từng dòng; bảng có đường kẻ dưới tiêu đề (NVIDIA tr.12) |
| `a7e69d8` | Dịch ngày trong ô bảng, tiêu đề viết hoa, "furniture" không lặp lại; thêm `scripts/audit_untranslated.py` |
| `7ee02d5` | Dịch trang bìa và mục lục theo từng mục (`line_run_groups`) |
| `afa0fd0` | Sửa bìa: ngắt dòng lại ở cỡ chữ cuối, dòng mới theo trục y, khoảng trắng đầu đoạn, gạch ô trống |
| `3f1f062` | Sửa theo bản soát 38 mục: số đoạn lề, tiêu đề nhiều dòng, gạch chân liên kết, ô bảng, ngày, thuật ngữ |

- Test: `.venv/bin/python -m unittest discover -s tests -q` cho 576 test.
  Có **5 lỗi cũ có sẵn** do môi trường (thiếu module `websocket`, thiếu font
  hoặc trình duyệt cho test Google). Không phải hồi quy, đừng mất công sửa.
- Thiết lập máy mới: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements-ocr.txt`
  (`babeldoc==0.2.33` chỉ cài được trên Python 3.10–3.12).
  Mô hình bố cục tự tải ở lần chạy đầu. Font Cabin nằm sẵn trong `colab/fonts/`.

### Quy tắc dự án phải tuân theo (từ `CLAUDE.md` / `agent-knowledge/index.md`)

- PDF nguồn bất biến; bản dịch ghi ra đường dẫn khác.
- Ưu tiên bản dịch thiếu mà đúng cấu trúc hơn là làm hỏng công thức, bảng,
  kiểu chữ hay cắt chữ.
- Dùng `pdf2zh/` có sẵn trong repo, không thay bằng gói PyPI.
- Không commit PDF đã dựng, cache, venv hay mô hình.
- Đọc `agent-knowledge/pdf-engine.md` và `agent-knowledge/regressions.md`
  trước khi sửa engine, và **cập nhật hai file này cùng commit**.
- Commit kết thúc bằng:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: <link phiên mới>
  ```
  Không ghi tên hay ID model vào commit.

## 3. Dữ liệu ngoài repo (sẽ mất khi container cũ bị thu hồi)

Phiên cũ để mọi thứ ở scratchpad `/tmp/claude-0/.../scratchpad`, gọi là `$S`.
Người dùng đã nhận file `handoff-data.zip`, gồm:

| Trong zip | Là gì | Ghi chú |
|---|---|---|
| `cache/translations_cache.jsonl` | **Cache bản dịch** (khoảng 2.000 đoạn, ~0,9 MB) | Giữ cache này để không phải dịch lại hàng nghìn đoạn. Mỗi dòng có dạng `{"src","dst",...}`; dòng sau ghi đè dòng trước |
| `corpus/*.pdf`, `nvda/…pdf`, `ifrs3.pdf` | 7 PDF test | IFRS 3 do người dùng cung cấp; các file khác là 10-Q và bài báo công khai |
| `tools/` | Script soát lỗi tự viết (mục 6) | Đường dẫn `S=...` cứng trong script; sửa lại theo scratchpad mới |
| `glossary/` | Không có | Glossary nằm trong repo: `colab/glossary/ifrs-acca.csv` |

Trong session mới: tạo scratchpad `$S`, giải nén zip vào đó (giữ cấu trúc
`cache/`, `corpus/`, `nvda/`, `ifrs3.pdf`, `tools/`), rồi chạy lại các lệnh ở
mục 4.

## 4. Quy trình dịch (skill `claude-translate`)

```bash
.venv/bin/python colab/claude_translate.py prepare --base $S --pdf FILE.pdf [--pages 1-8] --job $S/claude/<job>
```

- **Luôn** truyền `--base $S`, và `--job` phải là **đường dẫn tuyệt đối**
  dưới `$S/claude/`. Thiếu một trong hai thì `prepare` lỗi.
- Đoạn đã có trong cache không vào lô. Lô nằm ở `$S/claude/<job>/batches/lo-NNN.txt`.
- Trả lời vào `answers/lo-NNN.txt`, mỗi dòng `id<TAB>bản dịch`. Giữ đủ thẻ
  `<sN>…</sN>` (kiểu chữ) và `<bN></bN>` (công thức hoặc glyph được giữ),
  đúng số lượng.
- **Id thay đổi giữa các lần `prepare`.** Chỉ trả lời theo id của lô vừa
  đọc. Phiên cũ từng chép câu trả lời theo id của job cũ, làm bốn dòng mục
  lục IFRS 3 trang 3 bị đổi chỗ trong cache (đã sửa).
- Sửa câu dịch: ghi `answers/sua-1.txt` (cùng dạng). File này ghi đè `lo-*`
  và cập nhật cache.

```bash
.venv/bin/python colab/claude_translate.py finish --job $S/claude/<job> [--render 1,2]
```

- Kết quả ra `$S/output/<tên>-vi.pdf` và `-vi-dual.pdf` (song ngữ).
- Soát chữ còn tiếng Anh:
  `.venv/bin/python scripts/audit_untranslated.py $S/output/<tên>-vi.pdf --min-words 1`.
  Tên riêng, mã, email, nhãn "(ii)" còn lại là bình thường.
- Thư mục làm việc của engine: `tmp/colab-work/<tên>/translations.jsonl`
  (đã gitignore). `scripts/translate_pdf.py … --engine handoff --segments
  tmp/colab-work/<tên>/translations.jsonl` dựng lại một trang mà không cần dịch.

### Bộ tài liệu soát hồi quy và trạng thái cuối

| Job (phiên cũ: `*-c11`) | PDF | Phạm vi | Số đoạn | Dòng còn Latin (audit) |
|---|---|---|---|---|
| nvda | `nvda/nvda-10q-q1fy27.pdf` (45 tr.) | cả file | 760/760 | 50 |
| ifrs3 | `ifrs3.pdf` (56 tr.) | cả file | 680/680 | 197 (phần lớn là tên, nhãn (i)(ii), chân trang) |
| msft | `corpus/msft-10q-q2fy26.pdf` | trang 1–8 | 203/203 | 14 |
| aapl | `corpus/aapl-10q-q2fy26.pdf` | trang 1–9 | 207/207 | 32 |
| attention | `corpus/attention.pdf` | trang 1–2 | 28/28 | 25 |
| bert | `corpus/bert.pdf` | trang 1–2 | 24/24 | 15 |
| resnet | `corpus/resnet.pdf` | trang 1–2 | 27/27 | 21 |

Với cache trong zip, chạy `prepare` cho 7 job sẽ ra 0 đoạn cần dịch (trừ khi
engine đổi cách tách đoạn).

## 5. Bản đồ thay đổi engine (để biết sửa ở đâu)

### `pdf2zh/high_level.py` (gán lớp bố cục cho từng pixel, trước converter)
- Lớp được bảo vệ (`box=0`): abandon, figure, table, isolate_formula,
  formula_caption. Lớp chữ là `i+2`. `page_class_bounds` lưu biên theo tọa độ
  PDF, gốc ở dưới.
- Đầu trang căn giữa: `running_header_lines` → mỗi nhóm thành một lớp riêng.
- `abandon` không lặp lại (`abandon_is_unrepeated_text`, so với
  `repeated_page_lines` của cả tài liệu) thì dịch.
- `figure` toàn chữ (`figure_is_text`) và vùng chia cột (`region_has_columns`)
  → `run_regions`. Trang mục lục cũng vậy, trừ trang chỉ mục (index).
- `translate_items()` tô từng mục của `line_run_groups` thành một lớp riêng và
  ghi vào `device.item_classes`.
- `figure` hoặc `isolate_formula` chỉ gồm chữ và ô đánh dấu
  (`formula_is_words`) thì thành chữ. Figure được tô thành một lớp chữ riêng
  ("Yes ☐ No ☒").
- `majority_text_class`: dòng nằm vắt qua hai khung chữ thì giao cho khung
  chiếm phần lớn.

### `pdf2zh/pdfinterp.py`
- `take_underline()` trong `do_f`, `do_f_a`, `do_S`:
  - Thanh mảnh (cao ≤ 1,6pt) tô màu bất kỳ trừ trắng, và nét vẽ **có màu**,
    bị gỡ khỏi luồng gốc (trả về `"n"`) rồi đưa sang converter qua
    `page.underline_rules`. Chỉ áp dụng ngoài form XObject.
  - Nét vẽ đen vẫn đi đường cũ thành `LTLine` (ô trống "____", gạch phân số).
- Hàm hỗ trợ trong rules: `thin_filled_rule`, `thin_stroked_rule`,
  `fill_colour_operator` (trả về `(operator, có_phải_màu_liên_kết)`).

### `pdf2zh/converter.py` (dựng đoạn và vẽ lại chữ)

**Dựng đoạn** (`receive_layout`):
- `starts_lower_line`: glyph nằm hẳn dưới glyph trước là dòng mới.
- Khoảng trống quá 4 em trên cùng một dòng thì tách đoạn (cột).
- Đoạn mới khi:
  - `line_changes_size` (cỡ chữ lệch ≥ 15%);
  - `heading_lines_part`: dòng căn giữa chồng nhau (dòng trên ≥ 3 từ), dòng
    đổi kiểu chữ, hoặc dòng nhãn ngắn như "Appendix A" / "23". Không áp dụng
    trong ô bảng;
  - `ends_with_check_box`.
- Đoạn chỉ có khoảng trắng thì lấy dòng và cỡ chữ của chữ đầu tiên. Ô đánh
  dấu đứng xa (> 1,5 em) thì tách riêng.
- `blank_run`: công thức chỉ gồm dấu cách thì thành một dấu cách.
- Nối dòng sau gạch nối không chèn dấu cách, nhưng chỉ khi có chữ số
  ("13a-" / "15(f)").
- `resumable_paragraph`: phần sau của một mục được viết sau cột bên cạnh thì
  nối lại vào mục đó.
- `paragraph.source_chars`: glyph gốc. `paragraph.hang_x0`: lề của dòng
  ngắt. `paragraph.text_box`: hộp chữ trước khi bị ghi đè bằng biên ô.

**Vẽ lại:**
- Đoạn vẽ lại nguyên từ glyph gốc:
  - `is_reference_column` (cột số đoạn lề như "11 12 13");
  - đoạn nhiều dòng mà bản dịch giống hệt bản gốc (danh sách tên).
- `size_to_fit_lines`: chọn cỡ chữ **trước** rồi mới ngắt dòng, nên dòng vẫn
  dàn đủ bề ngang.
- `widest_word`: từ dài hơn ô thì giảm cỡ chữ, không cắt giữa từ.
- Ô bảng ưu tiên giữ số dòng của bản gốc, chấp nhận thu nhỏ tới 85%.
- Đoạn một dòng không còn được tràn 5% (trước đây chữ chạy ra ngoài mép trang).
- Nhãn một dòng trong ô bảng được chạy sang phần trống của hàng
  (`available_width_right`).
- `available_height_below(..., size=)`: chừa tối đa 0,4 em khoảng trống trên
  đoạn sau.
- Thụt lề treo cho đoạn bắt đầu bằng nhãn danh sách (`LIST_LABEL_START`,
  `tab_at`, `wrap_x`).
- Gạch chân:
  - `rendered_line_extents` và `source_line_extents` đo dòng đã vẽ và dòng gốc;
  - `underline_operations` vẽ gạch theo độ dài chữ dịch, thấp hơn dấu nặng
    0,26 em. Gạch đen chỉ được ánh xạ khi phủ trọn dòng và nằm ngoài ô bảng;
    mẩu gạch dưới dấu cách cuối dòng thì bỏ;
  - text op có thêm khóa `"text"` và `"width_em"` để đo.
- Ngày: `us_date_order_is_certain` và `day_first_date_cell` đổi ô ngày
  tháng/ngày/năm sang ngày/tháng/năm khi dịch sang tiếng Việt. Cờ
  `self.month_first_dates` giữ theo tài liệu.
- `centred_line_x`, `first_word_start`: căn giữa dòng tiêu đề; từ đầu không
  vừa thì bắt đầu ở lề trái.

### `pdf2zh/rules.py`
- Bảng: `matching_table_cells`, `table_heading_cells`, `table_top_below_line`.
- `should_translate_table_cell` nhận thêm: ngày, từ nối bằng gạch chéo, chữ
  hoa ≥ 5 chữ cái, từ ghép có gạch nối ("Vice-Chair"), ngày dạng số.
- Mục lục và bìa: `line_run_groups` (`gap_is_blank_to_fill`: khoảng trống có
  gạch vẽ bên dưới không cắt dòng), `figure_is_text`, `region_has_columns`.

### Glossary `colab/glossary/ifrs-acca.csv`
Đã thêm: in millions/billions, unaudited, three months ended, condensed
consolidated, marketable / non-marketable, gross margin (có ghi chú),
"Quantitative and Qualitative Disclosures About Market Risk", Compute &
Networking, Graphics, ACIE, certiorari, Vice-Chair.

## 6. Công cụ soát lỗi (trong `tools/` của zip)

| Script | Dùng để |
|---|---|
| `pg.sh DOC PAGES OUTDIR` | Dựng nhanh một vài trang (`DOC` = nvda, msft, aapl, ifrs) từ `tmp/colab-work/*/translations.jsonl`, font Cabin. Log đầy đủ ở `OUTDIR.log` |
| `pair.py NAME PAGE [y0 y1 dpi vi.pdf]` | Ảnh ghép gốc (trái) và bản dịch (phải) của một trang hoặc một dải y |
| `oldnew.py NAME PAGE …` | Ghép bản dịch cũ (`new8/`) và mới (`new10/`) để so hồi quy |
| `coverage.py PDF PAGES OUTDIR` | Tách đoạn bằng engine, liệt kê dòng gốc bị bỏ sót và đoạn bị cắt giữa từ |
| `lay.py PDF pages` | In các khung của mô hình bố cục (lớp, độ tin cậy, tọa độ trên xuống) |
| `regress7.sh` | Chạy `prepare` cho cả 7 tài liệu (đổi hậu tố job, ví dụ `-c11` thành `-c12`) |

**Quy trình soát hồi quy** đã dùng mỗi lần sửa engine:
1. Chạy unit test.
2. Chạy `regress7.sh`, rồi so `segments.jsonl` mới với cũ để xem đoạn nào đổi
   và đổi có hợp lý không.
3. Dịch các đoạn mới, giữ cách dịch cũ trong cache cho thống nhất.
4. Chạy `finish` cho 7 job, sao lưu PDF ra một thư mục riêng.
5. Chạy audit và so số dòng còn Latin với lần trước.
6. So ảnh từng trang ở 40–50 dpi (tỉ lệ pixel lệch > 0,05%) với lần trước, rồi
   xem bằng mắt các trang lệch nhiều.

Mẹo debug: chèn tạm một dòng `print` có điều kiện biến môi trường ngay sau
`brk: bool = pstk[id].brk` trong vòng vẽ của converter, để in `id`, `cls`,
`x`, `x0`, `x1`, `y0`, `y1`, `size`, `brk`, `sstk[id]`, `new` cho từng đoạn.
**Nhớ gỡ trước khi commit.**

## 7. Việc còn mở

Bản soát 38 mục của người dùng: các mục P0 đã sửa hết. Còn lại:

| Mục | Vấn đề | Gợi ý |
|---|---|---|
| #31 | "Gross margin": Apple dùng cho số tiền (đúng là "Lợi nhuận gộp"), NVIDIA dùng cho % (nên là "Biên lợi nhuận gộp"). Cùng chuỗi gốc nên cache chỉ giữ một bản | Cần người dùng quyết; hoặc cho converter nhận ngữ cảnh (ô cùng hàng có `%`) |
| #20 | Cỡ chữ không đều giữa các ô cùng bảng (mỗi ô tự co) | Hệ số co chung theo bảng hoặc theo hàng tiêu đề |
| #22 | "Tài / sản / khác" ba dòng trong ô NVIDIA rộng khoảng 22pt | Cho tiêu đề cột lấn sang lề trống bên cạnh |
| #23 | Chữ mồ côi ở tiêu đề bảng NVIDIA tr.39 ("…đã công / bố") | Bật `balance_short_orphan` cho tiêu đề cỡ nhỏ |
| #26 | Tiêu đề cột ngày không thống nhất ("Ngày 28 tháng 3, 2026" so với "28/3/2026") | Sửa câu dịch (sua) và thêm quy ước vào glossary |
| #28 | Font Cabin khác Arial hoặc serif của bản gốc | Quyết định sản phẩm, chưa làm |
| #29 | "Ký bởi: /s/ …" lệch so với "By:" (NVIDIA tr.41) | Chưa chẩn đoán |
| — | Bộ kiểm tự động người dùng đề xuất chưa làm: tỉ lệ cỡ chữ VI/EN < 0,8, số dòng và lề, độ dài gạch, số đoạn dính nhau, chữ tràn mép trang | Viết thành script trong `scripts/` dựa trên `audit_untranslated.py` |

Đã sửa ở phiên 2 (nhánh `claude/new-session-utrvcc`): khoảng trống "từ ⇥ đến"
của bìa Microsoft (ký hiệu `⇥` trong lô), hàng ô đánh dấu bìa NVIDIA (tách sau
ô đánh dấu khi khoảng trống > 1 em), tiêu đề nhiều dòng căn giữa (BERT).

Không phải lỗi (đã giải thích với người dùng): #25 (số trang Microsoft nằm
đúng vị trí gốc), #37 ("continued..." là "còn tiếp…", "...continued" là
"…tiếp theo").

## 8. Kinh nghiệm rút ra

- **Mỗi quy tắc tách đoạn mới đều phải chạy trên cả 7 tài liệu.** Ví dụ:
  - Quy tắc dòng căn giữa đã làm vỡ tiêu đề cột nhiều dòng của bảng NVIDIA
    ("Cost or / Amortized / Cost"), nên phải loại trừ ô bảng và yêu cầu dòng
    trên có ít nhất 3 từ.
  - Quy tắc bỏ dấu cách sau gạch nối làm đổi 30 đoạn của bài báo, nên phải
    giới hạn cho mã có chữ số.
  - Bắt gạch đen làm mất ô trống "____" trên bìa Apple, nên với nét vẽ chỉ
    lấy gạch có màu.
- So `segments.jsonl` cũ và mới là cách nhanh nhất để thấy một quy tắc đổi
  cách tách đoạn ở đâu.
- Mô hình bố cục hay gán nhiều lớp cho cùng một khung (title, plain text,
  isolate_formula). Khi khung bị bảo vệ mà khung chữ trùng nó lại có độ tin
  cậy cao hơn, thường nên coi là chữ.
- Khi `receive_layout` vẽ đoạn, `x0`, `x1`, `y0`, `y1` của đoạn trong ô bảng
  đã bị ghi đè bằng biên ô. Dùng `paragraph.text_box` để lấy hộp chữ thật.
- `pg.sh` cũ cắt log bằng `tail`; bản trong zip đã ghi đủ log ra `OUTDIR.log`.

## 9. Prompt mẫu cho phiên mới

> Đọc `docs/session-handoff-2026-11-17.md` trên nhánh
> `claude/determined-lovelace-39ctjx` rồi làm tiếp. Tôi đã upload
> `handoff-data.zip` (cache bản dịch, 7 PDF test, script soát lỗi): giải nén
> vào scratchpad và sửa biến `S` trong các script. Chạy unit test và
> `regress7.sh` để xác nhận trạng thái (0 đoạn cần dịch). Sau đó sửa tiếp các
> mục còn mở ở mục 7, theo đúng quy trình soát hồi quy ở mục 6. Sửa trong
> engine, không sửa tay. Commit, push lên nhánh, gửi lại file song ngữ và báo
> cáo bằng tiếng Việt.
