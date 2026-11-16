# Repository and Development

## Structure

- `app/`: CustomTkinter GUI, the self-updater, UI fonts, icons, bundled assets.
  `app/update.py` also carries the PowerShell helper that performs the swap.
  `app/i18n.py` holds every interface string in Vietnamese (the default) and
  English, error summaries and advice included; a key missing from one catalog
  fails `test_app_i18n.py`. `app/settings.py` remembers the interface
  language, target language and bilingual layout in
  `%LOCALAPPDATA%\PDFTranslate\settings.json` (macOS: Application Support;
  elsewhere `~/.config`), outside the folder the updater replaces. OCR is
  deliberately not remembered. QA scripts must patch `settings_path` rather
  than write the user's real file, and `App._check_for_update` so a
  screenshot never depends on GitHub.
- `app/icons.py` draws the queue's status badges, the spinner frames, the
  button glyphs and the empty-queue icon with Pillow at start-up, in light
  and dark variants, so no font decides how a status mark looks. `app/gui.py`
  relies on a few customtkinter 5.2.2 internals (`FieldMenu`'s canvas tags,
  the draw engine's even-height rounding in `_hairline`, the scrollable
  frame's scrollbar in `_fit_scrollbar`); before raising the pin in
  `requirements-app.txt`, re-check those and look at the window in light and
  dark mode.
- `scripts/translate_pdf.py`: supported CLI and stable output staging.
- `scripts/fetch_assets.py`: downloads the layout model and Unicode PDF font.
- `colab/`: fork-specific Google Colab workflow (notebook, LLM-provider
  failover, glossary checks, RapidOCR line layers for image-only scans
  (`ocr_layer.py`: no angle classifier, a failed line is re-read on its own
  crop, vertical tabs and still-failing lines stay as pixels, bold from stroke
  width, font size from the measured ink band divided by the glyphs' own em
  height; `ocr_check.py` corrects the Chinese model's confident misreadings
  by rule (3O→30, lone S→$, 二→–, fullwidth punctuation; leftover CJK stays
  pixels), compares every number with Tesseract (flag only, never chosen: it
  confuses 3/5/8 on Cabin) and, with `--ocr-reader paddle-vl` on a GPU,
  re-reads each line with PaddleOCR-VL at a pinned revision (used when it
  broadly agrees with RapidOCR, numbers that differ are flagged); flagged
  lines go to `<stem>-kiem-tra-ocr.pdf`), repainting with the dominant ring colour (bins pooled against JPEG
  tint, per-channel mode) while thin long horizontal rules are restored from
  the unpainted image, rebuild of image PDFs with an invisible OCR layer in the
  translation font (header/footer lines rebuilt only when they hold words),
  bundled OFL Cabin fonts; outputs are staged in the work folder and published
  by deleting the old file first so Google Drive gives them a new id and a
  fresh preview). It drives `scripts/translate_pdf.py` through the
  handoff engine and never imports engine internals except the handoff
  placeholder validators; `tests/test_colab_tools.py` covers it and skips the
  provider tests when `openai` is absent. `acca_translate.py` splits the run
  into `extract` (OCR, rebuild, segment extraction) and `build` (Cabin PDF,
  bilingual copy, publish; returns the segments left in English with a
  reason); `openai` and `router` load only when an API is called.
  `claude_translate.py` reuses both for translation without an API key:
  `prepare` writes `BASE/claude/<stem>/batches/lo-NNN.txt` (`id<TAB>src`,
  spaces collapsed, `\t`/`\n` escaped, at most 60 segments or 12,000
  characters, headed by the glossary terms the batch uses; cached segments are
  skipped), the translator writes `answers/*.txt` as `id<TAB>dst`, read in
  name order so `sua-*.txt` replaces `lo-*.txt`, and `finish` rejects answers
  with wrong tags, numbers or unknown ids (a glossary mismatch is only a
  warning), caches accepted ones as provider `claude`, builds through `build`
  and can render pages to 60 dpi PNGs. The workflow for a new session is
  `.claude/skills/claude-translate/SKILL.md`. With no CSV in `BASE/glossary`,
  `load_glossary` uses `colab/glossary/ifrs-acca.csv`; `terms_in` collapses
  whitespace first, since justified lines arrive with doubled spaces and
  multi-word terms never matched them.
- `pdf2zh/`: bundled extraction, translation, preservation, and PDF rendering.
  `pdf2zh/bilingual.py` builds the opt-in bilingual copy from two finished
  PDFs.
- `tests/`: standard-library `unittest` regressions.
- `references/`: product preservation contract.
- `app.spec`, `build.ps1`: Windows packaging.
- `app-macos.spec`, `build-macos.sh`: native macOS packaging.
- `.github/workflows/`: release and on-demand macOS builds.

## Commands

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe app\gui.py
.\.venv\Scripts\python.exe scripts\translate_pdf.py INPUT.pdf --output-dir OUT
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
```

macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-app.txt
.venv/bin/python app/gui.py
.venv/bin/python scripts/translate_pdf.py INPUT.pdf --output-dir OUT
.venv/bin/python -m unittest discover -s tests -v
bash build-macos.sh
```

## Code and Tests

Use four-space indentation, type annotations, concise docstrings,
`snake_case`, `PascalCase`, and `UPPER_CASE`. Prefer `pathlib.Path`, explicit
exceptions, and small helpers whose geometry can be unit-tested. Match nearby
imports and style; there is no repository-wide formatter.

Name tests `test_<feature>.py`, classes `<Feature>Tests`, and methods
`test_<behavior>`. Test observable behavior rather than comments or exact
implementation text. Preserve unrelated dirty-worktree changes.

Commit subjects are short and imperative. PRs state user impact, validation,
and representative visual evidence for GUI/PDF changes. Never commit local
test PDFs or packaged artifacts.
