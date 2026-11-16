"""Safety checks for OCR'd lines on image-only pages.

RapidOCR uses a Chinese recognition model, which never gets a digit wrong on
our test pages but does read 30 as "3O", "$" alone as "S", a table dash as
"二", and fullwidth punctuation. Its confidence score stays high on those,
so they are corrected by rule here, and every number is compared with a
second reader:

- Tesseract (option 1, CPU): reads the whole page once; a number it reads
  differently is only flagged, never chosen, because on the Cabin font of the
  BPP kits it confuses 3, 5 and 8 far more often than RapidOCR.
- PaddleOCR-VL (option 2, needs a GPU): re-reads every line RapidOCR found.
  Its text is used when it broadly agrees with RapidOCR; if their numbers
  differ the line is flagged; if it reads something unrelated, RapidOCR's
  text is kept and the line is flagged.

Flagged lines are listed in a review PDF (red boxes on the original page and
both readings beside it) so a person checks them; nothing is guessed.
"""
import difflib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pymupdf as fitz

FONT = Path(__file__).parent / "fonts" / "Cabin-Regular.ttf"

# Any CJK character in an English document is a misreading by the Chinese model.
CJK = re.compile(r"[⺀-⿿　-ヿ㐀-鿿가-힯豈-﫿]")
DASH_LINES = {"二", "一", "ー", "—", "―"}
NUMBER = re.compile(r"\d(?:[\d,.]*\d)?")
GROUPED = re.compile(r"\(?-?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?%?[.,;:]?")
CELL_CHARS = set("0123456789,.()$'%-– ")


def _fullwidth_to_ascii(text):
    return "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else
                   " " if c == "　" else c for c in text)


def _fix_token(tok):
    """O/o and l/I/| inside a number become 0 and 1 ("3O" -> "30", "2OX7" -> "20X7")."""
    if not re.search(r"\d", tok):
        return tok
    chars = list(tok)
    for i, c in enumerate(chars):
        prev = chars[i - 1] if i else ""
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        after_digit = prev.isdigit() and (nxt == "" or nxt.isdigit() or nxt in ",.)%XOo")
        before_digit = prev in ("", "(", "$", ",", ".") and nxt.isdigit()
        if c in "Oo" and (after_digit or before_digit):
            chars[i] = "0"
        elif c in "lI|" and ((prev.isdigit() and (nxt == "" or nxt.isdigit() or nxt in ",.)"))
                             or (prev in ("", "(", ",") and nxt.isdigit())):
            chars[i] = "1"
    return "".join(chars)


def clean_text(text):
    """Correct the misreadings the Chinese model makes on English pages."""
    text = " ".join(_fullwidth_to_ascii(text or "").split())
    if text in DASH_LINES:
        return "–"
    if text in ("S", "S'000", "S000", "$000"):
        return "$" if text == "S" else "$'000"
    return " ".join(_fix_token(t) for t in text.split())


def has_cjk(text):
    return bool(CJK.search(text or ""))


def numbers_in(text):
    return NUMBER.findall(text or "")


def number_problem(text):
    """Why TEXT does not look like a well-formed number, or None."""
    text = text or ""
    for tok in text.split():
        if re.search(r"\d,\d", tok) and not GROUPED.fullmatch(tok):
            return f"số sai định dạng: {tok}"
    compact = text.replace(" ", "")
    if compact and set(compact) <= CELL_CHARS and re.search(r"\d", compact):
        if compact.count("(") != compact.count(")"):
            return f"thiếu ngoặc: {text}"
        if re.search(r"\d\.\d{3}(?:\D|$)", compact) and "," not in compact:
            return f"dấu chấm ở chỗ thường là dấu phẩy: {text}"
    return None


def similar(a, b):
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def words_in(words, rect):
    """Words (text, (x0, y0, x1, y1)) whose centre lies in RECT, left to right."""
    x0, y0, x1, y1 = rect
    h = y1 - y0
    got = [(wx0, t) for t, (wx0, wy0, wx1, wy1) in words
           if x0 - 2 <= (wx0 + wx1) / 2 <= x1 + 2 and y0 - 0.25 * h <= (wy0 + wy1) / 2 <= y1 + 0.25 * h]
    return " ".join(t for _, t in sorted(got))


def judge(rapid, vl=None, tess=None):
    """Pick the text for one line and say why a person should look at it.

    RAPID: RapidOCR's text; VL: PaddleOCR-VL's (option 2) or None; TESS:
    Tesseract's words in the same box (option 1) or None when not run.
    Returns (text, readings, reasons)."""
    rapid = clean_text(rapid)
    readings, reasons, text = {"RapidOCR": rapid}, [], rapid
    if vl is not None:
        vl = clean_text(vl)
        readings["PaddleOCR-VL"] = vl
        if not vl or similar(rapid, vl) < 0.6:
            reasons.append("PaddleOCR-VL đọc khác hẳn RapidOCR (giữ bản RapidOCR)")
        else:
            text = vl
            if numbers_in(rapid) != numbers_in(vl):
                reasons.append("hai bộ đọc ra số khác nhau")
    if tess is not None and numbers_in(text):
        tess = clean_text(tess)
        readings["Tesseract"] = tess
        # With PaddleOCR-VL on, RapidOCR and PaddleOCR-VL agreeing outvotes
        # Tesseract, which misreads this kind of type the most.
        if vl is None and tess and numbers_in(tess) != numbers_in(text):
            reasons.append("hai bộ đọc ra số khác nhau")
    problem = number_problem(text)
    if problem:
        reasons.append(problem)
    return text, readings, reasons


# ------------------------------------------------------------ second readers
class Tesseract:
    """Option 1: word boxes from the tesseract command (apt install tesseract-ocr)."""

    @staticmethod
    def available():
        return shutil.which("tesseract") is not None

    def words(self, png, scale):
        out = subprocess.run(["tesseract", "stdin", "stdout", "--dpi", str(round(72 * scale)), "tsv"],
                             input=png, capture_output=True, timeout=300).stdout.decode("utf-8", "replace")
        words = []
        for row in out.splitlines()[1:]:
            c = row.split("\t")
            if len(c) == 12 and c[0] == "5" and c[11].strip():
                x, y, w, h = (int(v) for v in c[6:10])
                words.append((c[11].strip(), (x / scale, y / scale, (x + w) / scale, (y + h) / scale)))
        return words


class PaddleVL:
    """Option 2: PaddleOCR-VL reads each line crop. Needs a GPU to be usable
    (about 19 s per line on a CPU)."""

    MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
    REVISION = "132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd"  # the snapshot that was benchmarked

    def __init__(self, device="cuda", batch=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch, self.device, self.batch = torch, device, batch
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL, revision=self.REVISION, trust_remote_code=True, dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(self.MODEL, revision=self.REVISION, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"

    def _generate(self, prompt, images):
        inputs = self.processor(text=[prompt] * len(images), images=images, padding=True,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        return self.processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def read(self, crops):
        from PIL import Image

        # Batches of similar widths: every image in a batch is padded to the largest.
        order = sorted(range(len(crops)), key=lambda k: crops[k].shape[1])
        out = [""] * len(crops)
        for i in range(0, len(order), self.batch):
            part = order[i:i + self.batch]
            images = [Image.fromarray(crops[k]) for k in part]
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}]}],
                tokenize=False, add_generation_prompt=True)
            try:
                texts = self._generate(prompt, images)
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                self.batch = max(1, self.batch // 2)  # smaller batches from here on
                texts = [t for img in images for t in self._generate(prompt, [img])]
            for k, t in zip(part, texts):
                out[k] = " ".join(t.split())
        return out


def load_paddle_vl(log=print, allow_cpu=False):
    """PaddleVL on the GPU, or None (with the reason logged) so the run falls
    back to RapidOCR plus the option 1 checks."""
    try:
        import torch
    except ImportError:
        log("⚠️  PaddleOCR-VL chưa được cài (cần chọn OCR_READER = paddle-vl trong notebook) → dùng RapidOCR")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        log("⚠️  Không có GPU (Runtime → Change runtime type → T4 GPU) → dùng RapidOCR")
        return None
    try:
        return PaddleVL(device)
    except Exception as e:  # download or load failure must not stop the translation
        log(f"⚠️  Không tải được PaddleOCR-VL ({type(e).__name__}: {str(e)[:150]}) → dùng RapidOCR")
        return None


# ------------------------------------------------------------ review sheet
def write_review(source_pdf, flags, out_path):
    """One page per flagged source page: the page with numbered red boxes on
    the left, the readings of each box on the right. Returns the flag count."""
    if not flags:
        return 0
    src = fitz.open(source_pdf)
    doc = fitz.open()
    font = fitz.Font(fontfile=str(FONT))
    by_page = {}
    for f in flags:
        by_page.setdefault(f["page"], []).append(f)
    for pno in sorted(by_page):
        sp = src[pno - 1]
        w, h = sp.rect.width, sp.rect.height
        page = doc.new_page(width=w + 300, height=h)
        page.show_pdf_page(fitz.Rect(0, 0, w, h), src, pno - 1)
        page.insert_font(fontname="cabin", fontfile=str(FONT))
        writer = fitz.TextWriter(page.rect)
        writer.append((w + 14, 30), f"Trang {pno}: {len(by_page[pno])} dòng cần xem lại", font=font, fontsize=11)
        y = 52
        for n, f in enumerate(by_page[pno], 1):
            x0, y0, x1, y1 = f["rect"]
            page.draw_rect(fitz.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2), color=(0.85, 0, 0), width=1.2)
            page.insert_text((max(1, x0 - 14), y0 + 7), str(n), fontname="cabin", fontsize=8, color=(0.85, 0, 0))
            lines = [f"{n}. Dùng: {f['text']}"]
            lines += [f"    {k}: {v or '(không đọc được)'}" for k, v in f["readings"].items() if k != "RapidOCR" or
                      v != f["text"]]
            lines += [f"    → {r}" for r in f["reasons"]]
            for line in lines:
                if y > h - 20:
                    break
                writer.append((w + 14, y), line[:70], font=font, fontsize=8)
                y += 11
            y += 4
        writer.write_text(page)
    doc.save(out_path, garbage=3, deflate=True)
    return len(flags)
