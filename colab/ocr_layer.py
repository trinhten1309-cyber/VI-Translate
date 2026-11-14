"""Give image-only scan pages an invisible OCR text layer.

Pages that are pictures with no text at all are recognised with RapidOCR
(already installed by requirements-ocr.txt). Each recognised line is written as
invisible text stretched over the exact box the line occupies in the picture,
which is the same shape as the OCR layer that scanners and online readers
produce. rebuild_text_layer.py then repaints the picture under those lines and
turns them into real text, so tables are handled like any other page instead
of being left untouched.

Usage: python ocr_layer.py INPUT.pdf OUTPUT.pdf [--pages 1-5] [--dpi 200]
"""
import argparse
import sys

import numpy as np
import pymupdf as fitz

from rebuild_text_layer import parse_pages

MIN_SCORE = 0.85
# A line whose box is far wider than its recognised text lost characters
# (e.g. "Go  t   cith"): keep it as picture rather than replace real words.
MAX_STRETCH = 1.8


def page_is_image_only(page):
    return not page.get_text().strip() and bool(page.get_images())


def image_only_pages(path, pages=""):
    with fitz.open(path) as doc:
        return [i + 1 for i in sorted(parse_pages(pages, doc.page_count)) if page_is_image_only(doc[i])]


def _engine():
    from rapidocr import RapidOCR  # imported lazily: only scans need it

    return RapidOCR(params={"Global.log_level": "warning"})


def add_invisible_line(page, box, text, scale):
    """Write TEXT invisibly over BOX (picture pixels). Returns False when the
    recognition looks incomplete and the line is left as picture."""
    xs, ys = [float(p[0]) / scale for p in box], [float(p[1]) / scale for p in box]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    height = y1 - y0
    text = " ".join(text.split())
    if height <= 0 or x1 <= x0 or not text:
        return False
    size = height * 0.78
    natural = fitz.get_text_length(text, fontname="helv", fontsize=size)
    if natural <= 0 or (x1 - x0) / natural > MAX_STRETCH:
        return False
    origin = fitz.Point(x0, y1 - height * 0.22)
    stretch = fitz.Matrix((x1 - x0) / natural, 1)
    page.insert_text(origin, text, fontname="helv", fontsize=size, render_mode=3, morph=(origin, stretch))
    return True


def add_ocr_layer(input_path, output_path, pages="", dpi=200, engine=None, on_page=None):
    """Write a copy of INPUT where the selected image-only pages carry an
    invisible OCR layer. Returns the 1-based page numbers that were recognised."""
    doc = fitz.open(input_path)
    wanted = parse_pages(pages, doc.page_count)
    targets = [i for i in sorted(wanted) if page_is_image_only(doc[i])]
    if targets and engine is None:
        engine = _engine()
    scale = dpi / 72
    for n, i in enumerate(targets, 1):
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        result = engine(img)
        kept = skipped = 0
        for box, text, score in zip(result.boxes if result.boxes is not None else [],
                                    result.txts or [], result.scores or []):
            if not text or not text.strip():
                continue
            if score >= MIN_SCORE and add_invisible_line(page, box, text, scale):
                kept += 1
            else:
                skipped += 1
        if on_page:
            on_page(n, len(targets), i + 1, kept, skipped)
    doc.save(output_path, garbage=3, deflate=True)
    return [i + 1 for i in targets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--pages", default="")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    done = add_ocr_layer(a.input, a.output, a.pages, a.dpi,
                         on_page=lambda n, total, page, kept, skipped: print(
                             f"OCR {n}/{total}: page {page} - {kept} lines, {skipped} left as picture"))
    print(f"Recognised {len(done)} image-only pages" + (f": {done}" if done else ""))


if __name__ == "__main__":
    sys.exit(main())
