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

    return RapidOCR(params={"Global.log_level": "error"})


# Ink strokes this much thicker than the page's usual text mark a bold line.
BOLD_STROKE_RATIO = 1.9


def box_rect(box, scale=1.0):
    xs, ys = [float(p[0]) / scale for p in box], [float(p[1]) / scale for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def line_fits(rect, text):
    """Recognised TEXT plausibly fills RECT (points): horizontal, no lost letters."""
    x0, y0, x1, y1 = rect
    height, width = y1 - y0, x1 - x0
    if height <= 0 or width <= 0 or not text or height > width * 1.5:  # vertical side tabs stay picture
        return False
    natural = fitz.get_text_length(text, fontname="helv", fontsize=height * 0.78)
    return natural > 0 and width / natural <= MAX_STRETCH


def stroke_weight(img, rect_px):
    """Median horizontal ink run divided by line height: thicker for bold type."""
    x0, y0, x1, y1 = (int(round(v)) for v in rect_px)
    grey = img[max(0, y0):y1, max(0, x0):x1].astype(float).mean(axis=2)
    if grey.size == 0:
        return 0.0
    paper = np.percentile(grey, 95)
    ink = grey < paper - 0.45 * (paper - grey.min())
    runs = []
    for row in ink:
        edges = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        runs.extend(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1))
    return float(np.median(runs)) / max(1, y1 - y0) if runs else 0.0


def recognise_again(engine, page, rect, dpi):
    """Re-read one line on its own crop: whole-page passes sometimes garble a
    clear line that reads perfectly in isolation."""
    x0, y0, x1, y1 = rect
    pad = (y1 - y0) * 0.3
    clip = fitz.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad) & page.rect
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    crop = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    result = engine(crop, use_cls=False)
    scale = dpi / 72
    parts = []
    for box, text, score in zip(result.boxes if result.boxes is not None else [],
                                result.txts or [], result.scores or []):
        bx0, by0, bx1, by1 = box_rect(box, scale)
        cy = clip.y0 + (by0 + by1) / 2
        if y0 <= cy <= y1 and text and text.strip():
            parts.append((clip.x0 + bx0, text.strip(), score))
    if not parts or min(score for _, _, score in parts) < MIN_SCORE:
        return None
    return " ".join(text for _, text, _ in sorted(parts))


_FONTS = {}


def glyph_extent(text, font="helv"):
    """Top and bottom of TEXT's glyphs in em, baseline at 0 ("Section C" has no
    descender: about 0.74 and -0.02; "Co (Dec)" reaches -0.21)."""
    f = _FONTS.setdefault(font, fitz.Font(font))
    top = bottom = 0.0
    for ch in text:
        if not ch.isspace():
            r = f.glyph_bbox(ord(ch))
            top, bottom = max(top, r.y1), min(bottom, r.y0)
    return top, bottom


def ink_band(img, rect_px):
    """Top and bottom pixel rows of the ink of the line inside RECT_PX, or None.
    OCR boxes overlap the next line a little, so only the band of inked rows
    around the box centre counts; full-width rows (underlines) are ignored."""
    x0, y0, x1, y1 = (int(round(v)) for v in rect_px)
    x0, y0 = max(0, x0), max(0, y0)
    grey = img[y0:y1, x0:x1].astype(float).mean(axis=2)
    if grey.shape[0] < 4 or grey.shape[1] < 4:
        return None
    paper = np.percentile(grey, 95)
    if paper - grey.min() < 60:
        return None
    ink = grey < paper - 0.45 * (paper - grey.min())
    cover = ink.mean(axis=1)
    inked = (ink.sum(axis=1) >= 2) & (cover < 0.6)
    rows = np.flatnonzero(inked)
    if not len(rows):
        return None
    gap = max(2, int(0.12 * (y1 - y0)))  # the dot of an i is closer than the next line
    centre = (y1 - y0) / 2
    bands, start = [], rows[0]
    for a, b in zip(rows[:-1], rows[1:]):
        if b - a > gap:
            bands.append((start, a))
            start = b
    bands.append((start, rows[-1]))
    top, bottom = min(bands, key=lambda band: 0 if band[0] <= centre <= band[1]
                      else min(abs(band[0] - centre), abs(band[1] - centre)))
    return y0 + top, y0 + bottom + 1


def line_metrics(rect, text, font, band=None):
    """Font size and baseline (points) for TEXT drawn over RECT.
    The box height alone underestimates lines without descenders ("Section C")
    because the box only spans capital height; the measured ink band divided by
    the glyphs' own height in em gives the printed size."""
    x0, y0, x1, y1 = rect
    height = y1 - y0
    size, baseline = height * 0.78, y1 - height * 0.22
    if band:
        top, bottom = glyph_extent(text, font)
        ink_top, ink_bottom = band
        if top - bottom > 0.3 and ink_bottom > ink_top:
            measured = (ink_bottom - ink_top) / (top - bottom)
            if 0.7 * size <= measured <= 1.6 * size:  # otherwise the measure caught something else
                size, baseline = measured, ink_bottom + bottom * measured
    return size, baseline


def add_invisible_line(page, rect, text, bold=False, band=None):
    """Write TEXT invisibly, stretched over RECT (points).
    BAND: top and bottom of the line's ink in points, when measured."""
    x0, y0, x1, y1 = rect
    font = "hebo" if bold else "helv"
    size, baseline = line_metrics(rect, text, font, band)
    natural = fitz.get_text_length(text, fontname=font, fontsize=size)
    origin = fitz.Point(x0, baseline)
    stretch = fitz.Matrix((x1 - x0) / natural, 1)
    page.insert_text(origin, text, fontname=font, fontsize=size, render_mode=3, morph=(origin, stretch))


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
        # Book pages are upright: the angle classifier only turns short cells
        # such as "$'000" upside down into "000.$".
        result = engine(img, use_cls=False)
        lines, skipped = [], 0
        for box, text, score in zip(result.boxes if result.boxes is not None else [],
                                    result.txts or [], result.scores or []):
            text = " ".join((text or "").split())
            if not text:
                continue
            rect = box_rect(box, scale)
            if rect[3] - rect[1] > (rect[2] - rect[0]) * 1.5:
                skipped += 1  # vertical side tab: keep the picture
                continue
            if score < MIN_SCORE or not line_fits(rect, text):
                text = recognise_again(engine, page, rect, dpi)
                if not text or not line_fits(rect, text):
                    skipped += 1
                    continue
            rect_px = [v * scale for v in rect]
            band = ink_band(img, rect_px)
            band = band and (band[0] / scale, band[1] / scale)
            lines.append((rect, text, stroke_weight(img, rect_px), band))
        typical = float(np.median([w for _, _, w, _ in lines])) if lines else 0.0
        for rect, text, weight, band in lines:
            add_invisible_line(page, rect, text, bold=typical > 0 and weight >= typical * BOLD_STROKE_RATIO,
                               band=band)
        if on_page:
            on_page(n, len(targets), i + 1, len(lines), skipped)
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
