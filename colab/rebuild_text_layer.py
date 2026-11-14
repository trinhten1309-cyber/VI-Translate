"""Turn an 'image + invisible OCR text' PDF into a real text PDF.

Scanned or flattened PDFs (e.g. study kits exported page-by-page as images) show
their words inside pictures and keep an invisible OCR text layer for search.
A translator can only *add* text on top of such pages, so the English in the
picture stays visible underneath. This script:

  1. renders each page,
  2. paints over every OCR word with the surrounding background colour,
  3. writes the OCR words back as normal visible text in their original colour,

so VI-Translate can replace the text cleanly. Pages that already have visible
text are copied unchanged.

Usage: python rebuild_text_layer.py INPUT.pdf OUTPUT.pdf [--dpi 200]
"""
import argparse
import re
import sys

import numpy as np
import pymupdf as fitz


def page_needs_rebuild(page):
    """True when the page shows images and all its text is invisible (render mode 3)."""
    trace = page.get_texttrace()
    if not trace or not page.get_images():
        return False
    invisible = sum(len(t["chars"]) for t in trace if t["type"] == 3)
    total = sum(len(t["chars"]) for t in trace)
    return total > 0 and invisible / total > 0.9


def ring_pixels(arr, x0, y0, x1, y1, pad):
    h, w, _ = arr.shape
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    outer = arr[Y0:Y1, X0:X1].reshape(-1, 3)
    mask = np.ones((Y1 - Y0, X1 - X0), bool)
    mask[max(0, y0 - Y0):y1 - Y0, max(0, x0 - X0):x1 - X0] = False
    ring = arr[Y0:Y1, X0:X1][mask]
    return ring if len(ring) else outer


def dominant_colour(pixels):
    """Most common colour around a line. A median mixes the white page above a
    line with a grey table header below it and paints a grey band."""
    bins = (pixels // 16).astype(np.int32)
    keys = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    top = values[counts.argmax()]
    return pixels[keys == top].mean(axis=0)


def rebuild_page(src_page, out_doc, dpi):
    scale = dpi / 72
    pix = src_page.get_pixmap(dpi=dpi, alpha=False)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    spans = []
    H = src_page.rect.height
    for block in src_page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            line_spans = [s for s in line["spans"] if s["text"].strip()]
            if not line_spans:
                continue
            # leave running headers/footers (page numbers, logos) inside the picture
            if line["bbox"][1] > H * 0.92 or line["bbox"][3] < H * 0.03:
                continue
            # OCR gives every word its own slightly different size; use one size per line
            size = float(np.median([s["size"] for s in line_spans if len(s["text"].strip()) > 1]
                                   or [s["size"] for s in line_spans]))
            for s in line_spans:
                s["size"] = size
                s["_line"] = id(line)
                spans.append(s)
    # OCR sizes wobble from line to line: snap body-sized lines to the page's body size
    if spans:
        body = float(np.median([s["size"] for s in spans]))
        for s in spans:
            if 0.7 * body <= s["size"] <= 1.3 * body:
                s["size"] = body
    painted = []
    for s in spans:
        x0, y0, x1, y1 = (int(round(v * scale)) for v in s["bbox"])
        x0, y0, x1, y1 = max(0, x0 - 2), max(0, y0 - 2), min(arr.shape[1], x1 + 2), min(arr.shape[0], y1 + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        ring = ring_pixels(arr, x0, y0, x1, y1, pad=4)
        bg = dominant_colour(ring)
        inside = arr[y0:y1, x0:x1].reshape(-1, 3).astype(float)
        dist = np.linalg.norm(inside - bg, axis=1)
        far = inside[dist >= np.percentile(dist, 90)] if len(inside) else inside
        fg = far.mean(axis=0) if len(far) and dist.max() > 60 else (0, 0, 0)
        painted.append((x0, y0, x1, y1, bg))
        s["_fg"] = tuple(float(c) / 255 for c in fg)
    for x0, y0, x1, y1, bg in painted:  # paint after sampling so neighbours don't affect each other
        arr[y0:y1, x0:x1] = bg.astype(np.uint8)

    clean = fitz.Pixmap(fitz.csRGB, pix.width, pix.height, arr.tobytes(), False)
    page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
    page.insert_image(page.rect, stream=clean.tobytes("jpeg", jpg_quality=88))
    lines = {}
    for s in spans:
        if "_fg" in s:
            lines.setdefault(s["_line"], []).append(s)
    for group in lines.values():
        group.sort(key=lambda s: s["bbox"][0])
        text = ""
        for prev, s in zip([None] + group[:-1], group):
            if prev is not None and not text.endswith(" ") and not s["text"].startswith(" "):
                text += " "
            text += s["text"]
        text = " ".join(text.split())
        text = re.sub(r"\s+([.,;:!?)\]%])", r"\1", text)   # OCR puts spaces before punctuation
        text = re.sub(r"([(\[$])\s+", r"\1", text)
        first = group[0]
        bold = sum(("bold" in s["font"].lower() or s["flags"] & 16) for s in group) * 2 > len(group)
        italic = sum(("italic" in s["font"].lower() or s["flags"] & 2) for s in group) * 2 > len(group)
        font = {(False, False): "helv", (True, False): "hebo",
                (False, True): "heit", (True, True): "hebi"}[(bool(bold), bool(italic))]
        size = first["size"]
        box_w = group[-1]["bbox"][2] - first["bbox"][0]
        width = fitz.get_text_length(text, fontname=font, fontsize=size)
        if width > box_w * 1.05 and width > 0:
            size *= max(box_w / width, 0.85)   # shrink a little to fit, never far below the body size
        fg = tuple(float(np.median([s["_fg"][k] for s in group])) for k in range(3))
        page.insert_text(first["origin"], text, fontname=font, fontsize=size, color=fg)

def parse_pages(spec, count):
    """'1-5,8' -> {0,1,2,3,4,7}; empty -> all pages."""
    if not spec or not spec.strip():
        return set(range(count))
    out = set()
    for part in spec.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a) - 1, min(int(b), count)))
        elif part:
            out.add(int(part) - 1)
    return {i for i in out if 0 <= i < count}


def needs_rebuild(path, pages=""):
    with fitz.open(path) as src:
        return [i + 1 for i in sorted(parse_pages(pages, src.page_count)) if page_needs_rebuild(src[i])]


def rebuild(input_path, output_path, pages="", dpi=200):
    """Rebuild the selected pages that carry an invisible OCR layer; copy the rest.
    Returns the 1-based page numbers that were rebuilt."""
    src = fitz.open(input_path)
    wanted = parse_pages(pages, src.page_count)
    out = fitz.open()
    rebuilt = []
    for i, page in enumerate(src):
        if i in wanted and page_needs_rebuild(page):
            rebuild_page(page, out, dpi)
            rebuilt.append(i + 1)
        else:
            out.insert_pdf(src, from_page=i, to_page=i)
    out.save(output_path, garbage=3, deflate=True)
    return rebuilt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--pages", default="")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    rebuilt = rebuild(a.input, a.output, a.pages, a.dpi)
    print(f"Rebuilt {len(rebuilt)} pages with an invisible OCR layer"
          + (f": {rebuilt}" if rebuilt else " - nothing to do, file already has visible text"))


if __name__ == "__main__":
    sys.exit(main())
