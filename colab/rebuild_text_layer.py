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
    bins = (pixels // 4).astype(np.int32)  # fine bins: a coarse bin averages in anti-aliased edges
    keys = (bins[:, 0] * 64 + bins[:, 1]) * 64 + bins[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    # JPEG noise spreads plain paper over neighbouring bins; the single fullest
    # bin could be a tinted one (255, 249, 255) that shows as a pink patch.
    # Score each bin with its neighbours within 2 bins (8 levels) per channel.
    vb = np.stack([values // 4096, values // 64 % 64, values % 64], axis=1)
    near = (np.abs(vb[:, None, :] - vb[None, :, :]) <= 2).all(axis=2) if len(values) <= 4000 else None
    if near is None:
        top = vb[counts.argmax()]
    else:
        top = vb[(near * counts[None, :]).sum(axis=1).argmax()]
    cluster = pixels[(np.abs(bins - top) <= 2).all(axis=1)].astype(np.int64)
    # Most common value per channel: a median is pulled grey by the
    # anti-aliased edges of nearby letters and leaves a visible patch.
    return np.array([np.bincount(cluster[:, c], minlength=256).argmax() for c in range(3)], float)


def rule_mask(arr, bg, x0, y0, x1, y1, min_run, max_thick):
    """Pixels of horizontal rules (underlines, double underlines, table lines)
    inside the box x0..x1, y0..y1, as a boolean array of the box's shape.
    A rule is a run of non-background pixels at least MIN_RUN long, looked for
    in a window wider than the box so a rule longer than the number above it
    still counts, and at most MAX_THICK rows thick so filled shapes and text
    never qualify. Its lighter anti-aliased edge rows are included: dropping
    them left rules thin and faint next to the untouched part of the line."""
    H, W = arr.shape[:2]
    wx0, wx1 = max(0, x0 - min_run), min(W, x1 + min_run)
    dist = np.linalg.norm(arr[y0:y1, wx0:wx1].astype(float) - bg, axis=2)
    strong = dist > 40
    runs = np.zeros_like(strong)
    for r in range(strong.shape[0]):
        edges = np.diff(np.concatenate(([0], strong[r].astype(np.int8), [0])))
        for s, t in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if t - s >= min_run:
                runs[r, s:t] = True
    if not runs.any():
        return None
    # thickness of each vertical stack of run pixels
    up, down = np.zeros(runs.shape, int), np.zeros(runs.shape, int)
    for r in range(runs.shape[0]):
        up[r] = (up[r - 1] + 1) * runs[r] if r else runs[r]
    for r in range(runs.shape[0] - 1, -1, -1):
        down[r] = (down[r + 1] + 1) * runs[r] if r < runs.shape[0] - 1 else runs[r]
    runs &= (up + down - 1) <= max_thick
    soft = dist > 12
    for _ in range(2):  # grow into the anti-aliased rows just above and below
        grown = runs.copy()
        grown[1:] |= runs[:-1] & soft[1:]
        grown[:-1] |= runs[1:] & soft[:-1]
        runs = grown
    return runs[:, x0 - wx0:x1 - wx0]


FACES = ("regular", "bold", "italic", "bold-italic")
BASE14 = {"regular": "helv", "bold": "hebo", "italic": "heit", "bold-italic": "hebi"}


def rebuild_page(src_page, out_doc, dpi, fonts=None):
    """FONTS: optional {face: font file} (faces in FACES) for the rebuilt text;
    Helvetica when missing. Numbers and labels that are not translated keep
    this font, so it should match the translation font."""
    scale = dpi / 72
    pix = src_page.get_pixmap(dpi=dpi, alpha=False)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    orig = arr.copy()
    spans = []
    H = src_page.rect.height
    for block in src_page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            line_spans = [s for s in line["spans"] if s["text"].strip()]
            if not line_spans:
                continue
            # In the running header/footer band, page numbers and logo lettering
            # ("BPP") stay in the picture; words such as "Questions 127" are
            # rebuilt so they are translated like everything else.
            if line["bbox"][1] > H * 0.92 or line["bbox"][3] < H * 0.03:
                if not re.search(r"[a-z]{3}", "".join(s["text"] for s in line_spans)):
                    continue
            # OCR gives every word its own slightly different size; use one size per line
            size = float(np.median([s["size"] for s in line_spans if len(s["text"].strip()) > 1]
                                   or [s["size"] for s in line_spans]))
            for s in line_spans:
                s["size"] = size
                s["_line"] = id(line)
                spans.append(s)
    # OCR sizes wobble a little from line to line: snap body-sized lines to the
    # page's body size, but leave sub-headings a size or two larger alone
    if spans:
        body = float(np.median([s["size"] for s in spans]))
        for s in spans:
            if 0.88 * body <= s["size"] <= 1.12 * body:
                s["size"] = body
    painted = []
    rules = np.zeros(arr.shape[:2], bool)
    H, W = arr.shape[:2]
    for s in spans:
        cx0, cy0, cx1, cy1 = (int(round(v * scale)) for v in s["bbox"])
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        # OCR boxes hug the letters' core; ascenders, descenders and italic
        # overhang reach a little past it and would remain as faint ghosts.
        h = cy1 - cy0
        x0, x1 = max(0, cx0 - max(2, int(0.15 * h))), min(W, cx1 + max(2, int(0.15 * h)))
        y0, y1 = max(0, cy0 - max(2, int(0.2 * h))), min(H, cy1 + max(2, int(0.2 * h)))
        ring = ring_pixels(arr, x0, y0, x1, y1, pad=4)
        bg = dominant_colour(ring)
        inside = arr[cy0:cy1, cx0:cx1].reshape(-1, 3).astype(float)
        dist = np.linalg.norm(inside - bg, axis=1)
        far = inside[dist >= np.percentile(dist, 90)] if len(inside) else inside
        fg = far.mean(axis=0) if len(far) and dist.max() > 60 else (0, 0, 0)
        # Keep underlines, double underlines and table rules that cross the box.
        keep = rule_mask(arr, bg, x0, y0, x1, y1, min_run=max(24, 2 * h), max_thick=max(3, int(0.15 * h)))
        if keep is not None:
            rules[y0:y1, x0:x1] |= keep
        painted.append((x0, y0, x1, y1, bg))
        s["_fg"] = tuple(float(c) / 255 for c in fg)
    for x0, y0, x1, y1, bg in painted:  # paint after sampling so neighbours don't affect each other
        arr[y0:y1, x0:x1] = bg.astype(np.uint8)
    arr[rules] = orig[rules]

    clean = fitz.Pixmap(fitz.csRGB, pix.width, pix.height, arr.tobytes(), False)
    page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
    page.insert_image(page.rect, stream=clean.tobytes("jpeg", jpg_quality=88))
    faces = _faces(fonts)
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
        face = {(False, False): "regular", (True, False): "bold",
                (False, True): "italic", (True, True): "bold-italic"}[(bool(bold), bool(italic))]
        name, measure, fontfile = faces[face]
        size = first["size"]
        box_w = group[-1]["bbox"][2] - first["bbox"][0]
        width = measure.text_length(text, fontsize=size)
        if width > box_w * 1.05 and width > 0:
            size *= max(box_w / width, 0.85)   # shrink a little to fit, never far below the body size
        fg = tuple(float(np.median([s["_fg"][k] for s in group])) for k in range(3))
        page.insert_text(first["origin"], text, fontname=name, fontfile=fontfile, fontsize=size, color=fg)


def _faces(fonts):
    """face -> (PDF font name, fitz.Font to measure with, font file or None)."""
    out = {}
    for face in FACES:
        path = (fonts or {}).get(face) or (fonts or {}).get("regular")
        if path:
            out[face] = (f"vi-{face}", fitz.Font(fontfile=str(path)), str(path))
        else:
            out[face] = (BASE14[face], fitz.Font(BASE14[face]), None)
    return out

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


def rebuild(input_path, output_path, pages="", dpi=200, fonts=None):
    """Rebuild the selected pages that carry an invisible OCR layer; copy the rest.
    FONTS: optional {face: font file} for the rebuilt text (see rebuild_page).
    Returns the 1-based page numbers that were rebuilt."""
    src = fitz.open(input_path)
    wanted = parse_pages(pages, src.page_count)
    out = fitz.open()
    rebuilt = []
    for i, page in enumerate(src):
        if i in wanted and page_needs_rebuild(page):
            rebuild_page(page, out, dpi, fonts)
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
    for face in FACES:
        ap.add_argument(f"--font-{face}", help=f"{face} font file for the rebuilt text (default Helvetica)")
    a = ap.parse_args()
    fonts = {face: getattr(a, f"font_{face.replace('-', '_')}") for face in FACES}
    rebuilt = rebuild(a.input, a.output, a.pages, a.dpi, {k: v for k, v in fonts.items() if v})
    print(f"Rebuilt {len(rebuilt)} pages with an invisible OCR layer"
          + (f": {rebuilt}" if rebuilt else " - nothing to do, file already has visible text"))


if __name__ == "__main__":
    sys.exit(main())
