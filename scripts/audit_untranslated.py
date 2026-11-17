#!/usr/bin/env python3
"""List the source-language lines a translated PDF still shows, page by page.

usage: audit_untranslated.py TRANSLATED.pdf [--pages 3-8] [--min-words 2] [--json]

A translated page keeps protected text (figures, tables the engine could not
split, page furniture) as source glyphs. This reads every text line of the
translated PDF and reports the ones written wholly in Latin letters without
any Vietnamese diacritic, with at least --min-words words of two or more
letters. Names, units and codes show up too; the report is for a reviewer to
decide which lines are engine misses ("Debt securities issued by the U.S.
Treasury" in a table the engine left whole) and which are meant to stay.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata

import pymupdf

WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]+")
VIETNAMESE = re.compile(
    "[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ]"
)


def parse_pages(spec: str, count: int) -> list[int]:
    if not spec:
        return list(range(1, count + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        start, _, end = part.partition("-")
        pages.update(range(int(start), int(end or start) + 1))
    return sorted(page for page in pages if 1 <= page <= count)


def english_lines(page: pymupdf.Page, min_words: int) -> list[dict]:
    found = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = unicodedata.normalize("NFC", "".join(span["text"] for span in line["spans"])).strip()
            if not text or VIETNAMESE.search(text):
                continue
            words = WORD.findall(text)
            if len(words) < min_words:
                continue
            found.append({"y": round(line["bbox"][1]), "x": round(line["bbox"][0]), "text": text})
    found.sort(key=lambda item: (item["y"], item["x"]))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf")
    parser.add_argument("--pages", default="")
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="one JSON object per line")
    args = parser.parse_args()
    document = pymupdf.open(args.pdf)
    total = 0
    for number in parse_pages(args.pages, document.page_count):
        lines = english_lines(document[number - 1], args.min_words)
        total += len(lines)
        for line in lines:
            if args.json:
                print(json.dumps({"page": number, **line}, ensure_ascii=False))
        if lines and not args.json:
            print(f"page {number}: {len(lines)} line(s)")
            for line in lines:
                print(f"   y={line['y']:>4} x={line['x']:>4}  {line['text'][:100]}")
    if not args.json:
        print(f"== {total} source-language line(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
