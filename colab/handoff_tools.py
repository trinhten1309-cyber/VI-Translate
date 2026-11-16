"""Helpers for VI-Translate handoff + LLM: glossary filtering and output checks.

Run with the repo's venv python and the repo on sys.path, so the checks reuse
the repo's own placeholder/style validators (same rules as the loader).
"""
import csv
import re

from pdf2zh.translator import placeholders, validate_style_tags, FormulaPlaceholderError


def load_glossary(path):
    """CSV columns: en,vi,keep_en  (keep_en = 1 means leave the English term as is)."""
    with open(path, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["en"].strip()]
    # Longest terms first so "non-controlling interest" wins over "interest".
    return sorted(rows, key=lambda r: -len(r["en"]))


def terms_in(text, glossary):
    """Glossary rows whose English term appears in text (whole words, any case)."""
    # Justified lines arrive with doubled spaces ("business  combination").
    found, taken = [], " ".join(text.lower().split())
    for r in glossary:
        pat = r"\b" + re.escape(r["en"].lower()) + r"s?\b"
        if re.search(pat, taken):
            found.append(r)
            taken = re.sub(pat, " ", taken)  # a shorter term inside it is not counted again
    return found


def check(src, dst, glossary):
    """Return a list of problems; empty list means the record is safe to keep."""
    problems = []
    if not dst.strip():
        return ["empty translation"]
    if placeholders(src) != placeholders(dst):
        problems.append("formula placeholders <bN></bN> changed")
    try:
        validate_style_tags(src, dst)
    except FormulaPlaceholderError as e:
        problems.append(str(e))
    for n in re.findall(r"\d[\d,.]*\d|\d", src):
        if n not in dst:
            problems.append(f"number {n} missing or reformatted")
    for r in terms_in(src, glossary):
        want = r["en"] if r.get("keep_en") == "1" else r["vi"]
        if want.lower() not in dst.lower():
            problems.append(f'term "{r["en"]}" should be "{want}"')
    return problems
