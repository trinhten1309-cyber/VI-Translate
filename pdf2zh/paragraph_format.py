"""A paragraph's format, read from where its source glyphs stand.

This module is the engine's conceptual framework for paragraph layout. The
rest of the converter used to carry one rule per document ("(a) followed by a
space", "a centred title of one line"), and every new document found the gap
between two rules. Here a paragraph's format is read from geometry alone, the
way a typesetter would read it off the page, and the translation is set to the
same format. A specific rule elsewhere may still override a principle for a
case it knows better; where none applies, these principles decide.

The principles, in the order they are applied:

1. Faithful form. The translation keeps the source's form: where its first
   line starts, where its wrapped lines start, where the text after a label
   starts, and how its lines meet the margins (left, justified, centred).
2. Geometry over spelling. A label is whatever stands before a tab-like gap
   on the first line, not a pattern of characters: "(a)", "B12D", "1.", "•"
   and "Note 3" are all recognised the same way.
3. Siblings stay alike. A line that opens with a label in the same column as
   the label of the paragraph above starts a sibling paragraph, not a
   continuation.
4. Prudence. When the evidence is weak (one line, a table cell, a gap that
   would open too wide) the paragraph keeps the plain left-aligned form the
   engine always drew. A wrong guess must cost less than no guess.

Coordinates are PDF user space: x grows to the right, y grows upward.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

# A word space is about a quarter of an em; a gap wider than one em is a tab.
TAB_GAP_EM = 1.0
# A label is short: "(iii)", "B12D", "Note 3" fit, a sentence's first word
# followed by a wide justified gap rarely does.
LEAD_MAX_CHARS = 8
# Justification stretches word gaps by at most this many ems; a line that
# would need more is left as it is, as a typesetter would rather than show
# rivers of white.
JUSTIFY_MAX_GAP_EM = 0.6

STYLE_TAG = re.compile(r"</?s\d+>")


def opens_with_lead(first_word: str, first_word_end: float | None,
                    next_word_x0: float | None, size: float) -> bool:
    """Whether a line opens with a short label followed by a tab-like gap."""
    if first_word_end is None or next_word_x0 is None or size <= 0:
        return False
    word = first_word.strip()
    if not word or len(word) > LEAD_MAX_CHARS:
        return False
    return next_word_x0 - first_word_end > TAB_GAP_EM * size


def source_lines(characters: Iterable[Any]) -> list[list[Any]]:
    """A paragraph's glyphs by line, top line first, each line left to right."""
    lines: list[list[Any]] = []
    keyed: list[tuple[float, float]] = []
    for character in sorted(characters, key=lambda glyph: (-round(float(glyph.matrix[5]), 1), float(glyph.x0))):
        baseline = float(character.matrix[5])
        size = float(character.size)
        for index, (line_baseline, line_size) in enumerate(keyed):
            if abs(line_baseline - baseline) <= 0.3 * max(size, line_size):
                lines[index].append(character)
                break
        else:
            keyed.append((baseline, size))
            lines.append([character])
    order = sorted(range(len(lines)), key=lambda index: -keyed[index][0])
    return [sorted(lines[index], key=lambda glyph: float(glyph.x0)) for index in order]


def hanging_lead(characters: Iterable[Any], body_x: float) -> str | None:
    """The label that hangs left of a paragraph's wrapped lines, if any.

    A hanging indent is a first line whose text, after its label, starts
    where the wrapped lines start. The label is what stands before that
    point: "(a)" in "(a)  one or more businesses ..." over "assets of ...".
    """
    lines = source_lines(characters)
    if not lines:
        return None
    first = [glyph for glyph in lines[0] if glyph.get_text().strip()]
    lead = []
    for index, glyph in enumerate(first):
        if float(glyph.x0) >= body_x - 0.5:
            # The label stands apart: at least a word space before the text.
            if not index or float(glyph.x0) - float(first[index - 1].x1) < 0.15 * float(glyph.size):
                return None
            break
        if float(glyph.x1) > body_x + 0.5:
            return None
        lead.append(glyph.get_text())
    else:
        return None
    text = "".join(lead).strip()
    if not text or len(text) > LEAD_MAX_CHARS:
        return None
    return text


def first_line_tab(characters: Iterable[Any]) -> tuple[str, float] | None:
    """The label opening a paragraph's first line and where its text starts.

    "(d)    by providing ..." gives ("(d)", the x of "by"). The label is what
    stands before the first gap wider than TAB_GAP_EM; it may hold an
    ordinary space ("Note 3").
    """
    lines = source_lines(characters)
    if not lines:
        return None
    ink = [glyph for glyph in lines[0] if glyph.get_text().strip()]
    for index in range(1, len(ink)):
        gap = float(ink[index].x0) - float(ink[index - 1].x1)
        size = max(float(ink[index].size), float(ink[index - 1].size))
        if gap > TAB_GAP_EM * size:
            lead = "".join(glyph.get_text() for glyph in ink[:index])
            if len(lead) > LEAD_MAX_CHARS:
                return None
            return lead, float(ink[index].x0)
        if index > LEAD_MAX_CHARS:
            return None
    return None


def stands_at_tab_stop(x0: float, baseline: float, characters: Iterable[Any], size: float) -> bool:
    """Whether something at X0 on a line stood a tab-wide gap after the text before it.

    Apple's cover sets "Large accelerated filer" and its check box in two
    columns; the box belongs at its own x, not glued to the label.
    """
    before = [
        float(glyph.x1)
        for glyph in characters
        if glyph.get_text().strip()
        and float(glyph.x1) <= x0 + 0.1
        and abs(float(glyph.matrix[5]) - baseline) <= 0.3 * max(size, float(glyph.size))
    ]
    return bool(before) and x0 - max(before) > TAB_GAP_EM * size


def lead_end(text: str, lead: str) -> int:
    """Index of the space after LEAD at the start of TEXT, or -1.

    Style tags around the label are skipped: "(a)<s1> Input:" and
    "<s1>(a)</s1> Input:" both end their label at the space.
    """
    wanted = re.sub(r"\s+", "", lead)
    index = matched = 0
    while index < len(text) and matched < len(wanted):
        tag = STYLE_TAG.match(text, index)
        if tag:
            index = tag.end()
            continue
        if text[index].isspace():
            index += 1
            continue
        if text[index] != wanted[matched]:
            return -1
        matched += 1
        index += 1
    if matched < len(wanted):
        return -1
    while index < len(text):
        tag = STYLE_TAG.match(text, index)
        if tag:
            index = tag.end()
            continue
        return index if text[index] in " \t" else -1
    return -1


def collapse_doubled_spaces(text: str) -> str:
    """One space where a translation has two, even across a style tag.

    "tài sản <s2> có thể xác định được</s2>" drew two spaces around the
    italic words. A run of 8 spaces or more is a blank to fill in and stays.
    """
    out: list[str] = []
    previous_space = True  # a translation does not start with a space
    index = 0
    while index < len(text):
        tag = STYLE_TAG.match(text, index)
        if tag:
            out.append(tag.group())
            index = tag.end()
            continue
        if text[index] == " ":
            run = len(text[index:]) - len(text[index:].lstrip(" "))
            if run >= 8:
                out.append(text[index:index + run])
                index += run
                previous_space = False
                continue
            if not previous_space:
                out.append(" ")
            previous_space = True
            index += run
            continue
        out.append(text[index])
        previous_space = text[index] in "\n\t"
        index += 1
    return "".join(out).rstrip(" ")


def source_alignment(characters: Iterable[Any]) -> str:
    """How a paragraph's source lines meet their margins.

    "justify" when every line but the last ends at the same right edge, over
    at least two lines; "left" otherwise. Centred text is read separately
    (`centred_source_axis` in the converter), since it needs its own test of
    the left edges.
    """
    lines = source_lines(characters)
    if len(lines) < 2:
        return "left"
    edges = []
    size = 0.0
    for line in lines:
        ink = [glyph for glyph in line if glyph.get_text().strip()]
        if not ink:
            return "left"
        edges.append((float(ink[0].x0), max(float(glyph.x1) for glyph in ink)))
        size = max(size, max(float(glyph.size) for glyph in ink))
    right = max(edge[1] for edge in edges)
    tolerance = max(0.8, 0.1 * size)
    body = edges[:-1]
    if all(right - edge[1] <= tolerance for edge in body):
        # Every full line must be a full line: a short heading over one long
        # line is not justified text.
        widths = [edge[1] - edge[0] for edge in body]
        if min(widths) >= 0.6 * (right - min(edge[0] for edge in edges)):
            return "justify"
    return "left"


def justify_operations(
    operations: list[dict],
    measure: Callable[[str, int, float], float],
    encode: Callable[[Any, str], str],
    right: float,
    last_lines: set[int],
    is_text: Callable[[dict], bool],
) -> None:
    """Stretch the word gaps of each line not in LAST_LINES to end at RIGHT.

    A text run is split at its spaces so that each word can move. Runs that
    are not plain text (formulas, rules) move with the gaps before them.
    """
    by_line: dict[int, list[dict]] = {}
    for values in operations:
        by_line.setdefault(values["lidx"], []).append(values)
    rebuilt: list[dict] = []
    for index in sorted(by_line):
        line = sorted(by_line[index], key=lambda values: float(values["x"]))
        if index in last_lines:
            rebuilt.extend(line)
            continue
        pieces: list[tuple[dict, bool]] = []  # (operation, ends with a space)
        for values in line:
            text = values.get("text", "")
            if not is_text(values) or "width_em" in values or " " not in text.strip():
                pieces.append((values, is_text(values) and text.endswith(" ")))
                continue
            style = int(values.get("style", 0))
            size = float(values["size"])
            x = float(values["x"])
            for word in re.findall(r"[^ ]+ *| +", text):
                piece = dict(values)
                piece["x"] = x
                piece["text"] = word
                piece["rtxt"] = encode(values["font"], word)
                pieces.append((piece, word.endswith(" ")))
                x += sum(measure(character, style, size) for character in word)
        # Where the line's ink ends now.
        ink_right = None
        for values, _space in pieces:
            if not is_text(values):
                continue
            text = values.get("text", "").rstrip()
            if not text:
                continue
            size = float(values["size"])
            width = (
                values["width_em"] * size if "width_em" in values
                else sum(measure(character, int(values.get("style", 0)), size) for character in text)
            )
            end = float(values["x"]) + width
            ink_right = end if ink_right is None else max(ink_right, end)
        # Gaps are the spaces between words, not one at the end of the line.
        gaps = [position for position, (values, space) in enumerate(pieces[:-1]) if space]
        size = max((float(values["size"]) for values, _ in pieces if is_text(values)), default=0.0)
        if ink_right is None or not gaps or size <= 0:
            rebuilt.extend(line)
            continue
        extra = (right - ink_right) / len(gaps)
        if extra <= 0 or extra > JUSTIFY_MAX_GAP_EM * size:
            rebuilt.extend(line)
            continue
        shift = 0.0
        gap_set = set(gaps)
        for position, (values, _space) in enumerate(pieces):
            if shift:
                values = dict(values)
                values["x"] = float(values["x"]) + shift
            rebuilt.append(values)
            if position in gap_set:
                shift += extra
    operations[:] = rebuilt

