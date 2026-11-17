"""Preservation rules that distinguish the Code4Life PDF translation core."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

FORMULA_FONT_PATTERN = re.compile(
    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|"
    r"stmary|.*Mono|.*Code|.*Sym|.*Math|.*Typewriter|Cousine|Consolas|Menlo|"
    r"Monaco|Inconsolata|Source.?Code|Fira.?Code|DejaVu.?Sans.?Mono|"
    r"Liberation.?Mono|Courier)"
)

MATH_OPERATOR_PATTERN = re.compile(
    r"[=≤≥≈≠±×÷·∑∫√∞∝+*/^]"
)
PROSE_WORD_PATTERN = re.compile(r"[a-z]{3,}")
MATH_FUNCTION_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:sin|cos|tan|cot|sec|csc|log|ln|exp|min|max|lim|det|mod)(?![A-Za-z])",
    re.IGNORECASE,
)
STACKED_TOKEN_PATTERN = re.compile(
    r"[A-Za-z\u0370-\u03ff][A-Za-z\u0370-\u03ff0-9%]*"
)

BULLET_CHARACTERS = frozenset(
    ("•", "■", "□", "▪", "▸", "▹", "►", "▶", "●", "○", "◆", "◇", "★", "☆", "‣", "⬤")
)

PRIVATE_USE_BULLETS = frozenset(("\uf0b7", "\uf0d8", "\uf0fc"))

# A word that numbers a list item: "(a)", "b)", "(iv)", "3.", "(2)".
ITEM_MARKER_PATTERN = re.compile(
    r"\(?(?:\d{1,3}(?:\.\d{1,3})*|[A-Za-z]|[ivxlcdmIVXLCDM]{1,6})[.)]"
)

LANGUAGE_LINE_HEIGHT = {
    "zh-cn": 1.4,
    "zh-tw": 1.4,
    "zh-hans": 1.4,
    "zh-hant": 1.4,
    "zh": 1.4,
    "ja": 1.1,
    "ko": 1.2,
    "en": 1.2,
    "ar": 1.0,
    "ru": 0.8,
    "uk": 0.8,
    "ta": 0.8,
    "vi": 1.2,
}

# Measured ink extents, not preferences: see min_line_height_for_language.
DEFAULT_MIN_LINE_HEIGHT = 0.95
LANGUAGE_MIN_LINE_HEIGHT = {
    "vi": 1.10,
}


@dataclass(frozen=True)
class PreservationDecision:
    """A page classification whose layout must remain untouched."""

    kind: str
    detail: str


@dataclass(frozen=True)
class TableTextCluster:
    """A visual text group inside a table cell, separated from codes/units."""

    bbox: tuple[float, float, float, float]
    text: str
    words: tuple[Sequence[Any], ...]


def is_formula_font(font_name: str) -> bool:
    """Return whether a font name marks formula or code text."""
    return FORMULA_FONT_PATTERN.match(font_name) is not None


def is_bullet_character(text: str, font_name: str | bytes = "") -> bool:
    """Recognize Unicode bullets and common Symbol/Wingdings PUA bullets."""
    if text in BULLET_CHARACTERS:
        return True
    if isinstance(font_name, bytes):
        font_name = font_name.decode(errors="ignore")
    return (
        text in PRIVATE_USE_BULLETS
        and re.search(
            r"wingdings|webdings|symbol|dingbats", font_name, re.IGNORECASE
        )
        is not None
    )


def line_height_for_language(language: str) -> float:
    """Return the translation line-height multiplier for a target language."""
    return LANGUAGE_LINE_HEIGHT.get(language.lower(), 1.1)


def min_line_height_for_language(language: str) -> float:
    """Return the tightest leading that still keeps two lines from touching.

    A paragraph that grew in translation used to buy room by crushing its
    leading to 0.75, which is below the ink of the glyphs being drawn. Measured
    by rendering every letter in the output font and reading the real ink
    extent, in em above and below the baseline:

        English lowercase   0.695 up + 0.210 down = 0.905
        Vietnamese          0.890 up + 0.210 down = 1.100

    Stacked tone marks (e-circumflex-acute, o-horn-grave) reach far higher than
    a plain ascender, so Vietnamese needs more room than English rather than
    less. Below these values the lines overlap no matter what else is right.
    """
    return LANGUAGE_MIN_LINE_HEIGHT.get(language.lower(), DEFAULT_MIN_LINE_HEIGHT)


def _rect(value: Sequence[Any]) -> tuple[float, float, float, float] | None:
    if len(value) < 4:
        return None
    try:
        return tuple(float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None


def _inside_any(
    rectangle: tuple[float, float, float, float],
    regions: Iterable[Sequence[Any]],
) -> bool:
    x0, y0, x1, y1 = rectangle
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for region in regions:
        bounds = _rect(region)
        if bounds is None:
            continue
        rx0, ry0, rx1, ry1 = bounds
        if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
            return True
    return False


def formula_regions(
    blocks: Iterable[Sequence[Any]],
    words: Iterable[Sequence[Any]],
    *,
    stacked_exclusions: Iterable[Sequence[Any]] = (),
) -> list[tuple[float, float, float, float]]:
    """Return ordinary-font regions whose exact mathematical layout must survive.

    The layout model catches dedicated formula fonts well, but technical PDFs
    often typeset equations in the same font as their prose. Operator-heavy
    blocks without prose words are equations. A second geometry rule catches an
    inline stacked fraction such as F1/b0 even when it sits inside a prose block.
    """
    protected: list[tuple[float, float, float, float]] = []
    for block in blocks:
        bounds = _rect(block)
        if bounds is None or len(block) < 5:
            continue
        compact = " ".join(str(block[4]).split())
        prose_candidate = MATH_FUNCTION_PATTERN.sub("", compact)
        if (
            compact
            and MATH_OPERATOR_PATTERN.search(compact)
            and PROSE_WORD_PATTERN.search(prose_candidate) is None
        ):
            protected.append(bounds)

    candidates = list(words)
    exclusions = tuple(stacked_exclusions)
    for index, upper in enumerate(candidates):
        upper_bounds = _rect(upper)
        if upper_bounds is None or len(upper) < 8:
            continue
        upper_text = str(upper[4])
        if (
            len(upper_text) > 4
            or STACKED_TOKEN_PATTERN.fullmatch(upper_text) is None
            or not any(character.isdigit() for character in upper_text)
            or _inside_any(upper_bounds, exclusions)
        ):
            continue
        ux0, uy0, ux1, uy1 = upper_bounds
        for lower in candidates[index + 1 :]:
            lower_bounds = _rect(lower)
            if lower_bounds is None or len(lower) < 8:
                continue
            if upper[5] != lower[5] or upper[6] == lower[6]:
                continue
            lower_text = str(lower[4])
            if (
                len(lower_text) > 4
                or STACKED_TOKEN_PATTERN.fullmatch(lower_text) is None
                or not any(character.isdigit() for character in lower_text)
                or _inside_any(lower_bounds, exclusions)
            ):
                continue
            lx0, ly0, lx1, ly1 = lower_bounds
            overlap = max(0.0, min(ux1, lx1) - max(ux0, lx0))
            smaller_width = min(ux1 - ux0, lx1 - lx0)
            centre_gap = abs((uy0 + uy1) / 2 - (ly0 + ly1) / 2)
            max_height = max(uy1 - uy0, ly1 - ly0)
            if (
                smaller_width > 0
                and overlap / smaller_width >= 0.6
                and 2 < centre_gap <= 1.5 * max_height
            ):
                protected.append(
                    (
                        min(ux0, lx0),
                        min(uy0, ly0),
                        max(ux1, lx1),
                        max(uy1, ly1),
                    )
                )
    return protected


def matching_table_cells(
    model_bounds: Sequence[Any],
    tables: Iterable[Any],
    *,
    minimum_overlap: float = 0.5,
) -> list[tuple[float, float, float, float]]:
    """Return cells from the table whose area covers a model table detection.

    The model is the gate: PyMuPDF cell detection only enables translation when
    it can explain at least half of that already-recognised table. Unmatched
    tables keep the old, fully protected behaviour.
    """
    model = _rect(model_bounds)
    if model is None:
        return []
    mx0, my0, mx1, my1 = model
    model_area = max(0.0, mx1 - mx0) * max(0.0, my1 - my0)
    if model_area <= 0:
        return []

    best: Any = None
    best_overlap = 0.0
    for table in tables:
        bounds = _rect(getattr(table, "bbox", ()))
        if bounds is None:
            continue
        tx0, ty0, tx1, ty1 = bounds
        intersection = max(0.0, min(mx1, tx1) - max(mx0, tx0)) * max(
            0.0, min(my1, ty1) - max(my0, ty0)
        )
        overlap = intersection / model_area
        # A statement rules only its figures, so the grid can start well
        # below the model's top: NVIDIA's unrealized-loss table holds three
        # heading lines ("Apr 26, 2026" / "Less than 12 months" / "Estimated
        # Fair Value") over a grid of four rows, 42% of its height, and the
        # whole table stayed in English. A grid that runs the table's width
        # down to its bottom explains it; the band above is its headings.
        if (
            overlap < minimum_overlap
            and min(mx1, tx1) - max(mx0, tx0) >= 0.9 * (mx1 - mx0)
            and ty1 >= my1 - 0.1 * (my1 - my0)
            and my0 < ty0
            and ty1 - ty0 >= 0.3 * (my1 - my0)
        ):
            overlap = minimum_overlap
        if overlap > best_overlap:
            best, best_overlap = table, overlap
    if best is None or best_overlap < minimum_overlap:
        return []

    cells: list[tuple[float, float, float, float]] = []
    for cell in getattr(best, "cells", ()):
        bounds = _rect(cell)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if mx0 <= cx <= mx1 and my0 <= cy <= my1 and bounds not in cells:
            cells.append(bounds)
    return cells


def opens_item(line: Sequence[Sequence[Any]]) -> bool:
    """Whether a line, as words left to right, starts with a list item's marker.

    The marker must introduce text on its row: a lone "(5)" is a value.
    """
    return len(line) >= 2 and ITEM_MARKER_PATTERN.fullmatch(str(line[0][4])) is not None


def aligned_columns(
    items: Sequence[tuple[float, float, float, float, str]],
    rows: Sequence[Sequence[tuple[float, float, float, float, str]]],
    gutter: float,
) -> list[list[float]]:
    """Columns of words parted by white space that no word crosses."""
    columns: list[list[float]] = []
    for x0, _y0, x1, _y1, _text in sorted(items):
        if columns and x0 < columns[-1][1] + gutter:
            columns[-1][1] = max(columns[-1][1], x1)
        else:
            columns.append([x0, x1])

    def populated(left: float, right: float) -> list[Sequence[tuple[float, float, float, float, str]]]:
        return [
            row for row in rows
            if any(left <= (item[0] + item[2]) / 2 <= right for item in row)
        ]

    # A real column holds text on more than one row. A lone gap inside a
    # centred heading is not a column, so it joins the column beside it.
    merged = True
    while merged and len(columns) > 1:
        merged = False
        for index, (left, right) in enumerate(columns):
            holding = populated(left, right)
            if len(holding) >= 2:
                continue
            # A standard numbers its paragraphs in the margin, and a region
            # may hold just one of them ("B63" beside its paragraph and the
            # list under it). The number is a column when the text beside it
            # starts on the edge the rows below share.
            if index == 0 and len(holding) == 1:
                row = sorted(holding[0])
                inside = [item for item in row if left <= (item[0] + item[2]) / 2 <= right]
                beside = [item for item in row if item[0] > right]
                next_left, next_right = columns[1]
                if (
                    len(inside) == 1
                    and beside
                    and abs(beside[0][0] - next_left) <= 1.5
                    and len(populated(next_left, next_right)) >= 3
                ):
                    continue
            neighbour = index - 1 if index else index + 1
            columns[neighbour] = [
                min(columns[neighbour][0], left), max(columns[neighbour][1], right)
            ]
            del columns[index]
            merged = True
            break
    return columns


def rows_without_spanning_text(
    rows: Sequence[list[tuple[float, float, float, float, str]]],
    gutter: float,
    height: float,
) -> tuple[
    list[list[tuple[float, float, float, float, str]]],
    list[list[tuple[float, float, float, float, str]]],
] | None:
    """Set aside the few paragraphs printed across a table's columns.

    The layout model draws one table around a standard's numbered paragraphs
    together with the heading above them, around a glossary with the sentence
    that introduces it, around a list of signatories with the sentence before
    it. That one line crosses the gutter every other row keeps clear, so no
    column was found and the whole region stayed in the source language
    (IFRS 3 pages 20, 24, 55 and 56). A gutter is accepted here when at most a
    fifth of the rows cross it, and only when each crossing row belongs to a
    paragraph of unbroken lines: a row with a column gap of its own is a table
    row, and a table that crosses its own gutter is not divided.
    Returns the rows that keep the gutter and the spanning paragraphs' rows,
    or None when the region does not divide this way.
    """
    allowed = len(rows) // 5
    if allowed < 1:
        return None
    segments: list[list[list[float]]] = []
    for row in rows:
        parts: list[list[float]] = []
        for item in sorted(row):
            if parts and item[0] < parts[-1][1] + gutter:
                parts[-1][1] = max(parts[-1][1], item[2])
            else:
                parts.append([item[0], item[2]])
        segments.append(parts)
    # Where few rows reach, and none of them is a table row, the rows that do
    # reach may be text set across a gutter.
    edges = sorted({edge for parts in segments for part in parts for edge in part})
    crossing: set[int] = set()
    for start, end in zip(edges, edges[1:]):
        middle = (start + end) / 2
        covering = [
            index for index, parts in enumerate(segments)
            if any(part[0] <= middle <= part[1] for part in parts)
        ]
        if len(covering) <= allowed and all(len(segments[index]) == 1 for index in covering):
            crossing.update(covering)
    if not crossing:
        return None

    # A crossing line takes the rest of its paragraph with it.
    paragraphs: list[list[int]] = []
    for index, row in enumerate(rows):
        if paragraphs:
            previous = rows[paragraphs[-1][-1]]
            if min(word[1] for word in row) - max(word[3] for word in previous) <= height * 0.5:
                paragraphs[-1].append(index)
                continue
        paragraphs.append([index])
    candidates = [
        paragraph for paragraph in paragraphs
        if crossing.intersection(paragraph)
        and all(len(segments[index]) == 1 for index in paragraph)
    ]

    def without(chosen: Sequence[list[int]]):
        taken = {index for paragraph in chosen for index in paragraph}
        kept = [row for index, row in enumerate(rows) if index not in taken]
        columns = aligned_columns([item for row in kept for item in row], kept, gutter)
        return kept, columns

    # The long last line of a ragged column reaches where few rows do as well.
    # Only a paragraph that lies over two of the columns left without it spans.
    _kept, columns = without(candidates)
    spanning = [
        paragraph for paragraph in candidates
        if sum(
            any(segments[index][0][0] < right and segments[index][0][1] > left for index in paragraph)
            for left, right in columns
        ) >= 2
    ]
    if not spanning or sum(len(paragraph) for paragraph in spanning) > allowed:
        return None
    kept, columns = without(spanning)
    if len(columns) < 2:
        return None
    return kept, [[item for index in paragraph for item in rows[index]] for paragraph in spanning]


def table_bounds_clear_of_text(
    table_bounds: Sequence[float],
    table_confidence: float,
    text_regions: Iterable[tuple[Sequence[float], float]],
) -> tuple[float, float, float, float]:
    """Pull a table's top or bottom edge back off a paragraph it cuts through.

    The model's table around IFRS 3 paragraphs 64-64H began inside the last
    line of paragraph 63, which the model had also, and with more confidence,
    boxed as text: the paragraph's first lines were translated and its last
    line stayed in English. A text region the model trusts more than the table
    keeps its lines when it straddles the table's edge, provided that costs the
    table no more than 15% of its height at that end.
    """
    x0, y0, x1, y1 = (float(value) for value in table_bounds)
    height = y1 - y0
    for bounds, confidence in text_regions:
        if confidence <= table_confidence:
            continue
        tx0, ty0, tx1, ty1 = (float(value) for value in bounds)
        if tx1 <= x0 or tx0 >= x1:
            continue
        if ty0 < y0 < ty1 and ty1 - y0 <= height * 0.15:
            y0 = ty1
        elif ty0 < y1 < ty1 and y1 - ty0 <= height * 0.15:
            y1 = ty0
    return x0, y0, x1, y1


def table_top_below_line(
    table_bounds: Sequence[float],
    words: Iterable[Sequence[Any]],
) -> tuple[float, float, float, float]:
    """Lower a table's top edge out of the descenders of the line above it.

    A glyph takes its region from its bottom-left corner, which sits on the
    descender line. NVIDIA page 20 sets "Three Months Ended" at 104-112 pt
    and the model's table starts at 114 pt; protection is painted a point
    wider, and the glyph boxes reach below the word boxes, so the heading
    fell inside the table and stayed in English while the same heading on
    the table below it was translated. A line whose middle is above the
    table's top keeps its descenders; the edge moves down at most a third of
    that line's height.
    """
    x0, y0, x1, y1 = (float(value) for value in table_bounds)
    top = y0
    for word in words:
        bounds = _rect(word)
        if bounds is None:
            continue
        wx0, wy0, wx1, wy1 = bounds
        if wx1 <= x0 or wx0 >= x1:
            continue
        height = wy1 - wy0
        if height <= 0 or (wy0 + wy1) / 2 >= y0:
            continue
        reach = wy1 + height * 0.3 + 1.0
        if reach > top:
            top = min(reach, y0 + height / 3)
    return x0, top, x1, y1


FIGURE_PATTERN = re.compile(r"[$€£¥(]?[\d.,]*\d[\d.,]*%?\)?|[—–-]{1,2}|\$")
FOOTNOTE_MARKER_PATTERN = re.compile(r"(?:\d{1,3}|[*\u2217\u2020\u2021\u00a7\u00b6]{1,3}|\([a-z0-9]{1,2}\))")
# A footnote mark set tight against its first word: "\u2217Equal contribution",
# "5We used values", "1http://image-net.org".
GLUED_FOOTNOTE_MARKER_PATTERN = re.compile(r"(?:[*\u2217\u2020\u2021\u00a7\u00b6]{1,3}[A-Za-z]|\d{1,2}(?:[A-Z][a-z]|https?:)).*")


def table_heading_cells(
    model_bounds: Sequence[Any],
    grid_cells: Sequence[Sequence[Any]],
    words: Iterable[Sequence[Any]],
    rules: Sequence[Sequence[float]] = (),
) -> list[tuple[float, float, float, float]]:
    """Cells for a table's column headings, which its ruling grid leaves out.

    A financial statement draws rules only under its figures: PyMuPDF found
    NVIDIA's income statement grid from the first row of figures down, while
    the model's table also held "Three Months Ended" and the dates under it,
    so those stayed in English with every row label translated beside them.
    The words of the model table above the grid are grouped the way
    a heading is read: a word run stacked on the next line of the same column
    continues it ("Accumulated Other / Comprehensive / Income"), and a run
    that lies over two runs of another line is a spanning heading ("Three
    Months Ended" over two dates, "Common Stock / Outstanding" over "Shares"
    and "Amount"), so it is its own cell. Each group is one cell.
    """
    model = _rect(model_bounds)
    grid = [bounds for bounds in (_rect(cell) for cell in grid_cells) if bounds is not None]
    if model is None or not grid:
        return []
    mx0, my0, mx1, my1 = model
    gy0 = min(cell[1] for cell in grid)
    items = []
    for word in words:
        bounds = _rect(word)
        if bounds is None or len(word) < 5 or not str(word[4]).strip():
            continue
        x0, y0, x1, y1 = bounds
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not (mx0 <= cx <= mx1 and my0 <= cy <= my1):
            continue
        # Headings stand above the figures; what follows the grid is rows
        # the grid did not reach, and those are not headings.
        if cy >= gy0 or y1 > gy0 + 1.0:
            continue
        items.append((x0, y0, x1, y1, str(word[4])))
    if not items:
        return []
    height = median(item[3] - item[1] for item in items)
    gutter = max(3.0, height * 0.6)

    rows: list[list[tuple[float, float, float, float, str]]] = []
    for item in sorted(items, key=lambda item: ((item[1] + item[3]) / 2, item[0])):
        if rows and abs((item[1] + item[3]) / 2 - (rows[-1][0][1] + rows[-1][0][3]) / 2) <= height * 0.3:
            rows[-1].append(item)
        else:
            rows.append([item])
    runs: list[list[float]] = []  # x0, y0, x1, y1, row
    members: list[list[tuple[float, float, float, float, str]]] = []
    for index, row in enumerate(rows):
        for item in sorted(row):
            if runs and runs[-1][4] == index and item[0] < runs[-1][2] + gutter:
                run = runs[-1]
                run[1], run[2], run[3] = min(run[1], item[1]), max(run[2], item[2]), max(run[3], item[3])
                members[-1].append(item)
            else:
                runs.append([item[0], item[1], item[2], item[3], index])
                members.append([item])

    def overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
        return a[0] < b[2] and b[0] < a[2]

    def spans(run: Sequence[float], row: int) -> bool:
        return sum(overlaps(run, other) for other in runs if other[4] == row) >= 2

    parent = list(range(len(runs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, first in enumerate(runs):
        for b, second in enumerate(runs):
            if b <= a or first[4] == second[4] or not overlaps(first, second):
                continue
            upper, lower = (first, second) if first[1] < second[1] else (second, first)
            if lower[1] - upper[3] > height * 0.6:
                continue
            if spans(first, int(second[4])) or spans(second, int(first[4])):
                continue
            # A rule drawn between two lines closes the upper heading: "Apr
            # 26, 2026" is underlined above "Less than 12 months", and the two
            # read as one cell "Apr 26, 2026 Less than 12 months".
            if any(
                upper[3] - 0.5 <= (rule[1] + rule[3]) / 2 <= (lower[1] + lower[3]) / 2
                and rule[0] < min(upper[2], lower[2]) and max(upper[0], lower[0]) < rule[2]
                for rule in rules
            ):
                continue
            parent[find(a)] = find(b)

    groups: dict[int, list[int]] = {}
    for index in range(len(runs)):
        groups.setdefault(find(index), []).append(index)
    boxes = []
    for indices in groups.values():
        boxes.append((
            min(runs[index][0] for index in indices),
            min(runs[index][1] for index in indices),
            max(runs[index][2] for index in indices),
            max(runs[index][3] for index in indices),
        ))
    # A heading over a narrow figure column ("Retained Earnings") has room
    # for its translation only beside and above its words: it takes half the
    # gap to the headings on either side, and the height of the heading band
    # that no other heading over its column uses.
    band_top = min(box[1] for box in boxes)
    cells = []
    for box in boxes:
        x0, y0, x1, y1 = box
        beside = [other for other in boxes
                  if other is not box and other[1] < y1 and y0 < other[3]]
        lefts = [(other[2] + x0) / 2 for other in beside if other[2] <= x0]
        rights = [(x1 + other[0]) / 2 for other in beside if other[0] >= x1]
        left = min(x0 - 1.0, max(max(lefts) if lefts else x0 - 1.0, x0 - 40.0))
        right = max(x1 + 1.0, min(min(rights) if rights else x1 + 1.0, x1 + 40.0))
        above = [other[3] for other in boxes
                 if other is not box and other[3] <= y0 and other[0] < right and left < other[2]]
        top = max(above) + 0.5 if above else band_top
        cells.append((left, min(top, y0), right, y1))
    return sorted(cells, key=lambda cell: (cell[1], cell[0]))


def abandon_outranked_by_text(
    bounds: Sequence[float],
    confidence: float,
    text_regions: Iterable[tuple[Sequence[float], float]],
) -> bool:
    """Whether the model also boxed this region as text, and trusted that more.

    The model often returns two classes for one box. Microsoft's "Refer to
    accompanying notes." was `table_footnote` at 0.67 and `abandon` at 0.29;
    protection was applied after the text classes, so the weaker reading won
    and the line stayed in English on one statement while translated on the
    others.
    """
    x0, y0, x1, y1 = (float(value) for value in bounds)
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area <= 0:
        return False
    for other, other_confidence in text_regions:
        if other_confidence <= confidence:
            continue
        ox0, oy0, ox1, oy1 = (float(value) for value in other)
        intersection = max(0.0, min(x1, ox1) - max(x0, ox0)) * max(0.0, min(y1, oy1) - max(y0, oy0))
        union = area + max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0) - intersection
        if union > 0 and intersection / union >= 0.8:
            return True
    return False


def running_header_lines(
    lines: Iterable[Sequence[Any]],
    page_width: float,
    page_height: float,
    band: float = 0.1,
) -> list[list[tuple[float, float, float, float]]]:
    """The lines of a centred running header, grouped into the paragraphs they set.

    NVIDIA's notes repeat "NVIDIA Corporation and Subsidiaries / Notes to
    Condensed Consolidated Financial Statements (Continued) / (Unaudited)" on
    fifteen pages. The model boxed it `abandon` on some, `plain text` on others
    and not at all on the rest, so it stayed English on six pages and was
    translated on nine, where the three lines ran together into one paragraph
    that wrapped at the narrowest point ("... Thuyết minh báo / cáo tài
    chính ..."). Each line of such a header is its own paragraph, except where
    a centred title wraps: a line breaks deliberately when the next line's
    first word would still have fitted on it, when the next line is the wider
    of the two, when the line is a label of two words or fewer ("PART I"
    over "Item 1"), or when the next line is a parenthetical of its own
    ("(Unaudited)").

    The same block titles each statement ("NVIDIA Corporation and
    Subsidiaries / Condensed Consolidated Balance Sheets / (In millions) /
    (Unaudited)"), reaching below the top tenth of the page.

    `lines` are (x0, y0, x1, y1, text) in top-down page coordinates. The block
    is the run of lines centred on the page that starts with the page's first
    line, in its top band, each line set close under the one before, with a
    gap after it. Two or more such lines are returned as groups of line boxes; anything else
    returns [].
    """
    candidates = []
    for line in lines:
        x0, y0, x1, y1 = (float(value) for value in line[:4])
        text = str(line[4]).strip()
        if re.search(r"[A-Za-z]", text):
            candidates.append((x0, y0, x1, y1, text))
    if len(candidates) < 2:
        return []
    candidates.sort(key=lambda line: (line[1], line[0]))
    middle = page_width / 2
    tolerance = max(3.0, page_width * 0.01)

    def centred(line: tuple[float, float, float, float, str]) -> bool:
        # A line of prose across the measure is centred too; a heading line
        # leaves margins on both sides.
        return (
            abs((line[0] + line[2]) / 2 - middle) <= tolerance
            and line[2] - line[0] <= page_width * 0.7
        )

    if candidates[0][1] > page_height * band or not centred(candidates[0]):
        return []
    top = [candidates[0]]
    for line in candidates[1:]:
        previous = top[-1]
        if line[1] < previous[3] - 1.0:
            return []  # a second line beside the first: not a stacked block
        close = line[1] - previous[3] <= max(2.0, 0.8 * (previous[3] - previous[1]))
        if close and not centred(line):
            # A left-aligned line set close under it ends the same paragraph:
            # a caption across the measure ("Table 1: ...") is not a header.
            return []
        if not close:
            break
        top.append(line)
    if len(top) < 2:
        return []
    widest = max(line[2] - line[0] for line in top)
    groups: list[list[tuple[float, float, float, float]]] = []
    previous = None
    for line in top:
        x0, y0, x1, y1, text = line
        deliberate = previous is None
        if previous is not None:
            first_word = text.split()[0]
            first_width = (x1 - x0) * len(first_word) / max(1, len(text))
            space = (y1 - y0) * 0.3
            fits = previous[2] - previous[0] + space + first_width <= widest
            # A wrapped line holds what the line above had no room for, so
            # it is not the wider of the two.
            wider = x1 - x0 > previous[2] - previous[0] + space
            # Two words or fewer is a label ("PART I" over "Item 1"), not the
            # first line of a wrapped title.
            label = len(previous[4].split()) <= 2
            # A line in another type size is another line of the block
            # ("SECURITIES AND EXCHANGE COMMISSION" at 18 pt over "Washington,
            # D.C. 20549" at 12 pt).
            resized = abs((y1 - y0) - (previous[3] - previous[1])) > 0.2 * (previous[3] - previous[1])
            deliberate = fits or wider or label or resized or (text.startswith("(") and text.endswith(")"))
        if deliberate:
            groups.append([])
        groups[-1].append((x0, y0, x1, y1))
        previous = line
    return groups


def figure_is_text(
    bounds: Sequence[float],
    words: Iterable[Sequence[Any]],
    images: Iterable[Sequence[float]],
    strokes: Iterable[tuple[Sequence[float], bool]],
) -> bool:
    """Whether a region the model called a figure is a page of set text.

    The model boxes the top of an SEC cover page ("UNITED STATES / SECURITIES
    AND EXCHANGE COMMISSION / FORM 10-Q / ...") as one figure, so the whole
    cover stayed in English. A cover holds lines of several words, a logo at
    most and ruled lines; a chart or a diagram holds short labels, curves and
    slanted strokes. Forty words or more, three or more to a line, no more
    than four curves or slanted strokes and images over at most 15% of the
    region make it text. `strokes` are (bounds, curved_or_slanted) drawing
    items; all bounds are top-down.
    """
    x0, y0, x1, y1 = (float(value) for value in bounds)
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return False

    def inside(item: Sequence[float]) -> bool:
        cx = (float(item[0]) + float(item[2])) / 2
        cy = (float(item[1]) + float(item[3])) / 2
        return x0 <= cx <= x1 and y0 <= cy <= y1

    lines: dict[tuple[Any, Any], int] = {}
    count = 0
    for word in words:
        if len(word) < 7 or not inside(word):
            continue
        count += 1
        key = (word[5], word[6])
        lines[key] = lines.get(key, 0) + 1
    if count < 40 or count < 3 * len(lines):
        return False
    image_area = 0.0
    for image in images:
        ix0, iy0, ix1, iy1 = (float(value) for value in image)
        image_area += max(0.0, min(x1, ix1) - max(x0, ix0)) * max(0.0, min(y1, iy1) - max(y0, iy0))
    if image_area > 0.15 * area:
        return False
    drawn = sum(1 for stroke, slanted in strokes if slanted and inside(stroke))
    return drawn <= 4


LIST_LABEL_PATTERN = re.compile(r"^(?:\(?[a-z]{1,2}\)|\(?[ivx]{1,4}\)|\d{1,3}[.)]?|[A-Z]{1,2}\d{1,3}[A-Z]?|[•●▪◦*\-–—])$")
RUN_ITEM_MARKER_PATTERN = re.compile(r"^(?:\(?[a-z]{1,2}\)|\(?[ivx]{1,4}\)|\d{1,3}[.)])$")
CONTINUING_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
def region_has_columns(bounds: Sequence[float], words: Iterable[Sequence[Any]]) -> bool:
    """Whether a text region sets two or more lines in separate columns.

    Microsoft's cover is boxed as text, with "WASHINGTON", "91-1144442" and
    "(I.R.S. ID)" side by side, so the three columns read as one sentence.
    Prose never leaves a gap of three line heights between words in the same
    place on two lines; a region whose lines do is a list of items. A
    justified line's wide spaces fall anywhere.
    """
    x0, y0, x1, y1 = (float(value) for value in bounds)
    inside = []
    for word in words:
        if len(word) < 5 or not str(word[4]).strip():
            continue
        cx, cy = (float(word[0]) + float(word[2])) / 2, (float(word[1]) + float(word[3])) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            inside.append(word)
    inside.sort(key=lambda word: ((float(word[1]) + float(word[3])) / 2, float(word[0])))
    rows: list[list[Sequence[Any]]] = []
    for word in inside:
        middle = (float(word[1]) + float(word[3])) / 2
        height = float(word[3]) - float(word[1])
        if rows and abs(middle - (float(rows[-1][0][1]) + float(rows[-1][0][3])) / 2) <= height * 0.5:
            rows[-1].append(word)
        else:
            rows.append([word])
    gutters: list[list[tuple[float, float]]] = []
    height = 0.0
    for row in rows:
        row.sort(key=lambda word: float(word[0]))
        height = max(height, median(float(word[3]) - float(word[1]) for word in row))
        gaps = [
            (float(before[2]), float(after[0]))
            for position, (before, after) in enumerate(zip(row, row[1:]))
            if float(after[0]) - float(before[2]) > 3 * (float(before[3]) - float(before[1]))
            # The space after a list marker or a paragraph number ("(a)",
            # "B12A", "•") is an indent, not a gutter.
            and not (position == 0 and LIST_LABEL_PATTERN.match(str(before[4])))
        ]
        if gaps:
            gutters.append(gaps)
    # The gutter runs down the page: two lines leave the same stretch empty.
    # A justified line's wide spaces fall anywhere.
    for index, gaps in enumerate(gutters):
        for other in gutters[index + 1:]:
            for a0, a1 in gaps:
                for b0, b1 in other:
                    if min(a1, b1) - max(a0, b0) >= 3 * height:
                        return True
    return False


LEADER_PATTERN = re.compile(r"^[.·…_]{2,}$")
TRAILING_PAGE_REFERENCE = re.compile(r"^(?:[.·…_\-–—]{2,}|\d{1,4}|[ivxlcdm]{1,6}|[A-Z]?\d{1,4})$", re.IGNORECASE)


def line_run_groups(
    words: Iterable[Sequence[Any]],
    region: Sequence[float] | None = None,
) -> list[list[Sequence[Any]]]:
    """The words of a set page split into the runs that read as one item each.

    A contents page or a cover is a list of short items in columns, each line
    with a page number or a second column beside it: read as prose, items on
    one line ran together, and the whole page was protected instead. Each
    line is cut where a gap wider than one and a half times its height opens;
    dot leaders and the page number after them stay out of the run, and so
    does any run without a word of two letters. A run carries on into the run
    below it when that one starts in lower case, when it left a parenthesis
    open or ended on a word such as "of" or "in", or when its line filled the
    measure and the next starts at the same edge; a marker such as "a)"
    always opens a new item. `words` are PyMuPDF words
    (x0, y0, x1, y1, text, block, line, word), top-down.
    """
    chosen = []
    for word in words:
        if len(word) < 7 or not str(word[4]).strip():
            continue
        if region is not None:
            cx, cy = (float(word[0]) + float(word[2])) / 2, (float(word[1]) + float(word[3])) / 2
            if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
                continue
        chosen.append(word)
    lines: dict[tuple[Any, Any], list[Sequence[Any]]] = {}
    for word in chosen:
        lines.setdefault((word[5], word[6]), []).append(word)
    runs: list[tuple[list[Sequence[Any]], bool]] = []  # words, ended by a page reference
    for line_words in lines.values():
        line_words.sort(key=lambda word: float(word[0]))
        height = median(float(word[3]) - float(word[1]) for word in line_words)
        current: list[Sequence[Any]] = []
        pieces = []
        for word in line_words:
            if current and float(word[0]) - float(current[-1][2]) > 1.5 * height:
                pieces.append(current)
                current = []
            current.append(word)
        if current:
            pieces.append(current)
        line_runs: list[tuple[list[Sequence[Any]], bool]] = []
        for piece in pieces:
            ended = False
            # Leaders and the page number after them stay as they are; a
            # number that is part of the item ("IFRS 3") does not follow a
            # leader.
            while len(piece) > 1 and LEADER_PATTERN.match(str(piece[-1][4])):
                piece = piece[:-1]
                ended = True
            if len(piece) > 2 and LEADER_PATTERN.match(str(piece[-2][4])) and TRAILING_PAGE_REFERENCE.match(str(piece[-1][4])):
                piece = piece[:-2]
                ended = True
            if not any(re.search(r"[^\W\d_]{2,}", str(word[4])) for word in piece):
                # A page number standing in its own column ends the item
                # beside it on this line.
                if line_runs and all(TRAILING_PAGE_REFERENCE.match(str(word[4])) for word in piece):
                    line_runs[-1] = (line_runs[-1][0], True)
                continue
            line_runs.append((piece, ended))
        runs.extend(line_runs)
    runs.sort(key=lambda run: (float(run[0][0][1]), float(run[0][0][0])))

    def box(run_words: list[Sequence[Any]]) -> tuple[float, float, float, float]:
        return (
            min(float(word[0]) for word in run_words),
            min(float(word[1]) for word in run_words),
            max(float(word[2]) for word in run_words),
            max(float(word[3]) for word in run_words),
        )

    widest = max((box(run_words)[2] - box(run_words)[0] for run_words, _ended in runs), default=0.0)
    groups: list[list[Sequence[Any]]] = []
    open_groups: list[tuple[list[Sequence[Any]], bool]] = []
    for run_words, ended in runs:
        rx0, ry0, rx1, ry1 = box(run_words)
        height = ry1 - ry0
        first = str(run_words[0][4])
        joined = False
        for index in range(len(open_groups) - 1, -1, -1):
            group, group_ended = open_groups[index]
            gx0, gy0, gx1, gy1 = box(group)
            if group_ended or ry0 - gy1 > 0.6 * height or ry0 < gy1 - 0.5 * height:
                continue
            if rx0 >= gx1 or rx1 <= gx0:
                continue
            if RUN_ITEM_MARKER_PATTERN.match(first):
                break  # "a) Condensed ..." opens an item of its own
            text = " ".join(str(word[4]) for word in group)
            last_line = [word for word in group if float(word[1]) >= gy1 - height * 0.5]
            last_width = (
                max(float(word[2]) for word in last_line) - min(float(word[0]) for word in last_line)
                if last_line else 0.0
            )
            continues = (
                first[:1].islower()
                or text.count("(") > text.count(")")
                or (len(group) > 1 and str(group[-1][4]).lower().strip(",") in CONTINUING_WORDS)
                or (last_width >= 0.8 * widest and abs(rx0 - gx0) <= 2.0)
            )
            if continues:
                group.extend(run_words)
                open_groups[index] = (group, ended)
                joined = True
                break
        if not joined:
            group = list(run_words)
            groups.append(group)
            open_groups.append((group, ended))
    return groups


def page_line_key(text: str, top: float) -> tuple[str, int] | None:
    """A line's letters and its height on the page, to find repeated furniture."""
    letters = re.sub(r"[^a-z]", "", text.lower())
    if len(letters) < 3:
        return None
    return letters, int(round(top / 6.0))


def repeated_page_lines(pages: Iterable[Iterable[tuple[str, float]]]) -> set[tuple[str, int]]:
    """Lines that recur at the same height on two or more pages.

    `pages` gives, for each page, its lines as (text, top). Running headers
    and footers repeat where they stand; a section title or a signature
    block does not.
    """
    seen: dict[tuple[str, int], int] = {}
    for lines in pages:
        keys = {key for text, top in lines if (key := page_line_key(text, top)) is not None}
        for key in keys:
            seen[key] = seen.get(key, 0) + 1
    return {key for key, count in seen.items() if count >= 2}


def abandon_is_unrepeated_text(
    bounds: Sequence[float],
    lines: Iterable[Sequence[Any]],
    repeated: set[tuple[str, int]],
    max_lines: int | None = 4,
) -> bool:
    """Whether a region the model set aside as page furniture is a passage.

    The model boxed NVIDIA's "Part I. Financial Information" (page 3) and the
    signature blocks of its certifications ("Executive Vice President and
    Chief Financial Officer") `abandon`, so they stayed in English. Furniture
    repeats from page to page at the same height; text that appears once, with
    two words or more on at most four lines, is the page's own and is
    translated. `lines` are
    (x0, y0, x1, y1, text) in top-down coordinates.
    """
    x0, y0, x1, y1 = (float(value) for value in bounds)
    words = 0
    inside = 0
    for line in lines:
        lx0, ly0, lx1, ly1 = (float(value) for value in line[:4])
        cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        text = str(line[4])
        key = page_line_key(text, ly0)
        if key is None:
            continue
        if key in repeated:
            return False
        inside += 1
        words += len(re.findall(r"[A-Za-z]{2,}", text))
    # A long block set aside is a list the page carries over, such as the end
    # of a table of contents (IFRS 3 page 3); the caller translates it item by
    # item (max_lines=None) rather than as one passage.
    return inside > 0 and (max_lines is None or inside <= max_lines) and words >= 2


def abandon_is_table_caption(
    bounds: Sequence[float],
    tables: Iterable[Sequence[float]],
    gap: float = 16.0,
) -> bool:
    """Whether a region the model set aside as page furniture titles a table.

    The model labelled "NVIDIA Corporation and Subsidiaries / Condensed
    Consolidated Balance Sheets / (In millions) (Unaudited)" `abandon`, like a
    running header, on three of the five statements of a 10-Q; the income
    statement's identical layout was a `table_caption` and was translated. A
    caption sits directly above its table and within its width; a running
    header has the page's first heading or paragraph between it and any table.
    """
    x0, y0, x1, y1 = (float(value) for value in bounds)
    for table in tables:
        tx0, ty0, tx1, ty1 = (float(value) for value in table)
        if 0.0 <= ty0 - y1 <= gap and x0 >= tx0 - 2.0 and x1 <= tx1 + 2.0:
            return True
    return False


def overlapping_text_groups(
    regions: Sequence[tuple[Sequence[float], float]],
    minimum: float = 0.6,
) -> list[list[int]]:
    """Text regions the model drew twice over the same lines, grouped.

    The model often answers one paragraph with two "plain text" boxes a few
    points apart (NVIDIA 10-Q note 6: x 84-578 at 0.35 and x 100-594 at 0.51).
    Each became its own class, so the glyphs between their left edges formed a
    paragraph of their own: "Non-marketable e" / "quity securities had ...",
    "$1.0 billi" / "on of". Regions whose overlap covers most of the smaller
    one are one paragraph region. Returns groups of indices, largest
    confidence first.
    """
    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    boxes = [tuple(float(value) for value in bounds) for bounds, _confidence in regions]
    for a, first in enumerate(boxes):
        for b in range(a + 1, len(boxes)):
            second = boxes[b]
            width = min(first[2], second[2]) - max(first[0], second[0])
            height = min(first[3], second[3]) - max(first[1], second[1])
            if width <= 0 or height <= 0:
                continue
            smaller = min(
                (first[2] - first[0]) * (first[3] - first[1]),
                (second[2] - second[0]) * (second[3] - second[1]),
            )
            if smaller > 0 and width * height >= minimum * smaller:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for index in range(len(regions)):
        groups.setdefault(find(index), []).append(index)
    return [
        sorted(group, key=lambda index: -regions[index][1])
        for group in groups.values()
        if len(group) > 1
    ]


def lines_crossing_region(
    region: Sequence[float],
    lines: Iterable[Sequence[float]],
) -> list[tuple[float, float, float, float]]:
    """Lines a text region covers only in part, which it should own whole.

    On the cover of NVIDIA's 10-Q the model's text box for a check-box
    question began 66 pt into its first line, at "ark whether ...", so the
    glyphs before it ("Indicate by check m") fell to the catch-all class and
    became a paragraph of their own: the sentence was cut mid-word into four
    segments. A line whose middle lies in the region's height and that shares
    at least a third of its width with the region belongs to it.
    """
    rx0, ry0, rx1, ry1 = (float(value) for value in region)
    result = []
    for line in lines:
        x0, y0, x1, y1 = (float(value) for value in line[:4])
        width = x1 - x0
        if width <= 0 or not ry0 <= (y0 + y1) / 2 <= ry1:
            continue
        shared = min(x1, rx1) - max(x0, rx0)
        if shared >= width / 3 and (x0 < rx0 - 0.5 or x1 > rx1 + 0.5):
            result.append((x0, y0, x1, y1))
    return result


def is_footnote_text(
    bounds: Sequence[float],
    words: Iterable[Sequence[Any]],
    page_height: float,
) -> bool:
    """Whether a region the model set aside as page furniture is a footnote.

    The model's `abandon` class covers running headers, footers and page
    numbers, which stay as printed, and it also took the footnote of IFRS 3
    paragraph B56 ("1 In paragraphs B56-B62 the term ..."). A footnote opens
    with its reference mark, runs to a sentence of six words or more, sits in
    the lower half of the page, and is not the last text on it: the running
    footer lies below.
    """
    x0, y0, x1, y1 = (float(value) for value in bounds)
    if y0 < page_height / 2:
        return False
    inside = []
    below = False
    for word in words:
        wx0, wy0, wx1, wy1 = (float(value) for value in word[:4])
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            inside.append((round(cy), wx0, str(word[4])))
        elif cy > y1:
            below = True
    inside.sort()
    return (
        below
        and len(inside) >= 6
        and (FOOTNOTE_MARKER_PATTERN.fullmatch(inside[0][2]) is not None
             or GLUED_FOOTNOTE_MARKER_PATTERN.fullmatch(inside[0][2]) is not None)
        and any(character.islower() for _cy, _x, text in inside[1:] for character in text)
    )


def text_aligned_table_cells(
    model_bounds: Sequence[Any],
    words: Iterable[Sequence[Any]],
    rules: Iterable[Sequence[Any]] = (),
) -> list[tuple[float, float, float, float]]:
    """Cells of a table set without a grid, found from how its text is aligned.

    PyMuPDF finds cells from ruling lines, so a table with only a rule under
    its header - every table in the Combatting infectious diarrhea review -
    yields none and stayed in English. Its columns are still plain to see: a
    band of white space runs the table's full height between them. Inside a
    column a cell ends where the space between lines grows, a rule crosses, a
    bullet opens an item, or the left edge moves (a record's name set out, its
    details set in). Each cell reaches down to the next cell of its column, so
    a longer translation has the white space below its source text and never
    more. A table that does not divide this way returns no cells and stays
    protected, as before.
    """
    model = _rect(model_bounds)
    if model is None:
        return []
    mx0, my0, mx1, my1 = model
    items = []
    for word in words:
        bounds = _rect(word)
        if bounds is None or len(word) < 5 or not str(word[4]).strip():
            continue
        x0, y0, x1, y1 = bounds
        if mx0 <= (x0 + x1) / 2 <= mx1 and my0 <= (y0 + y1) / 2 <= my1:
            items.append((x0, y0, x1, y1, str(word[4])))
    if len(items) < 4:
        return []
    height = median(item[3] - item[1] for item in items)

    rows: list[list[tuple[float, float, float, float, str]]] = []
    for item in sorted(items, key=lambda item: ((item[1] + item[3]) / 2, item[0])):
        if rows and abs((item[1] + item[3]) / 2 - (rows[-1][0][1] + rows[-1][0][3]) / 2) <= height * 0.5:
            rows[-1].append(item)
        else:
            rows.append([item])
    # A gutter must be wider than the space between words, which justified
    # cells stretch, so it is measured against the spaces this table uses.
    spaces = [
        current[0] - previous[2]
        for row in rows
        for previous, current in zip(sorted(row), sorted(row)[1:])
        if current[0] > previous[2]
    ]
    # Only spaces narrower than a line is high are word spaces. A statement
    # with four figure columns has more gaps between figures than between
    # words: Microsoft's cash flow statement put the median at 19.5 pt, the
    # gutter at 39 pt, and every column merged into one.
    word_spaces = [space for space in spaces if space <= height]
    gutter = max(3.0, height * 0.4, 2.0 * median(word_spaces) if word_spaces else 0.0)

    columns = aligned_columns(items, rows, gutter)
    spanning: list[list[tuple[float, float, float, float, str]]] = []
    if len(columns) < 2:
        # A heading or a sentence set across the columns hides their gutter.
        divided = rows_without_spanning_text(rows, gutter, height)
        if divided is None:
            return []
        rows, spanning = divided
        items = [item for row in rows for item in row]
        columns = aligned_columns(items, rows, gutter)
    if not 2 <= len(columns) <= 12:
        return []
    # A row run across a gutter merges the columns it crosses. When other
    # rows still show that gutter inside the merged column, the column holds
    # a hidden one and its rows would be read as sentences across it.
    hidden_gap = max(gutter, height * 1.5)
    for left, right in columns:
        split_rows = 0
        for row in rows:
            inside = sorted(item for item in row if left <= (item[0] + item[2]) / 2 <= right)
            if len(inside) < 2 or opens_item([list(item) for item in inside]):
                continue
            # Figures under a heading that spans their columns ("Three
            # Months Ended" over two years) are never read as sentences.
            if any(
                after[0] - before[2] > hidden_gap
                and not FIGURE_PATTERN.fullmatch(before[4])
                and not FIGURE_PATTERN.fullmatch(after[4])
                for before, after in zip(inside, inside[1:])
            ):
                split_rows += 1
        if split_rows >= 2:
            return []

    horizontal_rules = []
    for rule in rules:
        bounds = _rect(rule)
        if bounds is not None and bounds[3] - bounds[1] <= 2.0:
            horizontal_rules.append(bounds)

    # A spanning paragraph closes the cells above it the way a rule does.
    spanning_bounds = [
        (
            min(word[0] for word in paragraph),
            min(word[1] for word in paragraph),
            max(word[2] for word in paragraph),
            max(word[3] for word in paragraph),
        )
        for paragraph in spanning
    ]
    tops = sorted({item[1] for item in items} | {bounds[1] for bounds in spanning_bounds})

    # Amounts count only in columns made of amounts: a glossary definition
    # that cites "IFRS 13" beside a term is not a statement row.
    figure_columns = []
    for left, right in columns:
        inside = [item for item in items if left <= (item[0] + item[2]) / 2 <= right]
        if inside and sum(bool(FIGURE_PATTERN.fullmatch(item[4])) for item in inside) >= 0.6 * len(inside):
            figure_columns.append((left, right))
    figures = [
        ((item[1] + item[3]) / 2, item[0])
        for item in items
        if FIGURE_PATTERN.fullmatch(item[4])
        and any(left <= (item[0] + item[2]) / 2 <= right for left, right in figure_columns)
    ]

    cells: list[tuple[float, float, float, float]] = []
    for x0, y0, _x1, y1 in spanning_bounds:
        below = [top for top in tops if top > y1]
        cells.append((x0 - 2.0, y0, mx1, below[0] - 1.0 if below else y1))
    for index, (left, right) in enumerate(columns):
        members = sorted(
            (item for item in items if left <= (item[0] + item[2]) / 2 <= right),
            key=lambda item: (item[1], item[0]),
        )
        lines: list[list[tuple[float, float, float, float, str]]] = []
        for item in members:
            if lines:
                top = min(word[1] for word in lines[-1])
                bottom = max(word[3] for word in lines[-1])
                if top <= (item[1] + item[3]) / 2 <= bottom:
                    lines[-1].append(item)
                    continue
            lines.append([item])

        statement = (left, right) not in figure_columns and any(x0 >= right for _middle, x0 in figures)
        widest = max((max(word[2] for word in line) - left for line in lines), default=1.0)
        blocks: list[list[list[tuple[float, float, float, float, str]]]] = []
        for line in lines:
            line.sort()
            if blocks:
                previous = blocks[-1][-1]
                # A statement's row ends where its figures stand: a label
                # with an amount to its right is a complete row, even when
                # the next label sits tight under it at the same indent.
                previous_middle = sum((word[1] + word[3]) / 2 for word in previous) / len(previous)
                closed = any(
                    abs(middle - previous_middle) <= height * 0.5 and x0 >= right
                    for middle, x0 in figures
                )
                previous_bottom = max(word[3] for word in previous)
                top = min(word[1] for word in line)
                crossed = any(
                    previous_bottom - 0.5 <= (rule[1] + rule[3]) / 2 <= top + 0.5
                    and rule[0] < right and rule[2] > left
                    for rule in horizontal_rules
                )
                # A list item's wrapped lines hang under its text, not its
                # marker: "(a) the accounting ... in the" over "financial
                # statements ..." is one item, so its edge is the second word's.
                edge = previous[0][0]
                if opens_item(previous) and line[0][0] - previous[0][0] > height:
                    edge = previous[1][0]
                if len(previous) <= 3 and len(line) <= 3 and not any(
                    FIGURE_PATTERN.fullmatch(word[4]) for word in previous + line
                ):
                    # A column heading in a table of amounts is set flush
                    # right or centred ("Cash / and Cash / Equivalents",
                    # "Fair / Value / Level"): its short lines share a right
                    # edge or a centre, not a left edge.
                    previous_right = max(w[2] for w in previous)
                    # Two headings can share a merged column; compare with
                    # the words under this one only.
                    under = [w for w in line if w[0] < previous_right + 2.0] or line
                    line_right = max(w[2] for w in under)
                    previous_centre = (previous[0][0] + previous_right) / 2
                    line_centre = (under[0][0] + line_right) / 2
                    tolerance = max(1.5, height * 0.2)
                    if (abs(previous_right - line_right) <= tolerance
                            or abs(previous_centre - line_centre) <= tolerance):
                        edge = line[0][0]
                if statement and not closed:
                    # Beside figures, a short line without amounts is a
                    # caption over its rows ("Operations", "Financing"), and
                    # a full-width one wraps into the indented line below it
                    # ("Acquisition of companies, net of cash acquired and
                    # divestitures," / "and purchases of intangible ...").
                    reach = (max(word[2] for word in previous) - left) / max(1.0, widest)
                    ends_heading = str(previous[-1][4]).endswith(":")
                    flush_left = previous[0][0] - left <= height
                    if flush_left and (reach < 0.6 or ends_heading):
                        blocks.append([line])
                        continue
                    if 0 < line[0][0] - previous[0][0] <= 2 * height:
                        edge = line[0][0]
                if not (
                    top - previous_bottom > height * 0.5
                    or crossed
                    or closed
                    or abs(line[0][0] - edge) > max(1.5, height * 0.2)
                    or line[0][4][:1] in BULLET_CHARACTERS
                    or opens_item(line)
                ):
                    blocks[-1].append(line)
                    continue
            blocks.append([line])

        right_edge = columns[index + 1][0] - 2.0 if index + 1 < len(columns) else mx1
        for position, block in enumerate(blocks):
            block_words = [word for line in block for word in line]
            x0 = min(word[0] for word in block_words) - 2.0
            y0 = min(word[1] for word in block_words)
            y1 = max(word[3] for word in block_words)
            limit = my1
            if position + 1 < len(blocks):
                limit = min(word[1] for word in blocks[position + 1][0]) - 1.0
            for rule in horizontal_rules:
                middle = (rule[1] + rule[3]) / 2
                if y1 <= middle < limit and rule[0] < right_edge and rule[2] > x0:
                    limit = middle - 0.5
            for bounds in spanning_bounds:
                if y1 <= bounds[1] < limit:
                    limit = bounds[1] - 1.0
            cells.append((x0, y0, max(right_edge, x0 + 1.0), max(y1, limit)))
    return cells if len(cells) >= 2 else []


def words_by_cell(
    cells: Sequence[Sequence[Any]],
    words: Iterable[Sequence[Any]],
) -> list[list[Sequence[Any]]]:
    """Give each word to one cell: the one that covers most of it.

    A cell grid can overlap itself. The inline-code backgrounds of a printed
    web page give PyMuPDF edges for small cells lying inside the real ones,
    and a word claimed by two cells went to whichever was painted last:
    "True", the first word of a description, became a cell of its own, too
    narrow to hold its translation, and the rest of the sentence was sent
    without it. On a tie the larger cell wins, since it is the one bounded by
    the table's own rules.
    """
    owned: list[list[Sequence[Any]]] = [[] for _ in cells]
    bounds = [_rect(cell) for cell in cells]
    for word in words:
        box = _rect(word)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best: tuple[tuple[float, float], int] | None = None
        for index, cell in enumerate(bounds):
            if cell is None:
                continue
            cx0, cy0, cx1, cy1 = cell
            if not (cx0 <= cx <= cx1 and cy0 <= cy <= cy1):
                continue
            covered = max(0.0, min(x1, cx1) - max(x0, cx0)) * max(
                0.0, min(y1, cy1) - max(y0, cy0)
            )
            key = (covered, (cx1 - cx0) * (cy1 - cy0))
            if best is None or key > best[0]:
                best = (key, index)
        if best is not None:
            owned[best[1]].append(word)
    return owned


MONTH_NAMES = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|"
    "april|june|july|august|september|october|november|december"
)
DATE_CELL_PATTERN = re.compile(
    rf"(?:(?:{MONTH_NAMES})\.?\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}\s+(?:{MONTH_NAMES})\.?,?\s+\d{{4}}"
    rf"|(?:{MONTH_NAMES})\.?,?\s+\d{{4}})",
    re.IGNORECASE,
)
SLASHED_WORDS_PATTERN = re.compile(r"[A-Za-z][a-z]{2,}(?:/[A-Za-z][a-z]{2,})+")


def should_translate_table_cell(text: str) -> bool:
    """Return whether a cell contains natural-language text rather than codes.

    Product identifiers and numeric cells are safer left as original PDF glyphs.
    Natural-language labels in the supported source documents contain lowercase
    letters, including Unicode lowercase letters outside English.

    A capitalised word of five letters or more with vowels is a heading word
    ("ASSETS:"), not a code. A date is a heading, not a code: "Apr" is a short
    capitalised token like a
    unit, so NVIDIA's 10-Q kept 62 column dates ("Apr 26, 2026") in English
    while every date in its prose was translated. Words joined by a slash
    ("Schedule/Form") are words too.
    """
    value = " ".join(text.split())
    if not value:
        return False
    if DATE_CELL_PATTERN.search(value) or SLASHED_WORDS_PATTERN.search(value):
        return True

    def natural_token(token: str) -> bool:
        token = token.strip("()[]{}:;,\"'“”")
        letters = "".join(character for character in token if character.isalpha())
        if token.lower() in {"dry", "wet"}:
            return True
        if token.lower() in {"max", "min"}:
            return False
        if re.search(r"[\u0370-\u03ff]", token):
            return False
        if len(letters) <= 2:
            return False
        if letters.isupper():
            # A capitalised heading word ("ASSETS:", "LIABILITIES AND
            # SHAREHOLDERS' EQUITY:") reads as a word; a code is short, holds
            # digits or has few vowels ("IFRS", "USD", "A1B2C").
            return (
                len(letters) >= 5
                and sum(character in "AEIOUY" for character in letters) >= 2
                and not any(character.isdigit() for character in token)
            )
        if (
            any(character.isdigit() for character in token)
            or re.search(r"[a-z][A-Z]", token)
            or sum(character.isupper() for character in letters) >= 2
            or re.search(r"[%._/·]", token)
        ):
            return False
        if len(letters) <= 3 and letters[:1].isupper():
            return False
        return any(character.islower() for character in letters)

    return any(natural_token(token) for token in value.split())


def justified_units(
    words: Sequence[Sequence[Any]],
    cell: tuple[float, float, float, float],
    gap_limit: float,
    height: float,
) -> list[list[Sequence[Any]]]:
    """The cell's words, left to right, with each justified line held together.

    A justified line set across the full measure stretches every space by the
    same amount. IFRS 3 sets "a combination of entities or businesses under
    common control" alone on its line, so each space grew to 6.5 pt, past the
    gap that separates a description from its code column; the line was cut
    into nine one-word clusters, each translated on its own. Spaces that are
    all alike, between words, on a line that reaches the cell's right edge
    are a justified line; a column gap is one space far wider than the rest.
    A list marker ahead of the line keeps its own gap.
    """
    cx0, _cy0, cx1, _cy1 = cell
    rows: list[list[Sequence[Any]]] = []
    for word in sorted(words, key=lambda item: (float(item[1]) + float(item[3])) / 2):
        middle = (float(word[1]) + float(word[3])) / 2
        if rows and abs(middle - (float(rows[-1][0][1]) + float(rows[-1][0][3])) / 2) <= height * 0.5:
            rows[-1].append(word)
        else:
            rows.append([word])
    units: list[list[Sequence[Any]]] = []
    for row in rows:
        row.sort(key=lambda item: float(item[0]))
        start = 1 if opens_item(row) else 0
        inner = row[start:]
        gaps = [float(b[0]) - float(a[2]) for a, b in zip(inner, inner[1:])]
        justified = (
            len(gaps) >= 3
            and min(gaps) > 0
            and max(gaps) > gap_limit
            and max(gaps) <= min(gaps) * 1.35 + 0.5
            and sum(any(c.isalpha() for c in str(w[4])) for w in inner) >= 0.75 * len(inner)
            and float(inner[-1][2]) >= cx1 - max(6.0, 0.05 * (cx1 - cx0))
        )
        if justified:
            units.extend([word] for word in row[:start])
            units.append(list(inner))
        else:
            units.extend([word] for word in row)
    units.sort(key=lambda unit: (float(unit[0][0]), float(unit[0][1])))
    return units


def cluster_table_words(
    words: Iterable[Sequence[Any]],
    cell: Sequence[Any],
) -> list[TableTextCluster]:
    """Split a visually merged table cell into prose and code-like x clusters.

    Some PDFs omit the rule between a description and its abbreviation column,
    so PyMuPDF returns both as one cell. Normal word spaces are small; the jump
    to a right-aligned code is much larger. X-overlap across wrapped lines keeps
    multi-line descriptions together.
    """
    bounds = _rect(cell)
    items = [word for word in words if _rect(word) is not None and len(word) >= 5]
    if bounds is None or not items:
        return []

    heights = [max(0.1, float(word[3]) - float(word[1])) for word in items]
    gap_limit = max(3.0, median(heights) * 0.6)
    groups: list[list[Sequence[Any]]] = []
    group_bounds: list[list[float]] = []
    for unit in justified_units(items, bounds, gap_limit, median(heights)):
        x0, y0, x1, y1 = (float(value) for value in unit[0][:4])
        for word in unit[1:]:
            x0, y0 = min(x0, float(word[0])), min(y0, float(word[1]))
            x1, y1 = max(x1, float(word[2])), max(y1, float(word[3]))
        matches = [
            index
            for index, current in enumerate(group_bounds)
            if x0 <= current[2] + gap_limit and x1 >= current[0] - gap_limit
        ]
        if not matches:
            groups.append(list(unit))
            group_bounds.append([x0, y0, x1, y1])
            continue
        target = matches[0]
        groups[target].extend(unit)
        current = group_bounds[target]
        current[:] = [
            min(current[0], x0),
            min(current[1], y0),
            max(current[2], x1),
            max(current[3], y1),
        ]
        for extra in reversed(matches[1:]):
            groups[target].extend(groups.pop(extra))
            other = group_bounds.pop(extra)
            current[:] = [
                min(current[0], other[0]),
                min(current[1], other[1]),
                max(current[2], other[2]),
                max(current[3], other[3]),
            ]

    ordered = sorted(zip(groups, group_bounds), key=lambda item: item[1][0])
    cx0, cy0, cx1, cy1 = bounds
    result: list[TableTextCluster] = []
    for index, (group, group_box) in enumerate(ordered):
        left = cx0 if index == 0 else (ordered[index - 1][1][2] + group_box[0]) / 2
        right = cx1 if index + 1 == len(ordered) else (
            group_box[2] + ordered[index + 1][1][0]
        ) / 2
        text = " ".join(
            str(word[4])
            for word in sorted(
                group,
                key=lambda item: (
                    int(item[5]) if len(item) > 5 else 0,
                    int(item[6]) if len(item) > 6 else 0,
                    int(item[7]) if len(item) > 7 else 0,
                ),
            )
        )
        result.append(
            TableTextCluster(
                (max(cx0, left), cy0, min(cx1, right), cy1),
                text,
                tuple(group),
            )
        )
    return result


def page_has_image(blocks: Iterable[Mapping[str, Any]]) -> bool:
    """Return whether the page draws any raster image at all.

    is_scanned_page asks whether one image covers half the page, which is the
    right question for backing rectangles but the wrong one for deciding a page
    held nothing to translate: scanners routinely emit a page as dozens of
    tiles, none of them large on its own. A page with no text and no image is
    simply blank, and saying so would be noise.
    """
    return any(block.get("type") == 1 for block in blocks)


def is_scanned_page(blocks: Iterable[Mapping[str, Any]], page_area: float) -> bool:
    """Return whether a rendered image covers more than half of the page."""
    if page_area <= 0:
        return False
    for block in blocks:
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox")
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (float(value) for value in bbox)
        if max(0.0, x1 - x0) * max(0.0, y1 - y0) > page_area * 0.5:
            return True
    return False


_GROUPED_AMOUNT = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def has_index_page_reference(line: str) -> bool:
    """True for back-of-book entries such as ``Leasing, 108, 127``.

    Thousands separators are not page references: a financial statement whose
    rows read ``3,463,760`` or ``(1,558,690)`` must stay translatable.
    """
    return bool(re.search(r",\s*\d{1,4}", _GROUPED_AMOUNT.sub("#", line)))


def classify_preserved_page(page_text: str) -> PreservationDecision | None:
    """Classify pages whose number-heavy structure must not be reflowed."""
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return None

    toc_score = 0
    standalone_nums = 0
    spaced_page_nums = 0
    emspace_page_nums = 0
    for line in lines:
        if (
            re.search(r"\.{5,}", line)
            or re.search(r"(\.\s){4,}", line)
            or re.search(r"[\x08\ufffd\u2500-\u257f]{3,}", line)
        ):
            toc_score += 3
        elif re.search(r"[\x08\ufffd\u2500-\u257f]+\s*\d{1,4}\s*$", line):
            toc_score += 2
        elif re.search(r"\S\s{5,}\d{1,4}\s*$", line):
            spaced_page_nums += 1
        elif re.fullmatch(r"\d{1,4}", line):
            standalone_nums += 1

        if re.search(r"[\u2002\u2003]+\s*\d{1,4}\s*$", line) or re.search(
            r"[\u2002\u2003]+\s*[ivxlcdm]+\s*$", line, re.IGNORECASE
        ):
            emspace_page_nums += 1

    has_contents_header = any(
        re.fullmatch(r"(table\s+of\s+)?contents?", line, re.IGNORECASE)
        for line in lines[:5]
    )
    if has_contents_header:
        toc_score += 5
    if spaced_page_nums >= 5:
        toc_score += spaced_page_nums
    if emspace_page_nums >= 5:
        toc_score += emspace_page_nums
    if standalone_nums >= 8 and toc_score > 0:
        toc_score += standalone_nums
    if len(lines) >= 15 and standalone_nums >= 10 and standalone_nums / len(lines) > 0.3:
        toc_score += standalone_nums
    if len(lines) >= 15:
        lines_ending_num = sum(
            1 for line in lines if re.search(r"\S\s+\d{1,4}\s*$", line)
        )
        if lines_ending_num / len(lines) > 0.8:
            toc_score += lines_ending_num
    if toc_score >= 8:
        return PreservationDecision("TOC", f"score={toc_score}")

    index_comma_numbers = sum(1 for line in lines if has_index_page_reference(line))
    if len(lines) >= 20 and index_comma_numbers / len(lines) > 0.4:
        return PreservationDecision(
            "INDEX", f"comma_num={index_comma_numbers}/{len(lines)}"
        )
    if re.fullmatch(r"index", lines[0], re.IGNORECASE):
        return PreservationDecision("INDEX", "header")

    has_nomenclature_header = any(
        re.fullmatch(
            r"(nomenclature|list\s+of\s+symbols|symbols?\s+and\s+abbreviations?|"
            r"glossary|notation)s?",
            line,
            re.IGNORECASE,
        )
        for line in lines[:5]
    )
    if has_nomenclature_header and len(lines) >= 10:
        symbol_definition_pairs = sum(
            1
            for index in range(len(lines) - 1)
            if len(lines[index]) <= 15
            and len(lines[index + 1]) > 5
            and not lines[index].isdigit()
        )
        if symbol_definition_pairs / len(lines) > 0.3:
            return PreservationDecision(
                "NOMENCLATURE",
                f"pairs={symbol_definition_pairs}/{len(lines)}",
            )

    has_reference_header = any(
        re.fullmatch(
            r"[\xad]?(references?|bibliography|suggested\s+reading|further\s+reading|"
            r"works?\s+cited)",
            line,
            re.IGNORECASE,
        )
        for line in lines[:10]
    )
    numbered_refs = sum(1 for line in lines if re.match(r"^\d{1,3}\.\s", line))
    author_year_refs = sum(
        1 for line in lines if re.match(r"^[A-Z][a-z]+,?\s.*\(\d{4}\)", line)
    )
    bracketed_refs = sum(1 for line in lines if re.match(r"^\[\d{1,3}\]", line))
    year_parentheses = sum(1 for line in lines if re.search(r"\(\d{4}\)", line))
    isbn_doi = sum(
        1
        for line in lines
        if re.search(r"ISBN|ISSN|doi\.org|https?://", line, re.IGNORECASE)
    )
    all_refs = numbered_refs + author_year_refs + bracketed_refs
    reference_signals = all_refs + year_parentheses + isbn_doi
    if has_reference_header and reference_signals >= 5:
        return PreservationDecision(
            "REFERENCES",
            f"header, refs={all_refs}, years={year_parentheses}, isbn_doi={isbn_doi}",
        )
    if len(lines) >= 10 and all_refs >= 5 and year_parentheses + isbn_doi >= 3:
        return PreservationDecision(
            "REFERENCES",
            f"refs={all_refs}, years={year_parentheses}, isbn_doi={isbn_doi}",
        )
    return None
