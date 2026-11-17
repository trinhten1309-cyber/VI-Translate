from __future__ import annotations

import unittest
from types import SimpleNamespace

import pymupdf
from pdfminer.pdfinterp import PDFResourceManager

from pdf2zh.converter import TranslateConverter, has_words, is_translatable_segment
from pdf2zh.rules import (
    abandon_is_table_caption,
    abandon_outranked_by_text,
    lines_crossing_region,
    overlapping_text_groups,
    table_heading_cells,
    BULLET_CHARACTERS,
    classify_preserved_page,
    cluster_table_words,
    formula_regions,
    has_index_page_reference,
    is_bullet_character,
    is_footnote_text,
    is_formula_font,
    is_scanned_page,
    line_height_for_language,
    page_has_image,
    matching_table_cells,
    min_line_height_for_language,
    should_translate_table_cell,
    table_bounds_clear_of_text,
    text_aligned_table_cells,
)
from tests.test_damaged_sources import translated


def _words(line: int, y: float, *runs: tuple[float, str], space: float = 2.0) -> list[tuple]:
    """Words of one 8 pt table line: 4.5 pt a character, `space` between words."""
    words = []
    for x, text in runs:
        for word in text.split():
            words.append((x, y, x + len(word) * 4.5, y + 8.0, word, 0, line, len(words)))
            x += len(word) * 4.5 + space
    return words


class TextAlignedTableTests(unittest.TestCase):
    """Tables drawn with no grid: PyMuPDF finds no cells, so the whole table stayed English."""

    def _cells_text(self, cells, words):
        return [
            " ".join(word[4] for word in words
                     if cell[0] <= (word[0] + word[2]) / 2 <= cell[2]
                     and cell[1] <= (word[1] + word[3]) / 2 <= cell[3])
            for cell in cells
        ]

    def test_columns_from_white_space_and_items_from_bullets_rows_and_rules(self):
        """Combatting Table 2: a header rule, bullet items, rows set a little apart."""
        words = (
            _words(0, 663, (53, "Infectious Diarrhoea"), (289, "Treatment approaches"), (525, "Refs"))
            + _words(1, 675, (53, "Acute diarrhea"), (155, "●Rehydration therapy"), (525, "[15]"))
            + _words(2, 684, (155, "●Antibiotic regime of ciprofloxacin"))
            + _words(3, 707, (53, "Chronic diarrhea"), (155, "●Rehydration therapy"), (525, "[51]"))
        )
        rules = [(49, 672, 156, 672.5), (156, 672, 525, 672.5), (525, 672, 562, 672.5)]
        cells = text_aligned_table_cells((49, 658, 564, 763), words, rules)
        self.assertEqual(
            self._cells_text(cells, words),
            ["Infectious Diarrhoea", "Acute diarrhea", "Chronic diarrhea",
             "Treatment approaches", "●Rehydration therapy",
             "●Antibiotic regime of ciprofloxacin", "●Rehydration therapy",
             "Refs", "[15]", "[51]"],
        )
        header, acute = cells[0], cells[1]
        self.assertLess(header[3], 672)  # the header cell stops at its rule
        self.assertGreater(acute[3], 692)  # a label may grow into the space beside its bullets
        self.assertLess(acute[3], 707)  # but never into the next row

    def test_a_record_name_set_out_is_its_own_cell_above_its_details(self):
        """Combatting Table 4: rows touch, and only the left edge marks a new record."""
        words = (
            _words(0, 77, (52, "RotaTeq"), (171, "Merck Sharp"))
            + _words(1, 86, (60, "Live attenuated oral"), (179, "Prequalified and licensed"))
            + _words(2, 95, (60, "vaccine"), (179, "by the WHO."))
            + _words(3, 104, (52, "Rotarix"), (171, "GlaxoSmithKline"))
            + _words(4, 113, (60, "Live attenuated oral"), (179, "Received prequalification"))
        )
        cells = text_aligned_table_cells((48, 63, 400, 130), words)
        self.assertEqual(
            self._cells_text(cells, words),
            ["RotaTeq", "Live attenuated oral vaccine", "Rotarix", "Live attenuated oral",
             "Merck Sharp", "Prequalified and licensed by the WHO.", "GlaxoSmithKline",
             "Received prequalification"],
        )

    def test_justified_spaces_and_a_centred_heading_are_not_gutters(self):
        words = (
            _words(0, 10, (200, "Treatment approaches"), space=9.0)
            + _words(1, 26, (10, "Label"), (155, "stretched words set wide"), space=5.0)
            + _words(2, 42, (10, "Other"), (155, "stretched words again"), space=5.0)
        )
        self.assertEqual(
            self._cells_text(text_aligned_table_cells((0, 0, 400, 50), words), words),
            ["Label", "Other", "Treatment approaches", "stretched words set wide",
             "stretched words again"],
        )

    def test_list_items_start_cells_and_their_wrapped_lines_hang_under_the_text(self):
        """IFRS 3 paragraph 5: numbered paragraphs the layout model calls a table.

        Items set tight were one cell, and an item's second line, set under its
        text rather than its marker, began a cell of its own mid-sentence."""
        words = (
            _words(0, 10, (10, "5"), (50, "Applying the acquisition method requires the following:"))
            + _words(1, 20, (50, "(a)"), (80, "identifying the acquirer;"))
            + _words(2, 30, (50, "(b)"), (80, "recognising the assets acquired, the"))
            + _words(3, 40, (80, "liabilities assumed; and"))
            + _words(4, 50, (10, "6"), (50, "The acquirer shall be identified."))
        )
        cells = text_aligned_table_cells((0, 0, 400, 70), words)
        self.assertEqual(
            self._cells_text(cells, words),
            ["5", "6", "Applying the acquisition method requires the following:",
             "(a) identifying the acquirer;",
             "(b) recognising the assets acquired, the liabilities assumed; and",
             "The acquirer shall be identified."],
        )

    def test_text_without_a_column_gutter_stays_protected(self):
        words = (_words(0, 10, (10, "a paragraph of ordinary prose that wraps"))
                 + _words(1, 20, (10, "onto a second line of the same width")))
        self.assertEqual(text_aligned_table_cells((0, 0, 300, 40), words), [])

    def test_a_line_spanning_the_gutter_merges_its_columns(self):
        words = (
            _words(0, 10, (10, "Name"), (200, "Value"))
            + _words(1, 20, (10, "A heading that runs straight across the gutter"))
            + _words(2, 30, (10, "Alpha"), (200, "First"))
        )
        # One column left means nothing divides cleanly, so the table stays whole.
        self.assertEqual(text_aligned_table_cells((0, 0, 400, 50), words), [])


    def test_a_heading_across_numbered_paragraphs_becomes_its_own_cell(self):
        """IFRS 3 page 20: one model table around "Effective date and
        transition" and paragraphs 64-64H; the heading hid the gutter and the
        whole page stayed in English."""
        words = (
            _words(0, 10, (10, "Effective date and transition"))
            + _words(1, 30, (10, "64"), (50, "This IFRS shall be applied prospectively to"))
            + _words(2, 40, (50, "business combinations."))
            + _words(3, 55, (10, "64A"), (50, "[Deleted]"))
            + _words(4, 70, (10, "64B"), (50, "Improvements to IFRSs amended paragraphs"))
            + _words(5, 80, (50, "19, 30 and B56."))
            + _words(6, 95, (10, "64C"), (50, "Paragraphs 65A to 65E were added."))
        )
        cells = text_aligned_table_cells((0, 0, 400, 110), words)
        self.assertEqual(
            self._cells_text(cells, words),
            ["Effective date and transition", "64", "64A", "64B", "64C",
             "This IFRS shall be applied prospectively to business combinations.",
             "[Deleted]", "Improvements to IFRSs amended paragraphs 19, 30 and B56.",
             "Paragraphs 65A to 65E were added."],
        )
        heading = cells[0]
        self.assertLess(heading[3], 30)  # the heading stops above paragraph 64
        self.assertGreaterEqual(heading[2], 400)  # and may use the full width

    def test_a_sentence_above_a_list_of_names_spans_both_columns(self):
        """IFRS 3 page 55: the approval sentence, then signatories and their roles."""
        words = (
            _words(0, 10, (10, "Definition of a Business was approved for issue by all of the 14 members"))
            + _words(1, 19, (10, "of the Board."))
            + _words(2, 35, (10, "Hans Hoogervorst"), (200, "Chairman"))
            + _words(3, 50, (10, "Suzanne Lloyd"), (200, "Vice-Chair"))
        )
        for line, name in enumerate(("Nick Anderson", "Martin Edelmann", "Gary Kabureck",
                                     "Jianqiao Lu", "Darrel Scott", "Thomas Scott",
                                     "Ann Tarca", "Mary Tokar"), start=4):
            words += _words(line, 5 + line * 15, (10, name))
        cells = text_aligned_table_cells((0, 0, 400, 200), words)
        texts = self._cells_text(cells, words)
        self.assertEqual(
            texts[0],
            "Definition of a Business was approved for issue by all of the 14 members of the Board.",
        )
        self.assertIn("Hans Hoogervorst", texts)
        self.assertEqual(texts[-2:], ["Chairman", "Vice-Chair"])

    def test_table_rows_that_cross_their_own_gutter_stay_protected(self):
        """A crossing row with a column gap of its own is a table row, not prose."""
        words = _words(0, 10, (10, "Description of the item sold in the year"), (300, "1,200"))
        for line in range(1, 7):
            words += _words(line, 10 + line * 10, (10, "Item"), (150, "Code"), (300, "100"))
        # The first row's text runs over the "Code" column and keeps its amount.
        self.assertEqual(text_aligned_table_cells((0, 0, 400, 90), words), [])

    def test_a_ragged_last_column_is_not_text_across_a_gutter(self):
        words = []
        for line in range(6):
            words += _words(line, 10 + line * 10, (10, "ordinary prose that wraps here"))
        words += _words(6, 70, (10, "ordinary prose that wraps and runs on further"))
        self.assertEqual(text_aligned_table_cells((0, 0, 400, 90), words), [])

    def test_a_lone_paragraph_number_keeps_its_margin_column(self):
        """IFRS 3 page 46: "B63" beside its paragraph and the list under it."""
        words = (
            _words(0, 10, (10, "B63"), (50, "Examples of other IFRSs include:"))
            + _words(1, 25, (50, "(a)"), (80, "IAS 38 prescribes the accounting;"))
            + _words(2, 40, (50, "(b)"), (80, "[deleted]"))
            + _words(3, 55, (50, "(c)"), (80, "IAS 12 prescribes the accounting."))
        )
        cells = text_aligned_table_cells((0, 0, 400, 70), words)
        self.assertEqual(
            self._cells_text(cells, words),
            ["B63", "Examples of other IFRSs include:", "(a) IAS 38 prescribes the accounting;",
             "(b) [deleted]", "(c) IAS 12 prescribes the accounting."],
        )

    def test_a_table_edge_inside_a_trusted_paragraph_moves_off_it(self):
        """IFRS 3 page 20: the table began inside the last line of paragraph 63."""
        text = [((112, 127, 459, 162), 0.70), ((127, 220, 460, 303), 0.71)]
        self.assertEqual(
            table_bounds_clear_of_text((108, 146, 462, 706), 0.56, text),
            (108.0, 162.0, 462.0, 706.0),
        )
        # A paragraph the model trusts less than the table does not move it,
        self.assertEqual(
            table_bounds_clear_of_text((108, 146, 462, 706), 0.80, text),
            (108.0, 146.0, 462.0, 706.0),
        )
        # nor does one that would take a large part of the table.
        self.assertEqual(
            table_bounds_clear_of_text((108, 146, 462, 300), 0.56, [((112, 100, 459, 250), 0.9)]),
            (108.0, 146.0, 462.0, 300.0),
        )

    def test_a_footnote_is_told_from_a_running_footer(self):
        """IFRS 3 page 44: the model set the footnote of B56 aside with the footer."""
        footer = _words(9, 731, (231, "© IFRS Foundation"))
        note = _words(8, 697, (112, "1"), (120, "In paragraphs B56 to B62 the term refers to awards."))
        words = note + footer
        self.assertTrue(is_footnote_text((111, 695, 459, 714), words, 842))
        self.assertFalse(is_footnote_text((231, 730, 311, 741), words, 842))  # the footer itself
        self.assertFalse(is_footnote_text((111, 695, 459, 714), note, 842))  # last text on the page
        header = _words(0, 99, (112, "1"), (120, "A running header of six words or more"))
        self.assertFalse(is_footnote_text((111, 97, 459, 110), header + footer, 842))
        glued = _words(8, 697, (112, "\u2217Equal contribution. Listing order is random and fair."))
        self.assertTrue(is_footnote_text((111, 695, 459, 714), glued + footer, 842))
        plain = _words(8, 697, (112, "Chapter 3 Business combinations and group accounts continued"))
        self.assertFalse(is_footnote_text((111, 695, 459, 714), plain + footer, 842))


class FinancialStatementTests(unittest.TestCase):
    """NVIDIA Form 10-Q for the quarter ended April 26, 2026, pages 1-7."""

    @staticmethod
    def _text(cells, words):
        return [
            " ".join(word[4] for word in words
                     if cell[0] <= (word[0] + word[2]) / 2 <= cell[2]
                     and cell[1] <= (word[1] + word[3]) / 2 <= cell[3])
            for cell in cells
        ]

    def test_a_spanning_heading_and_its_dates_above_the_grid(self):
        words = (_words(0, 173, (439, "Three Months Ended"))
                 + _words(1, 184, (396, "Apr 26, 2026"), (514, "Apr 27, 2025"))
                 + _words(2, 202, (18, "Revenue"), (400, "81,615")))
        grid = [(18, 200, 364, 212), (364, 200, 593, 212)]
        cells = table_heading_cells((16, 173, 595, 466), grid, words)
        self.assertEqual(self._text(cells, words),
                         ["Three Months Ended", "Apr 26, 2026", "Apr 27, 2025"])

    def test_stacked_column_headings_stay_together(self):
        words = (_words(0, 102, (277, "Common Stock"))
                 + _words(1, 105, (425, "Accumulated Other"))
                 + _words(2, 109, (282, "Outstanding"))
                 + _words(3, 112, (357, "Additional"), (425, "Comprehensive"))
                 + _words(4, 119, (268, "Shares"), (310, "Amount"), (350, "Paid-in Capital"),
                          (440, "Income"))
                 + _words(5, 145, (19, "Balances as of Jan 25, 2026")))
        grid = [(19, 141, 261, 151), (261, 141, 592, 151)]
        cells = table_heading_cells((18, 100, 594, 335), grid, words)
        texts = self._text(cells, words)
        self.assertIn("Common Stock Outstanding", texts)
        self.assertIn("Additional Paid-in Capital", texts)
        self.assertIn("Accumulated Other Comprehensive Income", texts)
        self.assertIn("Shares", texts)
        self.assertIn("Amount", texts)
        self.assertNotIn("Balances as of Jan 25, 2026", " ".join(texts))

    def test_a_table_without_a_grid_has_no_heading_cells(self):
        words = _words(0, 10, (10, "Three Months Ended"))
        self.assertEqual(table_heading_cells((0, 0, 400, 50), [], words), [])

    def test_cash_flow_rows_are_cut_at_their_figures(self):
        """Microsoft 10-Q, quarter ended December 31, 2025, page 6: four figure
        columns put the median space at 19.5 pt and merged every column; then
        tight rows at one indent ran together as one cell."""
        words = (
            _words(0, 132, (20, "Operations"))
            + _words(1, 144, (20, "Net income"), (370, "$"), (400, "38,458"), (470, "$"), (500, "24,108"))
            + _words(2, 155, (20, "Adjustments to reconcile net income to net cash:"))
            + _words(3, 166, (27, "Depreciation, amortization, and other"), (400, "9,198"), (500, "5,667"))
            + _words(4, 177, (27, "Stock-based compensation expense"), (400, "3,219"), (500, "3,089"))
            + _words(5, 189, (20, "Acquisition of companies, net of cash acquired and divestitures,"))
            + _words(6, 200, (27, "and purchases of intangible assets"), (400, "(455)"), (500, "(1,405)"))
            + _words(7, 211, (20, "Purchases of investments"), (400, "(9,845)"), (500, "(2,050)"))
        )
        cells = text_aligned_table_cells((18, 128, 577, 220), words)
        # A lone "$" joins the label column; cluster_table_words parts it off.
        texts = [text.replace(" $", "") for text in TextAlignedTableTests._cells_text(None, cells, words)
                 if any(character.islower() for character in text)]
        self.assertEqual(texts, [
            "Operations", "Net income", "Adjustments to reconcile net income to net cash:",
            "Depreciation, amortization, and other", "Stock-based compensation expense",
            "Acquisition of companies, net of cash acquired and divestitures, "
            "and purchases of intangible assets",
            "Purchases of investments",
        ])

    def test_a_right_aligned_column_heading_stays_one_cell(self):
        words = (
            _words(0, 145, (148, "Fair"), (440, "Cash"))
            + _words(1, 153, (145, "Value"), (424, "and Cash"))
            + _words(2, 162, (145, "Level"), (409, "Equivalents"))
        )
        for line, (level, amount) in enumerate((("Level 1", "1,234"), ("Level 2", "567"))):
            words += _words(3 + line, 180 + line * 12, (20, "Commercial paper"), (145, level), (430, amount))
        cells = text_aligned_table_cells((18, 140, 577, 210), words)
        texts = TextAlignedTableTests._cells_text(None, cells, words)
        self.assertIn("Cash and Cash Equivalents", texts)
        self.assertIn("Fair Value Level", texts)

    def test_a_grid_under_a_tall_heading_band_still_explains_the_table(self):
        """Page 12: the grid covers four rows, 42% of the model table."""
        grid = SimpleNamespace(bbox=(18, 403, 593, 448), cells=[(18, 403, 306, 414), (306, 403, 373, 414)])
        self.assertEqual(len(matching_table_cells((17, 344, 596, 450), [grid])), 2)
        # A grid under only part of the width is not the table.
        narrow = SimpleNamespace(bbox=(300, 403, 593, 448), cells=[(300, 403, 400, 414)])
        self.assertEqual(matching_table_cells((17, 344, 596, 450), [narrow]), [])

    def test_a_rule_between_two_headings_parts_them(self):
        words = (_words(0, 344, (353, "Apr 26, 2026"))
                 + _words(1, 356, (337, "Less than 12 months"))
                 + _words(2, 404, (18, "Total"), (330, "16,187")))
        grid = [(18, 403, 306, 414), (306, 403, 373, 414)]
        rule = (306, 357.1, 446, 358.4)
        joined = self._text(table_heading_cells((17, 344, 596, 450), grid, words), words)
        self.assertIn("Apr 26, 2026 Less than 12 months", joined)
        parted = self._text(table_heading_cells((17, 344, 596, 450), grid, words, [rule]), words)
        self.assertIn("Apr 26, 2026", parted)
        self.assertIn("Less than 12 months", parted)

    def test_a_centred_running_header_keeps_its_lines(self):
        from pdf2zh.rules import running_header_lines
        header = [
            (235, 20, 378, 29, "NVIDIA Corporation and Subsidiaries"),
            (173, 30, 439, 39, "Notes to Condensed Consolidated Financial Statements (Continued)"),
            (284, 40, 329, 49, "(Unaudited)"),
            (18, 98, 241, 110, "Note 4 - Amortizable Intangible Assets and Goodwill"),
        ]
        self.assertEqual(len(running_header_lines(header, 612, 792)), 3)
        # A centred title that wraps stays one paragraph,
        title = [
            (117, 71, 481, 85, "BERT: Pre-training of Deep Bidirectional Transformers for"),
            (221, 87, 377, 101, "Language Understanding"),
            (122, 131, 188, 143, "Jacob Devlin"),
        ]
        self.assertEqual(len(running_header_lines(title, 595, 842)), 1)
        # and a caption whose last line is left-aligned is no header.
        caption = [
            (108, 71, 504, 81, "Table 1: Maximum path lengths, per-layer complexity and minimum number"),
            (108, 82, 504, 92, "for different layer types. n is the sequence length, d is the"),
            (108, 93, 435, 103, "size of convolutions and r the size of the neighborhood."),
        ]
        self.assertEqual(running_header_lines(caption, 612, 792), [])
        # A two-word label over another is two paragraphs.
        label = [(286, 50, 310, 58, "PART I"), (288, 58, 309, 67, "Item 1"), (20, 80, 300, 90, "Body")]
        self.assertEqual(len(running_header_lines(label, 596, 842)), 2)
        # Prose across the measure is not centred text.
        prose = [(18, 20, 595, 30, "We expect our Rubin platform"), (18, 31, 595, 41, "to start shipping")]
        self.assertEqual(running_header_lines(prose, 612, 792), [])

    def test_dates_and_slashed_words_are_table_text(self):
        from pdf2zh.rules import should_translate_table_cell
        for text in ("Apr 26, 2026", "December 31, 2025", "June 2025", "Schedule/Form",
                     "ASSETS:", "LIABILITIES AND SHAREHOLDERS’ EQUITY:"):
            self.assertTrue(should_translate_table_cell(text), text)
        for text in ("Apr", "MAY", "A1/B2", "kg/m3", "10.1+", "IFRS", "USD", "XYZWQ"):
            self.assertFalse(should_translate_table_cell(text), text)

    def test_a_table_top_leaves_the_descenders_above_it(self):
        """Page 20: "Three Months Ended" at 104-112 pt over a table at 114 pt."""
        from pdf2zh.rules import table_top_below_line
        heading = _words(0, 104, (450, "Three Months Ended"))
        x0, top, x1, bottom = table_top_below_line((16, 114, 596, 234), heading)
        self.assertGreater(top, 114)
        self.assertLessEqual(top, 114 + 8 / 3)
        # A line inside the table, or beside it, does not move the edge.
        self.assertEqual(table_top_below_line((16, 114, 596, 234), _words(0, 120, (450, "Total")))[1], 114)
        self.assertEqual(table_top_below_line((300, 114, 596, 234), _words(0, 104, (18, "Note")))[1], 114)

    def test_unrepeated_page_furniture_is_text(self):
        from pdf2zh.rules import abandon_is_unrepeated_text, repeated_page_lines
        header = ("NVIDIA Corporation and Subsidiaries", 20.0)
        pages = [[header, ("Part I. Financial Information", 71.0)], [header], [header, ("12", 590.0)]]
        repeated = repeated_page_lines(pages)
        signature = [(17, 585, 120, 594, "/s/ COLETTE M. KRESS"),
                     (17, 596, 90, 605, "Colette M. Kress"),
                     (17, 607, 227, 616, "Executive Vice President and Chief Financial Officer")]
        self.assertTrue(abandon_is_unrepeated_text((17, 585, 227, 619), signature, repeated))
        running = [(235, 20, 378, 29, "NVIDIA Corporation and Subsidiaries")]
        self.assertFalse(abandon_is_unrepeated_text((173, 20, 439, 52), running, repeated))
        # A long block set aside is a list carried over (a contents page), not a passage.
        contents = [(137, 127 + 14 * n, 300, 136 + 14 * n, f"Contents entry number {n}") for n in range(8)]
        self.assertFalse(abandon_is_unrepeated_text((132, 125, 485, 255), contents, repeated))
        # A page number is not a passage.
        self.assertFalse(abandon_is_unrepeated_text((300, 588, 312, 598), [(301, 590, 311, 598, "12")], repeated))

    def test_a_first_word_that_does_not_fit_starts_at_the_left_edge(self):
        from pdf2zh.converter import first_word_start

        def measure(character, style, size):
            return character, size * 0.5

        # "$ / Change": the line starts at the "$", 6 pt from the right edge.
        self.assertEqual(first_word_start("<s1>Thay đổi $</s1>", 520, 495, 526, 8, measure, []), 495)
        self.assertEqual(first_word_start("Thay đổi", 500, 495, 526, 8, measure, []), 500)

    def test_a_cover_set_as_a_figure_is_text(self):
        from pdf2zh.rules import figure_is_text
        cover = []
        for line in range(12):
            cover += _words(line, 40 + line * 12, (60, "QUARTERLY REPORT PURSUANT TO SECTION 13 OF THE ACT"))
        logo = [(277, 153, 335, 196)]
        self.assertTrue(figure_is_text((33, 38, 565, 393), cover, logo, [((10, 200, 590, 200), False)]))
        # A plot: short labels and slanted strokes.
        labels = []
        for line in range(20):
            labels += _words(line, 40 + line * 12, (60, "56-layer"), (200, "20"))
        strokes = [((60 + n, 50, 70 + n, 60), True) for n in range(30)]
        self.assertFalse(figure_is_text((33, 38, 565, 393), labels, [], strokes))
        # A figure that is mostly a picture.
        self.assertFalse(figure_is_text((33, 38, 565, 393), cover, [(33, 38, 565, 300)], []))

    def test_contents_items_are_cut_from_their_page_numbers(self):
        from pdf2zh.rules import line_run_groups
        words = (
            _words(0, 100, (23, "Item 1."))
            + _words(1, 100, (82, "Financial Statements (Unaudited)"))
            + _words(2, 112, (82, "a) Condensed Consolidated Statements of Income"))
            + _words(3, 112, (584, "3"))
            + _words(4, 124, (82, "b) Condensed Consolidated Balance Sheets"))
            + _words(5, 136, (112, "Recognising and measuring the identifiable assets acquired, the liabilities"))
            + _words(6, 146, (112, "assumed and any non-controlling interest"))
            + _words(7, 146, (453, "10"))
        )
        texts = [" ".join(word[4] for word in group) for group in line_run_groups(words)]
        self.assertEqual(texts, [
            "Item 1.",
            "Financial Statements (Unaudited)",
            "a) Condensed Consolidated Statements of Income",
            "b) Condensed Consolidated Balance Sheets",
            "Recognising and measuring the identifiable assets acquired, the liabilities "
            "assumed and any non-controlling interest",
        ])
        # Two columns on one line of a cover stay apart; an open bracket
        # carries its item onto the next line.
        cover = (
            _words(0, 242, (120, "(State or other jurisdiction of"))
            + _words(1, 242, (415, "(I.R.S. Employer"))
            + _words(2, 252, (118, "incorporation or organization)"))
            + _words(3, 252, (412, "Identification No.)"))
        )
        self.assertEqual(
            sorted(" ".join(word[4] for word in group) for group in line_run_groups(cover)),
            ["(I.R.S. Employer Identification No.)",
             "(State or other jurisdiction of incorporation or organization)"],
        )

    def test_a_line_two_text_boxes_share_goes_to_the_larger_part(self):
        import numpy as np
        from pdf2zh.high_level import majority_text_class
        self.assertEqual(majority_text_class(np.array([5, 5, 7, 7, 7, 7])), 7)
        self.assertIsNone(majority_text_class(np.array([7, 7, 7])))
        self.assertIsNone(majority_text_class(np.array([5, 0, 7, 7])))

    def test_an_items_later_piece_rejoins_its_paragraph(self):
        from pdf2zh.converter import Paragraph, resumable_paragraph
        first, beside = Paragraph(0, 0, 0, 0, 0, 0, 8, False, cls=20), Paragraph(0, 0, 0, 0, 0, 0, 8, False, cls=21)
        self.assertEqual(resumable_paragraph([first, beside], 20, {20, 21}), 0)
        self.assertIsNone(resumable_paragraph([first, beside], 20, {21}))
        self.assertIsNone(resumable_paragraph([first, beside], 22, {20, 21, 22}))

    def test_a_line_in_another_type_size_starts_a_paragraph(self):
        from pdf2zh.converter import line_changes_size
        self.assertTrue(line_changes_size("F", 8.0, 10.0))
        self.assertFalse(line_changes_size("t", 9.6, 10.0))
        # A raised reference or a digit at the start of a line does not count.
        self.assertFalse(line_changes_size("1", 6.0, 10.0))

    def test_a_header_line_in_another_size_is_its_own_paragraph(self):
        from pdf2zh.rules import running_header_lines
        cover = [
            (227, 73, 370, 94, "UNITED STATES"),
            (107, 94, 489, 115, "SECURITIES AND EXCHANGE COMMISSION"),
            (230, 115, 366, 128, "Washington, D.C. 20549"),
            (20, 172, 559, 183, "QUARTERLY REPORT PURSUANT TO SECTION 13"),
        ]
        self.assertEqual(len(running_header_lines(cover, 596, 842)), 3)

    def test_a_region_set_in_columns_is_a_list_of_items(self):
        from pdf2zh.rules import region_has_columns
        cover = (
            _words(0, 300, (100, "WASHINGTON"), (300, "91-1144442"))
            + _words(1, 310, (90, "(STATE OF INCORPORATION)"), (420, "(I.R.S. ID)"))
        )
        self.assertTrue(region_has_columns((20, 290, 560, 330), cover))
        # A list with markers indents its text; that is no gutter.
        listed = (_words(0, 300, (100, "(a)"), (160, "during the current reporting period; or"))
                  + _words(1, 310, (100, "(b)"), (160, "after the end of the reporting period")))
        self.assertFalse(region_has_columns((20, 290, 560, 330), listed))
        # Wide justified spaces in different places are no gutter either.
        justified = (_words(0, 300, (100, "assets"), (160, "or"))
                     + _words(1, 310, (100, "interests to the"), (260, "former owners")))
        self.assertFalse(region_has_columns((20, 290, 560, 330), justified))

    def test_a_centred_line_stays_centred_when_translated(self):
        from pdf2zh.converter import centred_line_x
        self.assertAlmostEqual(centred_line_x(235, 378, 180, 306, 18, 595), 216)
        # Prose that starts at the margin keeps its start,
        self.assertIsNone(centred_line_x(18, 594, 400, 306, 18, 595))
        # and so does a translation too wide to centre between the margins.
        self.assertIsNone(centred_line_x(235, 378, 590, 306, 18, 595))
        # A long contents line across the middle keeps its start.
        self.assertIsNone(centred_line_x(82, 540, 470, 306, 18, 595))
        # but a heading set larger than the body may be that long.
        self.assertIsNotNone(centred_line_x(107, 489, 300, 298, 20, 578, True))

    def test_a_one_line_heading_takes_the_free_space_to_its_right(self):
        from pdf2zh.converter import available_width_right
        heading = (18, 463, 118, 474)
        self.assertEqual(available_width_right(heading, [heading], 594), 594)
        # Something further along the line stops it,
        beside = (300, 460, 400, 476)
        self.assertEqual(available_width_right(heading, [heading, beside], 594), 298)
        # but a paragraph below it does not.
        below = (18, 430, 594, 455)
        self.assertEqual(available_width_right(heading, [heading, below], 594), 594)

    def test_a_ligature_glyph_reads_as_its_letters(self):
        from pdf2zh.converter import expand_ligature
        character = SimpleNamespace(_text="\ufb01")
        expand_ligature(character)
        self.assertEqual(character._text, "fi")
        plain = SimpleNamespace(_text="x")
        expand_ligature(plain)
        self.assertEqual(plain._text, "x")

    def test_two_text_boxes_over_one_paragraph_are_one_region(self):
        """NVIDIA 10-Q note 6: boxes at x 84-578 and 100-594 over the same lines."""
        regions = [((100, 348, 594, 367), 0.51), ((84, 348, 578, 367), 0.35),
                   ((18, 378, 124, 388), 0.9)]
        self.assertEqual(overlapping_text_groups(regions), [[0, 1]])

    def test_a_text_reading_the_model_trusts_more_beats_abandon(self):
        note = (19, 628, 153, 640)
        self.assertTrue(abandon_outranked_by_text(note, 0.29, [((19, 628, 153, 640), 0.67)]))
        self.assertFalse(abandon_outranked_by_text(note, 0.8, [((19, 628, 153, 640), 0.67)]))
        self.assertFalse(abandon_outranked_by_text(note, 0.29, [((19, 500, 300, 640), 0.9)]))

    def test_a_statement_title_above_its_table_is_a_caption(self):
        tables = [(17, 113, 594, 317)]
        self.assertTrue(abandon_is_table_caption((225, 63, 387, 102), tables))
        # A running header with the page's first heading between it and the table.
        self.assertFalse(abandon_is_table_caption((196, 22, 415, 52), tables))
        # Wider than the table: not its caption.
        self.assertFalse(abandon_is_table_caption((5, 63, 600, 102), tables))

    def test_a_text_box_that_starts_partway_along_a_line_takes_the_line(self):
        lines = [(18, 452, 589, 460), (18, 461, 589, 468), (18, 400, 574, 408)]
        self.assertEqual(
            lines_crossing_region((84, 454, 578, 472), lines),
            [(18, 452, 589, 460), (18, 461, 589, 468)],
        )
        # A line beside the region in another column is not taken.
        self.assertEqual(lines_crossing_region((300, 0, 590, 50), [(18, 10, 280, 18)]), [])


class PreservationRuleTests(unittest.TestCase):
    def test_formula_rule_covers_math_and_monospace_code_fonts(self):
        for font in (
            "CMMI10",
            "TeX-math-symbols",
            "STIXMath",
            "Consolas",
            "CourierNewPSMT",
            "SourceCodePro-Regular",
        ):
            with self.subTest(font=font):
                self.assertTrue(is_formula_font(font))
        self.assertFalse(is_formula_font("TimesNewRomanPSMT"))

    def test_operator_only_ordinary_font_block_is_protected_as_formula(self):
        blocks = [(10, 20, 90, 40, "F1 / b0 ≤ C2 [N/mm]")]
        self.assertEqual(formula_regions(blocks, []), [(10.0, 20.0, 90.0, 40.0)])

    def test_trigonometric_functions_do_not_make_an_equation_look_like_prose(self):
        for equation in (
            "FWA = F1² + F2² - 2 · F1 · F2 · cos β [N]",
            "FR = 2 · F2 · cos γ / 2 - FTR [N]",
            "FW6 = √2 · F2 · sin (β/2) [N]",
        ):
            with self.subTest(equation=equation):
                blocks = [(10, 20, 190, 40, equation)]
                self.assertEqual(
                    formula_regions(blocks, []),
                    [(10.0, 20.0, 190.0, 40.0)],
                )

    def test_prose_containing_variables_is_not_protected_as_a_formula(self):
        blocks = [(10, 20, 190, 40, "If F1 is larger than C2, use another belt")]
        self.assertEqual(formula_regions(blocks, []), [])

    def test_stacked_numbered_identifiers_are_protected_inside_prose(self):
        words = [
            (10, 10, 20, 20, "F1", 7, 0, 0),
            (10, 20, 20, 30, "b0", 7, 1, 0),
            (30, 20, 60, 30, "value", 7, 1, 1),
        ]
        self.assertEqual(formula_regions([], words), [(10.0, 10.0, 20.0, 30.0)])

    def test_stacked_detection_does_not_cross_a_protected_table(self):
        words = [
            (10, 10, 20, 20, "F1", 7, 0, 0),
            (10, 20, 20, 30, "F2", 7, 1, 0),
        ]
        self.assertEqual(
            formula_regions([], words, stacked_exclusions=[(0, 0, 100, 100)]),
            [],
        )

    def test_table_cells_require_a_half_area_model_match(self):
        matching = SimpleNamespace(
            bbox=(0, 0, 100, 140),
            cells=[(0, 0, 50, 50), (50, 0, 100, 50), (0, 100, 100, 140)],
        )
        distant = SimpleNamespace(bbox=(200, 200, 300, 300), cells=[(200, 200, 300, 300)])
        self.assertEqual(
            matching_table_cells((0, 0, 100, 100), [distant, matching]),
            [(0.0, 0.0, 50.0, 50.0), (50.0, 0.0, 100.0, 50.0)],
        )
        self.assertEqual(
            matching_table_cells((0, 0, 100, 100), [SimpleNamespace(bbox=(0, 0, 40, 100), cells=[])]),
            [],
        )

    def test_table_codes_and_numbers_stay_as_original_glyphs(self):
        for value in ("E 2/1, E 3/1, NOVO", "180° 210° 240°", "2.0"):
            with self.subTest(value=value):
                self.assertFalse(should_translate_table_cell(value))
        self.assertTrue(should_translate_table_cell("Tension member"))
        self.assertTrue(should_translate_table_cell("Lớp phủ mặt dưới"))

    def test_merged_description_and_code_cell_splits_into_x_clusters(self):
        words = [
            (0, 0, 20, 10, "Drive", 1, 0, 0),
            (22, 0, 45, 10, "drum", 1, 0, 1),
            (47, 0, 80, 10, "diameter", 1, 0, 2),
            (160, 0, 170, 10, "dA", 1, 0, 3),
        ]
        clusters = cluster_table_words(words, (0, 0, 180, 12))
        self.assertEqual([cluster.text for cluster in clusters], ["Drive drum diameter", "dA"])
        self.assertTrue(should_translate_table_cell(clusters[0].text))
        self.assertFalse(should_translate_table_cell(clusters[1].text))

    def test_wrapped_description_lines_stay_in_one_cluster(self):
        words = [
            (0, 0, 30, 10, "Maximum", 1, 0, 0),
            (32, 0, 50, 10, "belt", 1, 0, 1),
            (0, 11, 20, 21, "pull", 1, 1, 0),
            (22, 11, 55, 21, "allowed", 1, 1, 1),
            (160, 5, 170, 15, "F1", 1, 2, 0),
        ]
        clusters = cluster_table_words(words, (0, 0, 180, 24))
        self.assertEqual(len(clusters), 2)
        self.assertIn("Maximum", clusters[0].text)
        self.assertIn("allowed", clusters[0].text)

    def test_a_justified_line_stays_one_cluster(self):
        """IFRS 3 paragraph 2(c), word boxes as PyMuPDF reads them: justification
        stretched every space to 6.5 pt and the line came back as nine words."""
        line = [(208.3, 212.6, "a"), (219.2, 268.0, "combination"), (274.5, 282.1, "of"),
                (288.7, 317.3, "entities"), (323.8, 332.0, "or"), (338.5, 378.8, "businesses"),
                (385.4, 408.4, "under"), (415.0, 449.0, "common"), (455.5, 483.3, "control")]
        words = [(180.0, 551.0, 188.6, 561.0, "(c)", 10, 0, 0)]
        words += [(x0, 551.0, x1, 561.0, text, 10, 1, i) for i, (x0, x1, text) in enumerate(line)]
        words += [(208.3, 562.9, 253.8, 572.9, "(paragraphs", 10, 2, 0),
                  (256.1, 562.9, 280.1, 572.9, "B1–B4", 10, 2, 1),
                  (282.4, 562.9, 311.3, 572.9, "guidance).", 10, 2, 2)]
        clusters = cluster_table_words(words, (178.0, 551.0, 486.4, 579.8))
        self.assertEqual(
            [cluster.text for cluster in clusters],
            ["(c)", "a combination of entities or businesses under common control "
                    "(paragraphs B1–B4 guidance)."],
        )

    def test_evenly_spaced_values_beside_a_label_still_split(self):
        words = [(0, 0, 30, 10, "Revenue", 1, 0, 0)]
        words += [(x, 0, x + 25, 10, value, 1, 0, i + 1)
                  for i, (x, value) in enumerate(((60, "45,400"), (105, "5,100"), (150, "3,000")))]
        clusters = cluster_table_words(words, (0, 0, 178, 12))
        self.assertEqual([cluster.text for cluster in clusters], ["Revenue", "45,400", "5,100", "3,000"])

    def test_vietnamese_line_height_and_extended_bullets_are_preserved(self):
        self.assertEqual(line_height_for_language("vi"), 1.2)
        self.assertTrue({"•", "■", "▸", "◆", "⬤"}.issubset(BULLET_CHARACTERS))

    def test_vietnamese_leading_never_goes_below_its_measured_ink(self):
        """Rendering every letter of the output font gives 0.890 em above the
        baseline for stacked tone marks and 0.210 em below for dot-below vowels,
        so lines closer than 1.10 em are drawn through each other. The leading
        used to be compressed to 0.75 to buy room for a longer translation."""
        self.assertGreaterEqual(min_line_height_for_language("vi"), 1.10)

    def test_every_target_can_be_typeset_without_lines_touching(self):
        for language in ("vi", "en", "fr", "de", "unknown-code"):
            with self.subTest(language=language):
                minimum = min_line_height_for_language(language)
                self.assertGreaterEqual(minimum, 0.9)
                self.assertLessEqual(minimum, line_height_for_language(language))

    def test_office_private_use_bullets_keep_their_dingbat_font(self):
        for character, font in (
            ("\uf0d8", "Wingdings"),
            ("\uf0b7", "Symbol"),
            ("\uf0fc", "Wingdings"),
        ):
            with self.subTest(character=hex(ord(character)), font=font):
                self.assertTrue(is_bullet_character(character, font))
        self.assertFalse(is_bullet_character("\uf0d8", "Times New Roman"))

    def test_full_page_image_is_classified_as_scanned(self):
        self.assertTrue(
            is_scanned_page([{"type": 1, "bbox": (0, 0, 80, 80)}], 10_000)
        )
        self.assertFalse(
            is_scanned_page([{"type": 1, "bbox": (0, 0, 20, 20)}], 10_000)
        )

    def test_table_of_contents_page_keeps_number_alignment(self):
        text = "Table of Contents\n" + "\n".join(
            f"Chapter {index} .......... {index * 3}" for index in range(1, 6)
        )
        decision = classify_preserved_page(text)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.kind, "TOC")

    def test_index_page_keeps_term_and_page_number_columns(self):
        text = "Index\nAlpha, 11\nBeta, 12\nGamma, 13"
        decision = classify_preserved_page(text)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.kind, "INDEX")

    def test_financial_statement_amounts_are_not_index_page_numbers(self):
        rows = ("Revenue", "Cost of sales", "Gross profit", "Administrative expenses",
                "Operating profit", "Profit before tax", "Income tax expense",
                "Profit for the year", "Retained earnings", "Total equity")
        lines = ["Draft statements of profit or loss for the year ended 30 September 20X8"]
        for n, row in enumerate(rows):
            lines += [row, f"{3 + n},463,760", f"({1 + n},558,690)", f"{n + 2}05,070"]
        self.assertIsNone(classify_preserved_page("\n".join(lines)))
        self.assertFalse(has_index_page_reference("3,463,760"))
        self.assertFalse(has_index_page_reference("(1,558,690)"))
        self.assertTrue(has_index_page_reference("Leasing 108, 127, 132"))
        self.assertTrue(has_index_page_reference("Goodwill, 45"))

    def test_nomenclature_page_keeps_symbol_definition_pairs(self):
        lines = ["Nomenclature"]
        for symbol, definition in (
            ("E", "Energy of the system"),
            ("m", "Mass of the particle"),
            ("c", "Speed of light"),
            ("F", "Applied force"),
            ("a", "Measured acceleration"),
        ):
            lines.extend((symbol, definition))
        decision = classify_preserved_page("\n".join(lines))
        self.assertIsNotNone(decision)
        self.assertEqual(decision.kind, "NOMENCLATURE")

    def test_reference_page_keeps_citation_numbering(self):
        text = "References\n" + "\n".join(
            f"[{index}] Author, A. ({2020 + index}). https://doi.org/10.1/{index}"
            for index in range(1, 6)
        )
        decision = classify_preserved_page(text)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.kind, "REFERENCES")

    def test_normal_prose_is_not_misclassified(self):
        self.assertIsNone(
            classify_preserved_page(
                "A short introduction\nThis paragraph explains a translation system."
            )
        )

    def test_converter_rejects_an_unregistered_service(self):
        with self.assertRaisesRegex(ValueError, "Unsupported translation service"):
            TranslateConverter(PDFResourceManager(), service="bing")

    def test_converter_accepts_every_registered_engine(self):
        for service in ("google", "handoff"):
            with self.subTest(service=service):
                converter = TranslateConverter(PDFResourceManager(), service=service)
                self.assertEqual(converter.translator.name, service)

    def test_converter_keeps_the_late_filled_cell_bounds_mapping(self):
        bounds = {}
        converter = TranslateConverter(
            PDFResourceManager(), service="google", layout_bounds=bounds
        )
        bounds[0] = {7: (1, 2, 3, 4)}
        self.assertIs(converter.layout_bounds, bounds)
        self.assertEqual(converter.layout_bounds[0][7], (1, 2, 3, 4))


class TranslatableSegmentTests(unittest.TestCase):
    """The count these drive decides whether a document is called a scan."""

    def test_real_text_is_translatable(self):
        self.assertTrue(is_translatable_segment("Chapter 1", set()))
        self.assertTrue(is_translatable_segment("<s1>Bold</s1> text", set()))

    def test_blank_and_placeholder_only_runs_are_not_work(self):
        for empty in ("", "   ", "<s1></s1>", "{v3}", " {v12} "):
            with self.subTest(segment=empty):
                self.assertFalse(is_translatable_segment(empty, set()))

    def test_preserved_segments_are_not_counted(self):
        self.assertFalse(is_translatable_segment("E = mc2", {"E = mc2"}))

    def test_a_figure_caption_beside_a_formula_still_counts(self):
        """A page of diagrams that carries one real line is not an image-only
        page, so a scan-heavy textbook is never refused for needing OCR."""
        self.assertTrue(is_translatable_segment("Figure 3: the lever arm", {"{v1}"}))

    def test_figures_alone_have_no_words(self):
        for figures in ("1.5 ", " 0.62", "5   4   3   2   1", "(12)", "{v3} = 0.5",
                        "<s1>2024</s1>", "{v0} {v1}", "—"):
            with self.subTest(segment=figures):
                self.assertFalse(has_words(figures))
        for words in ("Chapter 1", "<s1>Bold</s1> 2", "5 mm", "Hình 3", "图 3"):
            with self.subTest(segment=words):
                self.assertTrue(has_words(words))

    def test_figures_are_never_sent_yet_still_count_as_text(self):
        """Google returned "0.62" as "0,62" for Vietnamese while the table beside
        it kept "0.62". Counted all the same, a chart is not taken for a scan."""
        document = pymupdf.open()
        page = document.new_page(width=300, height=300)
        page.insert_text((20, 50), "Total weight of the sample", fontsize=11)
        page.insert_text((20, 150), "0.62", fontsize=11)
        page.insert_text((20, 250), "5   4   3   2   1", fontsize=11)
        mono, report, sent = translated(document.tobytes())
        self.assertEqual(sent, ["Total weight of the sample"])
        self.assertEqual(report.translatable_segments, 3)
        with pymupdf.open(stream=mono) as output:
            self.assertEqual(
                output[0].get_text().split("\n")[:3],
                ["vi:Total weight of the sample", "0.62", "5   4   3   2   1"],
            )


class ImageOnlyPageTests(unittest.TestCase):
    def test_any_image_counts_even_when_no_single_tile_is_large(self):
        """Scanners emit a page as dozens of tiles, so the half-page rule that
        drives backing rectangles misses most scanned pages entirely."""
        tiles = [{"type": 1, "bbox": (0, 0, 60, 60)} for _ in range(12)]
        self.assertTrue(page_has_image(tiles))
        self.assertFalse(is_scanned_page(tiles, page_area=600 * 800))

    def test_a_page_of_text_alone_carries_no_image(self):
        self.assertFalse(page_has_image([{"type": 0, "bbox": (0, 0, 10, 10)}]))
        self.assertFalse(page_has_image([]))

    def test_a_page_is_judged_on_everything_it_contributed(self):
        """receive_layout runs once per nested container and only the last call
        carries the text, so judging a page as each call arrives reported a
        fully translated page as an untranslated scan."""
        converter = TranslateConverter(PDFResourceManager(), service="google")
        converter.pages_with_images.update({0, 1})
        for count in (0, 0, 0, 14):  # one page, four calls
            converter.segments_by_page[0] += count
        converter.segments_by_page[1] += 0

        self.assertEqual(converter.image_only_pages, {1})

    def test_a_blank_page_is_not_called_a_scan(self):
        converter = TranslateConverter(PDFResourceManager(), service="google")
        converter.segments_by_page[3] += 0

        self.assertEqual(converter.image_only_pages, set())


if __name__ == "__main__":
    unittest.main()
