"""The paragraph-format framework: a paragraph's form read from its geometry."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from pdf2zh.paragraph_format import (
    collapse_doubled_spaces,
    first_line_tab,
    hanging_lead,
    justify_operations,
    lead_end,
    opens_with_lead,
    stands_at_tab_stop,
    source_alignment,
)


def glyph(text: str, x0: float, baseline: float, size: float = 8.5, width: float = 4.0):
    return SimpleNamespace(
        get_text=lambda: text, x0=x0, x1=x0 + width, size=size, matrix=(1, 0, 0, 1, x0, baseline)
    )


def line(text: str, x0: float, baseline: float, gaps: dict[int, float] | None = None):
    """Glyphs of TEXT from X0, 4 pt apart; GAPS widens the gap before an index."""
    glyphs, x = [], x0
    for index, character in enumerate(text):
        x += (gaps or {}).get(index, 0.0)
        glyphs.append(glyph(character, x, baseline))
        x += 4.0
    return glyphs


class LabelTests(unittest.TestCase):
    def test_a_label_is_what_stands_before_a_tab(self):
        """IFRS 3's "(d)    by providing ..." keeps its tab, one line or many."""
        chars = line("(d)by", 180, 700, {3: 16.0})
        self.assertEqual(first_line_tab(chars), ("(d)", 180 + 12 + 16))
        # A word space is no tab.
        self.assertIsNone(first_line_tab(line("A business", 180, 700)))
        # A long first word followed by a wide justified gap is no label.
        self.assertIsNone(first_line_tab(line("combination may", 180, 700, {11: 12.0})))

    def test_a_hanging_label_is_read_from_geometry_not_spelling(self):
        """"(a)<s1> Input:" missed the "(a) " pattern and lost its indent."""
        body = 208.0
        chars = line("(a)Input", 180, 700, {3: body - 192}) + line("more", body, 689)
        self.assertEqual(hanging_lead(chars, body), "(a)")
        chars = line("B12Dtext", 180, 700, {4: body - 196}) + line("more", body, 689)
        self.assertEqual(hanging_lead(chars, body), "B12D")
        # Text that does not start where the wrapped lines start does not hang.
        self.assertIsNone(hanging_lead(line("Abcdefghij", 180, 700) + line("more", body, 689), body))

    def test_the_label_is_found_in_the_translation_across_tags(self):
        self.assertEqual(lead_end("(a)<s1> Đầu vào:</s1>", "(a)"), 7)
        self.assertEqual(lead_end("<s1>(a)</s1> Đầu", "(a)"), 12)
        self.assertEqual(lead_end("(a) x", "(b)"), -1)
        self.assertEqual(lead_end("(a)x", "(a)"), -1)

    def test_only_a_short_first_word_and_a_wide_gap_open_with_a_label(self):
        self.assertTrue(opens_with_lead("(e)", 189.0, 208.0, 8.5))
        self.assertFalse(opens_with_lead("(e)", 189.0, 192.0, 8.5))
        self.assertFalse(opens_with_lead("combination", 189.0, 208.0, 8.5))
        self.assertFalse(opens_with_lead("(e)", None, 208.0, 8.5))


class TabStopTests(unittest.TestCase):
    def test_a_check_box_in_its_own_column_stands_at_a_tab_stop(self):
        """Apple's "Large accelerated filer" and its box sit 37 pt apart."""
        label = line("filer", 30, 700)
        self.assertTrue(stands_at_tab_stop(138.0, 700, label, 8.0))
        self.assertFalse(stands_at_tab_stop(52.0, 700, label, 8.0))
        # Nothing before it on its line: no stop to keep.
        self.assertFalse(stands_at_tab_stop(138.0, 650, label, 8.0))


class AlignmentTests(unittest.TestCase):
    def test_lines_that_share_a_right_edge_are_justified(self):
        justified = line("aaaaaaaaaa", 100, 700) + line("bbbbbbbbbb", 100, 689) + line("cc", 100, 678)
        self.assertEqual(source_alignment(justified), "justify")
        ragged = line("aaaaaaaaaa", 100, 700) + line("bbbbbbb", 100, 689) + line("cc", 100, 678)
        self.assertEqual(source_alignment(ragged), "left")
        # A short heading over one full line is not justified text.
        heading = line("aaaa", 136, 700) + line("bbbbbbbbbb", 100, 689)
        self.assertEqual(source_alignment(heading + line("c", 100, 678)), "left")
        self.assertEqual(source_alignment(line("one", 100, 700)), "left")

    def test_justified_lines_end_at_the_measure_and_the_last_does_not(self):
        measure = lambda character, style, size: 5.0
        encode = lambda font, text: text
        operations = [
            {"type": "t", "font": "f", "size": 10.0, "x": 100.0, "lidx": 0, "text": "ab cd ef", "rtxt": "", "style": 0},
            {"type": "t", "font": "f", "size": 10.0, "x": 100.0, "lidx": 1, "text": "gh ij", "rtxt": "", "style": 0},
        ]
        justify_operations(operations, measure, encode, 146.0, {1}, lambda values: values["type"] == "t")
        first = [values for values in operations if values["lidx"] == 0]
        self.assertEqual([values["text"] for values in first], ["ab ", "cd ", "ef"])
        # 8 glyphs end at 140; two gaps take 3 pt each.
        self.assertEqual([values["x"] for values in first], [100.0, 118.0, 136.0])
        self.assertEqual([values["text"] for values in operations if values["lidx"] == 1], ["gh ij"])

    def test_a_line_that_would_open_too_wide_is_left_alone(self):
        operations = [{"type": "t", "font": "f", "size": 10.0, "x": 100.0, "lidx": 0, "text": "ab cd", "rtxt": "", "style": 0}]
        justify_operations(operations, lambda c, s, z: 5.0, lambda f, t: t, 300.0, set(), lambda v: True)
        self.assertEqual([values["x"] for values in operations], [100.0])


class SpacingTests(unittest.TestCase):
    def test_a_space_on_each_side_of_a_style_tag_is_one_space(self):
        self.assertEqual(collapse_doubled_spaces("tài sản <s2> có thể</s2> đã"), "tài sản <s2>có thể</s2> đã")
        self.assertEqual(collapse_doubled_spaces("a  b"), "a b")
        self.assertEqual(collapse_doubled_spaces("từ" + " " * 18 + "đến"), "từ" + " " * 18 + "đến")


if __name__ == "__main__":
    unittest.main()
