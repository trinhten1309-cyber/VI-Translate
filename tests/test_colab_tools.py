"""Regressions for the Google Colab workflow in colab/."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pymupdf

COLAB = Path(__file__).resolve().parents[1] / "colab"
sys.path.insert(0, str(COLAB))

import handoff_tools  # noqa: E402
import ocr_check  # noqa: E402
import ocr_layer  # noqa: E402
import rebuild_text_layer  # noqa: E402

HAS_OPENAI = importlib.util.find_spec("openai") is not None
HAS_RAPIDOCR = importlib.util.find_spec("rapidocr") is not None


def image_pdf_with_hidden_text(path: Path, text: str = "Cost of capital") -> pymupdf.Rect:
    """A page whose words are pixels in a picture plus an invisible OCR layer."""
    drawn = pymupdf.open()
    page = drawn.new_page(width=300, height=200)
    page.draw_rect(pymupdf.Rect(0, 0, 300, 60), color=None, fill=(0, 0.7, 0.75))
    page.insert_text((20, 120), text, fontsize=16, color=(0, 0, 0))
    picture = page.get_pixmap(dpi=144)

    scan = pymupdf.open()
    out = scan.new_page(width=300, height=200)
    out.insert_image(out.rect, pixmap=picture)
    out.insert_text((20, 120), text, fontsize=16, render_mode=3)
    scan.save(path)
    return pymupdf.Rect(20, 104, 20 + pymupdf.get_text_length(text, fontsize=16), 124)


class RebuildTextLayerTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_hidden_ocr_text_becomes_visible_and_the_picture_text_is_removed(self):
        source = self.dir / "scan.pdf"
        box = image_pdf_with_hidden_text(source)
        self.assertEqual(rebuild_text_layer.needs_rebuild(source), [1])

        rebuilt_path = self.dir / "rebuilt.pdf"
        self.assertEqual(rebuild_text_layer.rebuild(source, rebuilt_path), [1])

        page = pymupdf.open(rebuilt_path)[0]
        traces = page.get_texttrace()
        self.assertTrue(traces)
        self.assertTrue(all(trace["type"] != 3 for trace in traces), "text must be visible")
        self.assertIn("Cost of capital", page.get_text())

        # The picture under the words is now plain background: no dark ink left.
        xref = page.get_images()[0][0]
        picture = pymupdf.Pixmap(page.parent, xref)
        scale = picture.width / page.rect.width
        pixels = np.frombuffer(picture.samples, np.uint8).reshape(
            picture.height, picture.width, picture.n
        )
        x0, y0, x1, y1 = (int(v * scale) for v in box)
        self.assertGreater(pixels[y0:y1, x0:x1, :3].min(), 200)
        # Colour outside the words (the teal band) is untouched.
        self.assertLess(int(pixels[int(30 * scale), int(150 * scale), 0]), 60)

    def test_pages_with_visible_text_are_copied_unchanged(self):
        source = self.dir / "text.pdf"
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "Ordinary text")
        doc.save(source)
        self.assertEqual(rebuild_text_layer.needs_rebuild(source), [])
        self.assertEqual(rebuild_text_layer.rebuild(source, self.dir / "copy.pdf"), [])

    def test_underlines_and_double_underlines_under_numbers_survive_whole(self):
        # A total with a rule longer than the number (single) and a final
        # total with a double rule, as in BPP financial statements.
        drawn = pymupdf.open()
        page = drawn.new_page(width=300, height=200)
        for y, text in ((60, "16,000"), (110, "56,600")):
            page.insert_text((200, y), text, fontsize=10)
        # The rules sit just under the digits, inside the line's text box
        # (which reaches down to the descender), like the book's totals.
        page.draw_line((180, 62.5), (240, 62.5), width=0.7)
        page.draw_line((180, 112.5), (240, 112.5), width=0.7)
        page.draw_line((180, 114.7), (240, 114.7), width=0.7)
        picture = page.get_pixmap(dpi=144)
        source = self.dir / "totals.pdf"
        scan = pymupdf.open()
        out = scan.new_page(width=300, height=200)
        out.insert_image(out.rect, pixmap=picture)
        for y, text in ((60, "16,000"), (110, "56,600")):
            out.insert_text((200, y), text, fontsize=10, render_mode=3)
        scan.save(source)

        rebuilt = self.dir / "rebuilt.pdf"
        self.assertEqual(rebuild_text_layer.rebuild(source, rebuilt), [1])
        doc = pymupdf.open(rebuilt)
        picture = pymupdf.Pixmap(doc, doc[0].get_images()[0][0])
        scale = picture.width / 300
        pixels = np.frombuffer(picture.samples, np.uint8).reshape(
            picture.height, picture.width, picture.n)[:, :, :3].mean(axis=2)

        def darkest_row_profile(y):
            rows = pixels[int((y - 1.2) * scale):int((y + 1.2) * scale) + 1,
                          int(182 * scale):int(238 * scale)]
            return rows.min(axis=0)

        for y in (62.5, 112.5, 114.7):
            profile = darkest_row_profile(y)
            # the whole rule is there, as dark under the number as beside it
            self.assertLess(profile.max(), 140, f"rule at y={y} has a gap")
            self.assertLess(profile.max() - profile.min(), 60, f"rule at y={y} is uneven")
        # while the digits themselves are gone from the picture
        digits = pixels[int(52 * scale):int(60 * scale), int(201 * scale):int(230 * scale)]
        self.assertGreater(digits.min(), 200)

    def test_rebuilt_text_uses_the_given_font(self):
        source = self.dir / "scan.pdf"
        image_pdf_with_hidden_text(source, "40,600")
        rebuilt = self.dir / "rebuilt.pdf"
        fonts = {"regular": COLAB / "fonts" / "Cabin-Regular.ttf",
                 "bold": COLAB / "fonts" / "Cabin-Bold.ttf"}
        rebuild_text_layer.rebuild(source, rebuilt, fonts=fonts)
        names = {span["font"] for block in pymupdf.open(rebuilt)[0].get_text("dict")["blocks"]
                 for line in block.get("lines", []) for span in line["spans"]}
        self.assertEqual(names, {"Cabin-Regular"})

    def test_a_tinted_jpeg_bin_does_not_colour_the_background(self):
        # Plain white paper whose JPEG noise put more pixels in a pinkish bin
        # than in pure white: the patch must still be white, not pink.
        paper = [[255, 255, 255]] * 1000
        noise = [[255, g, 255] for g in (248, 249, 250, 251) for _ in range(400)]
        colour = rebuild_text_layer.dominant_colour(np.array(paper + noise, float))
        self.assertEqual(colour.tolist(), [255, 255, 255])
        # a coloured table header still wins over the white page beside it
        header = [[0, 179, 191]] * 900 + [[2, 181, 190]] * 300
        colour = rebuild_text_layer.dominant_colour(np.array(header + paper[:800], float))
        self.assertEqual(colour.tolist(), [0, 179, 191])

    def test_page_selection_matches_the_cli(self):
        self.assertEqual(rebuild_text_layer.parse_pages("1-3,5", 10), {0, 1, 2, 4})
        self.assertEqual(rebuild_text_layer.parse_pages("", 3), {0, 1, 2})
        self.assertEqual(rebuild_text_layer.parse_pages("8-20", 10), {7, 8, 9})


def pure_scan_pdf(path: Path, lines: list[str]) -> float:
    """Image-only page (no text at all) showing LINES; returns the render scale."""
    drawn = pymupdf.open()
    page = drawn.new_page(width=400, height=300)
    for n, text in enumerate(lines):
        page.insert_text((30, 60 + 40 * n), text, fontsize=18)
    picture = page.get_pixmap(dpi=200)
    scan = pymupdf.open()
    out = scan.new_page(width=400, height=300)
    out.insert_image(out.rect, pixmap=picture)
    scan.save(path)
    return 200 / 72


class FakeOcr:
    """Stands in for RapidOCR: returns fixed boxes in 200-dpi pixel space."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, _image, **_options):
        result = type("Result", (), {})()
        result.boxes = [np.array(box, dtype=np.float32) for box, _, _ in self.rows]
        result.txts = [text for _, text, _ in self.rows]
        result.scores = [score for _, _, score in self.rows]
        return result


def pixel_box(x0, y0, x1, y1, scale):
    return [[x0 * scale, y0 * scale], [x1 * scale, y0 * scale],
            [x1 * scale, y1 * scale], [x0 * scale, y1 * scale]]


class OcrLayerTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_confident_lines_get_an_invisible_layer_over_their_box(self):
        source = self.dir / "scan.pdf"
        scale = pure_scan_pdf(source, ["Cost of capital", "Gearing"])
        self.assertEqual(ocr_layer.image_only_pages(source), [1])
        engine = FakeOcr([
            (pixel_box(30, 46, 150, 64, scale), "Cost of capital", 0.99),
            (pixel_box(30, 86, 100, 104, scale), "Gearing", 0.40),        # unsure: stays picture
            (pixel_box(30, 126, 380, 144, scale), "Go  t   cith", 0.95),  # lost letters: stays picture
        ])
        target = self.dir / "scan-ocr.pdf"
        self.assertEqual(ocr_layer.add_ocr_layer(source, target, engine=engine), [1])

        page = pymupdf.open(target)[0]
        traces = page.get_texttrace()
        self.assertTrue(traces and all(trace["type"] == 3 for trace in traces))
        self.assertEqual(page.get_text().split(), ["Cost", "of", "capital"])
        words = page.get_text("words")
        self.assertAlmostEqual(words[0][0], 30, delta=2)
        self.assertAlmostEqual(words[-1][2], 150, delta=3)

        # The rebuild step then treats it like any scanner OCR layer.
        self.assertEqual(rebuild_text_layer.needs_rebuild(target), [1])

    def test_font_size_follows_the_printed_letters_not_the_box(self):
        # "Section C" has no descender, so its OCR box only spans capital
        # height and box-based sizing printed the heading a size too small.
        source = self.dir / "heading.pdf"
        scale = pure_scan_pdf(source, ["Section C", "Pedantic Co (Dec)"])
        engine = FakeOcr([
            (pixel_box(28, 44, 112, 62, scale), "Section C", 0.99),
            (pixel_box(28, 84, 180, 106, scale), "Pedantic Co (Dec)", 0.99),
        ])
        target = self.dir / "heading-ocr.pdf"
        ocr_layer.add_ocr_layer(source, target, engine=engine)
        sizes = {line["spans"][0]["text"]: line["spans"][0]["size"]
                 for block in pymupdf.open(target)[0].get_text("dict")["blocks"]
                 for line in block.get("lines", [])}
        self.assertAlmostEqual(sizes["Section C"], 18, delta=1)
        self.assertAlmostEqual(sizes["Pedantic Co (Dec)"], 18, delta=1)

    def test_pages_that_already_have_text_are_not_recognised_again(self):
        source = self.dir / "text.pdf"
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "Ordinary text")
        doc.save(source)
        self.assertEqual(ocr_layer.image_only_pages(source), [])
        unused = FakeOcr([])
        self.assertEqual(ocr_layer.add_ocr_layer(source, self.dir / "copy.pdf", engine=unused), [])

    @unittest.skipUnless(HAS_RAPIDOCR, "rapidocr comes from requirements-ocr.txt")
    def test_rapidocr_reads_a_rendered_scan(self):
        source = self.dir / "scan.pdf"
        pure_scan_pdf(source, ["Weighted average cost of capital"])
        target = self.dir / "scan-ocr.pdf"
        self.assertEqual(ocr_layer.add_ocr_layer(source, target), [1])
        self.assertIn("capital", pymupdf.open(target)[0].get_text())


class FakeSecondReader:
    """Stands in for Tesseract: fixed words with boxes in points."""

    def __init__(self, words):
        self.words_ = words

    def words(self, png, _scale):
        assert png.startswith(b"\x89PNG")
        return self.words_


class FakeLineReader:
    """Stands in for PaddleOCR-VL: answers by line order."""

    def __init__(self, texts):
        self.texts, self.crops = texts, []

    def read(self, crops):
        self.crops = crops
        return self.texts[:len(crops)]


class OcrCheckTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_known_misreadings_of_the_chinese_model_are_corrected(self):
        # Seen on the BPP kit and the benchmark pages; RapidOCR's score stays high on all of them.
        fixes = {
            "STATEMENTS OF FINANCIAL POSITION AS AT 3O SEPTEMBER 20X8":
                "STATEMENTS OF FINANCIAL POSITION AS AT 30 SEPTEMBER 20X8",
            "On 1 October 2OX7, Strata Co": "On 1 October 20X7, Strata Co",
            "(Total = 2O marks)": "(Total = 20 marks)",
            "1,5OO": "1,500",
            "二": "–",
            "S": "$",
            "$000": "$'000",
            "（1,600）": "(1,600)",
        }
        for read, right in fixes.items():
            self.assertEqual(ocr_check.clean_text(read), right)
        for text in ("McIlroy Co acquired 80% of IAS 36", "Co", "Total assets", "(5) There were no losses",
                     "Sophistic Co's net assets", "$'000"):
            self.assertEqual(ocr_check.clean_text(text), text)

    def test_badly_formed_numbers_are_reported(self):
        for bad in ("45,40", "(1,600", "45.400", "1,6000"):
            self.assertTrue(ocr_check.number_problem(bad), bad)
        for good in ("1,600", "(3,100)", "$'000", "45,400", "–", "Profit for the year ended 30 September 20X8",
                     "Dividend of $1.76 per share", "(27 mins)", "$1,500,000"):
            self.assertIsNone(ocr_check.number_problem(good), good)

    def test_a_number_tesseract_reads_differently_is_flagged_but_never_chosen(self):
        text, readings, reasons = ocr_check.judge("(3,100)", tess="_ (8,100)")
        self.assertEqual(text, "(3,100)")
        self.assertEqual(readings["Tesseract"], "_ (8,100)")
        self.assertTrue(reasons)
        self.assertEqual(ocr_check.judge("4,700", tess="_ 4,700")[2], [])   # same number, stray underline mark
        self.assertEqual(ocr_check.judge("BPP", tess="pp")[2], [])          # words are not compared
        self.assertEqual(ocr_check.judge("5,100", tess="")[2], [])          # Tesseract saw nothing

    def test_paddle_vl_text_is_used_when_it_agrees_and_checked_against_rapidocr(self):
        self.assertEqual(ocr_check.judge("$000", vl="$'000")[:3:2], ("$'000", []))
        text, _, reasons = ocr_check.judge("5,100", vl="5,700")
        self.assertEqual(text, "5,700")
        self.assertTrue(reasons)
        # An unrelated answer (a hallucination) keeps RapidOCR's text and is flagged.
        text, _, reasons = ocr_check.judge("Revenue", vl="The quick brown fox")
        self.assertEqual(text, "Revenue")
        self.assertTrue(reasons)
        # Two readers agreeing outvote Tesseract.
        self.assertEqual(ocr_check.judge("5,100", vl="5,100", tess="8,100")[2], [])

    def test_ocr_layer_uses_both_readers_and_lists_the_lines_to_check(self):
        source = self.dir / "scan.pdf"
        scale = pure_scan_pdf(source, ["Revenue 5,100", "AS AT 30 JUNE", "(3,100)", "Spieth"])
        engine = FakeOcr([
            (pixel_box(30, 46, 160, 64, scale), "Revenue 5,100", 0.99),
            (pixel_box(30, 86, 165, 104, scale), "AS AT 3O JUNE", 0.99),
            (pixel_box(30, 126, 90, 144, scale), "(3,100)", 0.99),
            (pixel_box(30, 166, 90, 184, scale), "收入", 0.99),          # Chinese characters: stays picture
        ])
        second = FakeSecondReader([("Revenue", (30, 46, 95, 64)), ("5,100", (110, 46, 160, 64)),
                                   ("(8,100)", (30, 126, 90, 144))])
        flags = []
        target = self.dir / "scan-ocr.pdf"
        ocr_layer.add_ocr_layer(source, target, engine=engine, second=second, flags=flags)
        self.assertEqual(pymupdf.open(target)[0].get_text().split(),
                         ["Revenue", "5,100", "AS", "AT", "30", "JUNE", "(3,100)"])
        self.assertEqual([f["text"] for f in flags], ["(3,100)"])
        self.assertEqual(flags[0]["page"], 1)

        review = self.dir / "review.pdf"
        self.assertEqual(ocr_check.write_review(source, flags, review), 1)
        page = pymupdf.open(review)[0]
        self.assertGreater(page.rect.width, 400)              # the readings sit beside the page
        self.assertIn("(8,100)", page.get_text())
        self.assertFalse(ocr_check.write_review(source, [], self.dir / "none.pdf"))
        self.assertFalse((self.dir / "none.pdf").exists())

        reader = FakeLineReader(["Revenue 5,100", "AS AT 30 JUNE", "(3,100)", "收入"])
        flags = []
        ocr_layer.add_ocr_layer(source, self.dir / "vl.pdf", engine=engine, reader=reader, flags=flags)
        self.assertEqual(len(reader.crops), 3)                # the Chinese line never reaches it
        self.assertTrue(all(c.ndim == 3 and c.size for c in reader.crops))
        self.assertEqual(flags, [])


class HandoffCheckTests(unittest.TestCase):
    glossary = [
        {"en": "non-controlling interest", "vi": "lợi ích cổ đông không kiểm soát", "keep_en": "0"},
        {"en": "interest", "vi": "lãi", "keep_en": "0"},
        {"en": "NCI", "vi": "NCI", "keep_en": "1"},
    ]

    def test_longest_term_wins_and_is_not_counted_twice(self):
        found = handoff_tools.terms_in("Measure the non-controlling interest (NCI).", self.glossary)
        self.assertEqual([row["en"] for row in found], ["non-controlling interest", "NCI"])

    def test_a_good_translation_has_no_problems(self):
        source = "Goodwill <b0></b0> NCI is <s1>tested</s1> for $5,200,000"
        target = "Lợi thế thương mại <b0></b0> NCI được <s1>kiểm tra</s1> với $5,200,000"
        self.assertEqual(handoff_tools.check(source, target, self.glossary), [])

    def test_broken_markers_numbers_and_terms_are_reported(self):
        problems = handoff_tools.check(
            "Goodwill <b0></b0> non-controlling interest is <s1>tested</s1> for $5,200,000",
            "Lợi thế thương mại lợi ích thiểu số được kiểm tra với $5.200.000",
            self.glossary,
        )
        joined = " | ".join(problems)
        self.assertIn("placeholders", joined)
        self.assertIn("5,200,000", joined)
        self.assertIn("lợi ích cổ đông không kiểm soát", joined)


class ClaudeTranslateTests(unittest.TestCase):
    """Translating from batch files: the translator writes only id<TAB>dst."""

    glossary = [{"en": "goodwill", "vi": "lợi thế thương mại", "keep_en": "0"}]

    @classmethod
    def setUpClass(cls):
        global claude_translate
        import claude_translate  # noqa: F401

    def test_answers_join_by_id_and_a_later_file_replaces_an_earlier_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "lo-001.txt").write_text("# a comment\n1\tmột\n2\thai cũ\nno tab here\n", encoding="utf-8")
            (folder / "sua-1.txt").write_text("2\thai mới\n3\tdòng\\nhai\n", encoding="utf-8")
            answers, bad = claude_translate.read_answers(folder)
        self.assertEqual(answers, {1: "một", 2: "hai mới", 3: "dòng\nhai"})
        self.assertEqual([(name, n) for name, n, _ in bad], [("lo-001.txt", 4)])

    def test_tags_numbers_and_unknown_ids_are_errors_terms_only_warnings(self):
        segments = {1: "Goodwill is <s1>tested</s1> for $5,200,000", 2: "Goodwill arises", 3: "Plain text"}
        answers = {1: "Được <s1>kiểm tra với $5.200.000", 2: "Phát sinh lợi thế", 3: "Văn bản", 9: "lạ"}
        accepted, errors, warnings = claude_translate.judge(segments, answers, self.glossary)
        self.assertEqual(sorted(accepted), [2, 3])
        self.assertEqual(sorted(errors), [1, 9])
        joined = " ".join(errors[1])
        self.assertIn("thẻ", joined)
        self.assertIn("5,200,000", joined)
        self.assertIn("lợi thế thương mại", warnings[2][0])

    def test_batches_hold_compact_lines_and_only_the_terms_they_use(self):
        items = [(1, "Goodwill  is   tested\tannually"), (2, "Line one\nline two")] + [
            (n, "word " * 10) for n in range(3, 8)]
        batches = claude_translate.make_batches(items, size=3, chars=10_000)
        self.assertEqual([[i for i, _ in b] for b in batches], [[1, 2, 3], [4, 5, 6], [7]])
        # 1 and 2 share a batch; each 49-character line then fills one alone.
        by_chars = claude_translate.make_batches(items, size=60, chars=60)
        self.assertEqual([[i for i, _ in b] for b in by_chars], [[1, 2], [3], [4], [5], [6], [7]])
        text = claude_translate.batch_text(1, 3, batches[0], self.glossary)
        self.assertIn("goodwill = lợi thế thương mại", text)
        self.assertIn("1\tGoodwill is tested\\tannually\n", text)
        self.assertIn("2\tLine one\\nline two\n", text)
        self.assertNotIn("goodwill", claude_translate.batch_text(3, 3, batches[2], self.glossary))
        self.assertEqual(claude_translate.unescape("a\\tb\\nc\\\\n"), "a\tb\nc\\n")

    def test_finish_again_after_fixing_one_batch_keeps_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, job = Path(tmp) / "base", Path(tmp) / "base" / "claude" / "book"
            (job / "answers").mkdir(parents=True)
            (base / "glossary").mkdir()
            (base / "glossary" / "g.csv").write_text("en,vi\ngoodwill,lợi thế thương mại\n", encoding="utf-8")
            sources = {1: "Revenue was 4,600", 2: "Goodwill <s1>arises</s1>", 3: "Costs were 300"}
            with (job / "segments.jsonl").open("w", encoding="utf-8") as f:
                for i, s in sources.items():
                    f.write(json.dumps({"id": i, "src": s}) + "\n")
            (job / "state.json").write_text(json.dumps({
                "base": str(base), "mon": "", "batches": [[1, 2], [3]], "pages": "",
                "original": "x.pdf", "pdf": "x.pdf"}), encoding="utf-8")
            (job / "answers" / "lo-001.txt").write_text(
                "1\tDoanh thu là 4,600\n2\tLợi thế thương mại <s1>phát sinh\n", encoding="utf-8")
            (job / "answers" / "lo-002.txt").write_text("3\tChi phí là 300\n", encoding="utf-8")

            built = []
            original_build = claude_translate.at.build
            claude_translate.at.build = lambda state, out, tr, failures: (
                built.append((dict(tr), [r["id"] for r in failures])) or {})
            try:
                with unittest.mock.patch("sys.stdout"):
                    claude_translate.main(["finish", "--job", str(job)])
                    # Only the broken answer is rewritten, in a fix file.
                    (job / "answers" / "sua-1.txt").write_text(
                        "2\tLợi thế thương mại <s1>phát sinh</s1>\n", encoding="utf-8")
                    claude_translate.main(["finish", "--job", str(job)])
            finally:
                claude_translate.at.build = original_build

            first, second = built
            self.assertEqual(set(first[0]), {sources[1], sources[3]})
            self.assertEqual(first[1], [2])
            self.assertEqual(set(second[0]), set(sources.values()))
            self.assertEqual(second[1], [])
            cached = (base / "cache" / "translations_cache.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(cached), 3)  # accepted answers are cached once each
            self.assertTrue(all(json.loads(line)["provider"] == "claude" for line in cached))

    def test_the_bundled_glossary_is_used_when_the_drive_has_none(self):
        import acca_translate

        with tempfile.TemporaryDirectory() as tmp:
            rows = acca_translate.load_glossary(tmp)
        terms = {row["en"]: row["vi"] for row in rows}
        self.assertEqual(terms["goodwill"], "lợi thế thương mại")
        self.assertEqual(terms["non-controlling interest"], "lợi ích cổ đông không kiểm soát")
        found = handoff_tools.terms_in("the  business  combination  under  common  control", rows)
        self.assertEqual([row["en"] for row in found], ["business combination", "common control"])


@unittest.skipUnless(HAS_OPENAI, "openai is installed only for the Colab workflow")
class ProviderFailoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global openai, httpx, Router, acca_translate
        import openai  # noqa: F401

        try:
            import httpx2 as httpx
        except ImportError:
            import httpx
        from router import Router
        import acca_translate

    def error(self, cls, code, message, headers=None):
        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        response = httpx.Response(code, request=request, headers=headers or {})
        return cls(message, response=response, body=None)

    def provider(self, name, script):
        class Completions:
            def create(self, **_):
                outcome = script.pop(0) if script else "ok"
                if isinstance(outcome, Exception):
                    raise outcome
                message = type("M", (), {"content": outcome})()
                return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

        client = type("Client", (), {})()
        client.chat = type("Chat", (), {"completions": Completions()})()
        return {"name": name, "client": client, "model": "m"}

    def router(self, *providers):
        clock = [0.0]
        self.slept = []

        def sleep(seconds):
            self.slept.append(round(seconds))
            clock[0] += seconds

        return Router(list(providers), sleep=sleep, clock=lambda: clock[0], log=lambda *_: None), clock

    def test_daily_quota_switches_to_the_next_provider_for_the_session(self):
        quota = self.error(openai.RateLimitError, 429, "quota GenerateRequestsPerDayPerProjectPerModel")
        router, _ = self.router(self.provider("gemini", [quota]), self.provider("groq", ["một", "hai"]))
        self.assertEqual(router.chat([]), ("một", "groq"))
        self.assertEqual(router.chat([]), ("hai", "groq"))
        self.assertTrue(router.state["gemini"]["off"])

    def test_a_per_minute_limit_waits_and_retries_the_same_provider(self):
        limit = self.error(openai.RateLimitError, 429, "slow down", {"retry-after": "7"})
        router, _ = self.router(self.provider("gemini", [limit, "xong"]), self.provider("groq", []))
        self.assertEqual(router.chat([]), ("xong", "gemini"))
        self.assertEqual(self.slept, [7])

    def test_a_request_too_large_for_one_provider_moves_on(self):
        too_big = self.error(openai.APIStatusError, 413, "Request too large")
        router, _ = self.router(self.provider("groq", [too_big]), self.provider("openrouter", ["ok"]))
        self.assertEqual(router.chat([]), ("ok", "openrouter"))

    def test_every_provider_out_raises_with_the_reasons(self):
        bad_key = self.error(openai.AuthenticationError, 401, "invalid key")
        router, _ = self.router(self.provider("gemini", [bad_key]))
        with self.assertRaisesRegex(RuntimeError, "401"):
            router.chat([])

    def test_glossary_is_filtered_by_paper_and_answers_are_parsed(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "acca.csv").write_text(
            "en,vi,mon,keep_en\n"
            "goodwill,lợi thế thương mại,F7;chung,0\n"
            "standard costing,chi phí định mức,F5,0\n"
            "NCI,,chung,1\n",
            encoding="utf-8",
        )
        rows = acca_translate.load_glossary(folder, "F7")
        self.assertEqual({row["en"] for row in rows}, {"goodwill", "NCI"})
        parsed = acca_translate.parse_json_array('```json\n[{"id": 0, "dst": "xin chào"}]\n```')
        self.assertEqual(parsed, {0: "xin chào"})

    def test_publishing_replaces_old_outputs_with_new_files(self):
        # Overwriting in place keeps a Google Drive file's id, so Drive's preview
        # went on showing the previous run. The old file must be deleted and a
        # new one created; a hard link to the old file proves it was not rewritten.
        out = Path(tempfile.mkdtemp())
        stage = Path(tempfile.mkdtemp())
        (out / "book-vi.pdf").write_text("old run")
        (out / "book-can-xem-lai.jsonl").write_text("stale review list")
        (out / "old-handle").hardlink_to(out / "book-vi.pdf")
        (stage / "book-vi.pdf").write_text("new run")
        replaced = acca_translate.publish([stage / "book-vi.pdf"], out,
                                          stale=[out / "book-can-xem-lai.jsonl"])
        self.assertEqual(replaced, 1)
        self.assertEqual((out / "book-vi.pdf").read_text(), "new run")
        self.assertEqual((out / "old-handle").read_text(), "old run")
        self.assertFalse((out / "book-can-xem-lai.jsonl").exists())

    def test_rebuilt_numbers_use_the_translation_font(self):
        fonts = acca_translate.rebuild_fonts("Cabin")
        self.assertEqual(set(fonts), {"regular", "bold", "italic", "bold-italic"})
        self.assertEqual(fonts["bold"].name, "Cabin-Bold.ttf")
        self.assertIsNone(acca_translate.rebuild_fonts("default"))

    def test_outputs_keep_only_the_selected_pages(self):
        folder = Path(tempfile.mkdtemp())
        mono = folder / "book-vi.pdf"
        doc = pymupdf.open()
        for n in range(1, 6):
            doc.new_page().insert_text((72, 72), f"page {n}")
        doc.save(mono)
        acca_translate.keep_selected_pages(mono, "2-3")
        kept = pymupdf.open(mono)
        self.assertEqual([page.get_text().strip() for page in kept], ["page 2", "page 3"])
        kept.close()

        alternating = folder / "book-vi-dual.pdf"
        doc = pymupdf.open()
        for n in range(1, 4):
            for side in ("source", "vi"):
                doc.new_page().insert_text((72, 72), f"{side} {n}")
        doc.save(alternating)
        acca_translate.keep_selected_pages(alternating, "2", per_page=2)
        kept = pymupdf.open(alternating)
        self.assertEqual([page.get_text().strip() for page in kept], ["source 2", "vi 2"])

    def test_bundled_fonts_are_passed_to_the_engine(self):
        args = acca_translate.font_args("Cabin")
        self.assertEqual(args[0], "--font")
        self.assertTrue(Path(args[1]).is_file())
        self.assertEqual(len(args), 8)
        self.assertEqual(acca_translate.font_args("default"), [])


if __name__ == "__main__":
    unittest.main()
