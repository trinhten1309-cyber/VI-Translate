"""Regressions for the Google Colab workflow in colab/."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pymupdf

COLAB = Path(__file__).resolve().parents[1] / "colab"
sys.path.insert(0, str(COLAB))

import handoff_tools  # noqa: E402
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
