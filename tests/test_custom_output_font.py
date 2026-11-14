from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pdfminer.pdfinterp import PDFResourceManager

from pdf2zh import high_level
from pdf2zh.converter import TranslateConverter
from scripts import translate_pdf

FONT_VARS = tuple(high_level.CUSTOM_FONT_ENV.values()) + (high_level.SKIP_SCAN_COVER_ENV,)


class CustomOutputFontTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.regular = self.dir / "Cabin-Regular.ttf"
        self.bold = self.dir / "Cabin-Bold.ttf"
        for face in (self.regular, self.bold):
            face.write_bytes(b"not parsed in these tests")
        cleared = {name: "" for name in FONT_VARS}
        patcher = mock.patch.dict(os.environ, cleared)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_variables_means_no_custom_family(self):
        self.assertEqual(high_level.custom_output_fonts(), {})

    def test_regular_and_bold_are_used_and_missing_faces_left_to_synthesis(self):
        os.environ["VI_FONT_REGULAR"] = str(self.regular)
        os.environ["VI_FONT_BOLD"] = str(self.bold)
        faces = high_level.custom_output_fonts()
        self.assertEqual(faces, {0: str(self.regular), 1: str(self.bold)})

        paths = high_level.output_style_font_paths("vi", str(self.regular))
        self.assertEqual(paths[1], str(self.bold))
        # Italic faces were not given: they fall back to the regular file,
        # which the engine then slants itself.
        self.assertEqual(paths[2], str(self.regular))
        self.assertEqual(paths[3], str(self.regular))

    def test_a_mistyped_regular_face_falls_back_to_the_default_font(self):
        os.environ["VI_FONT_REGULAR"] = str(self.dir / "missing.ttf")
        os.environ["VI_FONT_BOLD"] = str(self.bold)
        self.assertEqual(high_level.custom_output_fonts(), {})

    def test_the_custom_regular_face_replaces_the_downloaded_font(self):
        os.environ["VI_FONT_REGULAR"] = str(self.regular)
        with mock.patch.object(high_level, "get_font_and_metadata") as download:
            chosen = high_level.download_remote_fonts("vi")
        download.assert_not_called()
        self.assertEqual(Path(chosen), self.regular)

    def test_converter_defaults_keep_existing_behaviour(self):
        device = TranslateConverter(PDFResourceManager(), service="google")
        self.assertFalse(device.uniform_output_font)
        self.assertTrue(device.cover_scanned_text)


class CustomOutputFontCliTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.source = self.dir / "guide.pdf"
        self.source.write_bytes(b"%PDF-1.4")
        self.regular = self.dir / "Cabin-Regular.ttf"
        self.regular.write_bytes(b"font")
        patcher = mock.patch.dict(os.environ, {name: "" for name in FONT_VARS})
        patcher.start()
        self.addCleanup(patcher.stop)

    def parse(self, *extra: str):
        return translate_pdf._parser().parse_args(
            [str(self.source), "--output-dir", str(self.dir), *extra]
        )

    def test_cli_and_engine_agree_on_variable_names(self):
        self.assertEqual(translate_pdf.CUSTOM_FONT_ENV, high_level.CUSTOM_FONT_ENV)
        self.assertEqual(translate_pdf.SKIP_SCAN_COVER_ENV, high_level.SKIP_SCAN_COVER_ENV)

    def test_font_options_reach_the_engine_environment(self):
        args = self.parse("--font", str(self.regular), "--no-scan-cover")
        translate_pdf._validate_arguments(args)
        translate_pdf._apply_output_options(args)
        self.assertEqual(Path(os.environ["VI_FONT_REGULAR"]), self.regular.resolve())
        self.assertEqual(os.environ["VI_SKIP_SCAN_COVER"], "1")
        self.assertEqual(os.environ["VI_FONT_BOLD"], "")

    def test_a_style_face_without_a_regular_face_is_refused(self):
        args = self.parse("--font-bold", str(self.regular))
        with self.assertRaisesRegex(translate_pdf.TranslationError, "need --font"):
            translate_pdf._validate_arguments(args)

    def test_a_missing_font_file_is_refused_before_translating(self):
        args = self.parse("--font", str(self.dir / "nope.ttf"))
        with self.assertRaisesRegex(translate_pdf.TranslationError, "font file not found"):
            translate_pdf._validate_arguments(args)


if __name__ == "__main__":
    unittest.main()
