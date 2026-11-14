"""Rebuild colab/fonts/*.ttf: static Cabin faces (Regular/Bold/Italic/BoldItalic,
normal + SemiCondensed) cut from the Google Fonts variable files (SIL OFL 1.1).

The repository already ships the result; run this only to refresh it:
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google/fonts gf
    git -C gf sparse-checkout set ofl/cabin
    python colab/make_cabin.py gf/ofl/cabin colab/fonts
"""
import sys
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for fname, italic in [("Cabin[wdth,wght].ttf", False), ("Cabin-Italic[wdth,wght].ttf", True)]:
    for wght in (400, 700):
        for wdth, wtag in ((100, ""), (87.5, "SemiCondensed")):
            inst = instantiateVariableFont(TTFont(src / fname), {"wght": wght, "wdth": wdth},
                                           updateFontNames=False)
            style = ("Bold" if wght == 700 else "") + ("Italic" if italic else "") or "Regular"
            family = "Cabin" + (" " + wtag if wtag else "")
            for rec in inst["name"].names:
                if rec.nameID in (1, 16):
                    rec.string = family
                elif rec.nameID in (2, 17):
                    rec.string = style
                elif rec.nameID == 4:
                    rec.string = f"{family} {style}"
                elif rec.nameID == 6:
                    rec.string = f"{family.replace(' ', '')}-{style}"
            inst.save(out / f"Cabin{wtag}-{style}.ttf")
print("Built", len(list(out.glob("*.ttf"))), "Cabin font files in", out)
