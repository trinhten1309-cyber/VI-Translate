"""ACCA Translate runner for Google Colab - runs inside the VI-Translate venv.

  test : send one test sentence to every configured provider
  run  : (rebuild image PDFs) -> extract segments -> translate with a glossary,
         provider failover and checks -> build the translated and bilingual PDFs

API keys come from environment variables (GEMINI_API_KEY, CEREBRAS_API_KEY,
GROQ_API_KEY, OPENROUTER_API_KEY). Optional per provider: <NAME>_MODEL,
<NAME>_BASE_URL. PROVIDER_ORDER="cerebras,gemini,..." changes priority.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = Path(os.environ.get("VI_REPO", TOOLS.parent))
FONTS = TOOLS / "fonts"
FONT_STYLES = ("Cabin", "CabinSemiCondensed", "default")
sys.path[:0] = [str(TOOLS), str(REPO)]

from openai import OpenAI  # noqa: E402
from router import Router  # noqa: E402
from handoff_tools import terms_in, check  # noqa: E402
import rebuild_text_layer as rtl  # noqa: E402
import ocr_layer  # noqa: E402

PROVIDERS = {  # name: (base url, default model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash"),
    "cerebras": ("https://api.cerebras.ai/v1", "gpt-oss-120b"),
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "openrouter": ("https://openrouter.ai/api/v1", ""),
}

SYSTEM = """Bạn là biên dịch viên chuyên ngành kế toán – kiểm toán, dịch tài liệu ôn thi ACCA từ tiếng Anh sang tiếng Việt cho sinh viên Việt Nam.

QUY TẮC BẮT BUỘC
1. Dùng đúng thuật ngữ trong BẢNG THUẬT NGỮ (nếu có). Không tự đổi sang từ đồng nghĩa.
2. Giữ nguyên: tên chuẩn mực (IFRS 3, IAS 36…), viết tắt (NCI, PUP, WACC…), tên công ty (P Co, S Co), năm giả định (20X4), mọi con số và cách viết số ($5,200,000 giữ nguyên dấu phẩy).
3. Giữ nguyên tuyệt đối các thẻ <b0></b0>, <b1></b1>…: cùng số lượng, cùng thứ tự.
4. Thẻ <s1>…</s1> (đậm), <s2>…</s2> (nghiêng), <s3>…</s3> (đậm nghiêng): giữ đủ cặp, đặt quanh cụm từ tiếng Việt tương ứng.
5. Văn phong giáo trình: rõ, gọn; không thêm giải thích, không bỏ ý.
6. Tiêu đề: giữ nguyên số mục, chỉ dịch phần chữ.
7. Văn bản có thể đến từ OCR và có lỗi nhận dạng nhỏ (ví dụ "NCl" thay cho "NCI", "0" thay cho "O"): hiểu và dịch theo nghĩa đúng, nhưng không thay đổi con số.

ĐẦU VÀO: mảng JSON [{"id": ..., "src": ...}].
ĐẦU RA: CHỈ một mảng JSON [{"id": ..., "dst": ...}], cùng số phần tử, cùng id. Không markdown, không chú thích."""


# ---------------------------------------------------------------- providers
def build_router():
    order = [n.strip() for n in os.environ.get("PROVIDER_ORDER", ",".join(PROVIDERS)).split(",") if n.strip()]
    providers = []
    for name in order:
        if name not in PROVIDERS:
            print(f"⚠️  Không biết nguồn '{name}', bỏ qua")
            continue
        url, model = PROVIDERS[name]
        key = os.environ.get(f"{name.upper()}_API_KEY")
        model = os.environ.get(f"{name.upper()}_MODEL") or model
        url = os.environ.get(f"{name.upper()}_BASE_URL") or url
        if not key:
            print(f"⏭️  {name}: chưa có key, bỏ qua")
            continue
        if not model:
            print(f"⏭️  {name}: chưa điền mô hình, bỏ qua")
            continue
        providers.append({"name": name, "model": model,
                          "client": OpenAI(api_key=key, base_url=url, max_retries=0, timeout=180)})
    if not providers:
        sys.exit("❌ Không có nguồn API nào dùng được — kiểm tra Secrets")
    print("Thứ tự ưu tiên:", " → ".join(f"{p['name']} ({p['model']})" for p in providers))
    return Router(providers)


# ---------------------------------------------------------------- glossary
def load_glossary(folder, mon=""):
    rows = []
    for path in sorted(Path(folder).glob("*.csv")):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
                if not r.get("en") or not (r.get("vi") or r.get("keep_en") == "1"):
                    continue
                tags = {t.strip().lower() for t in r.get("mon", "").replace(",", ";").split(";") if t.strip()}
                if mon and tags and not tags & {mon.lower(), "chung", "all"}:
                    continue
                rows.append({"en": r["en"], "vi": r.get("vi") or r["en"], "keep_en": r.get("keep_en", "0")})
    seen, unique = set(), []
    for r in sorted(rows, key=lambda r: -len(r["en"])):  # longest first; first file wins on duplicates
        if r["en"].lower() not in seen:
            seen.add(r["en"].lower())
            unique.append(r)
    return unique


def glossary_block(texts, glossary):
    found = {}
    for t in texts:
        for r in terms_in(t, glossary):
            found[r["en"]] = r
    if not found:
        return ""
    lines = [f"- {r['en']} → giữ nguyên \"{r['en']}\"" if r["keep_en"] == "1" else f"- {r['en']} → {r['vi']}"
             for r in found.values()]
    return "BẢNG THUẬT NGỮ:\n" + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------- translation
def parse_json_array(text):
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b <= a:
        raise ValueError("không thấy mảng JSON trong câu trả lời")
    data = json.loads(text[a:b + 1])
    return {int(d["id"]): str(d.get("dst", "")) for d in data if isinstance(d, dict) and "id" in d}


def call(router, items, glossary, feedback=""):
    payload = json.dumps([{"id": i, "src": s} for i, s in items], ensure_ascii=False)
    user = glossary_block([s for _, s in items], glossary) + feedback + "ĐOẠN CẦN DỊCH:\n" + payload
    text, used = router.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return parse_json_array(text), used


def translate_items(router, items, glossary, results, failures, depth=0):
    """items: list of (id, src). Fills results[src] = (dst, provider)."""
    try:
        got, used = call(router, items, glossary)
    except (ValueError, json.JSONDecodeError) as e:
        if len(items) > 1:  # malformed answer: split the batch and try again
            mid = len(items) // 2
            translate_items(router, items[:mid], glossary, results, failures, depth + 1)
            translate_items(router, items[mid:], glossary, results, failures, depth + 1)
        else:
            failures.append({"src": items[0][1], "problems": [f"trả lời không đúng định dạng: {e}"]})
        return
    for i, src in items:
        dst = got.get(i, "")
        problems = check(src, dst, glossary)
        if not problems:
            results[src] = (dst, used)
            continue
        # one retry for this segment, telling the model what was wrong
        fb = "BẢN DỊCH TRƯỚC BỊ LỖI: " + "; ".join(problems) + ". Hãy dịch lại và sửa đúng các lỗi này.\n\n"
        try:
            got2, used2 = call(router, [(i, src)], glossary, fb)
            dst2 = got2.get(i, "")
            problems2 = check(src, dst2, glossary)
            if not problems2:
                results[src] = (dst2, used2)
                continue
            problems, dst = problems2, dst2
        except (ValueError, json.JSONDecodeError):
            pass
        failures.append({"src": src, "dst": dst, "problems": problems})


# ---------------------------------------------------------------- pipeline
def run_repo(args, hint=""):
    cmd = [sys.executable, str(REPO / "scripts" / "translate_pdf.py"), *args]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    noise = ("onnxruntime", "%|", "[RapidOCR]", "not in git repo")
    lines = [l for l in (p.stdout + p.stderr).splitlines() if not any(n in l for n in noise)]
    if p.returncode != 0:
        print("\n".join(lines[-15:]))
        sys.exit(f"❌ translate_pdf.py lỗi (mã {p.returncode})" + (f"\n   {hint}" if hint else ""))
    return lines


def make_dual(original, translated, out_path, mode, pages=""):
    """Bilingual PDF from the TRUE original pages + the translated pages."""
    import pymupdf as fitz
    src, tr = fitz.open(original), fitz.open(translated)
    keep = sorted(rtl.parse_pages(pages, src.page_count))
    out = fitz.open()
    for i in keep:
        r = src[i].rect
        if mode == "side-by-side":
            page = out.new_page(width=r.width * 2, height=r.height)
            page.show_pdf_page(fitz.Rect(0, 0, r.width, r.height), src, i)
            page.show_pdf_page(fitz.Rect(r.width, 0, r.width * 2, r.height), tr, i)
        else:  # alternating
            out.insert_pdf(src, from_page=i, to_page=i)
            out.insert_pdf(tr, from_page=i, to_page=i)
    out.save(out_path, garbage=3, deflate=True)


def font_args(style):
    """--font* options for one of the bundled families ('default' = engine font)."""
    if style == "default":
        return []
    faces = {"--font": "Regular", "--font-bold": "Bold",
             "--font-italic": "Italic", "--font-bold-italic": "BoldItalic"}
    args = []
    for flag, face in faces.items():
        path = FONTS / f"{style}-{face}.ttf"
        if path.is_file():
            args += [flag, str(path)]
    return args


def load_cache(path):
    cache = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                cache[r["src"]] = (r["dst"], r.get("provider", "?"))
            except (json.JSONDecodeError, KeyError):
                pass
    return cache


def cmd_run(a):
    base, pdf = Path(a.base), Path(a.pdf)
    work = (Path("/content/work") if Path("/content").is_dir() else REPO / "tmp" / "colab-work") / pdf.stem
    work.mkdir(parents=True, exist_ok=True)
    pages = ["--pages", a.pages] if a.pages else []
    original = pdf

    # PDF dạng ảnh: (a) trang chỉ có ảnh -> nhận dạng chữ (OCR) thành lớp chữ ẩn;
    # (b) trang có lớp chữ ẩn -> xoá chữ trong ảnh, dựng lại thành chữ thật.
    source, rebuilt, engine_ocr = original, [], []
    if a.ocr == "auto":
        scans = ocr_layer.image_only_pages(original, a.pages)
        if scans:
            print(f"⓪ {len(scans)} trang chỉ có ảnh (scan) → nhận dạng chữ OCR, mỗi trang vài giây...")
            source = work / f"{pdf.stem}-ocr.pdf"
            ocr_layer.add_ocr_layer(
                original, source, a.pages,
                on_page=lambda n, total, page, kept, skipped: print(
                    f"   OCR {n}/{total}: trang {page} — {kept} dòng"
                    + (f", {skipped} dòng nhận dạng không chắc nên giữ nguyên ảnh" if skipped else "")))
    elif a.ocr == "engine":
        engine_ocr = ["--ocr", "standard"]
    if a.image_pdf != "off" or source != original:
        need = rtl.needs_rebuild(source, a.pages)
        if need:
            print(f"⓪ {len(need)} trang dạng ảnh có lớp chữ → xoá chữ trong ảnh, dựng lại thành chữ thật...")
            pdf = work / pdf.name
            rebuilt = rtl.rebuild(source, pdf, a.pages)
            print(f"   đã dựng lại {len(rebuilt)} trang")

    print(f"① Tách đoạn: {pdf.name} {('trang ' + a.pages) if a.pages else '(cả file)'}")
    seg_file = work / "segments.jsonl"
    seg_file.unlink(missing_ok=True)
    run_repo([str(pdf), *pages, "--engine", "handoff", "--emit-segments", str(seg_file),
              "--output-dir", str(work / "extract"), *engine_ocr, "--overwrite"],
             hint="Nếu PDF là ảnh scan: để OCR = auto (hoặc engine) rồi chạy lại.")
    srcs = list(dict.fromkeys(json.loads(l)["src"] for l in seg_file.open(encoding="utf-8")))
    print(f"   {len(srcs)} đoạn cần dịch")

    glossary = load_glossary(base / "glossary", a.mon)
    print(f"② Bảng thuật ngữ: {len(glossary)} mục" + (f" (lọc môn {a.mon})" if a.mon else "")
          + ("" if glossary else " — chưa có, dịch không kèm thuật ngữ"))

    cache_path = base / "cache" / "translations_cache.jsonl"
    cache = {} if a.force else load_cache(cache_path)
    todo = [s for s in srcs if s not in cache]
    print(f"③ Dịch: {len(srcs) - len(todo)} đoạn có sẵn trong bộ nhớ đệm, {len(todo)} đoạn mới")

    failures, stopped = [], None
    if todo:
        router = build_router()
        batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
        with cache_path.open("a", encoding="utf-8") as cf:
            for n, batch in enumerate(batches, 1):
                results, fails = {}, []
                try:
                    translate_items(router, list(enumerate(batch)), glossary, results, fails)
                except RuntimeError as e:
                    stopped = str(e)
                    print(f"⛔ Dừng ở lô {n}/{len(batches)}: {e}")
                for src, (dst, used) in results.items():
                    cache[src] = (dst, used)
                    cf.write(json.dumps({"src": src, "dst": dst, "provider": used,
                                         "t": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False) + "\n")
                cf.flush()
                failures += fails
                by = {}
                for _, used in results.values():
                    by[used] = by.get(used, 0) + 1
                print(f"   lô {n}/{len(batches)}: ✓ {len(results)} " + str(by) + (f"  ✗ {len(fails)}" if fails else ""))
                if stopped:
                    break
        router.report()

    print("④ Dựng PDF")
    tr_file = work / "translations.jsonl"
    with tr_file.open("w", encoding="utf-8") as f:
        for s in srcs:
            if s in cache:
                f.write(json.dumps({"src": s, "dst": cache[s][0]}, ensure_ascii=False) + "\n")
    out_dir = base / "output"
    missing = work / "still-missing.jsonl"
    missing.unlink(missing_ok=True)
    own_dual = bool(rebuilt) and a.bilingual in ("side-by-side", "alternating")
    bil = [] if a.bilingual == "off" or own_dual else ["--bilingual", a.bilingual]
    extra = font_args(a.style) + (["--no-scan-cover"] if rebuilt else []) + engine_ocr
    lines = run_repo([str(pdf), *pages, "--engine", "handoff", "--segments", str(tr_file),
                      "--output-dir", str(out_dir), "--emit-segments", str(missing), *bil,
                      *extra, "--overwrite"])
    for l in lines:
        if "PDF:" in l or "untranslated" in l:
            print("   " + l.replace(str(base) + "/", ""))
    if own_dual:
        mono = out_dir / f"{pdf.stem}-vi.pdf"
        dual = out_dir / f"{pdf.stem}-vi-dual.pdf"
        make_dual(original, mono, dual, a.bilingual, a.pages)
        print(f"   Bilingual PDF ({a.bilingual}, trang gốc thật): output/{dual.name}")

    n_missing = sum(1 for _ in missing.open(encoding="utf-8")) if missing.exists() else 0
    if failures:
        fp = out_dir / f"{pdf.stem}-can-xem-lai.jsonl"
        with fp.open("w", encoding="utf-8") as f:
            for r in failures:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"   {len(failures)} đoạn không qua kiểm tra → output/{fp.name}")
    print(f"\n{'✅' if not n_missing else '⚠️ '} Xong: {len(srcs) - n_missing}/{len(srcs)} đoạn đã dịch."
          + (f" {n_missing} đoạn còn tiếng Anh — chạy lại ô này để dịch bù." if n_missing else ""))
    if stopped:
        print("   Các nguồn đã hết lượt; phần đã dịch được lưu trong cache, lần chạy sau sẽ tiếp tục.")


def cmd_test(_):
    router = build_router()
    src = "Goodwill is tested for <s1>impairment</s1> at least annually under IAS 36."
    glossary = [{"en": "impairment", "vi": "suy giảm giá trị", "keep_en": "0"}]
    for p in router.providers:
        solo = Router([p], log=lambda *_: None)
        try:
            got, _ = call(solo, [(0, src)], glossary)
            dst = got.get(0, "")
            probs = check(src, dst, glossary)
            print(f"{'✅' if not probs else '⚠️ '} {p['name']}: {dst}" + (f"   ({'; '.join(probs)})" if probs else ""))
        except Exception as e:
            print(f"❌ {p['name']}: {type(e).__name__} — {str(e)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--pdf", required=True)
    r.add_argument("--pages", default="")
    r.add_argument("--bilingual", default="side-by-side")
    r.add_argument("--batch", type=int, default=25)
    r.add_argument("--mon", default="")
    r.add_argument("--force", action="store_true")
    r.add_argument("--style", default="Cabin", choices=FONT_STYLES,
                   help="font chữ dịch: Cabin, CabinSemiCondensed (hẹp hơn) hoặc default")
    r.add_argument("--ocr", default="auto", choices=["auto", "engine", "off"],
                   help="auto: trang chỉ có ảnh được OCR rồi dựng lại thành chữ (dịch cả bảng); "
                        "engine: OCR thận trọng của repo (chỉ đoạn văn); off: tắt")
    r.add_argument("--image-pdf", default="auto", choices=["auto", "off"],
                   help="auto: tự dựng lại lớp chữ cho PDF dạng ảnh có OCR ẩn")
    a = ap.parse_args()
    {"test": cmd_test, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
