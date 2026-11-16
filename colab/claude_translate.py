"""Translate a PDF with Claude (or any person) as the translator, no API key.

  prepare : the notebook's steps up to the API call (OCR for scans, rebuild,
            segment extraction), then batch files of `id<TAB>src` lines, each
            headed by the glossary terms it contains
  finish  : read the answers (`id<TAB>dst` lines), check tags, numbers and
            terms, build the translated and side-by-side PDFs, optionally
            render a few pages to small PNGs

The translator writes only ids and Vietnamese; the English is never copied
back, which is most of the saving over a JSONL of {"src", "dst"} records.
Answers stay on disk, so after fixing a few ids `finish` runs again without
touching the other batches. Accepted answers go to the same cache as the
notebook's, so a later prepare of the same text skips them.

    python colab/claude_translate.py prepare --base DIR --pdf FILE [--pages 5-6]
    python colab/claude_translate.py finish --job DIR/claude/<stem> [--render 5,6]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acca_translate as at  # noqa: E402
from handoff_tools import check, terms_in  # noqa: E402

BATCH_SEGMENTS = 60
BATCH_CHARS = 12000


# ---------------------------------------------------------------- file format
def escape(text):
    """One segment on one line: backslash, tab and newline become \\\\, \\t, \\n."""
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def unescape(text):
    return re.sub(r"\\(.)", lambda m: {"t": "\t", "n": "\n"}.get(m.group(1), m.group(1)), text)


def compact(src):
    """What the translator reads: runs of spaces from justified lines collapsed."""
    return re.sub(r" {2,}", " ", src.strip())


def make_batches(items, size=BATCH_SEGMENTS, chars=BATCH_CHARS):
    """Split [(id, src)] into batches of at most SIZE segments and about CHARS characters."""
    batches, current, used = [], [], 0
    for item in items:
        n = len(compact(item[1]))
        if current and (len(current) >= size or used + n > chars):
            batches.append(current)
            current, used = [], 0
        current.append(item)
        used += n
    if current:
        batches.append(current)
    return batches


def batch_text(number, total, items, glossary):
    lines = [f"# Lô {number}/{total}. Trả lời: answers/lo-{number:03d}.txt, mỗi dòng id<TAB>bản dịch"]
    found = {}
    for _, src in items:
        for r in terms_in(src, glossary):
            found[r["en"]] = r
    if found:
        lines.append("# Thuật ngữ: " + "; ".join(
            f"{r['en']} = {'giữ nguyên' if r['keep_en'] == '1' else r['vi']}" for r in found.values()))
    lines += [f"{i}\t{escape(compact(src))}" for i, src in items]
    return "\n".join(lines) + "\n"


def read_answers(folder):
    """{id: dst} from every *.txt in FOLDER, in name order, so a later file
    (sua-*.txt after lo-*.txt) replaces an earlier answer. Also returns the
    lines that could not be read, as (file, line number, reason)."""
    answers, bad = {}, []
    for path in sorted(Path(folder).glob("*.txt")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, tab, dst = line.partition("\t")
            if not tab or not key.strip().isdigit():
                bad.append((path.name, n, "không có dạng id<TAB>bản dịch"))
                continue
            answers[int(key)] = unescape(dst.strip())
    return answers, bad


SHORT = (
    (re.compile(r"^empty translation$"), "trống"),
    (re.compile(r"^formula placeholders"), "thẻ <bN> bị đổi"),
    (re.compile(r"^number (.+) missing or reformatted$"), r"thiếu/đổi số \1"),
    (re.compile(r'^term "(.+)" should be "(.+)"$'), r"thuật ngữ \1 → \2"),
)


def short_reason(problem):
    for pattern, text in SHORT:
        if pattern.search(problem):
            return pattern.sub(text, problem)
    return "thẻ <sN> sai" if "<s" in problem or "style" in problem.lower() else problem


def judge(segments, answers, glossary):
    """Check every answer. Returns (accepted {id: dst}, errors {id: [reasons]},
    warnings {id: [reasons]}). A wrong tag, number or an empty answer rejects
    it (the segment stays English); a term that differs from the glossary is
    only a warning, so a natural inflection does not cost another round."""
    accepted, errors, warnings = {}, {}, {}
    for i, dst in answers.items():
        if i not in segments:
            errors[i] = ["id không có trong lô nào"]
            continue
        problems = check(segments[i], dst, glossary)
        hard = [short_reason(p) for p in problems if not p.startswith("term ")]
        soft = [short_reason(p) for p in problems if p.startswith("term ")]
        if hard:
            errors[i] = hard
        else:
            accepted[i] = dst
            if soft:
                warnings[i] = soft
    return accepted, errors, warnings


# ---------------------------------------------------------------- commands
def cmd_prepare(a):
    base = Path(a.base)
    job = Path(a.job) if a.job else base / "claude" / Path(a.pdf).stem
    answers = job / "answers"
    if answers.is_dir() and any(answers.glob("*.txt")) and not a.fresh:
        sys.exit(f"❌ {answers} đã có câu trả lời. Chạy finish, hoặc thêm --fresh để chuẩn bị lại từ đầu.")
    state = at.extract(a)
    srcs = state.pop("srcs")
    glossary = at.load_glossary(base / "glossary", a.mon)
    cache = {} if a.force else at.load_cache(base / "cache" / "translations_cache.jsonl")

    batches_dir = job / "batches"
    for folder in (batches_dir, answers):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.txt"):
            old.unlink()
    segments = {i: s for i, s in enumerate(srcs, 1)}
    todo = [(i, s) for i, s in segments.items() if s not in cache]
    batches = make_batches(todo, a.batch, a.chars)
    for n, items in enumerate(batches, 1):
        (batches_dir / f"lo-{n:03d}.txt").write_text(
            batch_text(n, len(batches), items, glossary), encoding="utf-8")
    with (job / "segments.jsonl").open("w", encoding="utf-8") as f:
        for i, s in segments.items():
            f.write(json.dumps({"id": i, "src": s}, ensure_ascii=False) + "\n")
    state.update(base=str(base), mon=a.mon, batches=[[i for i, _ in b] for b in batches])
    (job / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    chars = sum(len(compact(s)) for _, s in todo)
    print(f"\n✅ {len(srcs)} đoạn: {len(todo)} cần dịch, {len(srcs) - len(todo)} có sẵn trong cache"
          f" → {len(batches)} lô, {chars:,} ký tự; thuật ngữ: {len(glossary)} mục")
    for n in range(1, len(batches) + 1):
        print(f"   {batches_dir / f'lo-{n:03d}.txt'}")
    print(f"   Trả lời vào: {answers}/lo-NNN.txt (mỗi dòng id<TAB>bản dịch)")
    print(f"   Rồi chạy: python colab/claude_translate.py finish --job {job}")


def render_pages(pdf, pages, kept, count, folder, dpi):
    """PNG of each page in PAGES (source numbering) from PDF, which holds only
    the KEPT pages of a COUNT-page source (all of them when KEPT is empty)."""
    import pymupdf as fitz
    import rebuild_text_layer as rtl

    doc = fitz.open(pdf)
    keep = sorted(rtl.parse_pages(kept, count))
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(rtl.parse_pages(pages, count)):
        if p in keep and keep.index(p) < doc.page_count:
            path = folder / f"trang-{p + 1}.png"
            doc[keep.index(p)].get_pixmap(dpi=dpi).save(path)
            out.append(path)
    return out


def cmd_finish(a):
    job = Path(a.job)
    state = json.loads((job / "state.json").read_text(encoding="utf-8"))
    base = Path(state["base"])
    segments = {r["id"]: r["src"] for r in map(json.loads, (job / "segments.jsonl").open(encoding="utf-8"))}
    state["srcs"] = list(segments.values())
    glossary = at.load_glossary(base / "glossary", state.get("mon", ""))
    answers, bad = read_answers(job / "answers")
    accepted, errors, warnings = judge(segments, answers, glossary)
    asked = [i for batch in state["batches"] for i in batch]
    unanswered = [i for i in asked if i not in answers]

    print(f"Trả lời: {len(answers)}/{len(asked)} id; nhận {len(accepted)}, lỗi {len(errors)}, "
          f"thiếu {len(unanswered)}, cảnh báo thuật ngữ {len(warnings)}")
    for name, n, why in bad:
        print(f"   ✗ {name}:{n} {why}")
    for i in sorted(errors):
        print(f"   ✗ {i}: {'; '.join(errors[i])}")
    if unanswered:
        print(f"   ✗ thiếu: {compact_ids(unanswered)}")
    for i in sorted(warnings):
        print(f"   ⚠ {i}: {'; '.join(warnings[i])}")
    if a.check_only:
        return

    cache_path = base / "cache" / "translations_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = at.load_cache(cache_path)
    translations = {s: cache[s][0] for s in segments.values() if s in cache}
    with cache_path.open("a", encoding="utf-8") as cf:
        for i, dst in accepted.items():
            src = segments[i]
            translations[src] = dst
            if cache.get(src, ("",))[0] != dst:
                cf.write(json.dumps({"src": src, "dst": dst, "provider": "claude",
                                     "t": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False) + "\n")
    failures = [{"id": i, "src": segments[i], "dst": answers[i], "problems": errors[i]}
                for i in sorted(errors) if i in segments]

    print("Dựng PDF...")
    out_dir = base / "output"
    left = at.build(state, out_dir, translations, failures)
    ids = {s: i for i, s in segments.items()}
    by_reason = {}
    for src, why in left.items():
        by_reason.setdefault(why, []).append(ids.get(src, 0))
    print(f"\n{'✅' if not left else '⚠️ '} {len(segments) - len(left)}/{len(segments)} đoạn đã dịch")
    for why, group in by_reason.items():
        print(f"   còn tiếng Anh ({why}): {compact_ids(group)}")
    if a.render:
        import pymupdf as fitz

        stem = Path(state["pdf"]).stem
        with fitz.open(state["original"]) as src:
            count = src.page_count
        pngs = render_pages(out_dir / f"{stem}-vi.pdf", a.render, state["pages"], count, job / "png", a.dpi)
        for p in pngs:
            print(f"   ảnh: {p}")


def compact_ids(ids):
    """[3, 4, 5, 9] -> '3-5, 9'"""
    ids, parts = sorted(ids), []
    for i in ids:
        if parts and i == parts[-1][1] + 1:
            parts[-1][1] = i
        else:
            parts.append([i, i])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="tách đoạn và xuất các file lô id<TAB>src")
    at.add_source_args(p)
    p.add_argument("--job", default="", help="thư mục làm việc (mặc định BASE/claude/<tên file>)")
    p.add_argument("--batch", type=int, default=BATCH_SEGMENTS, help="số đoạn tối đa mỗi lô")
    p.add_argument("--chars", type=int, default=BATCH_CHARS, help="số ký tự tối đa mỗi lô")
    p.add_argument("--force", action="store_true", help="dịch lại cả đoạn đã có trong cache")
    p.add_argument("--fresh", action="store_true", help="xoá câu trả lời cũ, chuẩn bị lại từ đầu")
    f = sub.add_parser("finish", help="kiểm tra câu trả lời, dựng PDF và bản song ngữ")
    f.add_argument("--job", required=True)
    f.add_argument("--check-only", action="store_true", help="chỉ kiểm tra câu trả lời, không dựng PDF")
    f.add_argument("--render", default="", help="trang cần xuất PNG để xem, ví dụ 5,6")
    f.add_argument("--dpi", type=int, default=60)
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "finish": cmd_finish}[a.cmd](a)


if __name__ == "__main__":
    main()
