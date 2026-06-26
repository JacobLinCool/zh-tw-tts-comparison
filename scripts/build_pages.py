"""Build the GitHub Pages showcase into site/.

Generates the data the static pages need, all streamed from the HF audio dataset CDN:
  - site/manifest.json  : 25 sentences x models x conditions, audio = HF resolve URLs
  - site/plots/*.png    : copied from outputs/plots
  - site/results.html   : outputs/REPORT.md rendered to HTML

The committed static pages (site/index.html browse, site/leaderboard.html live ranking)
read manifest.json; leaderboard.html additionally reads the votes dataset live via the
datasets-server /rows API. Serve site/ directly:

    uv run --group site python scripts/build_pages.py
    python -m http.server -d site 8000   # open http://localhost:8000/
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tts_compare.config import audio_url  # noqa: E402

RESULTS = ROOT / "outputs" / "results.jsonl"
QUICK = ROOT / "data" / "quick_test" / "sentences.jsonl"
PLOTS_SRC = ROOT / "outputs" / "plots"
REPORT = ROOT / "outputs" / "REPORT.md"
SITE = ROOT / "site"

MODEL_ORDER = ["breezyvoice", "omnivoice", "cosyvoice3", "voxcpm2", "moss", "qwen3tts", "chatterbox"]


def _round(v):
    return round(v, 3) if v is not None else None


def build_manifest() -> None:
    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sents = {json.loads(ln)["id"]: json.loads(ln)
             for ln in QUICK.read_text(encoding="utf-8").splitlines() if ln.strip()}

    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    models += sorted({r["model"] for r in rows} - set(models))
    present = {r["condition"] for r in rows}
    conditions = [c for c in ["raw", "controlled", "ensub", "ensub_ctrl"] if c in present]

    clips = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if "wav" not in r:
            continue
        asr = r.get("asr", {})
        clips[r["id"]][r["model"]][r["condition"]] = {
            "wav": audio_url(r["model"], r["condition"], r["id"]),  # HF CDN URL
            "mer_q": _round(asr.get("qwen3", {}).get("mer")),
            "mer_b": _round(asr.get("breeze", {}).get("mer")),
            "hyp_q": asr.get("qwen3", {}).get("hyp"),
            "hyp_b": asr.get("breeze", {}).get("hyp"),
            "synth_s": r.get("synth_s"),
            "error": r.get("error"),
        }

    sentences = []
    for sid in sorted(clips, key=lambda s: (sents.get(s, {}).get("bucket", ""), s)):
        m = sents.get(sid, {})
        sentences.append({
            "id": sid, "text": m.get("text", ""), "bucket": m.get("bucket", ""),
            "register": m.get("register", ""), "est_duration_s": m.get("est_duration_s"),
            "entities": [a["surface"] for a in m.get("annotations", [])],
            "clips": clips[sid],
        })

    SITE.mkdir(parents=True, exist_ok=True)
    manifest = {"models": models, "conditions": conditions, "sentences": sentences}
    (SITE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"manifest.json: {len(models)} models, {len(sentences)} sentences, "
          f"{sum(len(c) for s in sentences for c in s['clips'].values())} clips")


def copy_plots() -> None:
    dst = SITE / "plots"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in sorted(PLOTS_SRC.glob("*.png")):
        shutil.copy2(png, dst / png.name)
        n += 1
    print(f"plots/: {n} png copied")


REPORT_TEMPLATE = """<!doctype html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS comparison — results</title>
<style>
 :root {{ --bg:#0f1115; --card:#171a21; --line:#272b35; --fg:#e6e8ee; --mut:#9aa3b2; --acc:#6ea8fe; }}
 body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.7 system-ui,-apple-system,"Segoe UI",sans-serif; }}
 header {{ position:sticky; top:0; background:#0f1115ee; backdrop-filter:blur(8px); border-bottom:1px solid var(--line); padding:10px 16px; }}
 header a {{ color:var(--acc); text-decoration:none; margin-right:14px; }}
 main {{ max-width:920px; margin:0 auto; padding:24px 18px 80px; }}
 h1,h2,h3 {{ line-height:1.3; }} h1 {{ font-size:24px; }} h2 {{ font-size:19px; margin-top:32px; border-top:1px solid var(--line); padding-top:18px; }}
 a {{ color:var(--acc); }} code {{ background:#0d0f13; padding:1px 5px; border-radius:4px; font-size:13px; }}
 img {{ max-width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; margin:8px 0; }}
 th,td {{ border:1px solid var(--line); padding:5px 9px; text-align:left; }} th {{ background:var(--card); }}
 blockquote {{ border-left:3px solid var(--acc); margin:12px 0; padding:4px 14px; color:var(--mut); background:var(--card); border-radius:0 8px 8px 0; }}
</style></head><body>
<header><a href="index.html">← 試聽總覽</a><a href="leaderboard.html">人類偏好榜</a></header>
<main>{body}</main></body></html>
"""


def render_report() -> None:
    md = REPORT.read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    (SITE / "results.html").write_text(REPORT_TEMPLATE.format(body=body), encoding="utf-8")
    print("results.html: rendered from REPORT.md")


def main() -> None:
    build_manifest()
    copy_plots()
    render_report()
    print(f"done → {SITE}")


if __name__ == "__main__":
    main()
