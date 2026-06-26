"""Build outputs/site/manifest.json from results.jsonl + the quick-test sentences.

The static pages (outputs/site/index.html browse, outputs/site/blind.html test) read this
manifest. Serve the whole outputs/ dir so wav paths resolve:

    uv run python -m http.server 8000 --directory outputs
    # open http://localhost:8000/site/

Audio paths are written relative to outputs/site/ (e.g. ../breezyvoice/raw/ner-001252.wav).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results.jsonl"
QUICK = ROOT / "data" / "quick_test" / "sentences.jsonl"
SITE = ROOT / "outputs" / "site"

# display order (best-to-worst-ish on the quick test; browse page keeps it stable)
MODEL_ORDER = ["breezyvoice", "omnivoice", "cosyvoice3", "voxcpm2", "moss", "qwen3tts", "chatterbox"]


def _site_path(wav: str) -> str:
    # rec["wav"] is repo-relative, e.g. "outputs/breezyvoice/raw/ner-001252.wav"
    return "../" + wav[len("outputs/"):] if wav.startswith("outputs/") else wav


def main() -> None:
    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sents = {json.loads(ln)["id"]: json.loads(ln)
             for ln in QUICK.read_text(encoding="utf-8").splitlines() if ln.strip()}

    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    models += sorted({r["model"] for r in rows} - set(models))
    present = {r["condition"] for r in rows}
    conditions = [c for c in ["raw", "controlled", "ensub", "ensub_ctrl"] if c in present]

    clips = defaultdict(lambda: defaultdict(dict))  # sid -> model -> cond -> entry
    for r in rows:
        if "wav" not in r:
            continue
        asr = r.get("asr", {})
        clips[r["id"]][r["model"]][r["condition"]] = {
            "wav": _site_path(r["wav"]),
            "mer_q": round(asr.get("qwen3", {}).get("mer"), 3) if asr.get("qwen3", {}).get("mer") is not None else None,
            "mer_b": round(asr.get("breeze", {}).get("mer"), 3) if asr.get("breeze", {}).get("mer") is not None else None,
            "hyp_q": asr.get("qwen3", {}).get("hyp"),
            "hyp_b": asr.get("breeze", {}).get("hyp"),
            "synth_s": r.get("synth_s"),
            "error": r.get("error"),
        }

    sentences = []
    for sid in sorted(clips, key=lambda s: (sents.get(s, {}).get("bucket", ""), s)):
        meta = sents.get(sid, {})
        sentences.append({
            "id": sid,
            "text": meta.get("text", ""),
            "bucket": meta.get("bucket", ""),
            "register": meta.get("register", ""),
            "est_duration_s": meta.get("est_duration_s"),
            "entities": [a["surface"] for a in meta.get("annotations", [])],
            "clips": clips[sid],
        })

    SITE.mkdir(parents=True, exist_ok=True)
    manifest = {"models": models, "conditions": conditions, "sentences": sentences}
    (SITE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {SITE / 'manifest.json'}: {len(models)} models, {len(sentences)} sentences, "
          f"{sum(len(c) for s in sentences for c in s['clips'].values())} clips")


if __name__ == "__main__":
    main()
