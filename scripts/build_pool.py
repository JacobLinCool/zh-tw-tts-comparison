"""Build the blind A/B test pool: each model at its BEST REALISTIC condition.

For the head-to-head listening test we pit models against each other on the same
sentence, each playing its most realistic input: English brand substitution plus
phonetic control on remaining Chinese entities where supported (ensub_ctrl), else
plain substitution (ensub) — see config.best_condition().

Audio is streamed from the HF audio dataset CDN (config.audio_url), so the Space and
GitHub Pages share one source of truth. Writes spaces/blind-test/pool.json.

    uv run python scripts/build_pool.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tts_compare.config import (  # noqa: E402
    HF_AUDIO_DATASET,
    MODELS,
    audio_url,
    best_condition,
)

RESULTS = ROOT / "outputs" / "results.jsonl"
QUICK = ROOT / "data" / "quick_test" / "sentences.jsonl"
OUT = ROOT / "spaces" / "blind-test" / "pool.json"

# Stable display/iteration order (best-to-worst-ish on the quick test).
ORDER = ["breezyvoice", "omnivoice", "cosyvoice3", "voxcpm2", "moss", "qwen3tts", "chatterbox"]


def main() -> None:
    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    meta = {json.loads(ln)["id"]: json.loads(ln)
            for ln in QUICK.read_text(encoding="utf-8").splitlines() if ln.strip()}

    # index records by (model, condition, id)
    rec = {(r["model"], r["condition"], r["id"]): r for r in rows}
    models = [m for m in ORDER if m in MODELS]

    sentences = []
    skipped = 0
    for sid in sorted(meta, key=lambda s: (meta[s].get("bucket", ""), s)):
        m = meta[sid]
        clips = {}
        for model in models:
            cond = best_condition(model)
            r = rec.get((model, cond, sid))
            if r is None or r.get("error") or "wav" not in r:
                skipped += 1
                continue
            clips[model] = {"condition": cond, "url": audio_url(model, cond, sid)}
        if len(clips) < 2:  # need at least a pair
            continue
        sentences.append({
            "id": sid,
            "text": m.get("text", ""),
            "bucket": m.get("bucket", ""),
            "register": m.get("register", ""),
            "entities": [a["surface"] for a in m.get("annotations", [])],
            "clips": clips,
        })

    pool = {"audio_dataset": HF_AUDIO_DATASET, "models": models, "sentences": sentences}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")

    n_clips = sum(len(s["clips"]) for s in sentences)
    cond_mix = defaultdict(int)
    for s in sentences:
        for c in s["clips"].values():
            cond_mix[c["condition"]] += 1
    print(f"wrote {OUT}: {len(sentences)} sentences, {n_clips} clips, "
          f"conditions={dict(cond_mix)}, skipped={skipped}")


if __name__ == "__main__":
    main()
