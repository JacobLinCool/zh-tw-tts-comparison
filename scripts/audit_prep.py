"""Prepare per-model raw-vs-controlled audit data from outputs/results.jsonl.

For each controllable model, pair the raw and controlled record of every sentence and
emit, per sentence: the reference text, its proper-noun entities (with pinyin/zhuyin),
the controlled INPUT actually sent to the model, what the ASR heard for raw vs
controlled, and the two MERs. This lets an auditor judge whether "controlled made MER
worse" is a real model effect or a phonetic-injection format bug.

Writes data/audit/<model>.json and prints a regression summary.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results.jsonl"
QUICK = ROOT / "data" / "quick_test" / "sentences.jsonl"
OUT = ROOT / "data" / "audit"

CONTROLLABLE = ["breezyvoice", "omnivoice", "cosyvoice3", "voxcpm2", "moss"]


def main() -> None:
    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sents = {json.loads(ln)["id"]: json.loads(ln)
             for ln in QUICK.read_text(encoding="utf-8").splitlines() if ln.strip()}

    by = defaultdict(dict)  # (model, id) -> {cond: rec}
    for r in rows:
        by[(r["model"], r["id"])][r["condition"]] = r

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'model':12s} {'pairs':>5s} {'regressed':>9s} {'improved':>8s} {'errors':>6s}  worst Δ")
    for m in CONTROLLABLE:
        items = []
        for (mm, sid), d in by.items():
            if mm != m or "raw" not in d:
                continue
            raw, ctrl = d["raw"], d.get("controlled", {})
            ents = [
                {"surface": a["surface"], "zh": a.get("zh", ""), "en": a.get("en", ""),
                 "pinyin": a.get("pinyin", ""), "zhuyin": a.get("zhuyin", "")}
                for a in sents.get(sid, {}).get("annotations", [])
            ]
            raw_mer = raw.get("asr", {}).get("breeze", {}).get("mer")    # Taiwan-tuned ASR as baseline
            ctrl_mer = ctrl.get("asr", {}).get("breeze", {}).get("mer")
            items.append({
                "id": sid,
                "ref_text": raw["ref_text"],
                "entities": ents,
                "raw_input": raw.get("input_text"),
                "raw_asr_qwen": raw.get("asr", {}).get("qwen3", {}).get("hyp"),
                "raw_asr_breeze": raw.get("asr", {}).get("breeze", {}).get("hyp"),
                "raw_mer_breeze": raw_mer,
                "ctrl_input": ctrl.get("input_text"),
                "ctrl_asr_qwen": ctrl.get("asr", {}).get("qwen3", {}).get("hyp"),
                "ctrl_asr_breeze": ctrl.get("asr", {}).get("breeze", {}).get("hyp"),
                "ctrl_mer_breeze": ctrl_mer,
                "ctrl_error": ctrl.get("error"),
                "delta": (ctrl_mer - raw_mer) if (raw_mer is not None and ctrl_mer is not None) else None,
            })
        items.sort(key=lambda x: (x["delta"] is None, -(x["delta"] or 0)))
        (OUT / f"{m}.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        reg = sum(1 for x in items if x["delta"] and x["delta"] > 0.02)
        imp = sum(1 for x in items if x["delta"] and x["delta"] < -0.02)
        errs = sum(1 for x in items if x["ctrl_error"])
        worst = max((x["delta"] for x in items if x["delta"] is not None), default=0)
        print(f"{m:12s} {len(items):5d} {reg:9d} {imp:8d} {errs:6d}  {worst:+.3f}")

    # ---- BreezyVoice IndexError cases: show the exact controlled input ----
    print("\n=== BreezyVoice controlled IndexError inputs ===")
    for r in rows:
        if r["model"] == "breezyvoice" and r["condition"] == "controlled" and r.get("error"):
            print(f"[{r['id']}] {r['error']}")
            print(f"  ref : {r['ref_text']}")
            print(f"  sent: {r['input_text']}")


if __name__ == "__main__":
    main()
