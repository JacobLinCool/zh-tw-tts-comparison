"""Re-score existing ensub records against the SUBSTITUTED text (the real target), not the
original Chinese reference. Recomputes MER locally from the stored ASR hypotheses — no GPU.
Fixes records produced before the driver's ref_text fix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_compare.dataset import load_quick_test
from tts_compare.scoring import mer
from tts_compare.substitutions import build_ensub_text, load_sub_map

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results.jsonl"


def main() -> None:
    sents = {r["id"]: r for r in load_quick_test()}
    sub_map = load_sub_map()
    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]

    fixed = 0
    for rec in rows:
        if rec.get("condition") != "ensub" or "asr" not in rec:
            continue
        ref = build_ensub_text(sents[rec["id"]], sub_map)
        rec["ref_text"] = ref
        for a in rec["asr"].values():
            if "hyp" in a:
                a.update(mer(ref, a["hyp"]))   # overwrite mer + token details
        fixed += 1

    with RESULTS.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"re-scored {fixed} ensub records against the substituted reference")


if __name__ == "__main__":
    main()
