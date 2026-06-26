"""Merge a results fragment into outputs/results.jsonl by (model, condition, id).

Records from the fragment add to / overwrite the matching records in the base file, so a
partial re-run (e.g. only the new long sentences) folds into the full result set without
re-synthesizing everything. Usage: python scripts/merge_results.py <fragment.jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "outputs" / "results.jsonl"


def load(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    frag_path = ROOT / "outputs" / sys.argv[1] if len(sys.argv) > 1 else None
    if not frag_path or not frag_path.exists():
        raise SystemExit(f"fragment not found: {frag_path}")

    def key(r):
        return (r["model"], r["condition"], r["id"])

    merged = {key(r): r for r in load(BASE)}
    frag = load(frag_path)
    for r in frag:
        merged[key(r)] = r

    rows = sorted(merged.values(), key=lambda r: (r["id"], r["model"], r["condition"]))
    with BASE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"merged {len(frag)} fragment records into {BASE.name}: now {len(rows)} total "
          f"({len({r['id'] for r in rows})} sentences)")


if __name__ == "__main__":
    main()
