"""Select a small, diverse quick-test set from JacobLinCool/tw-codeswitch-ner.

Goal: pick ~16-20 SHORT zh-TW sentences that stress two things at once:
  1. Taiwan-Mandarin pronunciation of proper nouns  -> entities that carry pinyin/zhuyin
  2. Chinese-English code-switching                 -> sentences with native English
     tokens, and/or entities that have both a zh and an en form (so a code-switch
     variant can be produced by direct zh<->en substitution).

Each selected sentence is emitted with the phonetic/bilingual annotations of its
linked entities, so the TTS harness can later inject pinyin/zhuyin control or build
substitution variants without re-querying the dataset.

Output: data/quick_test/sentences.jsonl  (+ a readable summary on stdout)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

DATASET = "JacobLinCool/tw-codeswitch-ner"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "quick_test"
TARGET_TOTAL = 18
SHORT_BUCKETS = {"XS", "S"}            # keep quick-test utterances short
MAX_PER_REGISTER = 3                   # diversity guard
MAX_PER_ENTITY = 1                     # each proper noun appears in at most 1 picked sentence
LONG_BUCKETS = {"M"}                   # longer tier: 15-30 s (L = 30-61 s exceeds many TTS limits)
LONG_TARGET = 7                        # extra long sentences appended after the short set
ENGLISH_RE = re.compile(r"[A-Za-z]{2,}")


def _norm_text(t: str) -> str:
    return re.sub(r"[\s，。、！？,.!?]", "", t)


def main() -> None:
    sentences = load_dataset(DATASET, "sentences", split="train")
    entities = load_dataset(DATASET, "entities", split="train")

    ent_by_id = {e["id"]: e for e in entities}

    def annotate(row: dict) -> dict:
        anns = []
        for e in row["entities"]:
            rec = ent_by_id.get(e["entity_id"], {})
            anns.append(
                {
                    "surface": e["surface"],
                    "entity_id": e["entity_id"],
                    "start": e["start"],
                    "end": e["end"],
                    "type": e["type"],
                    "region": e["region"],
                    "zh": rec.get("zh", ""),
                    "en": rec.get("en", ""),
                    "aliases": rec.get("aliases", []),
                    "pinyin": rec.get("pinyin", ""),
                    "zhuyin": rec.get("zhuyin", ""),
                }
            )
        has_english = bool(ENGLISH_RE.search(row["text"]))
        has_phonetic = any(a["pinyin"] or a["zhuyin"] for a in anns)
        # an entity is "substitutable" if it has BOTH a zh surface and an en form
        substitutable = [a for a in anns if a["en"] and (a["zh"] or a["surface"])]
        return {
            "id": row["id"],
            "text": row["text"],
            "register": row["register"],
            "bucket": row["bucket"],
            "n_han": row["n_han"],
            "est_duration_s": row["est_duration_s"],
            "has_english": has_english,
            "has_phonetic_entity": has_phonetic,
            "n_substitutable": len(substitutable),
            "annotations": anns,
        }

    cand = [annotate(r) for r in sentences]
    short = [c for c in cand if c["bucket"] in SHORT_BUCKETS and c["annotations"]]

    # Pool A: native code-switch (English already present), with phonetic-annotated entities.
    pool_a = sorted(
        [c for c in short if c["has_english"] and c["has_phonetic_entity"]],
        key=lambda c: (-len(c["annotations"]), c["est_duration_s"], c["id"]),
    )
    # Pool B: no native English but rich in proper nouns that carry pinyin/zhuyin AND
    # have an en form (=> can build a code-switch variant by substitution).
    pool_b = sorted(
        [
            c
            for c in short
            if not c["has_english"] and c["has_phonetic_entity"] and c["n_substitutable"] >= 1
        ],
        key=lambda c: (-c["n_substitutable"], c["est_duration_s"], c["id"]),
    )

    # Longer tier (bucket M): same criteria, sorted to prefer the LONGEST clips first.
    longs = [c for c in cand if c["bucket"] in LONG_BUCKETS and c["annotations"]]
    long_a = sorted(
        [c for c in longs if c["has_english"] and c["has_phonetic_entity"]],
        key=lambda c: (-len(c["annotations"]), -c["est_duration_s"], c["id"]),
    )
    long_b = sorted(
        [c for c in longs if not c["has_english"] and c["has_phonetic_entity"] and c["n_substitutable"] >= 1],
        key=lambda c: (-c["n_substitutable"], -c["est_duration_s"], c["id"]),
    )

    selected: list[dict] = []
    seen: set[str] = set()
    seen_text: set[str] = set()
    reg_count: dict[str, int] = defaultdict(int)
    ent_count: dict[str, int] = defaultdict(int)

    def take(pool: list[dict], quota: int, reg: dict[str, int] | None = None) -> None:
        reg = reg_count if reg is None else reg
        n = 0
        for c in pool:
            if n >= quota:
                break
            if c["id"] in seen or _norm_text(c["text"]) in seen_text:
                continue
            if reg[c["register"]] >= MAX_PER_REGISTER:
                continue
            ids = [a["entity_id"] for a in c["annotations"]]
            if any(ent_count[e] >= MAX_PER_ENTITY for e in ids):
                continue
            selected.append(c)
            seen.add(c["id"])
            seen_text.add(_norm_text(c["text"]))
            for e in ids:
                ent_count[e] += 1
            reg[c["register"]] += 1
            n += 1

    take(pool_a, TARGET_TOTAL // 2)        # ~9 native code-switch
    take(pool_b, TARGET_TOTAL - len(selected))  # fill rest with substitutable
    # backfill from either pool if still short (relax register cap)
    if len(selected) < TARGET_TOTAL:
        for c in pool_a + pool_b:
            if len(selected) >= TARGET_TOTAL:
                break
            if c["id"] not in seen:
                selected.append(c)
                seen.add(c["id"])
    n_short = len(selected)

    # Longer tier: append LONG_TARGET bucket-M sentences. Fresh register budget, but they
    # share the entity/text de-dup so they avoid proper nouns already used by the short set.
    long_reg: dict[str, int] = defaultdict(int)
    take(long_a, (LONG_TARGET + 1) // 2, long_reg)
    take(long_b, n_short + LONG_TARGET - len(selected), long_reg)
    for c in long_a + long_b:              # backfill long tier (relax register cap)
        if len(selected) >= n_short + LONG_TARGET:
            break
        if c["id"] not in seen and _norm_text(c["text"]) not in seen_text:
            selected.append(c)
            seen.add(c["id"])
            seen_text.add(_norm_text(c["text"]))

    # short set first (preserves the original order / --limit behavior), then the long tier
    selected.sort(key=lambda c: (c["bucket"] in LONG_BUCKETS, c["id"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "sentences.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for c in selected:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- readable summary ----
    n_long = sum(1 for c in selected if c["bucket"] in LONG_BUCKETS)
    print(f"pools: short a={len(pool_a)} b={len(pool_b)} | long a={len(long_a)} b={len(long_b)}")
    print(f"selected: {len(selected)} ({len(selected) - n_long} short + {n_long} long)  -> {out_file}")
    print(f"total est_duration: {sum(c['est_duration_s'] for c in selected):.1f}s")
    print("=" * 100)
    for c in selected:
        tag = "CS-native" if c["has_english"] else f"sub×{c['n_substitutable']}"
        print(f"[{c['id']}] {c['register']}/{c['bucket']} {c['est_duration_s']:.1f}s ({tag})")
        print(f"  {c['text']}")
        for a in c["annotations"]:
            extra = f" | en={a['en']}" if a["en"] else ""
            ph = f" | {a['pinyin']} | {a['zhuyin']}" if a["pinyin"] or a["zhuyin"] else ""
            print(f"    - {a['surface']} ({a['type']}/{a['region']}){extra}{ph}")
        print("-" * 100)


if __name__ == "__main__":
    main()
