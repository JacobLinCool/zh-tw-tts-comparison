"""Render outputs/results.jsonl into a readable comparison report (dual-ASR aware).

Each record carries r["asr"]["qwen3"|"breeze"] = {"hyp", "mer", ...}. We report MEDIAN MER
(robust to Whisper-style ASR hallucinations that spike the mean on a few clips) from both
ASRs side by side — Qwen3-ASR-1.7B and MediaTek Breeze-ASR-25 (Taiwan-tuned). Timing/VRAM
use the mean. Writes outputs/REPORT.md.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results.jsonl"
REPORT = ROOT / "outputs" / "REPORT.md"
QUICK = ROOT / "data" / "quick_test" / "sentences.jsonl"
SHORT_BUCKETS = {"XS", "S"}
LONG_BUCKETS = {"M", "L"}

ORDER = ["breezyvoice", "omnivoice", "cosyvoice3", "voxcpm2", "moss", "qwen3tts", "chatterbox"]
ZH_TW_NATIVE = set()  # no country flags on the website (was {"breezyvoice"})
NO_CONTROL = {"chatterbox", "qwen3tts"}
ASRS = ["qwen3", "breeze"]


def mer_of(r, asr):
    return r.get("asr", {}).get(asr, {}).get("mer")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def _med(recs, asr):  # median MER over records for one ASR
    xs = [mer_of(r, asr) for r in recs]
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def main() -> None:
    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["model"], r["condition"])].append(r)
    models = [m for m in ORDER if any(k[0] == m for k in bucket)]
    models += sorted({k[0] for k in bucket} - set(models))

    L: list[str] = []
    w = L.append
    gpus = sorted({r.get("gpu") for r in rows if r.get("gpu")})
    w("# zh-TW TTS comparison — results\n")
    w(f"Dataset: JacobLinCool/tw-codeswitch-ner (quick test, {len({r['id'] for r in rows})} sentences). "
      f"Shared Taiwan reference voice. TTS GPU: **{', '.join(gpus) or 'n/a'}** (uniform → comparable timing/VRAM). "
      "Two ASRs, no context: **qwen3** = Qwen3-ASR-1.7B, **breeze** = MediaTek Breeze-ASR-25 (Taiwan-tuned, Whisper-v2). "
      "Metric: **median** MER = CER(zh)+WER(en), NFKC+OpenCC normalized. Lower is better. "
      "Median, not mean, because Breeze (Whisper) occasionally hallucinates trailing text and spikes a few clips.\n")

    w("![median MER by model × condition, 95% bootstrap CI](plots/mer_by_condition_ci.png)\n")
    w("![best model: each at its best condition, 95% CI](plots/best_per_model.png)\n")
    w("> Top 5 models are statistically tied on MER (paired-bootstrap CIs include 0); only "
      "chatterbox & breezyvoice are significantly worse. The #1 flips by ASR — MER alone cannot "
      "crown a single winner at n=25; use the blind listening test + the speed/VRAM tiebreakers.\n")

    # ---- main table ----
    w("## Per-model × condition (median MER)\n")
    w("| Model | Cond | n | MER qwen3 | MER breeze | synth (s) | RTF | VRAM (GB) | err |")
    w("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for m in models:
        for cond in ("raw", "controlled", "ensub", "ensub_ctrl"):
            recs = bucket.get((m, cond))
            if not recs:
                continue
            n = sum(1 for r in recs if "asr" in r)
            synth = [r.get("synth_s") for r in recs if "synth_s" in r]
            rtf = [r.get("rtf") for r in recs if "rtf" in r]
            vram = [r.get("vram_peak_gb") for r in recs if r.get("vram_peak_gb")]
            errs = sum(1 for r in recs if r.get("error"))
            tag = f"`{m}`" + (" 🇹🇼" if m in ZH_TW_NATIVE else "")
            w(f"| {tag} | {cond} | {n} | {_med(recs, 'qwen3'):.3f} | {_med(recs, 'breeze'):.3f} | "
              f"{_mean(synth):.1f} | {_mean(rtf):.2f} | {max(vram) if vram else float('nan'):.1f} | {errs} |")
    w("")

    # ---- length robustness (short vs long) ----
    bmap = {json.loads(ln)["id"]: json.loads(ln)["bucket"]
            for ln in QUICK.read_text(encoding="utf-8").splitlines() if ln.strip()}

    def tier_med(m, tiers, asr):
        xs = [mer_of(r, asr) for r in bucket.get((m, "raw"), []) if bmap.get(r["id"]) in tiers]
        xs = [x for x in xs if x is not None]
        return st.median(xs) if xs else float("nan")

    w("## Length robustness (raw, median MER: short XS/S vs long M)\n")
    w("| Model | short | long | Δ (long−short) |")
    w("|---|--:|--:|--:|")
    for m in models:
        s, lng = tier_med(m, SHORT_BUCKETS, "qwen3"), tier_med(m, LONG_BUCKETS, "qwen3")
        if s == s and lng == lng:  # both non-nan
            tag = f"`{m}`" + (" 🇹🇼" if m in ZH_TW_NATIVE else "")
            w(f"| {tag} | {s:.3f} | {lng:.3f} | {lng - s:+.3f} |")
    w("\n> Positive Δ = degrades on long utterances. BreezyVoice breaks down on long inputs "
      "(repetition/degeneration); CosyVoice3 / VoxCPM2 / MOSS / OmniVoice stay robust.\n")

    # ---- ASR cross-check ----
    w("## ASR cross-check (raw, median) — does the Taiwan-tuned ASR rate models differently?\n")
    w("| Model | MER qwen3 | MER breeze | breeze − qwen3 |")
    w("|---|--:|--:|--:|")
    for m in models:
        recs = bucket.get((m, "raw"), [])
        if not recs:
            continue
        q, b = _med(recs, "qwen3"), _med(recs, "breeze")
        tag = f"`{m}`" + (" 🇹🇼" if m in ZH_TW_NATIVE else "")
        w(f"| {tag} | {q:.3f} | {b:.3f} | {b - q:+.3f} |")
    w("\n> Both ASRs broadly agree on the median. Large gaps flag ASR disagreement, not necessarily TTS quality.\n")

    # ---- control lift per ASR ----
    w("## Phonetic-control lift (raw → controlled Δ median MER; negative = control helped)\n")
    w("| Model | Δ qwen3 | Δ breeze |")
    w("|---|--:|--:|")
    for m in models:
        if m in NO_CONTROL:
            continue
        raw, ctl = bucket.get((m, "raw"), []), bucket.get((m, "controlled"), [])
        if raw and ctl:
            w(f"| `{m}` | {_med(ctl, 'qwen3') - _med(raw, 'qwen3'):+.3f} | {_med(ctl, 'breeze') - _med(raw, 'breeze'):+.3f} |")
    w("\n> `chatterbox` and `qwen3tts` expose no phonetic-control path (raw only).\n")

    # ---- English-name substitution effect (raw -> ensub) ----
    if any(k[1] == "ensub" for k in bucket):
        w("## English-name substitution (raw → ensub, Δ median MER)\n")
        w("Sentences re-rendered with real English brand names (Coach, Adidas, Hermès, …) instead of "
          "Chinese transliterations (蔻馳, 愛迪達, 愛馬仕). Negative Δ = the model handles the English names better.\n")
        w("| Model | Δ qwen3 | Δ breeze |")
        w("|---|--:|--:|")
        for m in models:
            raw, en = bucket.get((m, "raw"), []), bucket.get((m, "ensub"), [])
            if raw and en:
                w(f"| `{m}` | {_med(en, 'qwen3') - _med(raw, 'qwen3'):+.3f} | {_med(en, 'breeze') - _med(raw, 'breeze'):+.3f} |")
        w("")

    # ---- ranking per ASR ----
    for asr in ASRS:
        w(f"## Intelligibility ranking — {asr} (best condition, median MER)\n")
        best = []
        for m in models:
            cand = [(_med(bucket[(m, c)], asr), c) for c in ("controlled", "raw") if bucket.get((m, c))]
            cand = [(v, c) for v, c in cand if v == v]  # drop nan
            if cand:
                mer, cond = min(cand)
                best.append((mer, m, cond))
        for rank, (mer, m, cond) in enumerate(sorted(best), 1):
            flag = " 🇹🇼" if m in ZH_TW_NATIVE else ""
            w(f"{rank}. **{m}** — MER {mer:.3f} ({cond}){flag}")
        w("")

    bad = [r for r in rows if r.get("error")]
    if bad:
        w(f"## Synthesis errors ({len(bad)})\n")
        for r in bad[:40]:
            w(f"- `{r['model']}/{r['condition']}` {r['id']}: {r['error']}")
        w("")
    w("## Speed & memory (A100-80GB)\n")
    w("![RTF and peak VRAM per model](plots/speed_vram.png)\n")
    w("![speed-quality trade-off](plots/speed_quality_tradeoff.png)\n")
    w("> RTF = container-side synth time / audio seconds (log axis; <1 = faster than real time). "
      "VRAM = median per-utterance peak device occupancy incl. framework cache (the MAX is an "
      "allocator high-water spike, e.g. breezyvoice ranges ~7–48 GB, so median ~13 GB is reported).\n")

    w("---\nAudio: `outputs/<model>/<condition>/<id>.wav` (human listening). "
      "Per-sentence records incl. both ASR hypotheses: `outputs/results.jsonl`.")

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT} | {len(rows)} records | scored {sum(1 for r in rows if 'asr' in r)}")


if __name__ == "__main__":
    main()
