"""Speed / VRAM plots on the uniform A100-80GB.

(1) speed_vram.png  — median RTF (95% bootstrap CI, log axis) + peak VRAM per model.
(2) speed_quality_tradeoff.png — RTF vs MER scatter, bubble area ∝ peak VRAM (efficiency frontier).

RTF = container-side synth seconds / dataset est_duration_s (same denominator per sentence
across models, so comparable). Peak VRAM = max device VRAM observed for that model's container.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results.jsonl"
OUT = ROOT / "outputs" / "plots"
RNG = np.random.default_rng(0)


def boot_median_ci(vals, nboot=10000):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(v))
    if v.size == 1:
        return med, med, med
    b = np.median(v[RNG.integers(0, v.size, size=(nboot, v.size))], axis=1)
    return med, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rtf = defaultdict(list)        # model -> [rtf]
    vram_vals = defaultdict(list)  # model -> [per-utterance peak GB]
    mer = defaultdict(list)        # model -> [breeze raw mer]  (quality proxy)
    for r in rows:
        m = r["model"]
        if r.get("rtf") is not None:
            rtf[m].append(r["rtf"])
        if r.get("vram_peak_gb"):
            vram_vals[m].append(r["vram_peak_gb"])
        if r["condition"] == "raw":
            mv = r.get("asr", {}).get("breeze", {}).get("mer")
            if mv is not None:
                mer[m].append(mv)

    # median per-utterance peak VRAM (robust; PyTorch's caching allocator makes the MAX a
    # misleading high-water spike — e.g. breezyvoice ranges ~7–48 GB but typically ~13 GB)
    vram = {m: float(np.median(v)) for m, v in vram_vals.items()}
    models = sorted(rtf, key=lambda m: np.median(rtf[m]))   # fastest first
    rtf_stat = {m: boot_median_ci(rtf[m]) for m in models}

    # ---- (1) RTF + VRAM bars ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    xs = np.arange(len(models))

    meds = [rtf_stat[m][0] for m in models]
    lo = [rtf_stat[m][0] - rtf_stat[m][1] for m in models]
    hi = [rtf_stat[m][2] - rtf_stat[m][0] for m in models]
    a1.bar(xs, meds, 0.6, yerr=[lo, hi], capsize=4, color="#4c78a8", error_kw={"elinewidth": 1})
    a1.axhline(1.0, ls="--", c="#e45756", lw=1)
    a1.text(len(models) - 0.5, 1.05, "real-time (RTF=1)", c="#e45756", ha="right", fontsize=9)
    a1.set_yscale("log")
    a1.set_ylabel("RTF = synth time / audio sec  (log, lower = faster)")
    a1.set_title("Synthesis speed (median RTF, 95% CI)")
    for i, m in enumerate(models):
        a1.text(i, meds[i] * 1.12, f"{meds[i]:.2f}", ha="center", fontsize=8)

    vorder = models  # keep same order for cross-reference
    a2.bar(np.arange(len(vorder)), [vram[m] for m in vorder], 0.6, color="#72b7b2")
    a2.set_ylabel("median peak VRAM (GB)")
    a2.set_title("VRAM (median per-utterance peak; incl. framework cache)")
    for i, m in enumerate(vorder):
        a2.text(i, vram[m] + 0.2, f"{vram[m]:.1f}", ha="center", fontsize=8)
    for ax, order in ((a1, models), (a2, vorder)):
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("zh-TW TTS — speed & memory on A100-80GB", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "speed_vram.png", dpi=150)

    # ---- (2) speed vs quality, bubble = VRAM ----
    fig2, ax = plt.subplots(figsize=(9, 6.5))
    for m in models:
        x = rtf_stat[m][0]
        y = float(np.median(mer[m])) if mer[m] else np.nan
        s = vram[m] * 90
        ax.scatter(x, y, s=s, alpha=0.55, color="#4c78a8", edgecolor="#1b3a5b", zorder=3)
        ax.annotate(f"{m}\n{vram[m]:.0f}GB", (x, y), textcoords="offset points", xytext=(0, 0),
                    ha="center", va="center", fontsize=8)
    ax.axvline(1.0, ls="--", c="#e45756", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("median RTF  (log, ← faster)")
    ax.set_ylabel("median MER (Breeze-ASR, raw)  (← better)")
    ax.set_title("Speed–quality trade-off (bubble area ∝ peak VRAM)\nbottom-left = fast + accurate + small")
    ax.grid(alpha=0.25, zorder=0)
    fig2.tight_layout()
    fig2.savefig(OUT / "speed_quality_tradeoff.png", dpi=150)

    print("wrote", OUT / "speed_vram.png", "and", OUT / "speed_quality_tradeoff.png")
    print(f"{'model':12s} {'medRTF':>7s} {'VRAM':>6s} {'MER(b,raw)':>10s}")
    for m in models:
        y = np.median(mer[m]) if mer[m] else float('nan')
        print(f"{m:12s} {rtf_stat[m][0]:7.2f} {vram[m]:6.1f} {y:10.3f}")


if __name__ == "__main__":
    main()
