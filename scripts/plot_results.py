"""Plot median MER per model x condition with 95% bootstrap CIs, one panel per ASR.

Median (robust to Whisper-style ASR hallucination outliers) with a percentile bootstrap
95% CI. Writes outputs/plots/mer_by_condition_ci.png and a stats CSV.
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

CONDITIONS = ["raw", "controlled", "ensub", "ensub_ctrl"]
COND_COLOR = {"raw": "#4c78a8", "controlled": "#f58518", "ensub": "#54a24b", "ensub_ctrl": "#b279a2"}
ASRS = [("qwen3", "Qwen3-ASR-1.7B"), ("breeze", "Breeze-ASR-25 (Taiwan-tuned)")]
RNG = np.random.default_rng(0)


def boot_median_ci(vals, nboot=10000):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(v))
    if v.size == 1:
        return med, med, med
    boots = np.median(v[RNG.integers(0, v.size, size=(nboot, v.size))], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med, float(lo), float(hi)


def main() -> None:
    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cell = defaultdict(lambda: defaultdict(list))  # (model,cond) -> asr -> [mer]
    for r in rows:
        for asr, _ in ASRS:
            m = r.get("asr", {}).get(asr, {}).get("mer")
            if m is not None:
                cell[(r["model"], r["condition"])][asr].append(m)

    models = sorted({k[0] for k in cell},
                    key=lambda m: boot_median_ci(cell[(m, "raw")]["breeze"])[0])

    OUT.mkdir(parents=True, exist_ok=True)
    csv = [["model", "condition", "asr", "n", "median_mer", "ci_lo", "ci_hi"]]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    width = 0.2
    x = np.arange(len(models))

    for ax, (asr, asr_label) in zip(axes, ASRS):
        for j, cond in enumerate(CONDITIONS):
            meds, los, his = [], [], []
            for m in models:
                vals = cell.get((m, cond), {}).get(asr, [])
                med, lo, hi = boot_median_ci(vals)
                meds.append(med); los.append(med - lo); his.append(hi - med)
                if not np.isnan(med):
                    csv.append([m, cond, asr, len(vals), f"{med:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
            xpos = x + (j - (len(CONDITIONS) - 1) / 2) * width
            ax.bar(xpos, meds, width, yerr=[los, his], capsize=3, label=cond,
                   color=COND_COLOR[cond], edgecolor="white", linewidth=0.5,
                   error_kw={"elinewidth": 1, "alpha": 0.8})
        ax.set_title(asr_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("median MER  (lower = better)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="condition", frameon=False)

    fig.suptitle("zh-TW TTS — median MER by model × condition (25 sentences, 95% bootstrap CI)",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = OUT / "mer_by_condition_ci.png"
    fig.savefig(png, dpi=150)
    (OUT / "mer_stats.csv").write_text("\n".join(",".join(map(str, r)) for r in csv), encoding="utf-8")
    print(f"wrote {png}")
    print(f"wrote {OUT / 'mer_stats.csv'}  ({len(csv) - 1} cells)")


if __name__ == "__main__":
    main()
