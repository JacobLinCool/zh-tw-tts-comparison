"""Determine the best model: each model at its BEST condition (lowest median MER), with
95% bootstrap CIs, plus a PAIRED bootstrap significance test (same 25 sentences) of the
winner vs the rest. Writes outputs/plots/best_per_model.png and prints a ranking.
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
ASRS = [("qwen3", "Qwen3-ASR-1.7B"), ("breeze", "Breeze-ASR-25 (Taiwan-tuned)")]
RNG = np.random.default_rng(0)


def ci(vals):
    v = np.asarray(vals, float)
    b = np.median(v[RNG.integers(0, v.size, size=(10000, v.size))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # per (model, cond, asr): {sentence_id: mer}
    vec = defaultdict(dict)
    for r in rows:
        for asr, _ in ASRS:
            m = r.get("asr", {}).get(asr, {}).get("mer")
            if m is not None:
                vec[(r["model"], r["condition"], asr)][r["id"]] = m

    models = sorted({k[0] for k in vec})
    best = {}   # (model, asr) -> (cond, {id:mer})
    for m in models:
        for asr, _ in ASRS:
            cands = [(c, vec[(m, c, asr)]) for (mm, c, a) in vec if mm == m and a == asr]
            cands = [(c, d) for c, d in cands if d]
            if cands:
                best[(m, asr)] = min(cands, key=lambda cd: np.median(list(cd[1].values())))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, (asr, label) in zip(axes, ASRS):
        order = sorted([m for m in models if (m, asr) in best],
                       key=lambda m: np.median(list(best[(m, asr)][1].values())))
        meds, los, his, conds = [], [], [], []
        for m in order:
            cond, d = best[(m, asr)]
            md, lo, hi = ci(list(d.values()))
            meds.append(md); los.append(md - lo); his.append(hi - md); conds.append(cond)
        xs = np.arange(len(order))
        bars = ax.bar(xs, meds, 0.6, yerr=[los, his], capsize=4, color="#4c78a8",
                      error_kw={"elinewidth": 1})
        bars[0].set_color("#54a24b")  # winner
        for i, (m, c) in enumerate(zip(order, conds)):
            ax.text(i, meds[i] + his[i] + 0.004, c, ha="center", fontsize=7.5, rotation=0)
        ax.set_xticks(xs); ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel("median MER at best condition  (lower = better)")
        ax.set_title(label); ax.grid(axis="y", alpha=0.25)

        # paired significance: winner vs each other (common sentence ids)
        win = order[0]
        wc, wd = best[(win, asr)]
        print(f"\n[{asr}] best-condition ranking:")
        for rank, m in enumerate(order, 1):
            c, d = best[(m, asr)]
            md = np.median(list(d.values()))
            tag = ""
            if rank > 1:
                ids = sorted(set(wd) & set(d))
                diff = np.array([d[i] - wd[i] for i in ids])           # other - winner (>0 = winner better)
                bd = np.median(diff[RNG.integers(0, diff.size, size=(10000, diff.size))], axis=1)
                lo, hi = np.percentile(bd, [2.5, 97.5])
                tag = f"  vs {win}: Δmed={np.median(diff):+.3f} [95% {lo:+.3f},{hi:+.3f}] {'SIG' if lo > 0 else 'ns'}"
            print(f"  {rank}. {m:12s} {md:.3f} ({c}){tag}")

    fig.suptitle("Best model: each at its best condition (25 sentences, 95% bootstrap CI; green = winner)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "best_per_model.png", dpi=150)
    print(f"\nwrote {OUT / 'best_per_model.png'}")


if __name__ == "__main__":
    main()
