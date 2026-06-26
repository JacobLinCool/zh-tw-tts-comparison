"""Definitive human-preference ranking from the blind-test votes (offline).

Reads the raw append-only vote log from the HF votes dataset (authoritative, not subject
to dataset-viewer lag), fits a Bradley-Terry model (same MM iteration as the live
leaderboard) and adds a percentile bootstrap 95% CI + pairwise separation test. Writes
outputs/human_ranking.json (and optionally a plot) for the REPORT / site.

    uv run python scripts/rank_votes.py                 # read HF votes dataset
    uv run python scripts/rank_votes.py --local v.jsonl # read a local jsonl
    uv run python scripts/rank_votes.py --mock --plot   # synthetic data, sanity check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tts_compare.config import HF_VOTES_DATASET  # noqa: E402

MODEL_ORDER = ["breezyvoice", "omnivoice", "cosyvoice3", "voxcpm2", "moss", "qwen3tts", "chatterbox"]
PRIOR = 0.1
OUT = ROOT / "outputs" / "human_ranking.json"
RNG = np.random.default_rng(0)


def load_hf_votes() -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = [f for f in api.list_repo_files(HF_VOTES_DATASET, repo_type="dataset")
             if f.startswith("data/") and f.endswith(".jsonl")]
    votes = []
    for f in files:
        p = hf_hub_download(HF_VOTES_DATASET, f, repo_type="dataset")
        votes += [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return votes


def mock_votes(n: int = 600) -> list[dict]:
    q = {"cosyvoice3": 0.9, "omnivoice": 0.85, "voxcpm2": 0.8, "moss": 0.7,
         "qwen3tts": 0.65, "chatterbox": 0.4, "breezyvoice": 0.35}
    out = []
    while len(out) < n:
        a, b = RNG.choice(MODEL_ORDER, 2, replace=False)
        pa = q[a] / (q[a] + q[b])
        r = RNG.random()
        out.append({"model_a": a, "model_b": b,
                    "winner": "a" if r < pa * 0.92 else ("tie" if r < pa * 0.92 + 0.08 else "b")})
    return out


def bradley_terry(votes: list[dict], models=MODEL_ORDER) -> np.ndarray:
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}
    beat = np.full((n, n), PRIOR)
    np.fill_diagonal(beat, 0.0)
    for v in votes:
        i, j = idx.get(v.get("model_a")), idx.get(v.get("model_b"))
        if i is None or j is None or i == j:
            continue
        w = v.get("winner")
        if w == "a":
            beat[i, j] += 1
        elif w == "b":
            beat[j, i] += 1
        else:
            beat[i, j] += 0.5
            beat[j, i] += 0.5
    W = beat.sum(axis=1)
    games = beat + beat.T
    p = np.ones(n)
    for _ in range(500):
        denom = np.array([np.sum(games[i] / (p[i] + p)) - games[i, i] / (2 * p[i]) for i in range(n)])
        with np.errstate(divide="ignore", invalid="ignore"):
            np_ = np.where(denom > 0, W / denom, p)
        p = np_ / np_.sum()
    return p, beat, games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=str, help="read votes from a local jsonl instead of HF")
    ap.add_argument("--mock", action="store_true", help="use synthetic votes")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.mock:
        votes, src = mock_votes(), "mock"
    elif args.local:
        votes = [json.loads(ln) for ln in Path(args.local).read_text(encoding="utf-8").splitlines() if ln.strip()]
        src = args.local
    else:
        votes, src = load_hf_votes(), HF_VOTES_DATASET
    n = len(votes)
    print(f"loaded {n} votes from {src}")

    p, beat, games = bradley_terry(votes)
    models = MODEL_ORDER

    # bootstrap CI
    boots = np.zeros((args.boot, len(models)))
    if n:
        varr = np.array(votes, dtype=object)
        for b in range(args.boot):
            sample = list(varr[RNG.integers(0, n, n)])
            boots[b] = bradley_terry(sample)[0]
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0) if n else (p * 0, p * 0)

    order = np.argsort(-p)
    rows = []
    print("\nHuman-preference ranking (Bradley-Terry, 95% bootstrap CI):")
    lead = order[0]
    for rank, i in enumerate(order, 1):
        gms = (beat[i].sum() + beat[:, i].sum()) - 2 * PRIOR * (len(models) - 1)
        sep = "" if rank == 1 else (" SEPARATED" if hi[i] < lo[lead] else " tied")
        print(f"  {rank}. {models[i]:12s} {p[i]*100:5.1f}%  [95% {lo[i]*100:4.1f},{hi[i]*100:4.1f}]  "
              f"games≈{gms:.0f}{sep}")
        rows.append({"model": models[i], "rank": rank, "strength": float(p[i]),
                     "ci_lo": float(lo[i]), "ci_hi": float(hi[i]), "games": float(max(0, gms))})

    OUT.write_text(json.dumps({
        "n_votes": n, "source": src, "prior": PRIOR,
        "ranking": rows,
        "beat_matrix": {models[i]: {models[j]: float(beat[i, j] - (PRIOR if i != j else 0))
                                    for j in range(len(models)) if j != i} for i in range(len(models))},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")

    if args.plot and n:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ms = [models[i] for i in order]
        ys = np.arange(len(ms))[::-1]
        ax.barh(ys, [p[i] * 100 for i in order],
                xerr=[[(p[i] - lo[i]) * 100 for i in order], [(hi[i] - p[i]) * 100 for i in order]],
                color="#6ea8fe", capsize=4)
        ax.set_yticks(ys)
        ax.set_yticklabels(ms)
        ax.set_xlabel("Bradley-Terry preference strength (%)  ·  95% bootstrap CI")
        ax.set_title(f"Human-preference ranking ({n} blind A/B votes)")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        png = ROOT / "outputs" / "plots" / "human_ranking.png"
        fig.savefig(png, dpi=150)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
