"""Publish the audio + metadata to the HF audio dataset, and create the votes dataset.

The audio dataset is the single source of truth: the Gradio Space and the GitHub Pages
site both stream wavs from its CDN. Layout pushed:

    <model>/<condition>/<id>.wav     (600 clips, kept from outputs/)
    sentences.jsonl                  (the 25 quick-test sentences)
    clips.jsonl                      (per-clip metadata: MER, ASR hyps, timing)

Also create the (empty) votes dataset the Space appends to.

Requires HF write auth: `hf auth login` or HF_TOKEN env. Use --dry-run to only build
clips.jsonl locally (no network).

    uv run python scripts/upload_audio_hf.py --dry-run     # build + validate metadata
    uv run python scripts/upload_audio_hf.py               # create repos + upload
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tts_compare.config import HF_AUDIO_DATASET, HF_VOTES_DATASET, audio_url  # noqa: E402

OUTPUTS = ROOT / "outputs"
RESULTS = OUTPUTS / "results.jsonl"
QUICK = ROOT / "data" / "quick_test" / "sentences.jsonl"
CLIPS = OUTPUTS / "clips.jsonl"


def build_clips() -> int:
    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for r in rows:
        if "wav" not in r:
            continue
        asr = r.get("asr", {})
        out.append({
            "id": r["id"],
            "model": r["model"],
            "condition": r["condition"],
            "wav_path": f"{r['model']}/{r['condition']}/{r['id']}.wav",
            "url": audio_url(r["model"], r["condition"], r["id"]),
            "mer_q": asr.get("qwen3", {}).get("mer"),
            "mer_b": asr.get("breeze", {}).get("mer"),
            "hyp_q": asr.get("qwen3", {}).get("hyp"),
            "hyp_b": asr.get("breeze", {}).get("hyp"),
            "synth_s": r.get("synth_s"),
            "rtf": r.get("rtf"),
            "vram_peak_gb": r.get("vram_peak_gb"),
            "gpu": r.get("gpu"),
            "error": r.get("error"),
        })
    CLIPS.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in out), encoding="utf-8")
    print(f"wrote {CLIPS}: {len(out)} clip records")
    return len(out)


DATASET_CARD = f"""---
license: cc-by-4.0
language:
- zh
- en
tags:
- text-to-speech
- taiwan-mandarin
- code-switching
- evaluation
pretty_name: zh-TW TTS comparison (audio + metadata)
---

# zh-TW TTS comparison — audio & metadata

Synthesized speech from 7 open-source TTS systems on Taiwan-Mandarin / Chinese-English
code-switch sentences, across 4 input conditions (raw, controlled, ensub, ensub_ctrl).
Single source of truth for the [blind-test Space](https://huggingface.co/spaces/{HF_VOTES_DATASET.rsplit('/', 1)[0]}/zh-tw-tts-arena)
and the project's GitHub Pages site.

- `<model>/<condition>/<id>.wav` — audio clips (16/24/48 kHz depending on model)
- `sentences.jsonl` — the 25 quick-test sentences (text, bucket, entities)
- `clips.jsonl` — per-clip metadata: dual-ASR MER + hypotheses, synth time, RTF, peak VRAM
"""

VOTES_CARD = """---
license: cc-by-4.0
tags:
- human-feedback
- text-to-speech
- arena
pretty_name: zh-TW TTS Arena — blind-test votes
configs:
- config_name: default
  data_files: data/*.jsonl
---

# zh-TW TTS Arena — votes

Append-only log of blind A/B preference votes collected by the
[zh-tw-tts-arena Space](https://huggingface.co/spaces). One JSON object per line under
`data/`. Schema: `ts, session, sentence_id, model_a, model_b, condition_a, condition_b,
winner(a|b|tie), dwell_s, ua, app_version`.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only build clips.jsonl, no upload")
    ap.add_argument("--wavs", action="store_true", help="(re)upload the wav files too")
    args = ap.parse_args()

    n = build_clips()
    if args.dry_run:
        print("dry-run: skipping HF upload")
        return

    from huggingface_hub import DatasetCard, HfApi

    api = HfApi()
    who = api.whoami()
    print(f"authenticated as {who.get('name')}")

    # --- audio dataset ---
    api.create_repo(HF_AUDIO_DATASET, repo_type="dataset", exist_ok=True)
    DatasetCard(DATASET_CARD).push_to_hub(HF_AUDIO_DATASET, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(QUICK), path_in_repo="sentences.jsonl",
                    repo_id=HF_AUDIO_DATASET, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(CLIPS), path_in_repo="clips.jsonl",
                    repo_id=HF_AUDIO_DATASET, repo_type="dataset")
    if args.wavs:
        print(f"uploading {n} wavs (this can take a few minutes)…")
        api.upload_large_folder(repo_id=HF_AUDIO_DATASET, repo_type="dataset",
                                folder_path=str(OUTPUTS), allow_patterns=["*/*/*.wav"])
    else:
        print("skipping wavs (pass --wavs to upload them)")
    print(f"audio dataset → https://huggingface.co/datasets/{HF_AUDIO_DATASET}")

    # --- votes dataset (empty, viewer-ready) ---
    api.create_repo(HF_VOTES_DATASET, repo_type="dataset", exist_ok=True)
    DatasetCard(VOTES_CARD).push_to_hub(HF_VOTES_DATASET, repo_type="dataset")
    print(f"votes dataset → https://huggingface.co/datasets/{HF_VOTES_DATASET}")


if __name__ == "__main__":
    main()
