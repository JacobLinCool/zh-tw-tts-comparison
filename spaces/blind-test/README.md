---
title: zh-TW TTS Arena (Blind A/B)
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
short_description: Blind A/B listening test for Taiwan-Mandarin TTS
---

# zh-TW TTS Arena — Blind A/B listening test

Pairwise (head-to-head) blind test that ranks open-source TTS systems on **Taiwan
Mandarin + Chinese–English code-switching**. Each trial plays the same sentence from two
systems, blinded as 語音 A / 語音 B (model→side randomized); the listener picks the better
one or 平手 (tie). ~5 trials per session to keep fatigue low.

Models play their most realistic input (`ensub_ctrl` where phonetic control is supported,
else `ensub`), defined in `pool.json` (built by `scripts/build_pool.py` in the source repo).

## How votes are stored

Votes append to the **[`JacobLinCool/zh-tw-tts-arena-votes`](https://huggingface.co/datasets/JacobLinCool/zh-tw-tts-arena-votes)**
dataset via `huggingface_hub.CommitScheduler` (append-only JSONL, UUID file per Space
instance, batched commits). A static GitHub Pages site reads this dataset live and
recomputes a Bradley-Terry ranking.

### Setup (maintainer)

Add a **Secret** named `HF_TOKEN` (write scope) in the Space settings so the app can
commit to the votes dataset. Without it (or with `LOCAL_VOTES=1`) the app writes
`votes_local.jsonl` locally instead — useful for UI testing.

Audio is streamed from the **[`JacobLinCool/zh-tw-tts-comparison`](https://huggingface.co/datasets/JacobLinCool/zh-tw-tts-comparison)**
dataset CDN; this Space ships only `app.py` + `pool.json`, no audio.

Vote record schema: `ts, session, sentence_id, model_a, model_b, condition_a,
condition_b, winner(a|b|tie), dwell_s, ua, app_version`.
