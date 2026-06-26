# Controlled-condition audit

**Verdict: the "phonetic control makes MER worse" headline is mostly a format-fidelity /
mis-use bug in 3 models (MOSS, BreezyVoice, CosyVoice3), not a real effect.** Where the
control syntax is simple and correctly applied (VoxCPM2, OmniVoice) the model honors it and
the condition is roughly neutral. Evidence below is from `data/audit/<model>.json`
(raw-vs-controlled ASR per sentence).

| Model | control syntax | controlled verdict | evidence |
|---|---|---|---|
| **VoxCPM2** | `{tone3}` curly braces | ✅ works | pinyin honored; regressions are English/ASR artifacts, not the controlled Chinese |
| **OmniVoice** | `TONE3` uppercase inline | ✅ mostly works | entities mostly correct; occasional first-token slip (張→商) |
| **CosyVoice3** | `[initial][final]` split | ❌ broken | `[zh][āng][yǒng][r][ú]` → garbled "张华一卢奇"; y/w/r syllable split is malformed |
| **BreezyVoice** | `字[:ㄅㄆㄇN]` per char | ❌ broken | dense per-char 注音 → "查睿智" (was 張永儒); model expects SPARSE annotation, not every char; + IndexError on code-switch |
| **MOSS** | space-sep `tone3` inline | ❌ broken (catastrophic) | generation collapses and parrots the reference transcript; inline pinyin is not the right control channel |

## Smoking guns

- **MOSS** `ner-001252` (Δ +1.10): controlled ASR = *"我覺得要選擇用哪一個 model 來做這件事是需要考慮的…"* — verbatim the **reference-voice transcript**. The inline `ai4 ma3 shi4` pinyin broke generation entirely.
- **BreezyVoice** `ner-002045` (Δ +0.79): `張[:ㄓㄤ1]永[:ㄩㄥ3]儒[:ㄖㄨ2]…` → ASR heard "查睿智". The 注音 *values* are correct; annotating **every** character (rather than only rare/polyphone ones, as BreezyVoice's own G2PW does) is out-of-distribution and garbles output.
- **CosyVoice3** `ner-002045` (Δ +0.64): `[zh][āng][yǒng][r][ú]老師…` → "张华一卢奇". The `[initial][final]` split mishandles zero-initial / y / w / r syllables (`[yǒng]`, `[r][ú]`).
- **VoxCPM2** `ner-006570` (Δ +0.25): `{lin2}{jia1}{long2}…` → "林佳龙…" **correct**. The only error is the English acronym "NTT DOCOMO" → ASR spelled it "N T T D O C O M O". Control worked; regression is a code-switch/ASR artifact.

## BreezyVoice IndexError (2 sentences)

Both failing sentences contain **inline English + space** (`SK 電訊`, `momo 購物網`). BreezyVoice's
`get_bopomofo_rare()` indexes characters while inserting auto-annotations; our pre-injected
`[:…]` brackets adjacent to Latin tokens push an index out of range. Fix: for the controlled
condition skip BreezyVoice's auto-annotation (we already supply the 注音), or guard the call.

## Recommended fixes

1. **BreezyVoice** — stop annotating every character. Annotate **sparsely** (only the entity, or only polyphone-risk chars) and let its native G2PW handle the rest; for the controlled condition, run `text_normalize_new` but skip `get_bopomofo_rare` (we already injected 注音). Fixes both the garbling and the IndexError.
2. **CosyVoice3** — fix the `[initial][final]` splitter for zero-initial / y / w / ü / r syllables (e.g. 永 yǒng → `[yǒng]` is wrong; verify the exact token inventory the model was trained on, ideally from its g2p code).
3. **MOSS** — re-research the actual pronunciation-control channel; inline space-separated tone3 pinyin collapses generation. It may need a different marker/mode (or MOSS may simply not support reliable inline pinyin control).
4. **General** — the dataset's proper nouns are mostly common enough that models already pronounce them acceptably, so even *correct* control is near-neutral on whole-sentence MER. To measure control's real value, score at the **entity span** and/or pick genuinely rare/ambiguous proper nouns.

## What this means for the results

- **Speed / VRAM** numbers (uniform A100-80GB, container-side timing) are solid and unaffected.
- **Raw-condition MER** is a valid intelligibility proxy (with the ASR-accent caveat — Qwen3-ASR likely favors mainland accent, so it under-rates native-Taiwan BreezyVoice).
- **Controlled-condition MER should NOT be compared across models yet** — for 3 of 5 it reflects broken injection, not the model's true response to phonetic control.

## Resolution (post-fix re-run, dual ASR, median MER)

All three injection bugs were root-caused and fixed (`tts_compare/phonetics.py` + the BreezyVoice/Moss adapters); the experiment was re-run with a second ASR (Breeze-ASR-25). Outcome:

- **BreezyVoice** — fixed (sparse 注音 override + bounds-safe `replace_blank`). 0 errors; control now **helps**: raw 0.139 → controlled 0.083 (qwen3 median). Its native G2PW already reads common nouns, so only the few Taiwan-specific/polyphone readings get overridden — and they help.
- **CosyVoice3** — fixed (`y`/`w` added to the onset inventory; `永`→`[y][ǒng]`). No more garbling; control is now ~neutral (+0.015 median).
- **MOSS** — format fixed (space-delimited pinyin). The catastrophic reference-parroting is gone on all but **1** sentence (ner-001252, which has two pinyin runs); overall MOSS inline pinyin is fragile and does not help (+0.019). Honest result, not a bug.
- **ASR caveat discovered**: Breeze-ASR (Whisper-v2) hallucinates trailing text on a few clips (chatterbox mean 0.656 but **median 0.119**, max 8.135). Report uses **median** MER to stay robust. The "Qwen3 under-rates Taiwan accent" hypothesis is NOT supported by the median — both ASRs rate BreezyVoice ~0.13–0.14 raw (breeze − qwen3 ≈ 0).
