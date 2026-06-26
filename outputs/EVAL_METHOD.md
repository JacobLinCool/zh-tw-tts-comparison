# Blind listening-test methodology — decision

**Chosen design: a MUSHRA-style multi-stimulus rating** — for each sentence, present ALL 7 systems
side-by-side, blind and shuffled (labelled 語音 A/B/C…), and rate each on a 0–100 slider; plus a
per-system **"發音有誤"** flag to capture proper-noun / code-switch correctness that a naturalness
slider misses. Implemented in `outputs/site/blind.html`.

## Why (from the research; 4 method families surveyed)

Our constraints: **7 systems**, **small listener pool (3–6, untrained)**, dual goal of
**naturalness/preference AND proper-noun correctness**, and **no natural human reference recordings**.

| Method | Verdict for our case |
|---|---|
| **Absolute MOS (1–5, one at a time)** | ✗ Underpowered — needs ≥30 listeners / ≥150 judgments per system; context/anchoring bias makes scores non-absolute. |
| **CMOS / AB pairwise** | ◐ Most sensitive per comparison, but 7 systems = C(7,2)=**21 pairs** × repeats — too many trials. |
| **Best-Worst Scaling / full ranking** | ◐ Efficient for N=7 and gives a Bradley-Terry ranking, but only RELATIVE order; ranking 7 items at once is cognitively heavy and produces ties. |
| **MUSHRA-style (all systems, 0–100, one screen)** | ✓ **Best fit.** Purpose-built to compare 5–9 systems at once (7 = the sweet spot); within-subject + continuous scale is sensitive and **data-efficient per listener** (≈6 ratings/file stabilise the mean; adding listeners helps more than utterances), which directly offsets a tiny pool. Sorting the sliders also yields the ranking. |

**Adaptations we make** (standard for TTS, where "a true MUSHRA does not exist"):
- **Drop the hidden reference + low-pass anchors** (they model audio-codec artifacts, not TTS errors, and we have no per-sentence human recording). The shared Taiwan reference voice acts as soft context.
- **Show the sentence text** (with entities highlighted) — the test is blind w.r.t. *which system*, not w.r.t. the words; the rater needs the text to judge whether proper nouns were said correctly.
- **Add a per-system correctness flag.** Naturalness and pronunciation-correctness are different axes (Blizzard pairs a naturalness test with a separate intelligibility/pronunciation task). The `發音有誤` checkbox captures the correctness signal in the same screen instead of a second pass.

## How to analyse the exported data

Export gives one row per (sentence, system, rater) with the 0–100 score, the error flag, and the
**real** system name (de-anonymised). Then: per-system mean + 95% CI, repeated-measures ANOVA across
the 7 systems, Holm-corrected pairwise tests; and an error-flag RATE per system for the correctness
axis. With 3–6 raters this reliably ORDERS the systems and separates clearly different ones; adjacent
mid-pack systems may not separate (note it rather than over-claim).

Sources: ITU-R BS.1534-3 (MUSHRA); ITU-T P.800/P.808 (MOS); "Rethinking MUSHRA" (arXiv 2411.12719);
MUSHRA-1S (arXiv 2509.19219); Cooper & Yamagishi 2023, Kirkland et al. 2023 (MOS bias); Blizzard
Challenge naturalness + intelligibility protocols.
