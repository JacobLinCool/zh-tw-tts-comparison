"""Mixed Error Rate (MER) for zh-TW + English code-switch ASR scoring.

Per the research recipe (docs/tts-systems.md): tokenize each CJK character as one
token and each contiguous alphanumeric run as one word token, then Levenshtein WER
over those tokens (= CER on Chinese, WER on English, unified). Normalize reference
and hypothesis IDENTICALLY before scoring: NFKC (full->half width), OpenCC script
unification (Traditional vs Simplified ASR output), lowercase English, drop
punctuation/whitespace. The TTS *input* (with phonetic annotations) is never scored —
only the original sentence text is the reference.
"""

from __future__ import annotations

import re
import unicodedata

import jiwer
from opencc import OpenCC

_T2S = OpenCC("t2s")  # unify both sides to Simplified so trad/simp ASR output compares fairly
_HAN = r"一-鿿㐀-䶿"
_TOKEN = re.compile(rf"[{_HAN}]|[a-z0-9]+")
# keep whitespace so English word boundaries survive (English scored as WER, not glued)
_PUNCT = re.compile(rf"[^{_HAN}a-z0-9\s]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _T2S.convert(text)
    text = _PUNCT.sub(" ", text)
    return text


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def mer(reference: str, hypothesis: str) -> dict:
    """Mixed Error Rate + the components, computed over normalized tokens."""
    ref_tok = tokenize(reference)
    hyp_tok = tokenize(hypothesis)
    ref_s = " ".join(ref_tok) or "∅"
    hyp_s = " ".join(hyp_tok) or "∅"
    out = jiwer.process_words(ref_s, hyp_s)
    return {
        "mer": out.wer,
        "ref_tokens": len(ref_tok),
        "hits": out.hits,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "ref_norm": ref_s,
        "hyp_norm": hyp_s,
    }
