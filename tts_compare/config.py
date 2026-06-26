"""Central registry: models, the shared reference voice, paths, and run conditions.

Phonetic-control syntax differs per model (see docs/tts-systems.md); the `control`
field selects the renderer in tts_compare.phonetics. `gpu` is the Modal GPU tier.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# Shared Taiwan-accented reference voice used by ALL models (fair comparison; also
# the main lever that pulls the 6 mainland-default models toward a Taiwan accent).
REFERENCE_WAV = DATA / "ref" / "ref_16k.wav"          # 16 kHz mono PCM
REFERENCE_MP3 = DATA / "ref" / "ref.mp3"
REFERENCE_TEXT = "我覺得要選擇用哪一個 model 來做這件事是需要考慮的"

QUICK_TEST = DATA / "quick_test" / "sentences.jsonl"

# control kinds (renderer keys in phonetics.render_char):
#   "zhuyin"        BreezyVoice: 字[:ㄅㄆㄇ<tone-digit>]        (keep char, append bracket)
#   "pinyin_split"  CosyVoice3:  [initial][final-with-diacritic] (replace char)
#   "pinyin_brace"  VoxCPM2:     {tone3}                         (replace char; needs normalize=False)
#   "pinyin_tone3"  MOSS:        tone3 (space-separated)         (replace char)
#   "pinyin_upper"  OmniVoice:   TONE3 (uppercase, inline)       (replace char)
#   "none"          Chatterbox / Qwen3-TTS-Base: no control path
MODELS: dict[str, dict] = {
    "breezyvoice": {
        "hf_id": "MediaTek-Research/BreezyVoice",
        "gpu": "L4",
        "control": "zhuyin",
        "zh_tw_native": True,
    },
    "chatterbox": {
        "hf_id": "ResembleAI/chatterbox",
        "gpu": "L4",
        "control": "none",
        "zh_tw_native": False,
    },
    "cosyvoice3": {
        "hf_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "gpu": "L4",
        "control": "pinyin_split",
        "zh_tw_native": False,
    },
    "voxcpm2": {
        "hf_id": "openbmb/VoxCPM2",
        "gpu": "L4",
        "control": "pinyin_brace",
        "zh_tw_native": False,
    },
    "moss": {
        "hf_id": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        "gpu": "A10G",  # ~18 GB weights + 48 kHz; needs more VRAM than L4 headroom
        "control": "pinyin_tone3",
        "zh_tw_native": False,
    },
    "qwen3tts": {
        "hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "gpu": "L4",
        "control": "none",
        "zh_tw_native": False,
    },
    "omnivoice": {
        "hf_id": "k2-fsa/OmniVoice",
        "gpu": "L4",
        "control": "pinyin_upper",
        "zh_tw_native": False,
    },
}

# Text conditions to synthesize per sentence.
#   "raw":        original sentence, no phonetic annotation (baseline)
#   "controlled": proper-noun surfaces annotated with the model's phonetic syntax
CONDITIONS = ("raw", "controlled")


def controllable(model_key: str) -> bool:
    return MODELS[model_key]["control"] != "none"


# ---- Hugging Face artifacts (single source of truth for audio + votes) ----
# Audio dataset holds all synthesized wavs + clip metadata; the Gradio Space and the
# GitHub Pages site both stream audio from its CDN `resolve` URLs (`<audio src>` needs
# no CORS). The votes dataset is an append-only log the Space writes and Pages reads.
HF_USER = "JacobLinCool"
HF_AUDIO_DATASET = f"{HF_USER}/zh-tw-tts-comparison"
HF_VOTES_DATASET = f"{HF_USER}/zh-tw-tts-arena-votes"
HF_SPACE = f"{HF_USER}/zh-tw-tts-arena"


def audio_url(model: str, condition: str, sid: str, *, dataset: str = HF_AUDIO_DATASET) -> str:
    """Public CDN URL for a synthesized clip in the audio dataset."""
    return f"https://huggingface.co/datasets/{dataset}/resolve/main/{model}/{condition}/{sid}.wav"


def best_condition(model_key: str) -> str:
    """The model's most realistic input condition for the blind A/B test:
    English brand substitution, plus phonetic control on remaining Chinese entities
    where the model supports it (ensub_ctrl), otherwise plain substitution (ensub)."""
    return "ensub_ctrl" if controllable(model_key) else "ensub"
