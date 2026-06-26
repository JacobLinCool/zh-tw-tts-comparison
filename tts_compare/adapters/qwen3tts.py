"""Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-1.7B-Base) adapter — runs inside its Modal container.

Zero-shot cloning via the official `qwen-tts` PyPI package (Qwen3TTSModel). The Base
variant is clone-only (ICL mode), so generate_voice_clone() needs both the reference clip
and its transcript. There is NO phonetic/pronunciation control (raw text -> Qwen2 BPE
tokenizer, pronunciation learned implicitly), so synthesize() ignores `controlled` and
just forwards the text. language="Chinese", 24 kHz output. See docs/tts-systems.md.
"""

from __future__ import annotations

from .base import SynthResult, TtsAdapter


class Qwen3TtsAdapter(TtsAdapter):
    name = "qwen3tts"
    sample_rate = 24000

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        # attn_implementation="sdpa" avoids the slow/fragile flash-attn build.
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    def synthesize(
        self, text: str, ref_wav_path: str, ref_text: str, controlled: bool = False
    ) -> SynthResult:
        import numpy as np

        # Base has no phonetic control; `controlled` is ignored (raw text only).
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="Chinese",
            ref_audio=ref_wav_path,
            ref_text=ref_text,
        )
        samples = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim > 1:  # downmix to mono if the model emits multi-channel
            audio = audio.mean(axis=int(np.argmin(audio.shape)))
        return SynthResult(
            audio=audio.reshape(-1),
            sample_rate=int(sr) if sr else self.sample_rate,
        )
