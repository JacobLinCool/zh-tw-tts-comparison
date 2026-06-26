"""Chatterbox Multilingual (ResembleAI/chatterbox) adapter — runs inside its Modal container.

Zero-shot cloning via the `chatterbox-tts` PyPI package (V3 multilingual T3 checkpoint,
`language_id="zh"`). The Chinese frontend is a shape-based Cangjie converter, so there is
NO phonetic-control path: synthesize() forwards `text` verbatim and ignores `controlled`
(config marks this model `control="none"`). No reference transcript is needed. 24 kHz output.
Every output carries a Perth neural watermark. See docs/tts-systems.md.
"""

from __future__ import annotations

from .base import SynthResult, TtsAdapter


class ChatterboxAdapter(TtsAdapter):
    name = "chatterbox"
    sample_rate = 24000

    def load(self) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        # The multilingual class loads the 23-language V3 checkpoint by default.
        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

    def synthesize(
        self, text: str, ref_wav_path: str, ref_text: str, controlled: bool = False
    ) -> SynthResult:
        import numpy as np

        # No phonetic control and no reference transcript: clone from the ref wav only.
        wav = self.model.generate(
            text,
            language_id="zh",
            audio_prompt_path=ref_wav_path,
        )
        arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:  # (channels, samples) — downmix to mono
            arr = arr.mean(axis=0)
        return SynthResult(audio=arr.reshape(-1), sample_rate=self.sample_rate)
