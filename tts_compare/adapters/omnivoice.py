"""OmniVoice (k2-fsa/OmniVoice) adapter — runs inside its Modal container.

Zero-shot cloning via the `omnivoice` PyPI package. Phonetic control is inline in the
text (uppercase pinyin+tone digit), already injected by the driver, so synthesize()
just forwards the text. 24 kHz output. See docs/tts-systems.md.
"""

from __future__ import annotations

from .base import SynthResult, TtsAdapter


class OmniVoiceAdapter(TtsAdapter):
    name = "omnivoice"
    sample_rate = 24000

    def load(self) -> None:
        import torch
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16
        )

    def synthesize(
        self, text: str, ref_wav_path: str, ref_text: str, controlled: bool = False
    ) -> SynthResult:
        import numpy as np

        audio = self.model.generate(
            text=text,
            language="zh",
            ref_audio=ref_wav_path,
            ref_text=ref_text,
            num_step=32,
        )
        samples = audio[0] if isinstance(audio, (list, tuple)) else audio
        return SynthResult(audio=np.asarray(samples, dtype=np.float32), sample_rate=self.sample_rate)
