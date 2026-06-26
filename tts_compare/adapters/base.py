"""Adapter contract shared by every TTS system.

An adapter is constructed and `load()`-ed once inside its Modal container (heavy
imports happen in `load()`, never at module top level, so the module stays importable
locally for tests). `synthesize()` returns mono float32 PCM in [-1, 1] plus its native
sample rate; the driver writes the wav locally and feeds it to ASR.

Phonetic control is resolved by the driver (tts_compare.phonetics): the adapter just
synthesizes whatever `text` it is given. `controlled` is passed so adapters that need a
mode toggle for inline phonemes (e.g. VoxCPM2 must disable text normalization) can react.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SynthResult:
    audio: "object"   # np.ndarray float32 mono, [-1, 1]
    sample_rate: int


class TtsAdapter:
    #: short key, must match tts_compare.config.MODELS
    name: str = ""

    def load(self) -> None:
        """Import deps and load weights. Called once per container."""
        raise NotImplementedError

    def synthesize(
        self,
        text: str,
        ref_wav_path: str,
        ref_text: str,
        controlled: bool = False,
    ) -> SynthResult:
        """Synthesize `text` cloning the voice in `ref_wav_path` (16 kHz mono)."""
        raise NotImplementedError
