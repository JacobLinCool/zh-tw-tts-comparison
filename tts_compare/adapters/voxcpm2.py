"""VoxCPM2 (openbmb/VoxCPM2) adapter — runs inside its Modal container.

Tokenizer-free diffusion-AR TTS (2B, MiniCPM-4 + AudioVAE V2) via the `voxcpm` PyPI
package. Phonetic control is inline curly-brace pinyin+tone ({tai2}{ji1}{dian4}),
already injected by the driver; the braces are only honored when text normalization is
OFF, so `controlled` toggles it (controlled -> normalize=False phoneme mode; raw ->
normalize=True for number/symbol expansion). Hi-Fi voice cloning conditions on the
reference wav + its exact transcript. 48 kHz output. See docs/tts-systems.md.
"""

from __future__ import annotations

from .base import SynthResult, TtsAdapter


class VoxCPM2Adapter(TtsAdapter):
    name = "voxcpm2"
    sample_rate = 48000

    def load(self) -> None:
        from voxcpm import VoxCPM

        # load_denoiser=False skips the ModelScope ZipEnhancer download; optimize=False
        # avoids the torch.compile warm-up for a robust cold start (weights cache in
        # HF_HOME). device="cuda" pins the model to the GPU.
        self.model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM2",
            load_denoiser=False,
            optimize=False,
            device="cuda",
        )

    def synthesize(
        self, text: str, ref_wav_path: str, ref_text: str, controlled: bool = False
    ) -> SynthResult:
        import numpy as np

        # Hi-Fi / Ultimate cloning: feed the reference clip + its transcript for the
        # best voice match (the same clip goes to both prompt_* and reference_wav_path,
        # per the usage guide). normalize=False keeps the {pinyin+tone} braces verbatim
        # in the controlled condition; normalize=True expands numbers/symbols for raw.
        audio = self.model.generate(
            text=text,
            prompt_wav_path=ref_wav_path,
            prompt_text=ref_text,
            reference_wav_path=ref_wav_path,
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=not controlled,
        )
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:  # downmix to mono if the model ever emits multi-channel
            arr = arr.mean(axis=int(np.argmin(arr.shape)))
        return SynthResult(audio=arr.reshape(-1), sample_rate=self.sample_rate)
