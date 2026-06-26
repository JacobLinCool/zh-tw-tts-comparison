"""CosyVoice3 (FunAudioLLM/Fun-CosyVoice3-0.5B-2512) adapter — runs inside its Modal container.

Zero-shot cloning via the cloned CosyVoice repo (not pip-installable). The Qwen2-0.5B
LLM predicts semantic tokens, a DiT flow decoder makes mel and a HiFi-GAN vocodes to
24 kHz. Phonetic control is "pronunciation inpainting": bracketed [initial][final]
pinyin tokens (tone as a Unicode diacritic) inlined in the text, already injected by the
driver, so synthesize() just forwards the text — the brackets survive text normalization
untouched. Zero-shot needs the reference wav plus its transcript; we wrap the transcript
as `<system><|endofprompt|><transcript>`, the prompt format the model card uses. See
docs/tts-systems.md.
"""

from __future__ import annotations

from .base import SynthResult, TtsAdapter

# Repo is git-cloned (with the third_party/Matcha-TTS submodule) into this path at image
# build time; both must be on sys.path for `import cosyvoice` to resolve.
_REPO_DIR = "/opt/CosyVoice"
_HF_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
# Zero-shot prompt format from the model card: a system prompt, the <|endofprompt|>
# separator token, then the exact transcript of the reference wav.
_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


class CosyVoice3Adapter(TtsAdapter):
    name = "cosyvoice3"
    sample_rate = 24000

    def load(self) -> None:
        import os
        import sys

        for p in (_REPO_DIR, os.path.join(_REPO_DIR, "third_party", "Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)

        import torch  # noqa: F401  (ensures CUDA torch is importable before model build)
        from huggingface_hub import snapshot_download

        from cosyvoice.cli.cosyvoice import AutoModel

        # Pull weights from HF (persisted on the cache volume) and hand the local path to
        # AutoModel — passing the bare HF id would route through modelscope instead.
        model_dir = snapshot_download(_HF_ID)
        # AutoModel detects cosyvoice3.yaml in the dir and builds CosyVoice3 (Qwen2 backbone
        # auto-loaded from the CosyVoice-BlankEN/ subdir); weights load onto cuda.
        self.model = AutoModel(model_dir=model_dir)

    def synthesize(
        self, text: str, ref_wav_path: str, ref_text: str, controlled: bool = False
    ) -> SynthResult:
        import numpy as np

        prompt_text = _PROMPT_PREFIX + (ref_text or "")
        # text_normalize may split the utterance into several chunks; concatenate them.
        segments = []
        for out in self.model.inference_zero_shot(
            text, prompt_text, ref_wav_path, stream=False
        ):
            seg = out["tts_speech"]  # torch tensor (channels, samples) at 24 kHz
            segments.append(np.asarray(seg.detach().cpu().numpy(), dtype=np.float32))
        if segments:
            audio = np.concatenate(segments, axis=1)
        else:
            audio = np.zeros((1, 0), dtype=np.float32)
        mono = audio.mean(axis=0)  # downmix if the model ever emits >1 channel
        return SynthResult(audio=np.asarray(mono, dtype=np.float32), sample_rate=self.sample_rate)
