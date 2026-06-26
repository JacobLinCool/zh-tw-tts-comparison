"""BreezyVoice (MediaTek-Research/BreezyVoice) adapter — runs inside its Modal container.

Zero-shot Taiwanese-Mandarin voice cloning built on a fine-tuned CosyVoice-300M. The model
is NOT a pip package: the Modal image git-clones github.com/mtkresearch/BreezyVoice and this
adapter imports `CustomCosyVoice` + `get_bopomofo_rare` from its `single_inference.py`.

Phonetic control is inline 注音 (Bopomofo) brackets placed after a character, e.g.
台積[:ㄐㄧ1]電, already injected by the driver. The fixed pipeline is
`frontend.text_normalize_new(text, split=False)` (protects [:...] spans from TN) then
`get_bopomofo_rare(text, G2PWConverter())` (auto-annotates rare/polyphone chars with 注音 but
SKIPS any char already followed by '[', so manual annotations win) on BOTH the target text and
the reference transcript, fed to `inference_zero_shot_no_normalize`. Because that path preserves
manual brackets, the raw and controlled conditions use the same code (no `controlled` branch).
22.05 kHz output. See docs/tts-systems.md.
"""

from __future__ import annotations

import os
import sys

from .base import SynthResult, TtsAdapter


class BreezyVoiceAdapter(TtsAdapter):
    name = "breezyvoice"
    sample_rate = 22050

    def load(self) -> None:
        # BreezyVoice runs from a clone of its GitHub repo (not pip-installable): put the repo
        # root + its vendored Matcha-TTS on sys.path BEFORE importing single_inference (the
        # module's own sys.path.append for Matcha-TTS happens only AFTER its top-level
        # `from cosyvoice...` imports, so we add both here to be safe).
        repo_dir = os.environ.get("BREEZYVOICE_DIR", "/opt/BreezyVoice")
        for p in (repo_dir, os.path.join(repo_dir, "third_party", "Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)

        from g2pw import G2PWConverter
        import single_inference
        from single_inference import CustomCosyVoice, get_bopomofo_rare

        # BreezyVoice's text_normalize_new() (single_inference.py) calls replace_blank(), which
        # does an unguarded `text[i + 1]` on each space. When a manually 注音-annotated Han char
        # sits at the end of a bracket-split segment AND follows inline "English<space>"
        # (e.g. 'SK 電[:ㄉㄧㄢ4]', 'momo 購[:ㄍㄡ4]'), text_normalize_no_split chops that trailing
        # char (`text = text[:-1]`), exposing a trailing space -> replace_blank reads past the
        # end -> "IndexError: string index out of range". Patch replace_blank to be bounds-safe
        # (a trailing/leading space is simply dropped). text_normalize_new resolves the bare name
        # `replace_blank` from single_inference's module globals, so patching it there suffices.
        def _safe_replace_blank(text: str) -> str:
            out, n = [], len(text)
            for i, c in enumerate(text):
                if c == " ":
                    nxt = text[i + 1] if i + 1 < n else ""
                    prv = text[i - 1] if i >= 1 else ""
                    if (nxt.isascii() and nxt != " ") and (prv.isascii() and prv != " "):
                        out.append(c)
                else:
                    out.append(c)
            return "".join(out)

        single_inference.replace_blank = _safe_replace_blank

        self._get_bopomofo_rare = get_bopomofo_rare
        # Weights auto-download from HF on first construction (snapshot_download honours
        # HF_HOME -> the mounted cache volume). The model loads onto cuda automatically
        # (CosyVoiceModel hardcodes torch.device('cuda' if available else 'cpu')).
        self.cosyvoice = CustomCosyVoice("MediaTek-Research/BreezyVoice")
        # G2PWConverter downloads its ONNX model on first construction; point it at the cache
        # volume (G2PW_MODEL_DIR) so cold starts reuse it instead of re-downloading.
        g2pw_dir = os.environ.get("G2PW_MODEL_DIR", "G2PWModel")
        self.bopomofo = G2PWConverter(model_dir=g2pw_dir)

    def synthesize(
        self, text: str, ref_wav_path: str, ref_text: str, controlled: bool = False
    ) -> SynthResult:
        import numpy as np
        from cosyvoice.utils.file_utils import load_wav

        prompt_speech_16k = load_wav(ref_wav_path, 16000)

        # Normalize (protects manual [:...] spans) then auto-annotate 注音 on rare/polyphone
        # chars, keeping any manual brackets. Same path for both the reference transcript and
        # the target text; `controlled` needs no special handling here.
        prompt_text = self.cosyvoice.frontend.text_normalize_new(ref_text, split=False)
        prompt_text = self._get_bopomofo_rare(prompt_text, self.bopomofo)
        tts_text = self.cosyvoice.frontend.text_normalize_new(text, split=False)
        tts_text = self._get_bopomofo_rare(tts_text, self.bopomofo)

        out = self.cosyvoice.inference_zero_shot_no_normalize(
            tts_text, prompt_text, prompt_speech_16k
        )

        # out['tts_speech'] is a torch tensor shaped [channels, samples] (mono -> [1, N]).
        audio = out["tts_speech"].squeeze().detach().cpu().numpy().astype(np.float32)
        if audio.ndim > 1:  # downmix to mono if the model ever emits multi-channel
            audio = audio.mean(axis=0)
        return SynthResult(audio=audio, sample_rate=self.sample_rate)
