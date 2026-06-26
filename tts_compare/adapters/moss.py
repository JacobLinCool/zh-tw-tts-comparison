"""MOSS-TTS-Local-Transformer-v1.5 (OpenMOSS-Team) adapter — runs in its Modal container.

HF custom-code model (arch ``moss_tts_local``, trust_remote_code=True): a Qwen3-4B
backbone emits 12 RVQ codes per frame, decoded by the MOSS-Audio-Tokenizer-v2 codec to
native 48 kHz STEREO audio. The processor does NO g2p/normalization — text (incl.
space-separated tone3 Pinyin) is BPE-tokenized as-is, so the driver's phonetic control
is forwarded unchanged; `controlled` toggles nothing (MOSS has no normalization mode).
Reference audio is passed for zero-shot cloning. Output is downmixed stereo -> mono.
See docs/tts-systems.md.
"""

from __future__ import annotations

from .base import SynthResult, TtsAdapter


class MossAdapter(TtsAdapter):
    name = "moss"
    sample_rate = 48000  # native 48 kHz (stereo, downmixed to mono below)

    def load(self) -> None:
        import soundfile as sf
        import torch
        import torchaudio
        from transformers import AutoModel, AutoProcessor

        # torchaudio>=2.9 routes load() through torchcodec, whose CUDA libs mismatch this
        # image (libnvrtc.so.13 missing). The processor only reads our wav reference, so
        # swap torchaudio.load for a soundfile reader returning (channels, frames).
        def _sf_load(path, *args, **kwargs):
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return torch.from_numpy(data.T).contiguous(), sr

        torchaudio.load = _sf_load

        # Some CUDA/PyTorch builds ship a broken cuDNN SDPA kernel; disable it.
        torch.backends.cuda.enable_cudnn_sdp(False)

        self.torch = torch
        self.device = "cuda"
        model_id = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            codec_attention_implementation="sdpa",  # don't require flash-attn on the codec
        )
        # The audio codec must live on the GPU alongside the backbone.
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(self.device)
        self.model = (
            AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation="sdpa",  # use "flash_attention_2" only if flash-attn is built
                torch_dtype=torch.bfloat16,
            )
            .to(self.device)
            .eval()
        )

    def synthesize(
        self, text: str, ref_wav_path: str, ref_text: str, controlled: bool = False
    ) -> SynthResult:
        import numpy as np

        torch = self.torch

        # Phonetic control (numbered tone3 Pinyin) is already inline in `text` (driver),
        # and the processor does no g2p, so forward it verbatim. Reference audio drives
        # zero-shot cloning toward the shared Taiwan-accented voice.
        conv = [
            self.processor.build_user_message(
                text=text, language="Chinese", reference=[ref_wav_path]
            )
        ]
        batch = self.processor([conv], mode="generation")

        with torch.no_grad():
            out = self.model.generate(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                max_new_tokens=4096,
                do_sample=True,
                audio_temperature=1.7,
                audio_top_p=0.8,
                audio_top_k=25,
                audio_repetition_penalty=1.0,
            )

        audio = None
        for msg in self.processor.decode(out):
            if msg is None:
                continue
            audio = msg.audio_codes_list[0]  # [channels, samples] @ 48 kHz
            break
        if audio is None:
            raise RuntimeError("MOSS produced no audio")

        # bf16 tensors cannot convert to numpy directly; cast to float32 first.
        wav = audio.detach().to(torch.float32).cpu().numpy()
        if wav.ndim == 2:
            wav = wav.mean(axis=0)  # downmix STEREO -> mono
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        return SynthResult(audio=wav, sample_rate=self.sample_rate)
