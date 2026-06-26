"""Modal app for the zh-TW TTS comparison.

Each TTS system is a Modal class with its own image/GPU; the model loads once in
`@enter` and stays warm. Phonetic control is resolved locally (tts_compare.phonetics)
and the controlled text is sent to the container, which only synthesizes. Audio comes
back as wav bytes and is saved LOCALLY under outputs/<model>/<condition>/<id>.wav, then
scored with Qwen3-ASR (no context) using Mixed Error Rate.

Pilot run (cheap, validates the whole pipeline end to end):
    uv run modal run modal_app.py --models omnivoice --limit 2

Full run:
    uv run modal run modal_app.py --models omnivoice --limit 0
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path

import modal

from tts_compare import config

app = modal.App("zh-tw-tts-comparison")

# Persistent HF weight cache shared across containers/runs.
hf_cache = modal.Volume.from_name("zh-tts-hf-cache", create_if_missing=True)
CACHE_DIR = "/cache"
CACHE_ENV = {"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}


def _encode_wav(audio, sample_rate: int) -> bytes:
    """float32 mono [-1,1] -> 16-bit PCM wav bytes (runs in-container)."""
    import numpy as np
    import soundfile as sf

    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 1.0:
        a = a / peak
    buf = io.BytesIO()
    sf.write(buf, a, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _write_ref(ref_wav_bytes: bytes) -> str:
    p = "/tmp/ref_16k.wav"
    Path(p).write_bytes(ref_wav_bytes)
    return p


# Peak device VRAM (bytes) used in this container — covers ALL frameworks (torch,
# onnxruntime, etc.) because each model owns its GPU, so device-level "used" == this model.
_vram = {"peak_bytes": 0}
_vram_started = {"on": False}


def _start_vram_sampler() -> None:
    if _vram_started["on"]:
        return
    _vram_started["on"] = True

    def _loop():
        import torch

        while True:
            try:
                free, total = torch.cuda.mem_get_info()
                used = total - free
                if used > _vram["peak_bytes"]:
                    _vram["peak_bytes"] = used
            except Exception:
                pass
            time.sleep(0.05)

    threading.Thread(target=_loop, daemon=True).start()


def _synth_and_measure(adapter, text: str, ref_wav_bytes: bytes, ref_text: str, controlled: bool):
    """Synthesize -> (wav bytes, peak VRAM GB, inference seconds). Timing is measured INSIDE
    the container around the model call — pure inference latency, no client network/queue time,
    so it stays valid even when models run concurrently. Peak VRAM is monotonic per container."""
    ref_path = _write_ref(ref_wav_bytes)
    t0 = time.perf_counter()
    res = adapter.synthesize(text, ref_path, ref_text, controlled)
    infer_s = round(time.perf_counter() - t0, 3)
    wav = _encode_wav(res.audio, res.sample_rate)
    return wav, round(_vram["peak_bytes"] / 1e9, 2), infer_s


# --------------------------------------------------------------------------- #
# Images                                                                       #
# --------------------------------------------------------------------------- #
omnivoice_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.8.0+cu128",
        "torchaudio==2.8.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("omnivoice", "soundfile", "numpy", "hf_transfer")
    .env(CACHE_ENV)
    .add_local_python_source("tts_compare")
)

breezyvoice_image = (
    # CUDA 11.8 + cudnn8 base matches the repo Dockerfile, torch 2.3.1+cu118, and
    # onnxruntime-gpu 1.16; nvidia/cuda has no python so add a standalone 3.10.
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "git-lfs", "ffmpeg", "libsndfile1",
        "curl", "ca-certificates", "build-essential",
    )
    .run_commands(
        # BreezyVoice is NOT pip-installable: clone the repo and run from inside it.
        "git clone --depth 1 https://github.com/mtkresearch/BreezyVoice.git /opt/BreezyVoice",
        # Drop the two trailing modelscope ttsfrd wheel URLs (optional, frequently fail);
        # the frontend auto-falls back to WeTextProcessing.
        "sed -i '/modelscope/d' /opt/BreezyVoice/requirements.txt",
        # setuptools>=81 removed pkg_resources, which pynini/WeTextProcessing import at
        # build time. Pin setuptools<81 and propagate it into pip's ISOLATED build envs
        # via PIP_CONSTRAINT so those legacy wheels build.
        "pip install -U pip 'setuptools<81' wheel",
        "printf 'setuptools<81\\n' > /opt/pipc.txt",
        # Pinned deps incl. torch==2.3.1+cu118 and onnxruntime-gpu==1.16.0 + WeTextProcessing.
        "PIP_CONSTRAINT=/opt/pipc.txt pip install --no-cache-dir "
        "-r /opt/BreezyVoice/requirements.txt "
        "--extra-index-url https://download.pytorch.org/whl/cu118",
    )
    # ruamel.yaml>=0.18 breaks HyperPyYAML's Loader (composer reads loader.max_depth,
    # which the old Loader lacks); pin <0.18 so single_inference's load_hyperpyyaml works.
    .pip_install("hf_transfer", "ruamel.yaml<0.18")
    .env({
        "PYTHONUTF8": "1",
        "BREEZYVOICE_DIR": "/opt/BreezyVoice",
        # Cache the G2PW ONNX model on the mounted volume (CACHE_DIR == /cache) so cold
        # starts reuse it instead of re-downloading.
        "G2PW_MODEL_DIR": "/cache/G2PWModel",
    })
    .env(CACHE_ENV)
    .add_local_python_source("tts_compare")
)

chatterbox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("chatterbox-tts==0.1.7", "soundfile", "numpy", "hf_transfer")
    .env(CACHE_ENV)
    .add_local_python_source("tts_compare")
)

cosyvoice3_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "sox", "libsox-dev", "ffmpeg", "unzip", "build-essential")
    # Install the repo's pinned torch FIRST so deepspeed/pyworld build against it.
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Clone the repo (NOT pip-installable) with the REQUIRED third_party/Matcha-TTS
    # submodule, then install its full requirements (the file carries its own
    # --extra-index-url lines for cu121 torch + cu12 onnxruntime-gpu).
    .run_commands(
        "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice",
        "cd /opt/CosyVoice && git submodule update --init --recursive",
        # setuptools<81 keeps pkg_resources available for legacy wheel builds (pynini etc.).
        "pip install -U pip 'setuptools<81' wheel",
        "printf 'setuptools<81\\n' > /opt/pipc.txt",
        "cd /opt/CosyVoice && PIP_CONSTRAINT=/opt/pipc.txt pip install -r requirements.txt",
        # deepspeed is training-only and JIT-compiles CUDA ops at import (needs nvcc/CUDA_HOME,
        # absent in debian_slim), which breaks the transformers import. Inference needs none of it.
        "pip uninstall -y deepspeed || true",
    )
    # ruamel.yaml<0.18 keeps HyperPyYAML's Loader working (same max_depth issue as BreezyVoice).
    .pip_install("huggingface_hub", "hf_transfer", "soundfile", "ruamel.yaml<0.18")
    .env(CACHE_ENV)
    .add_local_python_source("tts_compare")
)

voxcpm2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.8.0+cu128",
        "torchaudio==2.8.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("voxcpm", "soundfile", "numpy", "hf_transfer")
    .env(CACHE_ENV)
    .add_local_python_source("tts_compare")
)

moss_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch==2.9.1+cu128",
        "torchaudio==2.9.1+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers>=4.57",
        "accelerate",
        "huggingface_hub",
        "soundfile",
        "numpy",
        "hf_transfer",
    )
    .env(CACHE_ENV)
    .add_local_python_source("tts_compare")
)

qwen3tts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "sox")
    .pip_install(
        "torch==2.8.0+cu128",
        "torchaudio==2.8.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "qwen-tts",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "soundfile",
        "numpy",
        "hf_transfer",
    )
    .env(CACHE_ENV)
    .add_local_python_source("tts_compare")
)

asr_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("qwen-asr", "soundfile", "librosa", "numpy", "hf_transfer")
    .env(CACHE_ENV)
    .add_local_python_source("tts_compare")
)

# Second ASR: MediaTek Breeze-ASR-25 (Whisper-large-v2, tuned for Taiwan Mandarin + code-switch).
# Cross-checking against Qwen3-ASR is fairer — a mainland-biased ASR under-rates Taiwan accent.
breeze_asr_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.8.0+cu128",
        "torchaudio==2.8.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("transformers", "accelerate", "soundfile", "numpy", "hf_transfer")
    .env(CACHE_ENV)
    .add_local_python_source("tts_compare")
)


# --------------------------------------------------------------------------- #
# TTS classes                                                                  #
# --------------------------------------------------------------------------- #
class OmniVoice:
    @modal.enter()
    def _load(self):
        from tts_compare.adapters.omnivoice import OmniVoiceAdapter

        self.adapter = OmniVoiceAdapter()
        self.adapter.load()
        _start_vram_sampler()

    @modal.method()
    def synthesize(self, text: str, ref_wav_bytes: bytes, ref_text: str, controlled: bool):
        return _synth_and_measure(self.adapter, text, ref_wav_bytes, ref_text, controlled)


class BreezyVoice:
    @modal.enter()
    def _load(self):
        from tts_compare.adapters.breezyvoice import BreezyVoiceAdapter

        self.adapter = BreezyVoiceAdapter()
        self.adapter.load()
        _start_vram_sampler()

    @modal.method()
    def synthesize(self, text: str, ref_wav_bytes: bytes, ref_text: str, controlled: bool):
        return _synth_and_measure(self.adapter, text, ref_wav_bytes, ref_text, controlled)


class Chatterbox:
    @modal.enter()
    def _load(self):
        from tts_compare.adapters.chatterbox import ChatterboxAdapter

        self.adapter = ChatterboxAdapter()
        self.adapter.load()
        _start_vram_sampler()

    @modal.method()
    def synthesize(self, text: str, ref_wav_bytes: bytes, ref_text: str, controlled: bool):
        return _synth_and_measure(self.adapter, text, ref_wav_bytes, ref_text, controlled)


class CosyVoice3:
    @modal.enter()
    def _load(self):
        from tts_compare.adapters.cosyvoice3 import CosyVoice3Adapter

        self.adapter = CosyVoice3Adapter()
        self.adapter.load()
        _start_vram_sampler()

    @modal.method()
    def synthesize(self, text: str, ref_wav_bytes: bytes, ref_text: str, controlled: bool):
        return _synth_and_measure(self.adapter, text, ref_wav_bytes, ref_text, controlled)


class VoxCPM2:
    @modal.enter()
    def _load(self):
        from tts_compare.adapters.voxcpm2 import VoxCPM2Adapter

        self.adapter = VoxCPM2Adapter()
        self.adapter.load()
        _start_vram_sampler()

    @modal.method()
    def synthesize(self, text: str, ref_wav_bytes: bytes, ref_text: str, controlled: bool):
        return _synth_and_measure(self.adapter, text, ref_wav_bytes, ref_text, controlled)


class Moss:
    @modal.enter()
    def _load(self):
        from tts_compare.adapters.moss import MossAdapter

        self.adapter = MossAdapter()
        self.adapter.load()
        _start_vram_sampler()

    @modal.method()
    def synthesize(self, text: str, ref_wav_bytes: bytes, ref_text: str, controlled: bool):
        return _synth_and_measure(self.adapter, text, ref_wav_bytes, ref_text, controlled)


class Qwen3Tts:
    @modal.enter()
    def _load(self):
        from tts_compare.adapters.qwen3tts import Qwen3TtsAdapter

        self.adapter = Qwen3TtsAdapter()
        self.adapter.load()
        _start_vram_sampler()

    @modal.method()
    def synthesize(self, text: str, ref_wav_bytes: bytes, ref_text: str, controlled: bool):
        return _synth_and_measure(self.adapter, text, ref_wav_bytes, ref_text, controlled)


# Register only the models named in TTS_ACTIVE (default: all) by applying the Modal
# decorator functionally. A single failing image build no longer blocks the others —
# e.g. `TTS_ACTIVE=voxcpm2,chatterbox` builds/runs just those two.
_SPECS = [
    ("omnivoice", OmniVoice, omnivoice_image, "L4"),
    ("breezyvoice", BreezyVoice, breezyvoice_image, "L4"),
    ("chatterbox", Chatterbox, chatterbox_image, "L4"),
    ("cosyvoice3", CosyVoice3, cosyvoice3_image, "L4"),
    ("voxcpm2", VoxCPM2, voxcpm2_image, "L4"),
    ("moss", Moss, moss_image, "A10G"),
    ("qwen3tts", Qwen3Tts, qwen3tts_image, "L4"),
]
_ALL = [k for k, *_ in _SPECS]
ACTIVE = set(filter(None, os.environ.get("TTS_ACTIVE", ",".join(_ALL)).split(",")))
# SAME GPU for every model so timing / RTF / peak VRAM are comparable (override via TTS_GPU).
# A100-80GB: every model's CUDA stack (cu11.8–cu12.8) supports Ampere. Blackwell (RTX PRO
# 6000 / B200) would break the older-CUDA models (BreezyVoice cu11.8 / CosyVoice3 / Chatterbox).
GPU = os.environ.get("TTS_GPU", "A100-80GB")
TTS_CLASSES = {}
for _key, _cls, _img, _gpu in _SPECS:
    if _key in ACTIVE:
        TTS_CLASSES[_key] = app.cls(
            image=_img, gpu=GPU, volumes={CACHE_DIR: hf_cache},
            timeout=1800, scaledown_window=300,
        )(_cls)


# --------------------------------------------------------------------------- #
# ASR                                                                          #
# --------------------------------------------------------------------------- #
@app.cls(image=asr_image, gpu="L4", volumes={CACHE_DIR: hf_cache},
         timeout=1800, scaledown_window=300)
class ASR:
    @modal.enter()
    def _load(self):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=8,
            max_new_tokens=1024,
        )

    @modal.method()
    def recognize(self, wav_bytes: bytes, language: str = "Chinese") -> str:
        p = "/tmp/asr_in.wav"
        Path(p).write_bytes(wav_bytes)
        # NO context: pure pronunciation/intelligibility (per experiment decision)
        res = self.model.transcribe(audio=p, context="", language=language)
        return res[0].text


@app.cls(image=breeze_asr_image, gpu="L4", volumes={CACHE_DIR: hf_cache},
         timeout=1800, scaledown_window=300)
class BreezeASR:
    @modal.enter()
    def _load(self):
        import torch
        from transformers import (
            AutomaticSpeechRecognitionPipeline,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        proc = WhisperProcessor.from_pretrained("MediaTek-Research/Breeze-ASR-25")
        model = (
            WhisperForConditionalGeneration.from_pretrained(
                "MediaTek-Research/Breeze-ASR-25", torch_dtype=torch.float16
            )
            .to("cuda")
            .eval()
        )
        self.pipe = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=proc.tokenizer,
            feature_extractor=proc.feature_extractor,
            chunk_length_s=0,  # utterances are < 30 s, no chunking
            device="cuda",
            torch_dtype=torch.float16,
        )

    @modal.method()
    def recognize(self, wav_bytes: bytes) -> str:
        import io

        import numpy as np
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        # the HF ASR pipeline resamples to 16 kHz when given an explicit sampling_rate
        out = self.pipe({"raw": np.asarray(data, dtype=np.float32), "sampling_rate": int(sr)},
                        return_timestamps=True)
        return out["text"]


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(models: str = "omnivoice", limit: int = 0, conditions: str = "raw,controlled",
         asr: bool = True, ids: str = "", out: str = "results.jsonl"):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tts_compare.config import MODELS, REFERENCE_TEXT, REFERENCE_WAV, controllable
    from tts_compare.dataset import load_quick_test
    from tts_compare.phonetics import build_controlled_text
    from tts_compare.scoring import mer
    from tts_compare.substitutions import build_ensub_controlled_text, build_ensub_text, load_sub_map

    rows = load_quick_test()
    if ids:
        wanted = {i.strip() for i in ids.split(",") if i.strip()}
        rows = [r for r in rows if r["id"] in wanted]
    if limit:
        rows = rows[:limit]
    ref_bytes = REFERENCE_WAV.read_bytes()
    model_keys = [m.strip() for m in models.split(",") if m.strip() and m.strip() in TTS_CLASSES]
    cond_list = [c.strip() for c in conditions.split(",") if c.strip()]
    sub_map = load_sub_map()
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Each model owns its container, so run the models CONCURRENTLY (Modal keeps one warm
    # container per model — weights load once in @enter and are reused for every sentence).
    # Timing comes back from inside the container (pure inference), so concurrency between
    # models does not distort it.
    def run_model(mk):
        recs = []
        obj = TTS_CLASSES[mk]()
        wall0 = time.time()
        try:
            obj.synthesize.remote(rows[0]["text"], ref_bytes, REFERENCE_TEXT, False)  # warmup: load once
            print(f"[{mk}] warm after {time.time() - wall0:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"!! [{mk}] warmup failed: {type(e).__name__}: {e}")
        for cond in cond_list:
            if cond in ("controlled", "ensub_ctrl") and not controllable(mk):
                continue
            for row in rows:
                # Target (ref) is what should be SPOKEN; input may carry control annotations.
                if cond == "controlled":
                    text, ref_text = build_controlled_text(row, MODELS[mk]["control"]), row["text"]
                elif cond == "ensub":
                    text = ref_text = build_ensub_text(row, sub_map)
                elif cond == "ensub_ctrl":
                    text = build_ensub_controlled_text(row, sub_map, MODELS[mk]["control"])
                    ref_text = build_ensub_text(row, sub_map)
                else:
                    text = ref_text = row["text"]
                rec = {"model": mk, "condition": cond, "id": row["id"],
                       "ref_text": ref_text, "input_text": text}
                try:
                    # ensub_ctrl also carries phonetic-control annotations → must use the
                    # control path (e.g. VoxCPM2 needs normalize=False so {pinyin} isn't spoken).
                    has_control = cond in ("controlled", "ensub_ctrl")
                    wav, vram, infer_s = obj.synthesize.remote(text, ref_bytes, REFERENCE_TEXT, has_control)
                    rec["synth_s"] = infer_s            # measured inside the container
                    rec["vram_peak_gb"] = vram
                    rec["gpu"] = GPU
                    out = config.OUTPUTS / mk / cond
                    out.mkdir(parents=True, exist_ok=True)
                    wav_path = out / f"{row['id']}.wav"
                    wav_path.write_bytes(wav)
                    rec["wav"] = str(wav_path.relative_to(config.ROOT))
                    rec["rtf"] = round(infer_s / max(row["est_duration_s"], 0.1), 3)
                    print(f"[{mk}/{cond}] {row['id']} synth={infer_s}s vram={vram}GB")
                except Exception as e:  # noqa: BLE001 — keep the batch alive
                    rec["error"] = f"{type(e).__name__}: {e}"
                    print(f"!! [{mk}/{cond}] {row['id']} FAILED: {rec['error']}")
                recs.append(rec)
        return recs

    records = []
    with ThreadPoolExecutor(max_workers=max(1, len(model_keys))) as ex:
        for fut in as_completed([ex.submit(run_model, mk) for mk in model_keys]):
            records.extend(fut.result())

    # ---- ASR + scoring with BOTH ASRs (parallel; the ASRs are instruments, not compared) ----
    if asr:
        asr_models = {"qwen3": ASR(), "breeze": BreezeASR()}
        todo = [r for r in records if "wav" in r]
        for name, obj in asr_models.items():
            if todo:
                try:
                    obj.recognize.remote((config.ROOT / todo[0]["wav"]).read_bytes())  # warmup
                except Exception as e:  # noqa: BLE001
                    print(f"!! [{name}] warmup failed: {type(e).__name__}: {e}")

        def score(rec):
            rec["asr"] = {}
            wav_bytes = (config.ROOT / rec["wav"]).read_bytes()
            for name, obj in asr_models.items():
                try:
                    hyp = obj.recognize.remote(wav_bytes)
                    rec["asr"][name] = {"hyp": hyp, **mer(rec["ref_text"], hyp)}
                except Exception as e:  # noqa: BLE001
                    rec["asr"][name] = {"error": f"{type(e).__name__}: {e}"}
            shown = {n: round(d["mer"], 3) for n, d in rec["asr"].items() if "mer" in d}
            print(f"[ASR] {rec['model']}/{rec['condition']} {rec['id']} {shown}")
            return rec

        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(score, todo))

    results_path = config.OUTPUTS / out
    with results_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- summary ----
    def _mean_mer(sub, asr_name):
        xs = [r["asr"][asr_name]["mer"] for r in sub
              if "asr" in r and asr_name in r["asr"] and "mer" in r["asr"][asr_name]]
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 70)
    for mk in model_keys:
        for cond in cond_list:
            sub = [r for r in records if r["model"] == mk and r["condition"] == cond and "asr" in r]
            if not sub:
                continue
            synth = [r["synth_s"] for r in sub if "synth_s" in r]
            print(f"{mk:12s} {cond:11s} n={len(sub):2d}  "
                  f"MER qwen3={_mean_mer(sub, 'qwen3'):.3f} breeze={_mean_mer(sub, 'breeze'):.3f}  "
                  f"synth={(sum(synth) / len(synth)) if synth else 0:.2f}s")
        vrams = [r["vram_peak_gb"] for r in records if r["model"] == mk and "vram_peak_gb" in r]
        if vrams:
            print(f"{mk:12s} {'peak VRAM':22s} {max(vrams):.1f}GB  (GPU={GPU})")
    print(f"\nwrote {results_path}  | audio under {config.OUTPUTS}/<model>/<condition>/")
