export const meta = {
  name: 'tts-adapters',
  description: 'Write Modal adapters for the 6 remaining TTS systems against the proven OmniVoice contract',
  phases: [{ title: 'Adapters', detail: 'one agent per model writes its adapter file and returns Modal image and Cls code' }],
}

const MODELS = [
  { key: 'breezyvoice', cls: 'BreezyVoice', hint: 'NOT a pip package. The Modal image must git clone https://github.com/mtkresearch/BreezyVoice and run from inside it (adapter imports CustomCosyVoice and get_bopomofo_rare from single_inference.py). Python 3.10. requirements.txt pins torch 2.3.1 plus onnxruntime-gpu; the two trailing modelscope ttsfrd wheels are OPTIONAL and often fail (WeTextProcessing fallback) so remove them or tolerate failure. Needs cudnn. Zero-shot needs ref wav (16k) plus ref_text. Pipeline: cosyvoice.frontend.text_normalize_new(text, split=False), then get_bopomofo_rare(text, G2PWConverter()) on BOTH target text and ref_text, then inference_zero_shot_no_normalize(text, prompt_text, prompt_wav). get_bopomofo_rare PRESERVES manually injected zhuyin brackets and auto-annotates the rest, so the same path serves raw and controlled. Output 22050 Hz. Repo on PYTHONPATH; weights auto-download from MediaTek-Research/BreezyVoice.' },
  { key: 'chatterbox', cls: 'Chatterbox', hint: 'pip install chatterbox-tts. Use ChatterboxMultilingualTTS.from_pretrained(device=cuda); generate(text, language_id=zh, audio_prompt_path=ref_wav). NO phonetic control (driver only sends raw condition; ignore controlled flag). No ref_text. Output 24000 Hz. Default CUDA torch wheels fine.' },
  { key: 'cosyvoice3', cls: 'CosyVoice3', hint: 'NOT pip. git clone https://github.com/FunAudioLLM/CosyVoice with submodules (third_party/Matcha-TTS must be on PYTHONPATH). Load FunAudioLLM/Fun-CosyVoice3-0.5B-2512 weights. Zero-shot needs ref wav (16k) plus ref_text. Phonetic control already injected by driver as bracketed initial/final tokens; forward text as-is. Find the correct CosyVoice3 loader class in the cloned repo and the zero-shot call taking tts_text, prompt_text, prompt_speech_16k. Output 24000 Hz. Hardest one; flag the loader and call you chose and the uncertainty.' },
  { key: 'voxcpm2', cls: 'VoxCPM2', hint: 'pip install voxcpm. Load openbmb/VoxCPM2 (find exact from_pretrained or VoxCPM class API). generate(text=..., normalize=NOT controlled): when controlled is True the text has curly-brace pinyin so pass normalize=False; when False pass normalize=True. Voice cloning optional: pass our ref wav as the prompt/clone kwarg so the voice matches. Output 48000 Hz; downmix to mono.' },
  { key: 'moss', cls: 'Moss', hint: 'transformers with trust_remote_code=True (arch moss_tts_local). Load model plus processor from OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5. Use processor.build_user_message(text=..., language=zh), model.generate, processor.decode to get audio. Phonetic control already injected (space-separated tone3 pinyin); forward text as-is. About 18GB bf16 weights, needs A10G. Reference-less default voice works; if a ref/clone kwarg exists, pass our ref wav. Output 48000 Hz STEREO, downmix to mono float32. Study the model card example usage closely.' },
  { key: 'qwen3tts', cls: 'Qwen3Tts', hint: 'Find the official inference package/repo for Qwen/Qwen3-TTS-12Hz-1.7B-Base (github.com/QwenLM/Qwen3-TTS, likely a qwen-tts style package). Base variant is 3-second zero-shot clone via generate_voice_clone(ref_audio, ref_text) and ICL mode needs both ref_audio and ref_text. NO phonetic control (driver sends only raw; ignore controlled flag). language=Chinese. Output 24000 Hz. Determine the exact class/import and clone call from the repo README.' },
]

const ADAPTER_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['model_key', 'adapter_file', 'adapter_class', 'modal_image_code', 'modal_cls_code', 'gpu', 'python_version', 'needs_git_clone', 'risks', 'confidence'],
  properties: {
    model_key: { type: 'string' },
    adapter_file: { type: 'string', description: 'path of the file you wrote, e.g. tts_compare/adapters/voxcpm2.py' },
    adapter_class: { type: 'string', description: 'adapter class name you defined, e.g. VoxCPM2Adapter' },
    modal_image_code: { type: 'string', description: 'copy-paste-ready python defining KEY_image equals a modal.Image chain, matching modal_app.py conventions: end with .env(CACHE_ENV).add_local_python_source the tts_compare package; include apt_install, pip_install (add hf_transfer), and any run_commands git clone or PYTHONPATH or build steps. CACHE_ENV is already defined in modal_app.py.' },
    modal_cls_code: { type: 'string', description: 'copy-paste-ready python: an app.cls decorated class (image equals KEY_image, gpu set, volumes the CACHE_DIR mapped to hf_cache, timeout 1800, scaledown_window 300) with a modal.enter _load method that constructs and loads the adapter, and a modal.method synthesize(self, text, ref_wav_bytes, ref_text, controlled) returning bytes that calls _write_ref then adapter.synthesize then _encode_wav, mirroring the OmniVoice class.' },
    gpu: { type: 'string' },
    python_version: { type: 'string' },
    needs_git_clone: { type: 'boolean' },
    risks: { type: 'string', description: 'concrete uncertainties and likely failure points to iterate on during the Modal run' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const prompt = (m) => [
  'You are writing a Modal TTS adapter for "' + m.key + '" in an existing repo (cwd is the repo root). The OmniVoice adapter already works end to end; replicate its contract exactly.',
  '',
  'STEP 1, read the contract and the proven example with the Read tool:',
  '  - tts_compare/adapters/base.py        the TtsAdapter and SynthResult contract',
  '  - tts_compare/adapters/omnivoice.py   the working reference adapter',
  '  - modal_app.py                        study CACHE_ENV, CACHE_DIR, hf_cache, _write_ref, _encode_wav, the omnivoice_image definition, and the OmniVoice app.cls class. Your returned snippets MUST match these conventions.',
  '  - data/research/findings.json         find the object whose model_id matches ' + m.key + ' and use its how_to_run.install, how_to_run.entrypoint_snippet, voice_mode, and sample rate as the source of truth.',
  '  - docs/tts-systems.md                 the human-readable section for this model.',
  '',
  'Model-specific guidance (verify against the findings, do not blindly trust): ' + m.hint,
  '',
  'STEP 2, write the adapter file tts_compare/adapters/' + m.key + '.py:',
  '  - define class ' + m.cls + 'Adapter(TtsAdapter) with name set to "' + m.key + '" and sample_rate set to the native rate.',
  '  - ALL heavy imports (torch, the model package) go INSIDE load(), never at module top level, so the module imports cleanly on a machine without the deps, exactly like omnivoice.py.',
  '  - load(self) loads weights onto cuda.',
  '  - synthesize(self, text, ref_wav_path, ref_text, controlled=False) returns SynthResult with audio as a float32 mono numpy array in minus one to one and sample_rate set. Downmix to mono if the model emits stereo. The driver already injected any phonetic control into text; forward it as-is, except where the hint says the controlled flag must toggle a mode (for example VoxCPM2 normalize=False).',
  '  - match the docstring and style of omnivoice.py.',
  '',
  'STEP 3, return strict structured output. Do NOT edit modal_app.py; I integrate your snippets. modal_image_code and modal_cls_code must be copy-paste-ready and consistent with modal_app.py (reference CACHE_ENV, CACHE_DIR, hf_cache, _write_ref, _encode_wav, app).',
  '',
  'Constraints: research by reading files and curl or WebFetch of the model card or GitHub README only. DO NOT pip-install heavy packages and DO NOT run modal or the model (no GPU). Be faithful to the documented inference API; set confidence honestly and list real risks for the iteration phase.',
].join('\n')

phase('Adapters')
const results = await parallel(MODELS.map(m => () =>
  agent(prompt(m), { label: 'adapter:' + m.key, phase: 'Adapters', schema: ADAPTER_SCHEMA, agentType: 'general-purpose' })
))
const ok = results.filter(Boolean)
log('wrote ' + ok.length + ' of ' + MODELS.length + ' adapters')
return ok
