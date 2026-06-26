export const meta = {
  name: 'tts-fix-control',
  description: 'Research the correct phonetic-control format for the 3 broken models and propose concrete fixes',
  phases: [{ title: 'Fix', detail: 'one agent per broken model: read its real g2p/frontend code, propose the fix' }],
}

const MODELS = [
  {
    key: 'breezyvoice', kind: 'zhuyin',
    hint: 'Our controlled condition annotates EVERY entity character as char + bracket-colon + bopomofo + tone-digit (e.g. 張 then [:bopomofo1]). On dense per-char annotation the model garbles output (張永儒 came out as a different word), and it raises IndexError: string index out of range on sentences containing inline English + space (SK 電訊, momo 購物網). Read github.com/mtkresearch/BreezyVoice single_inference.py get_bopomofo_rare() and text_normalize_new() to (a) find why pre-injected brackets next to Latin tokens cause the IndexError, and (b) determine the INTENDED annotation density — BreezyVoice has a G2PW frontend that auto-annotates only rare/polyphone chars; the manual override is meant to be SPARSE. Decide the right controlled strategy (likely: skip get_bopomofo_rare when we pre-inject, and/or only annotate sparingly) and whether to guard the adapter.',
  },
  {
    key: 'cosyvoice3', kind: 'pinyin_split',
    hint: 'Our controlled condition rewrites each entity syllable as bracketed [initial][final-with-tone-diacritic] tokens. It garbles zero-initial / y / w / ü / r syllables (e.g. 永 yong became [yong] as one token, 儒 ru became [r][u]); ASR heard a different word. Read github.com/FunAudioLLM/CosyVoice text frontend / tokenizer (the CosyVoice3 "pronunciation inpainting" or pinyin handling code, e.g. cosyvoice/tokenizer or the frontend g2p) to find the EXACT token inventory the model was trained on: how are initials/finals tokenized, how are y/w/yu/zero-initial syllables represented (does w map to u, y to i, etc.), and the exact tone-diacritic convention. Give the corrected splitter.',
  },
  {
    key: 'moss', kind: 'pinyin_tone3',
    hint: 'Our controlled condition inlines space-separated tone3 pinyin (e.g. ai4 ma3 shi4) inside the text. This CATASTROPHICALLY breaks MOSS: generation collapses and the model parrots the reference-voice transcript instead of synthesizing the target. Read the OpenMOSS-Team/MOSS-TTS model card and github.com/OpenMOSS/MOSS-TTS for the ACTUAL pronunciation-control mechanism (the v1.0/v1.5 capability docs claim pinyin support — find the real required syntax: a tag? a special token? a separate field? Style.TONE3 with specific delimiters?). Determine the correct format, OR conclude that reliable inline pinyin control is unsupported and recommend dropping control for MOSS (run raw only).',
  },
]

const FIX_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['model_key', 'root_cause', 'viable', 'fix_target', 'phonetics_fix', 'adapter_fix', 'correct_format_example', 'confidence', 'evidence'],
  properties: {
    model_key: { type: 'string' },
    root_cause: { type: 'string', description: 'precise reason the controlled condition broke' },
    viable: { type: 'boolean', description: 'can this model do reliable inline phonetic control at all, with the corrected format?' },
    fix_target: { type: 'string', enum: ['phonetics', 'adapter', 'both', 'drop-control'] },
    phonetics_fix: { type: 'string', description: 'concrete change to tts_compare/phonetics.py render_surface (or its helpers) for this control kind: include corrected Python, or "none". If the rendering syntax itself is correct and only the adapter must change, say so.' },
    adapter_fix: { type: 'string', description: 'concrete change to tts_compare/adapters/<model>.py: include corrected Python, or "none".' },
    correct_format_example: { type: 'string', description: 'the correct control syntax with a real worked example (e.g. for 台積電 = tai2 ji1 dian4)' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    evidence: { type: 'string', description: 'what you found in the model\'s actual code/docs, with file paths / URLs' },
  },
}

const prompt = (m) => [
  'You are fixing the phonetic-control ("controlled" condition) for the TTS model "' + m.key + '" in this repo (cwd = repo root). An audit found its controlled output is broken; the phonetic VALUES are correct (from the dataset) but the FORMAT/USAGE is wrong.',
  '',
  'Read these first (Read tool):',
  '  - outputs/AUDIT.md                              the audit verdict and evidence',
  '  - data/audit/' + m.key + '.json                 per-sentence raw-vs-controlled ASR for this model (see the regressions)',
  '  - tts_compare/phonetics.py                      render_surface() and helpers; this model uses control kind "' + m.kind + '"',
  '  - tts_compare/adapters/' + m.key + '.py         the adapter (how text is fed to the model)',
  '  - docs/tts-systems.md                           this model\'s documented control syntax',
  '',
  'Then investigate the model\'s ACTUAL code to find the correct format (curl/clone the GitHub repo, WebFetch the model card / docs, read the real g2p / tokenizer / processor):',
  m.hint,
  '',
  'Deliver a STRICT structured fix: root cause, whether reliable control is viable, what to change in phonetics.py and/or the adapter (concrete Python), and a worked example of the correct syntax. If the model genuinely cannot do reliable inline control, set fix_target="drop-control" and viable=false and explain. Do NOT run modal or the model (no GPU); read code/docs only. Cite evidence (files/URLs) and set confidence honestly.',
].join('\n')

phase('Fix')
const fixes = await parallel(MODELS.map(m => () =>
  agent(prompt(m), { label: 'fix:' + m.key, phase: 'Fix', schema: FIX_SCHEMA, agentType: 'general-purpose' })
))
const ok = fixes.filter(Boolean)
log('collected ' + ok.length + ' fix proposals')
return ok
