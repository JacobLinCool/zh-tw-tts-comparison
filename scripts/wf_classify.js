export const meta = {
  name: 'classify-entities',
  description: 'Classify which test-set entities a Taiwanese speaker would code-switch to the Latin/English name',
  phases: [{ title: 'Classify', detail: 'parallel agents classify batches of entities' }],
}

const BATCHES = [[0, 29], [29, 57], [57, 85]]

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['decisions'],
  properties: {
    decisions: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['entity_id', 'surface', 'use_english', 'english_form', 'category', 'confidence', 'reason'],
        properties: {
          entity_id: { type: 'string' },
          surface: { type: 'string' },
          use_english: { type: 'boolean', description: 'true if the sentence should use the Latin/English name instead of the Chinese surface' },
          english_form: { type: 'string', description: 'EXACT Latin string to substitute into the sentence (natural spoken form, e.g. "Adidas", "JPMorgan", "SK Telecom"); empty string if use_english=false' },
          category: { type: 'string', enum: ['western-brand', 'western-person', 'cjk-origin', 'taiwan', 'acronym', 'gray'] },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
          reason: { type: 'string' },
        },
      },
    },
  },
}

const prompt = (start, end) => [
  'You are deciding, for a Taiwan-Mandarin TTS code-switch test, whether each entity should appear in the sentence as the LATIN/English name instead of its Chinese surface.',
  '',
  'Read the file data/entities_in_test.json (a JSON array). Classify EXACTLY the entities at 0-based array indices [' + start + ', ' + end + ') — that is, array slice from index ' + start + ' up to but not including ' + end + '.',
  '',
  'THE TEST (apply Taiwan usage, not Mainland): use_english = true ONLY IF a typical Taiwanese speaker, reading this sentence aloud naturally, would CODE-SWITCH to the Latin/English name rather than pronounce the Chinese characters.',
  '',
  'Guidance:',
  '- TRUE for Western brands Taiwanese normally say in English, especially when the Chinese is a bookish transliteration nobody uses: 蔻馳=Coach, 耐吉=Nike, 愛迪達=Adidas, 思愛普=SAP, 超微=AMD, 歐特克=Autodesk, 谷歌=Google (Taiwan says Google, not 谷歌), GitHub, Visa, Slack, Pinterest, Workday, 保時捷=Porsche, 百思買=Best Buy, etc.',
  '- FALSE when the established Taiwan name IS Chinese even though the brand is Western: 麥當勞 (Taiwanese say 麥當勞, never "McDonald\'s").',
  '- FALSE for CJK-origin entities Taiwanese read in Mandarin: Japanese 花王 (not "Kao"), Korean actors/groups 玄彬/奉俊昊/少女時代, and Chinese people whose name is natively Chinese — 馬雲 (say 馬雲, not "Jack Ma"), 黃仁勳 (Taiwanese-American, Taiwan media say 黃仁勳, not "Jensen Huang"), 陳福陽.',
  '- Western PEOPLE: Taiwanese media usually use the Chinese transliteration, so default FALSE for 舒馬克/珍·古德/譚德塞/拉夫羅夫/菲爾普斯/內馬爾/勞埃德·布蘭克梵 unless the person is overwhelmingly known by the Latin name in Taiwan.',
  '- TW entities (台積電, 聯發科, 張永儒, 魏哲家…): FALSE — Taiwanese say the Chinese. (Even though 台積電≈TSMC, the natural spoken form is 台積電.)',
  '- Korean/Japanese firms with Latin global brands are GRAY: 三星 vs Samsung, SK電訊 vs SK Telecom, NTT DOCOMO — judge by dominant Taiwan usage; mark confidence medium/low and category gray when unsure.',
  '',
  'For english_form give the NATURAL SPOKEN Latin form (may be shorter than the formal en field: e.g. "JPMorgan" not "JPMorgan Chase", "the Fed" or "Federal Reserve", "Reuters"). Empty string when use_english=false.',
  '',
  'Return STRICT structured output with one decision per entity in your assigned slice. Set confidence honestly; use category "gray" + low/medium confidence for genuine judgment calls so a human can review.',
].join('\n')

phase('Classify')
const out = await parallel(BATCHES.map(([s, e]) => () =>
  agent(prompt(s, e), { label: 'classify:' + s + '-' + e, phase: 'Classify', schema: SCHEMA, agentType: 'general-purpose' })
))
const decisions = out.filter(Boolean).flatMap(o => o.decisions)
log('classified ' + decisions.length + ' entities')
return decisions
