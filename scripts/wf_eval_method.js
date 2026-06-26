export const meta = {
  name: 'tts-eval-method',
  description: 'Research the best subjective listening-test methodology for comparing 7 TTS systems on zh-TW pronunciation + code-switch',
  phases: [{ title: 'Research', detail: 'one agent per method family + one on current TTS practice' }],
}

const TOPICS = [
  { key: 'mushra', q: 'MUSHRA (ITU-R BS.1534): multi-stimulus rating with hidden reference and anchor. How it works, the 0-100 scale, why it is used to compare MANY systems at once for the SAME stimulus, post-screening of listeners, statistical analysis, and its known limitations (designed for intermediate-quality audio coding; debates about using it for TTS naturalness). Cite the standard and TTS papers that use it.' },
  { key: 'mos_cmos', q: 'Absolute MOS and Comparison MOS (ITU-T P.800 / P.808 crowdsourced; CMOS / DMOS). One-at-a-time 1-5 absolute rating vs side-by-side comparison on a -3..+3 scale. Range/anchoring bias, why CMOS detects small differences better than absolute MOS, how many ratings/listeners are needed, and how TTS papers report MOS with confidence intervals.' },
  { key: 'preference_ranking', q: '2-alternative forced choice (AB / ABX preference), best-worst scaling, and full ranking of N systems. Statistical efficiency and number of trials for N=7 systems (pairwise = 21 pairs is expensive; best-worst scaling and full ranking are more efficient), Thurstonian / Bradley-Terry models to turn pairwise/rank data into scores, cognitive load of ranking 7 items, and when preference beats absolute rating.' },
  { key: 'tts_practice', q: 'CURRENT (2023-2026) practice for subjectively evaluating multiple TTS systems, especially Mandarin / Taiwan Mandarin / code-switching / pronunciation-correctness studies. What do recent TTS papers and challenges (e.g. Blizzard Challenge) actually use (MOS? CMOS? MUSHRA? preference? side-by-side?), typical listener counts and screening, how they evaluate PROPER-NOUN / pronunciation correctness specifically (often an intelligibility or error-count task, not naturalness MOS), and significance testing.' },
]

const METHOD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['method_family', 'how_it_works', 'scale', 'handles_many_systems', 'sensitivity', 'effort', 'best_for', 'pros', 'cons', 'suitability', 'recommendation', 'citations'],
  properties: {
    method_family: { type: 'string' },
    how_it_works: { type: 'string' },
    scale: { type: 'string', description: 'rating scale / response format' },
    handles_many_systems: { type: 'string', description: 'how it scales to comparing 7 systems at once' },
    sensitivity: { type: 'string', description: 'power to detect small differences + how scores are analyzed statistically' },
    effort: { type: 'string', description: 'listener effort and number of trials/listeners needed' },
    best_for: { type: 'string' },
    pros: { type: 'array', items: { type: 'string' } },
    cons: { type: 'array', items: { type: 'string' } },
    suitability: { type: 'string', description: 'fit for OUR case: compare 7 TTS on Taiwan-Mandarin pronunciation + Chinese-English code-switch, small listener pool, both naturalness AND proper-noun correctness' },
    recommendation: { type: 'string', enum: ['strongly-recommend', 'recommend', 'situational', 'avoid'] },
    citations: { type: 'array', items: { type: 'string' } },
  },
}

const prompt = (t) => [
  'You are researching subjective listening-test methodology for a study that compares SEVEN text-to-speech systems on Taiwan-Mandarin pronunciation and Chinese-English code-switching. A human will listen and judge. The listener pool is small (the researcher + a few colleagues). We care about BOTH overall naturalness/preference AND whether proper nouns are pronounced correctly.',
  '',
  'Research this method family thoroughly and cite primary sources (ITU standards, papers, Blizzard Challenge reports). Use WebSearch / WebFetch.',
  '',
  'TOPIC: ' + t.q,
  '',
  'Return STRICT structured output assessing this method for OUR case (7 systems, zh-TW + code-switch, small listener pool). Be concrete about scale, how many trials/listeners are needed, sensitivity, and whether it suits comparing many systems at once. Set recommendation honestly and cite URLs.',
].join('\n')

phase('Research')
const out = await parallel(TOPICS.map(t => () =>
  agent(prompt(t), { label: 'method:' + t.key, phase: 'Research', schema: METHOD_SCHEMA, agentType: 'general-purpose' })
))
return out.filter(Boolean)
