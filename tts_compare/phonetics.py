"""Phonetic-control rendering.

The dataset gives each entity a Taiwan-standard reading as diacritic Pinyin
("tái jī diàn") and tone-marked Zhuyin ("ㄊㄞˊ ㄐㄧ ㄉㄧㄢˋ"). Each TTS model wants a
DIFFERENT control syntax. We reduce every Han character to a canonical
`(pinyin_tone3, zhuyin_digit)` pair — preferring the dataset reading when the entity
surface aligns char-for-char, else falling back to pypinyin — then render that pair
into the per-model syntax and inject it at the proper-noun surface inside the sentence.

Sources of truth for the syntaxes: docs/tts-systems.md.
"""

from __future__ import annotations

import re

from pypinyin import Style, pinyin

# diacritic vowel -> (base vowel, tone 1-4)
_TONE = {
    "ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4),
    "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3), "è": ("e", 4),
    "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4),
    "ō": ("o", 1), "ó": ("o", 2), "ǒ": ("o", 3), "ò": ("o", 4),
    "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4),
    "ǖ": ("ü", 1), "ǘ": ("ü", 2), "ǚ": ("ü", 3), "ǜ": ("ü", 4),
    "ń": ("n", 2), "ň": ("n", 3), "ǹ": ("n", 4), "ḿ": ("m", 2),
}
# Zhuyin tone marks -> digit (tone 1 carries no mark).
_ZH_TONE = {"ˊ": "2", "ˇ": "3", "ˋ": "4", "˙": "5"}

# Syllable onsets that CosyVoice3 registers as their own bracket tokens (see
# cosyvoice/tokenizer/tokenizer.py CosyVoice3Tokenizer.special_tokens). `y` and `w`
# ARE trained onset tokens ([y], [w]); a zero-consonant glide syllable is written as
# the onset token plus the BARE-vowel final (yǒng -> [y][ǒng], wáng -> [w][àng]),
# mirroring surface pinyin. Omitting them made the splitter emit a single bracket
# (e.g. [yǒng]) that is NOT in the inventory, so the LLM tokenizer byte-fell-back on
# the literal text and garbled the whole entity span. ü after j/q/x/y is carried by
# the u-form final ([q][uàn], [x][ué], [y][ún]); the y/j/q/x onset disambiguates ü vs u.
_INITIALS = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w",
]
_HAN = re.compile(r"[一-鿿]")

# MOSS feeds `text` verbatim to a Qwen3 BPE tokenizer (no g2p). Its model card requires
# inline tone3 pinyin runs to be SPACE-DELIMITED from surrounding Han text
# ("您好，请问您来自哪 zuo4 cheng2 shi4？") and strips spaces hugging Chinese punctuation
# via fix_punctuation_spacing. Gluing romanized pinyin straight onto a Han character
# ("刷了ai4") is out-of-distribution and can collapse generation. These match the card.
_CN_PUNCT = "，。、！？：；（）「」『』《》〈〉【】…—·～"


def moss_fix_spacing(s: str) -> str:
    """Apply MOSS's documented spacing: single spaces between tokens, none around CN punct."""
    s = re.sub(r"[ \t]+", " ", s)                            # collapse multi-space runs
    s = re.sub(rf"\s+([{re.escape(_CN_PUNCT)}])", r"\1", s)  # no space before CN punctuation
    s = re.sub(rf"([{re.escape(_CN_PUNCT)}])\s+", r"\1", s)  # no space after CN punctuation
    return s.strip()


def diacritic_to_tone3(syllable: str) -> str:
    """'tái' -> 'tai2'; 'jī' -> 'ji1'; unmarked -> tone 5 (neutral)."""
    out, tone = [], None
    for ch in syllable.strip():
        if ch in _TONE:
            base, tone = _TONE[ch]
            out.append(base)
        else:
            out.append(ch)
    return "".join(out) + str(tone if tone is not None else 5)


def zhuyin_to_digit(syllable: str) -> str:
    """'ㄊㄞˊ' -> 'ㄊㄞ2'; tone-1 (no mark) -> 'ㄊㄞ1'. Neutral '˙' may lead -> trailing 5."""
    s = syllable.strip()
    tone = "1"
    glyphs = []
    for ch in s:
        if ch in _ZH_TONE:
            tone = _ZH_TONE[ch]
        else:
            glyphs.append(ch)
    return "".join(glyphs) + tone


def _split_initial_final(base: str) -> tuple[str, str]:
    """'chong' -> ('ch', 'ong'); 'yong' -> ('y', 'ong'); 'wang' -> ('w', 'ang'); 'ai' -> ('', 'ai')."""
    for ini in _INITIALS:
        if base.startswith(ini):
            return ini, base[len(ini):]
    return "", base


def tone3_to_diacritic(tone3: str) -> str:
    """'tai2' -> 'tái'; 'lv3' -> 'lǚ'. Places the mark by standard pinyin rule."""
    m = re.match(r"^([a-zü]+?)([1-5])$", tone3)
    if not m:
        return tone3
    body, tone = m.group(1), int(m.group(2))
    body = body.replace("v", "ü")
    if tone == 5:
        return body
    marks = {
        "a": "āáǎà", "e": "ēéěè", "i": "īíǐì",
        "o": "ōóǒò", "u": "ūúǔù", "ü": "ǖǘǚǜ",
    }
    # priority: a, e, then 'ou' -> o, else last vowel among i/o/u/ü
    if "a" in body:
        target = "a"
    elif "e" in body:
        target = "e"
    elif "ou" in body:
        target = "o"
    else:
        target = next((c for c in reversed(body) if c in marks), None)
    if target is None:
        return body
    return body.replace(target, marks[target][tone - 1], 1)


# ---- canonical reading per Han character ------------------------------------

def _pinyin_tone3(ch: str) -> str:
    r = pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore")
    return r[0][0] if r and r[0] else ""


def _zhuyin_digit(ch: str) -> str:
    r = pinyin(ch, style=Style.BOPOMOFO, errors="ignore")
    return zhuyin_to_digit(r[0][0]) if r and r[0] else ""


def char_readings(surface: str, ds_pinyin: str, ds_zhuyin: str) -> list[dict]:
    """Per-character (pinyin_tone3, zhuyin_digit) for an entity surface.

    Uses the dataset reading when the surface is all-Han and aligns 1:1 with the
    dataset syllables (the curated Taiwan reading, incl. polyphone disambiguation);
    otherwise falls back to pypinyin per character. Non-Han chars get empty readings
    and are emitted verbatim by the renderers.
    """
    py = ds_pinyin.split()
    zh = ds_zhuyin.split()
    han_only = all(_HAN.match(c) for c in surface)
    aligned = han_only and len(surface) == len(py) == len(zh)
    readings = []
    for i, ch in enumerate(surface):
        if not _HAN.match(ch):
            readings.append({"char": ch, "han": False, "tone3": "", "zhuyin": ""})
        elif aligned:
            readings.append({
                "char": ch, "han": True,
                "tone3": diacritic_to_tone3(py[i]), "zhuyin": zhuyin_to_digit(zh[i]),
            })
        else:
            readings.append({
                "char": ch, "han": True,
                "tone3": _pinyin_tone3(ch), "zhuyin": _zhuyin_digit(ch),
            })
    return readings


# ---- per-model rendering of one annotated surface ---------------------------

def render_surface(surface: str, ds_pinyin: str, ds_zhuyin: str, control: str) -> str:
    readings = char_readings(surface, ds_pinyin, ds_zhuyin)
    out = []
    for r in readings:
        if not r["han"]:
            out.append(r["char"])
            continue
        t3, zy = r["tone3"], r["zhuyin"]
        if control == "zhuyin":                       # BreezyVoice: SPARSE [:ㄅㄆㄇN] override
            # BreezyVoice's G2PW frontend already auto-annotates rare/polyphone chars, and the
            # LLM was trained with manual 注音 brackets ONLY on such chars. Annotating every
            # (incl. common, default-read) char is out-of-distribution and garbles output
            # (張永儒 -> 查睿智), so override ONLY when the Taiwan reading differs from the
            # model's default reading; leave default-read chars bare for the native G2PW.
            if zy and zy != _zhuyin_digit(r["char"]):
                out.append(f"{r['char']}[:{zy}]")
            else:
                out.append(r["char"])
        elif control == "pinyin_brace":               # VoxCPM2: {tone3}
            out.append(f"{{{t3}}}" if t3 else r["char"])
        elif control == "pinyin_tone3":               # MOSS: tone3, space-delimited from Han
            out.append(f" {t3} " if t3 else r["char"])
        elif control == "pinyin_upper":               # OmniVoice: TONE3 uppercase
            out.append(t3.upper() if t3 else r["char"])
        elif control == "pinyin_split":               # CosyVoice3: [initial][final-diacritic]
            if not t3:
                out.append(r["char"])
            else:
                body = re.sub(r"[1-5]$", "", t3)
                ini, fin = _split_initial_final(body)
                fin_d = tone3_to_diacritic(fin + t3[-1]) if fin else ""
                toks = (f"[{ini}]" if ini else "") + (f"[{fin_d}]" if fin_d else "")
                out.append(toks or r["char"])
        else:
            out.append(r["char"])
    return "".join(out)


def build_controlled_text(row: dict, control: str) -> str:
    """Rebuild the sentence with each proper-noun surface rendered in `control` syntax.

    `row` is a quick_test sentence with `text` and `annotations` (each carrying
    start/end/surface/pinyin/zhuyin). Non-entity spans are left raw.
    """
    if control == "none":
        return row["text"]
    text = row["text"]
    spans = sorted(
        [a for a in row["annotations"] if "start" in a and "end" in a],
        key=lambda a: a["start"],
    )
    pieces, cursor = [], 0
    for a in spans:
        s, e = a["start"], a["end"]
        if s < cursor:        # overlapping annotation; skip to keep text well-formed
            continue
        pieces.append(text[cursor:s])
        surface = text[s:e]
        pieces.append(render_surface(surface, a.get("pinyin", ""), a.get("zhuyin", ""), control))
        cursor = e
    pieces.append(text[cursor:])
    rebuilt = "".join(pieces)
    if control == "pinyin_tone3":  # MOSS: enforce its documented inline-pinyin spacing
        rebuilt = moss_fix_spacing(rebuilt)
    return rebuilt
