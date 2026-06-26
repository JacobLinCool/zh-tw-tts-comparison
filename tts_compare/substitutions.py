"""Build the "ensub" sentence variant: entities whose Latin/English name is how a
Taiwanese speaker would actually say them (per data/sub_final.json, curated by the user)
are substituted from their Chinese surface to the English form — a natural code-switch
rendering. Spacing between CJK and inserted Latin is normalized for natural typography.
"""

from __future__ import annotations

import json
import re

from .config import DATA

_SUB_PATH = DATA / "sub_final.json"


def load_sub_map() -> dict:
    """surface -> {"use": bool, "en": str}."""
    return json.loads(_SUB_PATH.read_text(encoding="utf-8"))


def _norm_spacing(s: str) -> str:
    s = re.sub(r"([一-鿿])([A-Za-z0-9])", r"\1 \2", s)   # Han then Latin -> add space
    s = re.sub(r"([A-Za-z0-9])([一-鿿])", r"\1 \2", s)   # Latin then Han -> add space
    s = re.sub(r" {2,}", " ", s)
    s = re.sub(r" +([，。、！？：；）」』】])", r"\1", s)    # no space before CJK closing punct
    s = re.sub(r"([（「『【]) +", r"\1", s)                # no space after CJK opening punct
    return s.strip()


def _is_sub(sub_map: dict, surface: str) -> bool:
    d = sub_map.get(surface)
    return bool(d and d.get("use") and d.get("en"))


def build_ensub_text(row: dict, sub_map: dict) -> str:
    """Replace substituted entity surfaces with their English form (by char position)."""
    text = row["text"]
    spans = sorted([a for a in row["annotations"] if "start" in a and "end" in a],
                   key=lambda a: a["start"])
    out, cursor = [], 0
    for a in spans:
        s, e = a["start"], a["end"]
        if s < cursor:
            continue
        out.append(text[cursor:s])
        out.append(sub_map[a["surface"]]["en"] if _is_sub(sub_map, a["surface"]) else text[s:e])
        cursor = e
    out.append(text[cursor:])
    return _norm_spacing("".join(out))


def build_ensub_controlled_text(row: dict, sub_map: dict, control: str) -> str:
    """Best-of-both input: English-origin entities -> English name (no control); entities that
    stay Chinese -> phonetic control (pinyin/zhuyin) in the model's syntax. Target (for scoring)
    is build_ensub_text — control annotations only guide pronunciation, they are not spoken.
    """
    from .phonetics import render_surface

    text = row["text"]
    spans = sorted([a for a in row["annotations"] if "start" in a and "end" in a],
                   key=lambda a: a["start"])
    out, cursor = [], 0
    for a in spans:
        s, e = a["start"], a["end"]
        if s < cursor:
            continue
        out.append(text[cursor:s])
        if _is_sub(sub_map, a["surface"]):
            out.append(sub_map[a["surface"]]["en"])                    # English, no control
        elif control != "none":
            out.append(render_surface(text[s:e], a.get("pinyin", ""), a.get("zhuyin", ""), control))
        else:
            out.append(text[s:e])
        cursor = e
    out.append(text[cursor:])
    return _norm_spacing("".join(out))
