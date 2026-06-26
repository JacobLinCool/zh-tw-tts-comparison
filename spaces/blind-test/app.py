"""zh-TW TTS Arena — blind A/B listening test (Hugging Face Space).

Each trial plays the SAME sentence synthesized by two different TTS systems, blinded as
語音 A / 語音 B (model→side randomized). The listener picks the better one or 平手 (tie).
Each session is ~5 trials to keep fatigue low; votes append to a HF Dataset and a static
GitHub Pages site recomputes a Bradley-Terry ranking from them.

Models play their most realistic input (ensub_ctrl where the model supports phonetic
control, else ensub) — baked into pool.json by scripts/build_pool.py.

Persistence: huggingface_hub.CommitScheduler appends an append-only JSONL to the votes
dataset (UUID file per Space instance, batched commits, thread-safe). Set the HF_TOKEN
secret (write scope) in the Space settings. For local UI testing run with LOCAL_VOTES=1
(or simply no HF_TOKEN) to write ./votes_local.jsonl instead.

    LOCAL_VOTES=1 python app.py
"""

from __future__ import annotations

import html
import json
import os
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

# ---- config (mirrors tts_compare.config; the Space is self-contained) ----
VOTES_DATASET = "JacobLinCool/zh-tw-tts-arena-votes"
APP_VERSION = "1"
TRIALS_PER_ROUND = 5

HERE = Path(__file__).resolve().parent
POOL = json.loads((HERE / "pool.json").read_text(encoding="utf-8"))
SENTENCES = POOL["sentences"]
MODELS = POOL["models"]

# ---- vote persistence ----
HF_TOKEN = os.environ.get("HF_TOKEN")
LOCAL = os.environ.get("LOCAL_VOTES") == "1" or not HF_TOKEN

if LOCAL:
    scheduler = None
    _lock = threading.Lock()
    VOTES_FILE = HERE / "votes_local.jsonl"
    print(f"[votes] LOCAL mode → {VOTES_FILE}")
else:
    from huggingface_hub import CommitScheduler

    votes_dir = HERE / "votes"
    votes_dir.mkdir(exist_ok=True)
    VOTES_FILE = votes_dir / f"votes_{uuid.uuid4()}.jsonl"
    scheduler = CommitScheduler(
        repo_id=VOTES_DATASET,
        repo_type="dataset",
        folder_path=votes_dir,
        path_in_repo="data",
        every=5,
        token=HF_TOKEN,
    )
    _lock = scheduler.lock
    print(f"[votes] HF mode → {VOTES_DATASET} ({VOTES_FILE.name})")

# Cross-session pair-balance counter (resets on Space restart; best-effort coverage).
GLOBAL_PAIR_USE: dict[frozenset, int] = {}
_pair_lock = threading.Lock()


def save_vote(trial: dict, winner: str, session: str, dwell_s: float, request: gr.Request) -> None:
    ua = ""
    if request is not None:
        try:
            ua = request.headers.get("user-agent", "")[:300]
        except Exception:
            ua = ""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": session,
        "sentence_id": trial["sid"],
        "model_a": trial["model_a"],
        "model_b": trial["model_b"],
        "condition_a": trial["cond_a"],
        "condition_b": trial["cond_b"],
        "winner": winner,  # "a" | "b" | "tie"
        "dwell_s": round(dwell_s, 1),
        "ua": ua,
        "app_version": APP_VERSION,
    }
    with _lock:
        with VOTES_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---- sampling ----
def _pairs(models: list[str]) -> list[tuple[str, str]]:
    return [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]


def build_round() -> list[dict]:
    trials, used = [], set()
    attempts = 0
    while len(trials) < TRIALS_PER_ROUND and attempts < 200:
        attempts += 1
        s = random.choice(SENTENCES)
        cand = [p for p in _pairs(list(s["clips"].keys())) if (s["id"], frozenset(p)) not in used]
        if not cand:
            continue
        random.shuffle(cand)
        with _pair_lock:
            cand.sort(key=lambda p: GLOBAL_PAIR_USE.get(frozenset(p), 0))
            a, b = cand[0]
            GLOBAL_PAIR_USE[frozenset((a, b))] = GLOBAL_PAIR_USE.get(frozenset((a, b)), 0) + 1
        used.add((s["id"], frozenset((a, b))))
        if random.random() < 0.5:  # randomize which model is shown as A (left)
            a, b = b, a
        trials.append({
            "sid": s["id"], "text": s["text"], "entities": s.get("entities", []),
            "model_a": a, "model_b": b,
            "cond_a": s["clips"][a]["condition"], "cond_b": s["clips"][b]["condition"],
            "url_a": s["clips"][a]["url"], "url_b": s["clips"][b]["url"],
        })
    return trials


# ---- rendering ----
def _highlight(text: str, entities: list[str]) -> str:
    out = html.escape(text)
    for e in sorted({x for x in entities if x and any("一" <= c <= "鿿" for c in x)}, key=len, reverse=True):
        out = out.replace(html.escape(e), f"<mark>{html.escape(e)}</mark>")
    return out


def _sentence_html(t: dict) -> str:
    return (f"<div class='sent'><div class='txt'>{_highlight(t['text'], t['entities'])}</div></div>")


def _players_html(t: dict) -> str:
    def card(lab, url):
        return (f"<div class='pcard'><div class='plab'>語音 {lab}</div>"
                f"<audio controls preload='none' src='{html.escape(url)}'></audio></div>")
    return f"<div class='ab'>{card('A', t['url_a'])}{card('B', t['url_b'])}</div>"


def _reveal_html(t: dict, winner: str) -> str:
    pick = {"a": "語音 A 較好", "b": "語音 B 較好", "tie": "平手 / 分不出"}[winner]
    return (f"<div class='reveal'><div class='pick'>你的選擇：<b>{pick}</b></div>"
            f"<div class='who'>語音 A = <code>{t['model_a']}</code> ({t['cond_a']})　·　"
            f"語音 B = <code>{t['model_b']}</code> ({t['cond_b']})</div></div>")


# outputs order shared by every handler
def view(state: dict, phase: str, winner: str | None = None):
    trials, idx = state["trials"], state["idx"]
    t = trials[idx]
    last = idx == len(trials) - 1
    prog = f"第 {idx + 1} / {len(trials)} 題　·　本場已投 {state['count']} 票"
    voting = phase == "vote"
    return (
        state,
        gr.update(value=prog),
        gr.update(value=_sentence_html(t)),
        gr.update(value=_players_html(t)),
        gr.update(interactive=voting),  # btn_a
        gr.update(interactive=voting),  # btn_tie
        gr.update(interactive=voting),  # btn_b
        gr.update(value="" if voting else _reveal_html(t, winner), visible=not voting),
        gr.update(visible=not voting, value=("🎧 再聽 5 題" if last else "下一題 →")),
    )


def on_load(state):
    state = {"session": uuid.uuid4().hex[:16], "count": 0,
             "trials": build_round(), "idx": 0, "t0": time.time()}
    return view(state, "vote")


def _vote(state, winner, request):
    t = state["trials"][state["idx"]]
    save_vote(t, winner, state["session"], time.time() - state.get("t0", time.time()), request)
    state["count"] += 1
    return view(state, "revealed", winner)


def vote_a(state, request: gr.Request):
    return _vote(state, "a", request)


def vote_tie(state, request: gr.Request):
    return _vote(state, "tie", request)


def vote_b(state, request: gr.Request):
    return _vote(state, "b", request)


def on_next(state):
    if state["idx"] < len(state["trials"]) - 1:
        state["idx"] += 1
    else:
        state["trials"] = build_round()
        state["idx"] = 0
    state["t0"] = time.time()
    return view(state, "vote")


CSS = """
#wrap { max-width: 820px; margin: 0 auto; }
.sent { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
        border-radius: 12px; padding: 16px 18px; margin: 6px 0 14px; }
.sent .txt { font-size: 20px; line-height: 1.8; }
.sent mark { background: #2b3a55; color: #cfe0ff; padding: 0 3px; border-radius: 4px; }
.ab { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.pcard { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
         border-radius: 12px; padding: 14px; text-align: center; }
.plab { font-size: 17px; font-weight: 700; margin-bottom: 10px; }
.pcard audio { width: 100%; }
.reveal { background: var(--block-background-fill); border: 1px dashed var(--border-color-accent, #6ea8fe);
          border-radius: 12px; padding: 12px 16px; margin-top: 4px; }
.reveal .pick { font-size: 15px; } .reveal .who { color: var(--body-text-color-subdued); margin-top: 4px; }
@media (max-width: 560px) { .ab { grid-template-columns: 1fr; } }
"""

with gr.Blocks(title="zh-TW TTS Arena — 盲測") as demo:
    with gr.Column(elem_id="wrap"):
        gr.Markdown(
            "## 🎧 台灣中文 TTS 盲測 (Blind A/B)\n"
            "聽**同一句話**由兩個系統合成的版本（已匿名為 語音 A / 語音 B），選出你覺得**比較自然、好聽、發音正確**的一個，"
            "分不出來就按「平手」。每場約 5 題，投票後才會揭曉是哪個系統。你的偏好會匿名收集，用來排出人類偏好榜。"
        )
        progress = gr.Markdown()
        sent = gr.HTML()
        players = gr.HTML()
        with gr.Row():
            btn_a = gr.Button("◀ 語音 A 較好", variant="primary")
            btn_tie = gr.Button("平手 / 分不出")
            btn_b = gr.Button("語音 B 較好 ▶", variant="primary")
        reveal = gr.HTML(visible=False)
        next_btn = gr.Button("下一題 →", visible=False)
        state = gr.State()

    OUTPUTS = [state, progress, sent, players, btn_a, btn_tie, btn_b, reveal, next_btn]
    demo.load(on_load, inputs=[state], outputs=OUTPUTS)
    btn_a.click(vote_a, inputs=[state], outputs=OUTPUTS)
    btn_tie.click(vote_tie, inputs=[state], outputs=OUTPUTS)
    btn_b.click(vote_b, inputs=[state], outputs=OUTPUTS)
    next_btn.click(on_next, inputs=[state], outputs=OUTPUTS)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
