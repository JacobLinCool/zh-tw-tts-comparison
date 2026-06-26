"""Integrate the 6 generated adapter image/Cls snippets into modal_app.py.

Reads the adapter specs (from the tts-adapters workflow), inserts each `<key>_image`
definition before `asr_image`, inserts each `@app.cls` class before `TTS_CLASSES`, and
rewrites the TTS_CLASSES registry to include all 7 models. Idempotent-ish: re-running
after manual edits is not safe; regenerate from a clean modal_app.py if needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SPECS = json.loads(Path("/tmp/adapters.json").read_text(encoding="utf-8"))
ORDER = ["breezyvoice", "chatterbox", "cosyvoice3", "voxcpm2", "moss", "qwen3tts"]
by = {s["model_key"]: s for s in SPECS}

app_path = Path("modal_app.py")
src = app_path.read_text(encoding="utf-8")


def clean_cls(code: str) -> str:
    lines = [ln for ln in code.strip().splitlines() if not ln.strip().startswith("# register")]
    return "\n".join(lines).rstrip()


# 1) images before asr_image
images = "\n\n".join(by[k]["modal_image_code"].strip() for k in ORDER)
assert src.count("asr_image = (") == 1
src = src.replace("asr_image = (", images + "\n\n" + "asr_image = (", 1)

# 2) classes before TTS_CLASSES
classes = "\n\n\n".join(clean_cls(by[k]["modal_cls_code"]) for k in ORDER)
assert src.count("TTS_CLASSES = {") == 1
src = src.replace("TTS_CLASSES = {", classes + "\n\n\n" + "TTS_CLASSES = {", 1)

# 3) rewrite the registry dict
clsname = {k: re.search(r"class (\w+)\b", by[k]["modal_cls_code"]).group(1) for k in ORDER}
new_dict = (
    "TTS_CLASSES = {\n"
    '    "omnivoice": OmniVoice,\n'
    + "".join(f'    "{k}": {clsname[k]},\n' for k in ORDER)
    + "}\n"
)
src = re.sub(r"TTS_CLASSES = \{.*?\n\}\n", new_dict, src, count=1, flags=re.S)

app_path.write_text(src, encoding="utf-8")
print("integrated", len(ORDER), "adapters")
print("registry:", ["omnivoice"] + [f"{k}->{clsname[k]}" for k in ORDER])
