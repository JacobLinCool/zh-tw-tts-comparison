"""Create/update the Gradio blind-test Space and push spaces/blind-test/ to it.

Requires HF write auth (`hf auth login` or HF_TOKEN). The Space needs an `HF_TOKEN`
secret (write scope) so it can append votes to the votes dataset — set it in the Space
UI, or pass --set-token to copy it from the HF_WRITE_TOKEN env var here.

    uv run python scripts/deploy_space.py
    HF_WRITE_TOKEN=hf_xxx uv run python scripts/deploy_space.py --set-token
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tts_compare.config import HF_SPACE  # noqa: E402

SPACE_DIR = ROOT / "spaces" / "blind-test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-token", action="store_true",
                    help="copy HF_WRITE_TOKEN env var into the Space's HF_TOKEN secret")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"authenticated as {api.whoami().get('name')}")

    api.create_repo(HF_SPACE, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=HF_SPACE,
        repo_type="space",
        ignore_patterns=["votes/*", "votes_local.jsonl", "__pycache__/*", "*.pyc"],
    )
    print(f"pushed {SPACE_DIR} → https://huggingface.co/spaces/{HF_SPACE}")

    if args.set_token:
        tok = os.environ.get("HF_WRITE_TOKEN")
        if not tok:
            print("--set-token given but HF_WRITE_TOKEN env var is empty; skipping")
        else:
            api.add_space_secret(repo_id=HF_SPACE, key="HF_TOKEN", value=tok)
            print("set HF_TOKEN secret on the Space")
    else:
        print("reminder: set the HF_TOKEN secret (write scope) in the Space settings, "
              "otherwise votes are not persisted")


if __name__ == "__main__":
    main()
