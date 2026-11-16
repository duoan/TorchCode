"""Where this copy of TorchCode is hosted on GitHub, read from repo.json at the repo root.

Every "Open in Colab" badge and every absolute GitHub file link is built from this one
setting. Links between files inside the repo are relative and need no setting.

To host TorchCode somewhere else, edit repo.json, then run `make repo-links`.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_cfg = json.loads((ROOT / "repo.json").read_text(encoding="utf-8"))

REPO: str = _cfg["github_repo"]        # "owner/name"
BRANCH: str = _cfg["branch"]

GITHUB_BLOB = f"https://github.com/{REPO}/blob/{BRANCH}"
COLAB_BLOB = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}"


def github_url(path: str) -> str:
    """Absolute GitHub URL for a repo-relative path, e.g. 'templates/01_relu.ipynb'."""
    return f"{GITHUB_BLOB}/{path}"


def colab_url(path: str) -> str:
    """'Open in Colab' URL for a repo-relative notebook path."""
    return f"{COLAB_BLOB}/{path}"
