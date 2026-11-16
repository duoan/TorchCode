#!/usr/bin/env python3
"""Point every Colab badge and absolute GitHub file link at the repo named in repo.json.

Rewrites, in place:
  README.md, *.md at the root, templates/*.ipynb, solutions/*.ipynb   Colab + GitHub file links
  labextension/src/index.ts                                         the toolbar Colab button's repo/branch

A link is rewritten only if it points at a file that exists in this repo (templates/,
solutions/, or a root Markdown file), so links to other projects are left alone. Links
that identify the upstream project rather than a file (Docker image, Hugging Face Space,
stars badge, contributors) are not file links and are not touched.

Usage:
    python scripts/sync_repo_links.py           rewrite
    python scripts/sync_repo_links.py --check   exit 1 if anything points elsewhere

The prerequisite notebooks and LEARNING_PATH.md are generated; scripts/prereqs/build.py
reads the same setting, so run `make repo-links` (sync + rebuild) rather than this alone.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_config import BRANCH, COLAB_BLOB, GITHUB_BLOB, REPO, ROOT  # noqa: E402

SEG = r"[^/\s\"'<>()\\]+"   # stop at backslashes too: links inside JSON strings are followed by \"
# Any GitHub/Colab "blob" URL followed by a path inside this repo.
LINK = re.compile(
    rf"(?P<prefix>https://colab\.research\.google\.com/github/{SEG}/{SEG}/blob/{SEG}"
    rf"|https://github\.com/{SEG}/{SEG}/blob/{SEG})"
    rf"/(?P<path>(?:templates|solutions)/{SEG}|{SEG}\.md)"
)
TS_REPO = re.compile(r"^const GITHUB_REPO = '[^']*';$", re.M)
TS_BRANCH = re.compile(r"^const GITHUB_BRANCH = '[^']*';$", re.M)


def rewrite_links(text: str) -> str:
    def sub(m: re.Match) -> str:
        if not (ROOT / m["path"]).exists():
            return m[0]                                   # not a file of this repo: leave it
        base = COLAB_BLOB if m["prefix"].startswith("https://colab") else GITHUB_BLOB
        return f"{base}/{m['path']}"
    return LINK.sub(sub, text)


def targets() -> list[Path]:
    files = sorted(ROOT.glob("*.md"))
    files += sorted((ROOT / "templates").glob("*.ipynb")) + sorted((ROOT / "solutions").glob("*.ipynb"))
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    changed = []
    for path in targets():
        old = path.read_text(encoding="utf-8")
        new = rewrite_links(old)
        if new != old:
            changed.append(path)
            if not args.check:
                path.write_text(new, encoding="utf-8")

    ts = ROOT / "labextension/src/index.ts"
    old = ts.read_text(encoding="utf-8")
    new = TS_BRANCH.sub(f"const GITHUB_BRANCH = '{BRANCH}';", TS_REPO.sub(f"const GITHUB_REPO = '{REPO}';", old))
    assert TS_REPO.search(new) and TS_BRANCH.search(new), "GITHUB_REPO / GITHUB_BRANCH constants not found in index.ts"
    if new != old:
        changed.append(ts)
        if not args.check:
            ts.write_text(new, encoding="utf-8")

    for p in changed:
        print(f"{'stale' if args.check else 'updated'}: {p.relative_to(ROOT)}")
    if not changed:
        print(f"all links point at {REPO}@{BRANCH}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
