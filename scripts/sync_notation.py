#!/usr/bin/env python3
"""Insert or update the "📐 Notation" box in the hand-written problem notebooks.

The symbols come from scripts/notation.py (PROBLEMS). The box goes at the end of the first
cell (the problem statement, which the web UI also shows) of both the template and the
solution. Notebooks without notation entries get any old box removed.

Usage:
    python scripts/sync_notation.py           update notebooks
    python scripts/sync_notation.py --check   exit 1 if any notebook is out of date
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from notation import PROBLEMS, strip_box, used_entries, with_box  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    stems = {p.stem for p in (ROOT / "templates").glob("[0-9][0-9]_*.ipynb") if not p.stem.startswith("00_")}
    unknown = set(PROBLEMS) - stems
    if unknown:
        raise SystemExit(f"notation.PROBLEMS names notebooks that don't exist: {sorted(unknown)}")

    stale = []
    for stem in sorted(stems):
        entries = PROBLEMS.get(stem, [])
        paths = (ROOT / "templates" / f"{stem}.ipynb", ROOT / "solutions" / f"{stem}_solution.ipynb")
        notebooks = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        bodies = [strip_box("".join(nb["cells"][0]["source"])) + "\n" +
                  "\n".join("".join(c["source"]) for c in nb["cells"][1:]) for nb in notebooks]
        never_used = [s for s, _ in entries if not any(used_entries([(s, "")], b) for b in bodies)]
        if never_used:
            raise SystemExit(f"{stem}: notation lists symbols used in neither notebook: {never_used}")
        for path, nb, body in zip(paths, notebooks, bodies):
            cell = nb["cells"][0]
            if cell["cell_type"] != "markdown":
                raise SystemExit(f"{path.name}: first cell is not markdown")
            old = "".join(cell["source"])
            here = used_entries(entries, body)            # only the symbols this file uses
            new = with_box(old, here) if here else (with_box(old, []) if "📐 Notation" in old else old)
            if new != old:
                stale.append(path)
                if not args.check:
                    cell["source"] = new.splitlines(keepends=True)
                    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    for p in stale:
        print(f"{'stale' if args.check else 'updated'}: {p.relative_to(ROOT)}")
    if not stale:
        print(f"notation boxes up to date ({sum(1 for s in stems if s in PROBLEMS)} problems)")
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
