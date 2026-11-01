#!/usr/bin/env python3
"""Build every chapter's exercise and solution notebooks.

    python3 build/build_notebooks.py            # writes into notebooks/
    python3 build/build_notebooks.py --check     # also runs the solution notebooks

Add a chapter by writing build/chNN.py with CELLS, TITLE and SOLUTION_BANNER,
then listing it in CHAPTERS below.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))

from nbtools import build  # noqa: E402

CHAPTERS = ["ch01", "ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08", "ch09", "ch10", "ch11", "ch12", "ch13", "ch14", "ch15", "ch16", "ch17"]
OUT_DIR = ROOT / "notebooks"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="execute each solution notebook after building")
    args = parser.parse_args()

    written = []
    for name in CHAPTERS:
        module = __import__(name)
        paths = build(module.CELLS, OUT_DIR, module.TITLE, module.SOLUTION_BANNER)
        written.extend(paths)
        for p in paths:
            print(f"wrote {p.relative_to(ROOT)}")

    if args.check:
        from check_notebooks import run_solutions
        failures = run_solutions([p for p in written if "_SOLUTIONS" in p.name])
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
