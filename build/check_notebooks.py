#!/usr/bin/env python3
"""Run the solution notebooks as scripts, so a broken exercise is caught before print.

    python3 build/check_notebooks.py notebooks/Chapter_01_*_SOLUTIONS.ipynb

Every code cell is concatenated in order and executed in a fresh interpreter. The
solution notebooks end with progress(), and this script fails if any check did not
pass — which means a reader could not have passed it either.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SKIP_PREFIXES = ("%pip", "!pip", "%matplotlib")


def to_script(nb_path):
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    parts = ["import matplotlib\nmatplotlib.use('Agg')\n"]
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        lines = [ln for ln in src.split("\n")
                 if not ln.strip().startswith(SKIP_PREFIXES)]
        parts.append("\n".join(lines))
    parts.append(
        "\nimport sys\n"
        "_bad = {k: v for k, v in _RESULTS.items() if v != 'pass'}\n"
        "if _bad:\n"
        "    print('CHECKS NOT PASSING:', _bad); sys.exit(1)\n"
        "print(f'all {len(_RESULTS)} checks passed')\n"
    )
    return "\n\n".join(parts)


def run_solutions(paths):
    failures = []
    for nb_path in paths:
        script = to_script(nb_path)
        with tempfile.TemporaryDirectory() as tmp:
            run_py = Path(tmp) / "run_notebook.py"
            run_py.write_text(script, encoding="utf-8")
            print(f"--- running {Path(nb_path).name}")
            proc = subprocess.run([sys.executable, "-I", str(run_py)],
                                  capture_output=True, text=True, cwd=tmp)
        tail = (proc.stdout or "").strip().split("\n")[-12:]
        print("\n".join(tail))
        if proc.returncode != 0:
            print((proc.stderr or "").strip()[-3000:])
            failures.append(nb_path)
            print(f"FAILED: {Path(nb_path).name}\n")
        else:
            print(f"OK: {Path(nb_path).name}\n")
    return failures


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(str(p) for p in
                      (Path(__file__).resolve().parent.parent / "notebooks")
                      .glob("*_SOLUTIONS.ipynb"))
    sys.exit(1 if run_solutions(args) else 0)
