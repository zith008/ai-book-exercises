"""Tiny notebook builder.

Each chapter is described once, as a list of cells. Cells created with `ex()`
carry two versions of the same code: the version the reader gets (with TODOs)
and the worked solution. `build()` writes both notebooks from that one source,
so an exercise and its answer can never drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

EXERCISE = "exercise"
SOLUTION = "solution"

# Where the notebooks are published. Change these three lines and rebuild and every
# "Open in Colab" badge follows. Set REPO to None while the repository is private or
# unpublished: no badge is written, rather than one that 404s for readers.
REPO = "zith008/ai-book-exercises"
BRANCH = "main"
NOTEBOOK_DIR = "notebooks"

BADGE = ("https://colab.research.google.com/assets/colab-badge.svg")


class Cell:
    def __init__(self, kind, exercise, solution=None, form=False):
        self.kind = kind                      # "markdown" | "code"
        self.exercise = exercise
        self.solution = solution if solution is not None else exercise
        self.form = form                      # Colab: hide the code, show title only

    def source(self, variant):
        return self.exercise if variant == EXERCISE else self.solution


def md(text, solution_text=None):
    """A markdown cell."""
    return Cell("markdown", text, solution_text)


def code(src, form=False):
    """A code cell that is identical in both notebooks."""
    return Cell("code", src, form=form)


def ex(todo_src, solution_src):
    """An exercise: TODO version for readers, worked version for the key."""
    return Cell("code", todo_src, solution_src)


def _lines(text):
    """nbformat wants a list of lines, each keeping its trailing newline."""
    text = text.strip("\n")
    if not text:
        return []
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _cell_json(cell, variant):
    src = _lines(cell.source(variant))
    if cell.kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    meta = {"cellView": "form"} if cell.form else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": src,
    }


def _badge_cell(name):
    """A markdown cell with the Colab badge for this notebook, or None."""
    if not REPO:
        return None
    url = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{NOTEBOOK_DIR}/{name}"
    return md(f'<a href="{url}" target="_parent">'
              f'<img src="{BADGE}" alt="Open this notebook in Colab"/></a>')


def notebook(cells, variant, title):
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": title, "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [_cell_json(c, variant) for c in cells],
    }


def build(cells, out_dir, stem, solution_banner):
    """Write <stem>.ipynb and <stem>_SOLUTIONS.ipynb. Returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for variant, suffix in ((EXERCISE, ""), (SOLUTION, "_SOLUTIONS")):
        body = list(cells)
        if variant == SOLUTION:
            body = [md(solution_banner)] + body
        name = f"{stem}{suffix}.ipynb"
        badge = _badge_cell(name)
        if badge is not None:
            body = [badge] + body
        path = out_dir / name
        path.write_text(
            json.dumps(notebook(body, variant, name), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written
