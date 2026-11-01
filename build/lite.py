"""The beginner template: one small structure, used by every chapter from 3 onward.

Chapters 1 and 2 were written cell by cell and are deliberately deeper. From chapter 3
the shape is fixed and much lighter:

    title -> harness -> given code -> (concept, exercise, check, hint, notice) x3 -> recap

Keeping the shape in one place means a chapter file only has to supply its content, and
every notebook a reader opens behaves identically.
"""

from harness import HARNESS
from nbtools import code, ex, md


class Exercise:
    """One exercise: the idea, the gap to fill, the check, the hint, the takeaway."""

    def __init__(self, name, section, concept, todo, solution, check, hint, notice=None):
        self.name = name            # short label, e.g. "The agent loop"
        self.section = section      # book section, e.g. "3.1"
        self.concept = concept      # markdown: analogy first, then the mechanism
        self.todo = todo            # code the reader gets
        self.solution = solution    # the worked version
        self.check = check          # code: @check(...) blocks
        self.hint = hint            # list of lines for hint(...)
        self.notice = notice        # optional markdown: what to notice afterwards


def _hint_cell(lines):
    body = ",\n     ".join(repr(line) for line in lines)
    return code(f"hint({body})")


def make_chapter(number, title, intro, exercises, given=None, learned=(),
                 minutes=25, needs_key=False, next_up=None, installs=None):
    stem = f"Chapter_{number:02d}_" + title.replace(" ", "_").replace(":", "")

    rows = "\n".join(
        f"| {i} | {e.section} | {e.name} |" for i, e in enumerate(exercises, 1)
    )
    key_line = (
        "This notebook calls a real model, so it needs an API key — the setup cell "
        "explains how to add one in Colab."
        if needs_key else
        "**No API key needed.** A stand-in model is built in, so everything here runs "
        "free and offline — exactly as the book's own scripts do."
    )

    cells = [
        md(f"""
# Chapter {number} — {title}
### Exercise notebook · *Developer Guide: Building with AI*

{intro}

| Exercise | Book section | What you write |
|---|---|---|
{rows}

**About {minutes} minutes.** {key_line}

**How this works:** run the cells in order. Each exercise has `# TODO` comments and ends
with `raise NotImplementedError` — write your code, delete that line, run the cell. Then
run the check cell underneath: it prints ✅, or it tells you what is wrong. Stuck for ten
minutes? Run the `hint(...)` cell. Nobody is marking you.
"""),
        code(HARNESS, form=True),
    ]

    if installs:
        cells.append(code(f"%pip install -q {installs}"))

    if given:
        cells.append(md("""
---
## Setup — the code you are given

Read this, then run it. You will use these pieces in the exercises; you do not need to
write them.
"""))
        cells.append(code(given))

    for i, e in enumerate(exercises, 1):
        cells.append(md(f"""
---
## Exercise {i} — {e.name}  ·  *book §{e.section}*

{e.concept}
"""))
        cells.append(ex(e.todo, e.solution))
        cells.append(code(e.check))
        cells.append(_hint_cell(e.hint))
        if e.notice:
            cells.append(md(e.notice))

    bullets = "\n".join(f"* {item}" for item in learned)
    nxt = f"\n**Next:** {next_up}\n" if next_up else ""
    cells.append(md(f"""
---
## What you built

{bullets}
{nxt}"""))
    cells.append(code("progress()"))

    banner = f"""
# ⚠️ Solutions — Chapter {number}

The worked key for *Chapter {number} — {title}*. Every cell is complete and the notebook
runs top to bottom.

Check your own attempt against it, or use it to unblock one line. Reading a solution you
have not attempted feels like learning and is not.
"""
    return cells, stem, banner
