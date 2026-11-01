# Author's guide to the exercise notebooks

How the seventeen notebooks are built, why they are built that way, and what to keep doing
if you add or change one. Written for the author of *Developer Guide: Building with AI*.

---

## 1. What the notebooks are for

The book already contains complete, working code. A notebook that reprints it adds
nothing — the reader scrolls, presses Shift+Enter, watches it work, and learns very little.
People learn what they have struggled with.

So every notebook here follows one rule:

> **The reader writes the part that carries the idea. Everything else is given.**

In Chapter 1 the reader writes `sigmoid`, the three lines of `Layer.backward`, the four
steps of the training loop, and a gradient check. The plotting, the data loading and the
class skeletons are given, because wrestling with `meshgrid` teaches nothing about
backpropagation. Each exercise is small enough to hold in your head and large enough to
be able to go wrong.

**Why notebooks at all, when the book ships `.py` scripts?** The scripts are the reference
implementation; the notebooks are the teaching surface. Colab gives a reader a GPU-capable
Python environment with numpy, sklearn, matplotlib and torch already installed, reachable
from a phone, with nothing to install and nothing to break. The gap between "I bought a
book" and "I am running code" becomes one click. That gap is where most readers are lost.

---

## 2. The five design rules

### Rule 1 — One idea per exercise, with a check under it

Each exercise cell is followed by a check cell. The check is not a grade; it is the
feedback loop that lets a reader work alone at 11pm without you. It tests only what the
exercise taught, and when it fails it says what to do:

```python
assert cold[0] > at1[0] > hot[0], (
    "lower temperature must make the top choice MORE likely and higher temperature "
    f"less: you got {cold[0]:.3f} / {at1[0]:.3f} / {hot[0]:.3f} at T=0.2 / 1.0 / 1.5. "
    "Check you divide the logits by T rather than multiplying."
)
```

Three parts, every time: **what should be true**, **what actually happened (with the
reader's own numbers)**, and **the most likely cause**. An assertion that just says
`assert acc > 0.8` tells a stuck reader nothing and sends them to the solutions.

### Rule 2 — Analogy first, then the formula

The sigmoid is introduced as "a weighted vote, squashed into a verdict" before
$1/(1+e^{-z})$ appears. Temperature is "how strictly the model commits to its own
favourite" before the softmax. A reader who has the picture can decode the notation; a
reader given the notation first usually just scrolls past it.

Where a derivative is needed, it is **handed over in a table** rather than derived. The
chapter is about building a network, not about calculus — and gatekeeping on calculus loses
the practitioners who are the book's audience.

### Rule 3 — Every notebook must run end to end with no setup

No local files, no `pip install -r requirements.txt`, no dataset downloads, no accounts.
Chapter 1 uses `make_moons` because sklearn generates it in memory. If a chapter genuinely
needs data, generate it in a cell or fetch it from a URL you control — never a path on your
machine, and never a link that can rot.

**And no API key.** See §4.

### Rule 4 — Hints are cheap, solutions are separate

Every exercise has a `hint(...)` cell that spells out the lines. This is deliberate. A
reader stuck for twenty minutes does not persevere; they close the tab. Hints let them pay
a small price to continue, instead of jumping to the full solution and learning nothing.

Solutions live in a **separate notebook**. Colab has no reliable way to hide a cell —
`#@title` forms are a visual nicety, not a lock. Separate files mean the reader has to
make a decision to look, which is exactly the friction you want.

### Rule 5 — Teach the failures, not only the successes

The strongest exercise in Chapter 1 is Exercise 6: the reader is handed a plausible,
broken first attempt and asked to find the shape bug. The strongest point in Chapter 2 is
that the budgeted chatbot is cheaper **and has forgotten things**.

The book already has this instinct — the "Limitations" and "Claims I could not verify"
sections are its best feature. Carry it into every notebook. Readers who have only ever
seen code work cannot debug, and debugging is the job.

---

## 3. The mechanics

```
build/
  nbtools.py          tiny notebook writer: md(), code(), ex()
  harness.py          the check/hint/progress/need harness pasted into every notebook
  lite.py             the beginner template used by chapters 3-17
  kit.py              shared stand-in embedder and documents for chapters 7-10
  ch01.py ... ch17.py one file per chapter: the cells, in order
  build_notebooks.py  builds every chapter
  check_notebooks.py  runs the solution notebooks and fails if any check does not pass
notebooks/            the generated .ipynb files (these are what readers open)
```

**Two shapes of chapter.** Chapters 1 and 2 are written cell by cell and are deliberately
deeper — six or seven exercises, longer explanations — because they are where a reader
decides whether to continue. Chapters 3 to 17 use `lite.py`: a fixed structure of **three
exercises**, each with concept → skeleton → check → hint → takeaway, about fifteen cells and
thirty-five minutes. A chapter file then only supplies content, and every notebook a reader
opens behaves identically.

Three exercises per chapter is a deliberate ceiling. Covering every section of a 1,500-line
chapter would produce a notebook nobody finishes; three well-chosen gaps, each carrying one
idea the chapter insists on, get worked all the way through.

**Exercises and solutions come from one source.** `ex(todo_src, solution_src)` holds both
versions of a cell, and the builder writes two notebooks from it. This matters more than it
looks: hand-maintained solution notebooks drift out of step with the exercises within about
two edits, and a wrong solutions file destroys a reader's trust faster than a missing one.

```bash
python3 build/build_notebooks.py           # regenerate the notebooks
python3 build/check_notebooks.py           # run the solutions, fail if a check fails
```

**Run the checker before every release.** It flattens each solution notebook into a script,
runs it in a clean interpreter, and exits non-zero if any check is not ✅. A broken exercise
that reaches a reader costs you a support email and some of their faith in the book; this
takes about ninety seconds to catch.

Treat `notebooks/*.ipynb` as build output. Edit `build/chNN.py`, never the `.ipynb` — an
edit made in Colab is lost at the next build.

---

## 4. API keys: the problem that decides your reader's first five minutes

From Chapter 2 onward the book's code needs an OpenAI key, and Chapter 9 needs Pinecone.
This is where technical books lose readers in bulk. Someone sitting on a train with your
book does not have a funded API account.

The pattern used in Chapter 2's notebook, and which should be used in every later chapter:

```python
client, MODE = get_client()     # LIVE if a key is found, MOCK otherwise
```

`MOCK` is a stand-in object with the same call signature and the same response shape
(`response.choices[0].message.content`, `response.usage.prompt_tokens`, `finish_reason`).
It returns canned replies, but it computes **real token counts with `tiktoken`**, honours
`max_tokens` and reports `finish_reason == "length"` when it truncates, and answers "what
was my first question?" only when that message is genuinely still in the list it was given.

The result: every exercise and every check in Chapter 2 passes with no key, no network and
no spend. Only the prose is fake. The reader learns tokens, temperature, context management
and the fact that cost grows every turn — all of which are mechanics, not model wording.

**Rules for writing checks that work in both modes:** assert on structure and behaviour
(message counts, token arithmetic, eviction order, which fields are present), never on what
the model says. `assert "Paris" in reply` is a check that fails for reasons the reader
cannot fix.

Your book already notes that several scripts "were not run against the live API while this
guide was written" and were checked with a stand-in client. The notebooks make that
stand-in a feature the reader can use, rather than a caveat.

How each later chapter's obstacle was handled, and why:

| Chapter | The obstacle | What was done |
|---|---|---|
| 3–6 | agents need a model | the book's own `ScriptedLLM` — a list of prepared replies. An agent only needs the model to be "a function from messages to text", so this teaches the loop perfectly at no cost |
| 7–10 | embeddings cost money per call | one shared stand-in embedder in `kit.py`: a hashing bag-of-words over a 16,384-slot vocabulary. Texts sharing words land close; texts sharing nothing are at distance exactly 1. Chunking, cutoffs, recall, truncation and quantisation all behave as they do in production — only paraphrase matching is cruder, and each notebook says so where it matters |
| 9 | Pinecone needs an account | the exercise is two *fake* vendors with deliberately different filter dialects, and the reader writes the adapter. That teaches the chapter's actual point — one contract, same tests — without an account. The notebook is explicit that a fake proves the contract, not the database |
| 11 | MCP servers are subprocesses over stdio | the reader builds the **wire format by hand** (§11.1), which is in-process and the chapter's strongest material anyway. The SDK is recommended in prose for real servers |
| 14 | LangGraph / LlamaIndex / CrewAI churn fast | the notebook installs **nothing**. The reader builds state + routing, retries + checkpoints, and human approval in about forty lines, so any framework's documentation then reads as recognisable engineering. The book's Ex-67 to Ex-70 remain the place to run the real libraries — pin exact versions there |
| 15–17 | safety, governance, production | the *logic* with fakes: a tool gateway, a hashed approval, a budget that reserves, span nesting, cost per success, a circuit-breaker state machine. Nobody deploys from a notebook, but every one of these is testable in one |

**One embedder across four chapters, not four.** Chapters 7 to 10 share `kit.py`. If each
chapter had its own fake, chapter 7's and chapter 10's would disagree in some detail and a
reader working straight through would hit a contradiction they could not resolve.

Also: always show the cost before a cell that spends money (`"this cell makes 12 calls,
about $0.002"`), and never put a live key in a cell. The notebook should tell the reader
how to use Colab's secrets manager, as Chapter 2's does.

---

## 5. Getting readers into the notebooks

**Publish to a public GitHub repository.** Colab opens a notebook straight from GitHub,
with no download step:

```
https://colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/Chapter_01_Neural_Network_From_Scratch.ipynb
```

Put an **Open in Colab badge** at the top of every notebook and in the README:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/REPO/blob/main/notebooks/Chapter_01_Neural_Network_From_Scratch.ipynb)
```

The badge in each notebook is **generated**, not pasted. `build/nbtools.py` holds three
lines:

```python
REPO = "zith008/ai-book-exercises"
BRANCH = "main"
NOTEBOOK_DIR = "notebooks"
```

Change them, rebuild, and all 34 badges follow. Set `REPO = None` to drop the badge
entirely — better than one pointing at a 404.

**While the repository is private, a badge only opens for someone signed in to GitHub with
access to it.** That is right for review and wrong for readers, so flip the repository to
public before the book ships, and check one badge in a logged-out browser.

In the **printed book**, at the end of each chapter:

> **Practise this chapter.** Work through the Chapter 1 exercise notebook at
> `github.com/USER/REPO` — you write the neuron, the backward pass and the training loop,
> and the notebook checks your work. No installation required.

One short, memorable URL for the repository (or a redirect you own) beats per-chapter
links a reader has to type from paper.

**Versioning.** Tag the repository to match the book's edition, and keep a `CHANGELOG.md`.
Someone reading the first edition in two years should be able to reach notebooks that still
run. Pin framework versions (§4) so they do.

---

## 6. Checklist for each new chapter

Writing `build/chNN.py` with `lite.make_chapter(...)`:

- [ ] **Three exercises**, each mapped to one book section and each carrying one idea the
      chapter insists on. Prefer the section with the sharpest "key idea" over the longest
      section.
- [ ] An **intro** saying what the reader will have built, and whether a key is needed.
- [ ] A **given** cell with everything the exercise is not about — class skeletons, data,
      stand-ins. Generous scaffolding is what makes a small gap teachable.
- [ ] Each exercise: **analogy or mechanism first**, then the formula; a skeleton with
      `# TODO`; a check cell; a hint cell; a short "what to notice".
- [ ] At least one **debugging or failure exercise** across the chapter — the leaky filter,
      the self-grading loop, the breaker that cannot close. These are the ones readers
      remember.
- [ ] Every check message names **what should be true, what happened, and the likely cause**.
- [ ] `need(...)` at the top of any check that depends on names from the exercise cell, and
      `need_data(...)` where the state exists but is empty until the exercise runs. Both
      report ⬜ rather than a confusing error.
- [ ] Register the chapter in `CHAPTERS` in `build_notebooks.py`.
- [ ] `python3 build/build_notebooks.py && python3 build/check_notebooks.py` is green —
      and **run the checker more than once**, to catch anything timing-dependent.
- [ ] Run the **exercise** notebook with nothing filled in and read what a reader sees:
      every check ⬜ with a useful message, no confusing `NameError`.
- [ ] Hand it to one person who has not read the chapter. Watch where they stall. That is
      the only test that matters.

## 7. What to tell readers about AI assistance

Readers will paste these exercises into an AI assistant — many of them will. It is worth
one honest paragraph in the book's introduction rather than pretending otherwise:

> The assistant will produce a correct `Layer.backward` in seconds, and you will have
> learned nothing from it. Use it the way you would use a tutor: ask it *why* your version
> fails the check, ask it what a shape mismatch means, ask it to explain the chain rule
> again differently. Do not ask it for the answer before you have written a wrong one.

This is also a fair description of how the book's own audience works professionally, which
makes it credible rather than preachy.

---

## 8. Known gaps, honestly

* **The repository is private**, so the Colab badges work for you and not yet for readers.
  Making it public is the last step before publication (§5).
* **Chapters 1 and 2 are denser than 3 to 17.** They were written first, to a different
  spec. They work, and a reader moving from chapter 2 to chapter 3 will notice the notebooks
  get lighter. Worth either trimming 1 and 2 or accepting it as a deliberate on-ramp.
* **The stand-in models are canned.** A reader expecting genuinely intelligent replies
  without a key will be disappointed; each notebook says so plainly in its setup cell, and
  should keep saying so.
* **The stand-in embedder matches on shared words, not meaning.** It cannot show that
  "refund window" and "return within 30 days" are the same thing — a real embedder can.
  Chapters 7 and 10 state this where a reader might otherwise draw the wrong conclusion.
  Distances, and therefore the cutoff chosen in chapter 7, differ with a real model; the
  method does not.
* **`tiktoken` downloads its vocabulary on first use**, so chapter 2's token exercises need
  a network connection even with no key. Fine in Colab; worth a line for offline readers.
* **Chapter 1's Experiment C contradicts the chapter slightly.** Run for 10,000 epochs, the
  "stalled" `[2, 8, 8, 1]` network reaches about 0.975 — it was slow, not stuck. The
  notebook leaves this as the reader's discovery, which is better teaching than the book's
  current wording. Consider softening §1.3's "has stalled, not finished" to match.
* **Nothing is tested against a live API.** Chapter 2's LIVE path shares all its code with
  its MOCK path, but run it once with a real key before release.
* **Three exercises per chapter leaves material on the table.** Sections not covered by an
  exercise are still in the book; the notebooks do not pretend to be complete coverage, and
  the mapping table at the top of each one says which sections it takes.

## 9. Current state

All 17 notebooks build and all 17 solution notebooks run top to bottom with **130 checks
green**, verified with `build/check_notebooks.py`. Every exercise notebook has also been run
with nothing filled in: each check reports ⬜ with an actionable message, and no cell fails
with a confusing error.

Four bugs were found this way and fixed, all of them in checks or stand-ins rather than in
the teaching code — including one flaky test (a zero TTL compared with `>` instead of `>=`)
that passed four times and failed the fifth. That is the argument for running the checker
before every release rather than trusting a single green run.
