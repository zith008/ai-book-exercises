"""Chapter 13 — Multi-Agent Systems: pipelines, orchestration, and testing a judge."""

from lite import Exercise, make_chapter

GIVEN = r'''
import json, time, tempfile
from pathlib import Path

WORKDIR = Path(tempfile.mkdtemp(prefix="pipeline_"))

# ---- Four "agents". Each is a function from input to a long result ----------
# In production each would be a model call. Here they are deterministic, so the
# exercises are about the plumbing between agents, which is where the bugs live.
def researcher(topic):
    return {"topic": topic,
            "findings": [f"finding {i} about {topic}" for i in range(1, 9)],
            "notes": "x" * 4000}                      # deliberately bulky

def analyst(research):
    return {"themes": ["cost", "speed", "risk"],
            "detail": "y" * 4000,
            "based_on": research["topic"]}

def writer(analysis):
    return {"draft": f"Report on {analysis['based_on']}: "
                     + ", ".join(analysis["themes"]),
            "words": 850}

def editor(draft):
    return {"final": draft["draft"].upper(), "approved": True}


class Flaky:
    """A worker that fails a given number of times before succeeding."""
    def __init__(self, failures, error=TimeoutError("upstream timed out")):
        self.remaining, self.error, self.attempts = failures, error, 0

    def __call__(self, task):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.error
        return f"done: {task}"


print("workdir:", WORKDIR)
print("agents: researcher -> analyst -> writer -> editor")
'''

EX1 = Exercise(
    name="A pipeline that does not lose work",
    section="13.1",
    concept=r"""
The simplest multi-agent system is a **pipeline**: researcher → analyst → writer → editor.
Each agent's output is the next one's input.

Two things go wrong, and both are about plumbing rather than intelligence.

**1. Passing everything to everyone.** The researcher produces 4,000 characters of notes.
If that is pasted into the analyst's prompt, and the analyst's output into the writer's,
every agent pays for everyone's output. Instead pass a **reference** — "the research is in
`research.json`" — and let the next agent read what it needs.

**2. Losing three stages when the fourth fails.** If the editor crashes, you do not want to
re-run the researcher. **Write each stage's output to disk as you go**, and a failure costs
one stage, not four.

> Start with a single agent. Add agents only when you can name a specific failure one agent
> cannot solve — and measure what the coordination costs.
""",
    todo=r'''
STAGES = [("research", researcher), ("analysis", analyst),
          ("draft", writer), ("final", editor)]


def run_pipeline(topic, workdir, resume=True):
    """Run the four stages, saving each one. Returns a record of the run.

    {"result", "stages_run", "stages_skipped", "failed_at"}
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    value = topic
    stages_run, stages_skipped = [], []

    for name, agent in STAGES:
        artifact = workdir / f"{name}.json"

        # TODO 1: if resume is True and this stage's artifact already exists, load it
        #         with json.loads(artifact.read_text()), record the name in
        #         stages_skipped, use it as `value`, and continue. That is what makes
        #         a re-run cheap.
        # TODO 2: otherwise run agent(value). If it raises, return
        #         {"result": None, "stages_run": stages_run,
        #          "stages_skipped": stages_skipped, "failed_at": name}
        # TODO 3: save the output: artifact.write_text(json.dumps(output)),
        #         record the name in stages_run, and set value = output.
        raise NotImplementedError

    return {"result": value, "stages_run": stages_run,
            "stages_skipped": stages_skipped, "failed_at": None}


first = run_pipeline("vector databases", WORKDIR)
print("first run:   ", {k: v for k, v in first.items() if k != "result"})
print("final text:  ", first["result"]["final"][:60])

second = run_pipeline("vector databases", WORKDIR)
print("\nsecond run:  ", {k: v for k, v in second.items() if k != "result"})
print("             ^ everything was already on disk, so nothing re-ran")

# How much would passing everything cost, against passing references?
sizes = {p.stem: p.stat().st_size for p in sorted(WORKDIR.glob("*.json"))}
print("\nartifact sizes:", sizes)
print("passing all of it to every stage:", sum(sizes.values()) * len(sizes), "characters")
print("passing a filename instead:      ", sum(len(str(WORKDIR / f'{k}.json'))
                                               for k in sizes) * len(sizes))
''',
    solution=r'''
STAGES = [("research", researcher), ("analysis", analyst),
          ("draft", writer), ("final", editor)]


def run_pipeline(topic, workdir, resume=True):
    """Run the four stages, saving each one. Returns a record of the run.

    {"result", "stages_run", "stages_skipped", "failed_at"}
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    value = topic
    stages_run, stages_skipped = [], []

    for name, agent in STAGES:
        artifact = workdir / f"{name}.json"

        if resume and artifact.exists():
            value = json.loads(artifact.read_text())
            stages_skipped.append(name)
            continue

        try:
            output = agent(value)
        except Exception:
            return {"result": None, "stages_run": stages_run,
                    "stages_skipped": stages_skipped, "failed_at": name}

        artifact.write_text(json.dumps(output))
        stages_run.append(name)
        value = output

    return {"result": value, "stages_run": stages_run,
            "stages_skipped": stages_skipped, "failed_at": None}


first = run_pipeline("vector databases", WORKDIR)
print("first run:   ", {k: v for k, v in first.items() if k != "result"})
print("final text:  ", first["result"]["final"][:60])

second = run_pipeline("vector databases", WORKDIR)
print("\nsecond run:  ", {k: v for k, v in second.items() if k != "result"})
print("             ^ everything was already on disk, so nothing re-ran")

# How much would passing everything cost, against passing references?
sizes = {p.stem: p.stat().st_size for p in sorted(WORKDIR.glob("*.json"))}
print("\nartifact sizes:", sizes)
print("passing all of it to every stage:", sum(sizes.values()) * len(sizes), "characters")
print("passing a filename instead:      ", sum(len(str(WORKDIR / f'{k}.json'))
                                               for k in sizes) * len(sizes))
''',
    check=r'''
@check("13.1  the pipeline runs and saves every stage")
def _():
    need("run_pipeline")
    fresh = Path(tempfile.mkdtemp(prefix="p1_"))
    record = run_pipeline("agents", fresh)
    assert record["failed_at"] is None, f"the run failed at {record['failed_at']}"
    assert record["stages_run"] == ["research", "analysis", "draft", "final"], \
        f"stages_run is {record['stages_run']}"
    assert record["result"]["approved"] is True, \
        "the final stage's output should come back as the result"
    saved = sorted(p.name for p in fresh.glob("*.json"))
    assert saved == ["analysis.json", "draft.json", "final.json", "research.json"], (
        f"files on disk: {saved}. Every stage must be written as it completes — that is "
        "what makes a later failure cheap."
    )


@check("13.1  a second run reuses the saved work")
def _():
    need("run_pipeline")
    fresh = Path(tempfile.mkdtemp(prefix="p2_"))
    run_pipeline("agents", fresh)
    again = run_pipeline("agents", fresh)
    assert again["stages_skipped"] == ["research", "analysis", "draft", "final"], (
        f"the second run re-ran {again['stages_run']} instead of loading from disk. "
        "Check the artifact's existence before calling the agent."
    )
    assert again["stages_run"] == [], "nothing should need re-running"
    assert again["result"]["approved"] is True, \
        "a resumed run must still produce the final result, loaded from disk"

    forced = run_pipeline("agents", fresh, resume=False)
    assert forced["stages_run"] == ["research", "analysis", "draft", "final"], \
        "resume=False should re-run everything"


@check("13.1  a failure costs one stage, not four")
def _():
    need("run_pipeline", "STAGES")
    fresh = Path(tempfile.mkdtemp(prefix="p3_"))
    original = list(STAGES)
    try:
        def broken_writer(_):
            raise RuntimeError("the writer agent fell over")
        STAGES[2] = ("draft", broken_writer)

        record = run_pipeline("agents", fresh)
        assert record["failed_at"] == "draft", (
            f"failed_at is {record['failed_at']!r} — it should name the stage that "
            "broke, so an operator knows where to look"
        )
        assert record["result"] is None, (
            "a failed pipeline must not return a result: a half-finished report that "
            "looks finished is worse than an obvious failure"
        )
        assert sorted(p.stem for p in fresh.glob("*.json")) == ["analysis", "research"], (
            "the two completed stages should still be on disk, so the re-run starts at "
            f"the writer; found {sorted(p.stem for p in fresh.glob('*.json'))}"
        )

        STAGES[2] = original[2]
        recovered = run_pipeline("agents", fresh)
        assert recovered["stages_skipped"] == ["research", "analysis"], \
            f"the fixed re-run should skip the first two stages; got {recovered}"
        assert recovered["stages_run"] == ["draft", "final"], \
            f"and run only the remaining two; got {recovered['stages_run']}"
        print("     failed at 'draft', re-ran 2 stages instead of 4")
    finally:
        STAGES[:] = original
''',
    hint=[
        "Inside the loop:",
        "  if resume and artifact.exists():",
        "      value = json.loads(artifact.read_text())",
        "      stages_skipped.append(name)",
        "      continue",
        "",
        "  try:",
        "      output = agent(value)",
        "  except Exception:",
        "      return {'result': None, 'stages_run': stages_run,",
        "              'stages_skipped': stages_skipped, 'failed_at': name}",
        "",
        "  artifact.write_text(json.dumps(output))",
        "  stages_run.append(name)",
        "  value = output",
    ],
    notice=r"""
**What to notice.** Compare the two numbers printed at the end. Passing every stage's full
output to every other stage costs tens of thousands of characters; passing filenames costs
a few hundred. With real model calls, that difference is the bill.

And the third check is the one that matters operationally: the writer broke, and the re-run
started at the writer. In a pipeline where each stage is a minute of model time, that is
the difference between a four-minute retry and a one-minute one.
""",
)

EX2 = Exercise(
    name="An orchestrator that retries the right things",
    section="13.3",
    concept=r"""
Instead of a fixed line, an **orchestrator** decides which worker handles each task, runs
them, and deals with failures. It is ordinary software wrapped around model calls — and the
interesting part is the failure handling.

**Retry only what can succeed.** This distinction is worth internalising:

| Error | Retry? | Why |
|---|---|---|
| timeout, rate limit, 503 | **yes** | the same request may work in a second |
| invalid argument, unknown worker, bad schema | **no** | it will fail identically, for ever |

Retrying a permanent error is not resilience. It is the same mistake, three times, at three
times the cost — and it delays the error reaching someone who could fix it.

And the rule from Chapter 12, which matters even more with several workers:

> **Never let a missing result look like a finished one.**

Partial results must be labelled as partial, or whatever consumes them will quietly treat
missing work as completed work.
""",
    todo=r'''
RETRYABLE = (TimeoutError, ConnectionError)


class Orchestrator:
    def __init__(self, workers, max_attempts=3):
        self.workers = workers          # name -> callable
        self.max_attempts = max_attempts

    def route(self, task):
        """Pick a worker by the task's 'kind'. Given to you."""
        return task.get("kind")

    def run_task(self, task):
        """Run one task with retries. Returns a result record."""
        name = self.route(task)

        # TODO 1: if there is no such worker, that is permanent. Return
        #         {"status": "failed", "result": None, "attempts": 0,
        #          "error": f"no worker for {name!r}", "retryable": False}
        # TODO 2: try up to self.max_attempts times:
        #         - on success return {"status": "ok", "result": ...,
        #           "attempts": the attempt number, "error": None, "retryable": False}
        #         - on an error that is an instance of RETRYABLE, try again
        #         - on any OTHER error, stop immediately: retrying a permanent error
        #           just wastes money. Return status "failed" with retryable False.
        # TODO 3: if every attempt was used up, return status "failed",
        #         retryable True, and the last error's message.
        raise NotImplementedError

    def run_all(self, tasks):
        """Run several tasks. Returns (records, summary)."""
        records = [self.run_task(t) for t in tasks]

        # TODO 4: summary must make a partial run obvious:
        #         {"total", "ok", "failed", "complete": True only if every task is ok}
        raise NotImplementedError


orchestrator = Orchestrator({
    "research": lambda t: f"researched {t['topic']}",
    "flaky": Flaky(failures=2),                      # fails twice, then works
    "invalid": lambda t: (_ for _ in ()).throw(ValueError("bad argument")),
})

tasks = [{"kind": "research", "topic": "RAG"},
         {"kind": "flaky", "topic": "retries"},
         {"kind": "invalid", "topic": "permanent"},
         {"kind": "nonexistent", "topic": "missing worker"}]

records, summary = orchestrator.run_all(tasks)
for task, record in zip(tasks, records):
    print(f"{task['kind']:>12}: {record['status']:>6} after {record['attempts']} "
          f"attempt(s)  {record['error'] or ''}")
print("\nsummary:", summary)
''',
    solution=r'''
RETRYABLE = (TimeoutError, ConnectionError)


class Orchestrator:
    def __init__(self, workers, max_attempts=3):
        self.workers = workers          # name -> callable
        self.max_attempts = max_attempts

    def route(self, task):
        """Pick a worker by the task's 'kind'. Given to you."""
        return task.get("kind")

    def run_task(self, task):
        """Run one task with retries. Returns a result record."""
        name = self.route(task)

        if name not in self.workers:
            return {"status": "failed", "result": None, "attempts": 0,
                    "error": f"no worker for {name!r}", "retryable": False}

        worker = self.workers[name]
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return {"status": "ok", "result": worker(task),
                        "attempts": attempt, "error": None, "retryable": False}
            except RETRYABLE as err:
                last_error = err                     # may work next time
            except Exception as err:
                # Permanent: the same call will fail the same way. Stop now, so the
                # real error reaches someone instead of being retried into a timeout.
                return {"status": "failed", "result": None, "attempts": attempt,
                        "error": str(err), "retryable": False}

        return {"status": "failed", "result": None, "attempts": self.max_attempts,
                "error": str(last_error), "retryable": True}

    def run_all(self, tasks):
        """Run several tasks. Returns (records, summary)."""
        records = [self.run_task(t) for t in tasks]

        ok = sum(1 for r in records if r["status"] == "ok")
        summary = {"total": len(records), "ok": ok, "failed": len(records) - ok,
                   "complete": ok == len(records)}
        return records, summary


orchestrator = Orchestrator({
    "research": lambda t: f"researched {t['topic']}",
    "flaky": Flaky(failures=2),                      # fails twice, then works
    "invalid": lambda t: (_ for _ in ()).throw(ValueError("bad argument")),
})

tasks = [{"kind": "research", "topic": "RAG"},
         {"kind": "flaky", "topic": "retries"},
         {"kind": "invalid", "topic": "permanent"},
         {"kind": "nonexistent", "topic": "missing worker"}]

records, summary = orchestrator.run_all(tasks)
for task, record in zip(tasks, records):
    print(f"{task['kind']:>12}: {record['status']:>6} after {record['attempts']} "
          f"attempt(s)  {record['error'] or ''}")
print("\nsummary:", summary)
''',
    check=r'''
@check("13.3  temporary failures are retried, permanent ones are not")
def _():
    need("Orchestrator")
    flaky = Flaky(failures=2)
    o = Orchestrator({"flaky": flaky,
                      "broken": lambda t: (_ for _ in ()).throw(ValueError("bad input"))})

    good = o.run_task({"kind": "flaky"})
    assert good["status"] == "ok", f"it succeeds on the third attempt; got {good}"
    assert good["attempts"] == 3, \
        f"it should take 3 attempts, the record says {good['attempts']}"

    permanent = o.run_task({"kind": "broken"})
    assert permanent["status"] == "failed", "a ValueError is a failure"
    assert permanent["attempts"] == 1, (
        f"a permanent error was attempted {permanent['attempts']} times. It will fail "
        "identically every time — retrying costs money and delays the real error."
    )
    assert permanent["retryable"] is False, \
        "and it must be marked as not retryable, so nothing upstream tries again"

    never = Orchestrator({"always": Flaky(failures=99)}, max_attempts=3)
    exhausted = never.run_task({"kind": "always"})
    assert exhausted["attempts"] == 3 and exhausted["retryable"] is True, (
        "a retryable error that never clears should use every attempt and stay marked "
        f"retryable; got {exhausted}"
    )


@check("13.3  an unknown worker is a failure, not a crash")
def _():
    need("Orchestrator")
    o = Orchestrator({"known": lambda t: "fine"})
    record = o.run_task({"kind": "mystery"})
    assert record["status"] == "failed", "routing to a worker that does not exist fails"
    assert record["attempts"] == 0, \
        "nothing ran, so no attempts were made"
    assert "mystery" in record["error"], \
        f"the error should name the worker asked for; got {record['error']!r}"
    assert record["retryable"] is False, "a missing worker will still be missing later"


@check("13.3  a partial run cannot pass as a complete one")
def _():
    need("Orchestrator")
    o = Orchestrator({"good": lambda t: "ok",
                      "bad": lambda t: (_ for _ in ()).throw(ValueError("no"))})

    _, mixed = o.run_all([{"kind": "good"}, {"kind": "bad"}, {"kind": "good"}])
    assert mixed["complete"] is False, (
        "one task failed, so the run is not complete. This is the flag that stops a "
        "downstream stage treating 2 of 3 results as the whole job."
    )
    assert (mixed["ok"], mixed["failed"], mixed["total"]) == (2, 1, 3), \
        f"the counts are wrong: {mixed}"

    _, whole = o.run_all([{"kind": "good"}, {"kind": "good"}])
    assert whole["complete"] is True, "when everything succeeds, complete is True"
    assert whole["failed"] == 0, f"nothing failed, yet summary says {whole}"
    print(f"     2 of 3 tasks succeeded and the summary says complete=False")
''',
    hint=[
        "run_task, after the unknown-worker check:",
        "  for attempt in range(1, self.max_attempts + 1):",
        "      try:",
        "          return {'status': 'ok', 'result': worker(task),",
        "                  'attempts': attempt, 'error': None, 'retryable': False}",
        "      except RETRYABLE as err:",
        "          last_error = err          # try again",
        "      except Exception as err:",
        "          return {...'failed'..., 'attempts': attempt, 'retryable': False}",
        "",
        "Two except clauses, and the order matters: the specific one first.",
        "",
        "run_all:",
        "  ok = sum(1 for r in records if r['status'] == 'ok')",
        "  summary = {'total': len(records), 'ok': ok,",
        "             'failed': len(records) - ok, 'complete': ok == len(records)}",
    ],
    notice=r"""
**What to notice.** Two `except` clauses is the whole lesson. The flaky worker got three
chances; the one with a bad argument got one. Both behaviours are correct, and a single
`except Exception: retry` would have got both wrong.

`complete: False` is the other half. Multi-agent systems fail *partially* far more often
than they fail completely, and a partial result that does not say so is the most expensive
kind of bug — nothing downstream can detect it.
""",
)

EX3 = Exercise(
    name="Testing the judge",
    section="13.4",
    concept=r"""
**Debate**: two agents argue opposite sides, a third judges. It sounds rigorous. It is
rigorous exactly as far as the judge is, and no further.

Model judges have a well-documented flaw: **position bias**. Present the same two arguments
in the other order and the verdict can change — because the judge is responding to position
rather than to content.

So there is a test, and it is cheap:

1. Judge arguments in the order (A, B).
2. Judge them again as (B, A).
3. If the winner **flips**, the verdict is not evidence of anything. Discard it.

A judge that passes this test is at least consistent. One that fails is a coin you are
paying for.

> Prefer a verifier that checks facts. If you must use a model as judge, swap the order and
> distrust any verdict that flips.
""",
    todo=r'''
ARGUMENTS = {
    "A": "Use Postgres: the team knows it, and 50GB fits comfortably on one machine.",
    "B": "Use a distributed store: it scales to petabytes, which we may need one day.",
}

# Two judges with the same interface. One reads the arguments; one just prefers
# whatever came first.
def biased_judge(first_label, first_text, second_label, second_text):
    """Always picks whatever is presented first. Given to you."""
    return first_label

def evidence_judge(first_label, first_text, second_label, second_text):
    """Prefers the argument that cites a concrete number. Given to you."""
    import re
    first_has = bool(re.search(r"\d+\s*(GB|TB|machine)", first_text))
    second_has = bool(re.search(r"\d+\s*(GB|TB|machine)", second_text))
    if first_has == second_has:
        return first_label                     # no evidence either way: position wins
    return first_label if first_has else second_label


def judge_both_ways(judge, arguments):
    """Run the judge twice, swapping the order. Return a verdict record.

    {"first_order", "second_order", "flipped", "verdict"}
    where verdict is the winner if both runs agree, and None if they do not.
    """
    labels = list(arguments)
    a, b = labels[0], labels[1]

    # TODO 1: run the judge with a first, then b.
    # TODO 2: run it again with b first, then a.
    # TODO 3: flipped is True when the two runs disagree.
    # TODO 4: verdict is the agreed winner, or None when they disagree. A flipped
    #         verdict is not a weak signal — it is no signal, so do not pick one.
    raise NotImplementedError


print("the judge that only looks at position:")
print("  ", judge_both_ways(biased_judge, ARGUMENTS))
print("\nthe judge that looks for evidence:")
print("  ", judge_both_ways(evidence_judge, ARGUMENTS))
''',
    solution=r'''
ARGUMENTS = {
    "A": "Use Postgres: the team knows it, and 50GB fits comfortably on one machine.",
    "B": "Use a distributed store: it scales to petabytes, which we may need one day.",
}

# Two judges with the same interface. One reads the arguments; one just prefers
# whatever came first.
def biased_judge(first_label, first_text, second_label, second_text):
    """Always picks whatever is presented first. Given to you."""
    return first_label

def evidence_judge(first_label, first_text, second_label, second_text):
    """Prefers the argument that cites a concrete number. Given to you."""
    import re
    first_has = bool(re.search(r"\d+\s*(GB|TB|machine)", first_text))
    second_has = bool(re.search(r"\d+\s*(GB|TB|machine)", second_text))
    if first_has == second_has:
        return first_label                     # no evidence either way: position wins
    return first_label if first_has else second_label


def judge_both_ways(judge, arguments):
    """Run the judge twice, swapping the order. Return a verdict record.

    {"first_order", "second_order", "flipped", "verdict"}
    where verdict is the winner if both runs agree, and None if they do not.
    """
    labels = list(arguments)
    a, b = labels[0], labels[1]

    forward = judge(a, arguments[a], b, arguments[b])
    backward = judge(b, arguments[b], a, arguments[a])
    flipped = forward != backward

    return {"first_order": forward, "second_order": backward,
            "flipped": flipped, "verdict": None if flipped else forward}


print("the judge that only looks at position:")
print("  ", judge_both_ways(biased_judge, ARGUMENTS))
print("\nthe judge that looks for evidence:")
print("  ", judge_both_ways(evidence_judge, ARGUMENTS))
''',
    check=r'''
@check("13.4  the swap test exposes a position-biased judge")
def _():
    need("judge_both_ways")
    result = judge_both_ways(biased_judge, ARGUMENTS)
    assert result["flipped"] is True, (
        "a judge that always picks whatever is first MUST flip when you swap the order. "
        f"Your record says {result} — check that the second run really swaps both the "
        "labels and the texts."
    )
    assert result["verdict"] is None, (
        "when the verdict flips there is no verdict. Returning one anyway hands a coin "
        "toss to whatever makes the decision next."
    )
    assert {result["first_order"], result["second_order"]} == {"A", "B"}, \
        f"the two runs should pick opposite sides; got {result}"


@check("13.4  a consistent judge gives a usable verdict")
def _():
    need("judge_both_ways")
    result = judge_both_ways(evidence_judge, ARGUMENTS)
    assert result["flipped"] is False, (
        f"the evidence judge should give the same answer both ways; got {result}"
    )
    assert result["verdict"] == "A", (
        f"the verdict should be A — it is the argument citing a concrete number (50GB) "
        f"— but you got {result['verdict']!r}"
    )
    assert result["first_order"] == result["second_order"] == "A", \
        "both runs should agree"


@check("13.4  the test works whichever side is right")
def _():
    need("judge_both_ways")
    swapped = {"B": ARGUMENTS["B"], "A": ARGUMENTS["A"]}       # B listed first
    result = judge_both_ways(evidence_judge, swapped)
    assert result["verdict"] == "A", (
        "the evidence judge should still choose A even when A is passed second — if "
        f"your result is {result}, the two runs are not really swapping the order"
    )

    tie = {"A": "no numbers here", "B": "none here either"}
    tied = judge_both_ways(evidence_judge, tie)
    assert tied["flipped"] is True and tied["verdict"] is None, (
        "with no evidence on either side this judge falls back to position, so the "
        f"swap test should catch it; got {tied}"
    )
    print("     biased judge flipped (no verdict), evidence judge consistent (A wins)")
''',
    hint=[
        "judge_both_ways is four lines:",
        "  forward = judge(a, arguments[a], b, arguments[b])",
        "  backward = judge(b, arguments[b], a, arguments[a])",
        "  flipped = forward != backward",
        "  return {'first_order': forward, 'second_order': backward,",
        "          'flipped': flipped, 'verdict': None if flipped else forward}",
        "",
        "The second call must swap BOTH the labels and their texts. Swapping only the",
        "labels tests nothing.",
    ],
    notice=r"""
**What to notice.** The last check is the subtle one: when neither argument carries
evidence, the evidence judge falls back on position — and the swap test catches it there
too. The test does not ask whether the judge is clever. It asks whether the verdict depends
on something other than the content.

Run it on any model-as-judge you deploy. It costs one extra call and tells you whether the
first one meant anything.

> A debate is only as good as its judge, and a judge that does not use evidence is no
> better than a coin.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=13,
    title="Multi-Agent Systems",
    intro=r"""
More agents means more coordination, and coordination is where multi-agent systems actually
fail — not in the reasoning. So this notebook is about the plumbing: passing work between
agents without paying for it twice, retrying the right failures, and testing a judge before
trusting its verdict.

**Start with one agent.** Add another only when you can name a failure one agent cannot fix.
""",
    minutes=35,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "Pass **references**, not whole outputs — otherwise every agent pays for every "
        "other agent's verbosity.",
        "**Write each stage to disk** as it finishes, so a failure in stage four costs "
        "one stage instead of four.",
        "**Retry only what can succeed**: timeouts yes, invalid arguments never.",
        "**Never let a missing result look like a finished one** — label partial runs as "
        "partial.",
        "A **debate is only as good as its judge**. Swap the order of the arguments and "
        "distrust any verdict that flips.",
        "Model judges show **position bias**; the swap test costs one extra call and "
        "tells you whether the verdict means anything.",
    ],
    next_up="Chapter 14 asks what a framework does for you — and what it costs.",
)
