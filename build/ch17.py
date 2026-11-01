"""Chapter 17 — Production: tracing, cost per success, and the circuit breaker."""

from lite import Exercise, make_chapter

GIVEN = r'''
import time, secrets, json
from collections import defaultdict

PRICE_PER_1K = {"small": 0.0002, "large": 0.003}


def usd(model, prompt_tokens, completion_tokens):
    """Cost in dollars from token counts. Given to you."""
    return ((prompt_tokens + completion_tokens) / 1000) * PRICE_PER_1K[model]


class Service:
    """A dependency that fails for a while, then recovers."""
    def __init__(self, fail_first=0):
        self.remaining, self.calls = fail_first, 0

    def __call__(self):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ConnectionError("service unavailable")
        return "ok"


# A day's worth of runs, as a tracing system would have recorded them
RUNS = [
    {"task": "summarise", "model": "small", "prompt": 800, "completion": 200, "ok": True},
    {"task": "summarise", "model": "small", "prompt": 800, "completion": 200, "ok": True},
    {"task": "summarise", "model": "small", "prompt": 900, "completion": 50, "ok": False},
    {"task": "summarise", "model": "small", "prompt": 900, "completion": 220, "ok": True},
    {"task": "extract", "model": "large", "prompt": 1200, "completion": 300, "ok": True},
    {"task": "extract", "model": "large", "prompt": 1200, "completion": 80, "ok": False},
    {"task": "extract", "model": "large", "prompt": 1200, "completion": 90, "ok": False},
    {"task": "extract", "model": "large", "prompt": 1300, "completion": 310, "ok": True},
    {"task": "classify", "model": "small", "prompt": 200, "completion": 10, "ok": True},
    {"task": "classify", "model": "small", "prompt": 200, "completion": 10, "ok": True},
]

print(f"{len(RUNS)} recorded runs across "
      f"{len({r['task'] for r in RUNS})} task types")
'''

EX1 = Exercise(
    name="Spans that connect",
    section="17.1",
    concept=r"""
A log line tells you something happened. A **trace** tells you what it was part of.

One user request becomes a model call, three tool calls, a retrieval, another model call.
Logged flat, those are seven unrelated lines. What you want is a tree:

```
request abc123                        1.42s   ok
  retrieve                            0.31s   ok
  model.call (small)                  0.55s   ok
  tool.search                         0.48s   error
  model.call (small)                  0.08s   ok
```

Three ideas do all the work:

* every span has an **id** and a **parent id** — that is what makes it a tree
* a span records **status** (ok or error), so a failure has a location
* durations come from a **monotonic clock** (`time.perf_counter()`), not the wall clock,
  which can jump backwards when the machine syncs its time

> Observability earns its place when it **changes what the system does** — ties a failure to
> a request, a cost to an outcome, a limit to a refusal. Test each of those by running it.
""",
    todo=r'''
class Tracer:
    def __init__(self):
        self.spans = []
        self.stack = []        # the spans currently open, innermost last

    def start(self, name, **attributes):
        """Open a span. Returns its id."""
        span_id = secrets.token_hex(4)

        # TODO 1: build the span:
        #         {"id": span_id,
        #          "parent": the id of the innermost open span, or None,
        #          "name": name, "attributes": dict(attributes),
        #          "started": time.perf_counter(),   <- monotonic, not time.time()
        #          "duration": None, "status": None}
        # TODO 2: append it to self.spans, push its id onto self.stack, return the id.
        raise NotImplementedError

    def end(self, span_id, status="ok"):
        """Close a span, recording how long it took."""
        # TODO 3: find the span, set its duration to
        #         time.perf_counter() - span["started"], set its status, and pop it
        #         off the stack.
        raise NotImplementedError

    def span(self, name, **attributes):
        """A context manager, so a failing block still closes its span. Given to you."""
        tracer = self

        class _Span:
            def __enter__(inner):
                inner.id = tracer.start(name, **attributes)
                return inner
            def __exit__(inner, exc_type, exc, tb):
                tracer.end(inner.id, "error" if exc_type else "ok")
                return False        # never swallow the exception

        return _Span()

    def tree(self):
        """Render the spans as an indented tree. Given to you."""
        children = defaultdict(list)
        for span in self.spans:
            children[span["parent"]].append(span)

        lines = []
        def walk(parent, depth):
            for span in children[parent]:
                lines.append(f"{'  ' * depth}{span['name']:<28} "
                             f"{span['duration']:.3f}s  {span['status']}")
                walk(span["id"], depth + 1)
        walk(None, 0)
        return "\n".join(lines)


tracer = Tracer()

with tracer.span("request", user="alice"):
    with tracer.span("retrieve", documents=12):
        time.sleep(0.01)
    with tracer.span("model.call", model="small"):
        time.sleep(0.02)
    try:
        with tracer.span("tool.search"):
            raise TimeoutError("search timed out")
    except TimeoutError:
        pass
    with tracer.span("model.call", model="small"):
        time.sleep(0.01)

print(tracer.tree())
''',
    solution=r'''
class Tracer:
    def __init__(self):
        self.spans = []
        self.stack = []        # the spans currently open, innermost last

    def start(self, name, **attributes):
        """Open a span. Returns its id."""
        span_id = secrets.token_hex(4)

        self.spans.append({
            "id": span_id,
            "parent": self.stack[-1] if self.stack else None,
            "name": name,
            "attributes": dict(attributes),
            "started": time.perf_counter(),      # monotonic: never jumps backwards
            "duration": None,
            "status": None,
        })
        self.stack.append(span_id)
        return span_id

    def end(self, span_id, status="ok"):
        """Close a span, recording how long it took."""
        span = next(s for s in self.spans if s["id"] == span_id)
        span["duration"] = time.perf_counter() - span["started"]
        span["status"] = status
        if span_id in self.stack:
            self.stack.remove(span_id)

    def span(self, name, **attributes):
        """A context manager, so a failing block still closes its span. Given to you."""
        tracer = self

        class _Span:
            def __enter__(inner):
                inner.id = tracer.start(name, **attributes)
                return inner
            def __exit__(inner, exc_type, exc, tb):
                tracer.end(inner.id, "error" if exc_type else "ok")
                return False        # never swallow the exception

        return _Span()

    def tree(self):
        """Render the spans as an indented tree. Given to you."""
        children = defaultdict(list)
        for span in self.spans:
            children[span["parent"]].append(span)

        lines = []
        def walk(parent, depth):
            for span in children[parent]:
                lines.append(f"{'  ' * depth}{span['name']:<28} "
                             f"{span['duration']:.3f}s  {span['status']}")
                walk(span["id"], depth + 1)
        walk(None, 0)
        return "\n".join(lines)


tracer = Tracer()

with tracer.span("request", user="alice"):
    with tracer.span("retrieve", documents=12):
        time.sleep(0.01)
    with tracer.span("model.call", model="small"):
        time.sleep(0.02)
    try:
        with tracer.span("tool.search"):
            raise TimeoutError("search timed out")
    except TimeoutError:
        pass
    with tracer.span("model.call", model="small"):
        time.sleep(0.01)

print(tracer.tree())
''',
    check=r'''
@check("17.1  spans nest, and the tree reflects the real structure")
def _():
    need("Tracer", "tracer")
    need_data(tracer.spans, "the Tracer cell above")
    root = [s for s in tracer.spans if s["parent"] is None]
    assert len(root) == 1 and root[0]["name"] == "request", (
        f"there should be exactly one root span; found "
        f"{[s['name'] for s in root]}. Each span's parent is whichever span was open "
        "when it started."
    )
    children = [s for s in tracer.spans if s["parent"] == root[0]["id"]]
    assert len(children) == 4, \
        f"the request has 4 direct children; found {len(children)}"
    assert [s["name"] for s in children] == \
        ["retrieve", "model.call", "tool.search", "model.call"], \
        f"children in order: {[s['name'] for s in children]}"
    assert all(s["duration"] is not None for s in tracer.spans), \
        "every span must be closed, with a duration recorded"
    assert all(s["id"] != s["parent"] for s in tracer.spans), \
        "a span cannot be its own parent"


@check("17.1  a failure is recorded where it happened")
def _():
    need("Tracer", "tracer")
    need_data(tracer.spans, "the Tracer cell above")
    failed = [s for s in tracer.spans if s["status"] == "error"]
    assert len(failed) == 1 and failed[0]["name"] == "tool.search", (
        f"exactly one span should have failed (tool.search); got "
        f"{[(s['name'], s['status']) for s in failed]}"
    )
    assert all(s["status"] == "ok" for s in tracer.spans if s["name"] != "tool.search"), (
        "only the span that actually raised should be marked error. The request "
        "handled the failure, so it succeeded."
    )
    retrieve = next(s for s in tracer.spans if s["name"] == "retrieve")
    assert retrieve["attributes"]["documents"] == 12, \
        "attributes passed to span() must be kept — they are how you query traces later"


@check("17.1  durations are measured with a monotonic clock")
def _():
    need("Tracer")
    t = Tracer()
    with t.span("outer"):
        time.sleep(0.03)
        with t.span("inner"):
            time.sleep(0.02)

    outer = next(s for s in t.spans if s["name"] == "outer")
    inner = next(s for s in t.spans if s["name"] == "inner")
    assert inner["duration"] >= 0.015, \
        f"the inner span slept 0.02s but recorded {inner['duration']:.4f}s"
    assert outer["duration"] > inner["duration"], (
        "the outer span contains the inner one, so it must take longer. If these are "
        "equal or reversed, the start time is being captured in the wrong place."
    )
    assert all(s["duration"] >= 0 for s in t.spans), (
        "a negative duration means time.time() was used — it can jump backwards when "
        "the machine syncs its clock. Use time.perf_counter()."
    )
    assert t.stack == [], (
        "every span was closed, so the stack should be empty — a leaking stack makes "
        "every later span a child of something that already finished"
    )
    print(f"     nested spans, 1 error located, outer {outer['duration']:.3f}s > "
          f"inner {inner['duration']:.3f}s")
''',
    hint=[
        "start:",
        "  self.spans.append({'id': span_id,",
        "                     'parent': self.stack[-1] if self.stack else None,",
        "                     'name': name, 'attributes': dict(attributes),",
        "                     'started': time.perf_counter(),",
        "                     'duration': None, 'status': None})",
        "  self.stack.append(span_id)",
        "  return span_id",
        "",
        "end:",
        "  span = next(s for s in self.spans if s['id'] == span_id)",
        "  span['duration'] = time.perf_counter() - span['started']",
        "  span['status'] = status",
        "  if span_id in self.stack: self.stack.remove(span_id)",
    ],
    notice=r"""
**What to notice.** `self.stack[-1]` is the entire parenting mechanism. Whatever span is
open becomes the parent of the next one — which is why the context manager matters: a block
that raises still closes its span, and the stack stays honest.

**In production, use the OpenTelemetry SDK** rather than this. It does the ids, parents,
status and export for you, and exports to tools that already know how to draw the tree.
Keep your attribute names in one place, though: the conventions for GenAI attributes are
still changing.
""",
)

EX2 = Exercise(
    name="Cost per success",
    section="17.3",
    concept=r"""
"We spent $400 on the API last month" is not a useful number. Nor is "the average request
costs $0.004". The number that drives decisions is:

$$\text{cost per success} = \frac{\text{everything you spent}}{\text{the successes}}$$

**All** the spending, including the failed attempts, divided by **only** the outcomes you
wanted. Failures are not free and they are not separate — they are overhead on the work
that did succeed. Charge the waste to the wins.

Then break it down **by task type**, because the average hides where the money goes. A
summarise task at $0.0005 per success and an extract task at $0.02 per success average out
to something that tells you nothing, while the breakdown points straight at the task to fix.

And one rule with no exceptions: **refuse a model that has no price.** If you cannot price
it, you cannot count it, and it will turn up on the invoice anyway.
""",
    todo=r'''
def cost_per_success(runs):
    """Overall cost per success, and the breakdown by task.

    Returns {"total_cost", "successes", "cost_per_success", "by_task"} where by_task
    maps a task name to its own {"cost", "successes", "cost_per_success", "waste"}.
    "waste" is the money spent on failed runs for that task.
    """
    # TODO 1: for each run, work out its cost with
    #         usd(run["model"], run["prompt"], run["completion"]).
    #         Every run counts, successful or not.
    # TODO 2: accumulate per task: total cost, number of successes, and the cost of
    #         the failed runs (the waste).
    # TODO 3: per task, cost_per_success = that task's TOTAL cost (failures included)
    #         divided by its successes. If a task has no successes, use float("inf") —
    #         money spent with nothing to show for it is not a cost of zero.
    # TODO 4: return the overall figures and the breakdown.
    raise NotImplementedError


def price_or_refuse(model, prompt, completion):
    """Cost for a model, or a refusal if we do not have its price."""
    # TODO 5: if model is not in PRICE_PER_1K, raise ValueError naming the model.
    #         A model you cannot price is one you cannot account for.
    raise NotImplementedError


report = cost_per_success(RUNS)
print(f"total spent      ${report['total_cost']:.5f}")
print(f"successes        {report['successes']} of {len(RUNS)}")
print(f"cost per success ${report['cost_per_success']:.5f}")
print(f"\n{'task':>12} {'cost':>10} {'ok':>4} {'per success':>13} {'wasted':>10}")
for task, figures in sorted(report["by_task"].items(),
                            key=lambda kv: -kv[1]["cost_per_success"]):
    print(f"{task:>12} ${figures['cost']:.5f} {figures['successes']:>4} "
          f"${figures['cost_per_success']:.5f} ${figures['waste']:.5f}")

print("\nan unpriced model:")
try:
    price_or_refuse("mystery-model-v9", 100, 100)
except ValueError as err:
    print("  refused:", err)
''',
    solution=r'''
def cost_per_success(runs):
    """Overall cost per success, and the breakdown by task.

    Returns {"total_cost", "successes", "cost_per_success", "by_task"} where by_task
    maps a task name to its own {"cost", "successes", "cost_per_success", "waste"}.
    "waste" is the money spent on failed runs for that task.
    """
    by_task = {}
    total_cost = 0.0
    successes = 0

    for run in runs:
        cost = usd(run["model"], run["prompt"], run["completion"])
        total_cost += cost
        successes += bool(run["ok"])

        figures = by_task.setdefault(run["task"],
                                     {"cost": 0.0, "successes": 0, "waste": 0.0})
        figures["cost"] += cost
        if run["ok"]:
            figures["successes"] += 1
        else:
            figures["waste"] += cost

    for figures in by_task.values():
        figures["cost_per_success"] = (figures["cost"] / figures["successes"]
                                       if figures["successes"] else float("inf"))

    return {"total_cost": total_cost, "successes": successes,
            "cost_per_success": total_cost / successes if successes else float("inf"),
            "by_task": by_task}


def price_or_refuse(model, prompt, completion):
    """Cost for a model, or a refusal if we do not have its price."""
    if model not in PRICE_PER_1K:
        raise ValueError(f"no price for model {model!r}; refusing to run it untracked")
    return usd(model, prompt, completion)


report = cost_per_success(RUNS)
print(f"total spent      ${report['total_cost']:.5f}")
print(f"successes        {report['successes']} of {len(RUNS)}")
print(f"cost per success ${report['cost_per_success']:.5f}")
print(f"\n{'task':>12} {'cost':>10} {'ok':>4} {'per success':>13} {'wasted':>10}")
for task, figures in sorted(report["by_task"].items(),
                            key=lambda kv: -kv[1]["cost_per_success"]):
    print(f"{task:>12} ${figures['cost']:.5f} {figures['successes']:>4} "
          f"${figures['cost_per_success']:.5f} ${figures['waste']:.5f}")

print("\nan unpriced model:")
try:
    price_or_refuse("mystery-model-v9", 100, 100)
except ValueError as err:
    print("  refused:", err)
''',
    check=r'''
@check("17.3  failures are charged to the successes")
def _():
    need("cost_per_success", "report")
    every_run = sum(usd(r["model"], r["prompt"], r["completion"]) for r in RUNS)
    assert abs(report["total_cost"] - every_run) < 1e-12, (
        f"total_cost is ${report['total_cost']:.6f} but all ten runs cost "
        f"${every_run:.6f}. Count every run — a call that failed was still billed."
    )
    assert report["successes"] == 7, \
        f"seven runs succeeded, report says {report['successes']}"
    assert abs(report["cost_per_success"] - every_run / 7) < 1e-12, (
        "cost per success is ALL the spending divided by the successes — not the "
        "average cost of the successful runs."
    )
    successful_only = sum(usd(r["model"], r["prompt"], r["completion"])
                          for r in RUNS if r["ok"]) / 7
    assert report["cost_per_success"] > successful_only, \
        "which means it must be higher than the average successful run"


@check("17.3  the breakdown shows where the money goes")
def _():
    need("cost_per_success", "report")
    by_task = report["by_task"]
    assert set(by_task) == {"summarise", "extract", "classify"}, \
        f"the breakdown covers {sorted(by_task)}"

    extract, classify = by_task["extract"], by_task["classify"]
    assert extract["cost_per_success"] > report["cost_per_success"] > \
        classify["cost_per_success"], (
        "extract should be well above the average and classify well below. If every "
        "task looks the same, the per-task totals are not being accumulated "
        f"separately. Got extract ${extract['cost_per_success']:.5f}, average "
        f"${report['cost_per_success']:.5f}, classify ${classify['cost_per_success']:.5f}"
    )
    assert extract["waste"] > 0, \
        "two extract runs failed, so some of its spend was wasted"
    assert classify["waste"] == 0, \
        f"no classify run failed, so its waste is 0; got {classify['waste']}"
    assert extract["successes"] == 2 and classify["successes"] == 2, \
        "two successes each for extract and classify"
    print(f"     average ${report['cost_per_success']:.5f}/success, but extract is "
          f"${extract['cost_per_success']:.5f} and classify ${classify['cost_per_success']:.5f}")


@check("17.3  all-failure and unpriced cases are handled")
def _():
    need("cost_per_success", "price_or_refuse")
    doomed = cost_per_success([
        {"task": "doomed", "model": "large", "prompt": 1000, "completion": 100,
         "ok": False}])
    assert doomed["cost_per_success"] == float("inf"), (
        "a task that never succeeds has an infinite cost per success, not zero. "
        "Reporting 0 would make the most broken task look like the cheapest."
    )
    assert doomed["by_task"]["doomed"]["cost_per_success"] == float("inf"), \
        "the same in the per-task breakdown"
    assert cost_per_success([])["successes"] == 0, "an empty list should not raise"

    assert price_or_refuse("small", 1000, 1000) > 0, "a priced model should work"
    for unknown in ("mystery-model-v9", "gpt-whatever", ""):
        try:
            price_or_refuse(unknown, 100, 100)
        except ValueError as err:
            assert unknown in str(err) or "price" in str(err), \
                f"name the model in the refusal; got {err!r}"
        else:
            raise AssertionError(
                f"{unknown!r} has no price and was allowed. A model you cannot price "
                "is one you cannot account for — and it still appears on the invoice."
            )
    print("     infinite cost for a task with no successes, unpriced models refused")
''',
    hint=[
        "In the loop:",
        "  cost = usd(run['model'], run['prompt'], run['completion'])",
        "  total_cost += cost",
        "  successes += bool(run['ok'])",
        "  figures = by_task.setdefault(run['task'],",
        "                               {'cost': 0.0, 'successes': 0, 'waste': 0.0})",
        "  figures['cost'] += cost",
        "  if run['ok']: figures['successes'] += 1",
        "  else:         figures['waste'] += cost",
        "",
        "Afterwards, per task:",
        "  figures['cost_per_success'] = (figures['cost'] / figures['successes']",
        "                                 if figures['successes'] else float('inf'))",
    ],
    notice=r"""
**What to notice.** `float("inf")` rather than `0` for a task that never succeeds. Report
zero and the most broken task in your system appears, on a dashboard, as the cheapest one.

And look at the spread: the average hides an order of magnitude between `classify` and
`extract`. "Our agent costs $0.004 per request" is a sentence that prevents the conversation
you need to have, which is about `extract` specifically.
""",
)

EX3 = Exercise(
    name="A circuit breaker that can close again",
    section="17.4",
    concept=r"""
A dependency goes down. Without protection, every request waits for a timeout, your
latency collapses, and you keep hammering a service that is trying to recover.

A **circuit breaker** is a small state machine that stops this:

```
CLOSED  ──too many failures──▶  OPEN  ──after a cooling-off period──▶  HALF_OPEN
  ▲                                                                       │
  └──────────────────── a test request succeeds ◀─────────────────────────┘
                              (it fails ──▶ back to OPEN)
```

* **CLOSED** — normal. Calls go through; failures are counted.
* **OPEN** — refuse immediately. Fail fast instead of waiting for a timeout, and let the
  dependency breathe.
* **HALF_OPEN** — after the cooling-off period, let **one** request through as a test. It
  succeeds: close. It fails: open again.

The half-open state is the part people leave out, and leaving it out is worse than having no
breaker at all:

> **A breaker that cannot close is an outage waiting for a trigger.**
""",
    todo=r'''
class CircuitBreaker:
    def __init__(self, failure_threshold=3, cooloff_seconds=0.05):
        self.failure_threshold = failure_threshold
        self.cooloff = cooloff_seconds
        self.state = "CLOSED"
        self.failures = 0
        self.opened_at = None
        self.refused = 0

    def _allow(self):
        """Decide whether a call may proceed now, moving state if the cooloff passed."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            # TODO 1: if the cooloff has passed (time.monotonic() - self.opened_at
            #         >= self.cooloff), move to HALF_OPEN and allow ONE test call.
            #         Otherwise refuse.
            raise NotImplementedError

        return True          # HALF_OPEN: the single test call is allowed

    def call(self, function, *args):
        """Run `function` through the breaker. Returns {"ok", "result", "state"}."""
        if not self._allow():
            self.refused += 1
            return {"ok": False, "result": "circuit open", "state": self.state}

        try:
            result = function(*args)
        except Exception as err:
            # TODO 2: a failure. Add one to self.failures. If we were HALF_OPEN, the
            #         test failed, so go straight back to OPEN (record opened_at with
            #         time.monotonic()). If we are CLOSED and self.failures has reached
            #         the threshold, open the circuit the same way.
            raise NotImplementedError

        # TODO 3: a success. If we were HALF_OPEN, the dependency is back: close the
        #         circuit and reset the failure count. If already CLOSED, reset the
        #         failure count too — a run of successes should clear old failures.
        raise NotImplementedError


service = Service(fail_first=5)
breaker = CircuitBreaker(failure_threshold=3, cooloff_seconds=0.05)

print("the service is down:")
for i in range(6):
    outcome = breaker.call(service)
    print(f"  call {i}: {str(outcome['ok']):>5}  state {outcome['state']}")

print(f"\nthe service was called {service.calls} times, "
      f"{breaker.refused} requests were refused without touching it")

print("\nwaiting out the cooloff; the service has recovered:")
time.sleep(0.06)
for i in range(3):
    outcome = breaker.call(service)
    print(f"  call {i}: {str(outcome['ok']):>5}  state {outcome['state']}")
''',
    solution=r'''
class CircuitBreaker:
    def __init__(self, failure_threshold=3, cooloff_seconds=0.05):
        self.failure_threshold = failure_threshold
        self.cooloff = cooloff_seconds
        self.state = "CLOSED"
        self.failures = 0
        self.opened_at = None
        self.refused = 0

    def _allow(self):
        """Decide whether a call may proceed now, moving state if the cooloff passed."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if time.monotonic() - self.opened_at >= self.cooloff:
                self.state = "HALF_OPEN"       # let exactly one request test the water
                return True
            return False

        return True          # HALF_OPEN: the single test call is allowed

    def _open(self):
        self.state = "OPEN"
        self.opened_at = time.monotonic()

    def call(self, function, *args):
        """Run `function` through the breaker. Returns {"ok", "result", "state"}."""
        if not self._allow():
            self.refused += 1
            return {"ok": False, "result": "circuit open", "state": self.state}

        try:
            result = function(*args)
        except Exception as err:
            self.failures += 1
            if self.state == "HALF_OPEN" or self.failures >= self.failure_threshold:
                self._open()
            return {"ok": False, "result": str(err), "state": self.state}

        if self.state == "HALF_OPEN":
            self.state = "CLOSED"              # recovered: this is the part that matters
        self.failures = 0
        return {"ok": True, "result": result, "state": self.state}


service = Service(fail_first=5)
breaker = CircuitBreaker(failure_threshold=3, cooloff_seconds=0.05)

print("the service is down:")
for i in range(6):
    outcome = breaker.call(service)
    print(f"  call {i}: {str(outcome['ok']):>5}  state {outcome['state']}")

print(f"\nthe service was called {service.calls} times, "
      f"{breaker.refused} requests were refused without touching it")

print("\nwaiting out the cooloff; the service has recovered:")
time.sleep(0.06)
for i in range(3):
    outcome = breaker.call(service)
    print(f"  call {i}: {str(outcome['ok']):>5}  state {outcome['state']}")
''',
    check=r'''
@check("17.4  the circuit opens and then fails fast")
def _():
    need("CircuitBreaker")
    down = Service(fail_first=99)
    b = CircuitBreaker(failure_threshold=3, cooloff_seconds=10)

    for _ in range(3):
        b.call(down)
    assert b.state == "OPEN", (
        f"after 3 failures with a threshold of 3 the circuit should be OPEN, it is "
        f"{b.state}"
    )
    calls_so_far = down.calls

    for _ in range(10):
        outcome = b.call(down)
        assert outcome["ok"] is False and outcome["result"] == "circuit open", \
            f"an open circuit refuses immediately; got {outcome}"
    assert down.calls == calls_so_far, (
        f"the service was called {down.calls - calls_so_far} more times while the "
        "circuit was open. The point is to stop touching it — both to fail fast and to "
        "let it recover."
    )
    assert b.refused == 10, f"all 10 refusals should be counted; got {b.refused}"


@check("17.4  the circuit can close again")
def _():
    need("CircuitBreaker")
    service = Service(fail_first=3)
    b = CircuitBreaker(failure_threshold=3, cooloff_seconds=0.02)

    for _ in range(3):
        b.call(service)
    assert b.state == "OPEN", "three failures should open it"

    assert b.call(service)["result"] == "circuit open", \
        "immediately after opening, calls are refused"

    time.sleep(0.03)
    recovered = b.call(service)
    assert recovered["ok"] is True, (
        f"after the cooloff, one test call should be allowed through and this one "
        f"succeeds; got {recovered}"
    )
    assert b.state == "CLOSED", (
        f"the test call succeeded, so the circuit must CLOSE — it is {b.state}. A "
        "breaker that cannot close is an outage waiting for a trigger."
    )
    assert b.failures == 0, "closing should reset the failure count"
    assert b.call(service)["ok"] is True, "and normal traffic resumes"


@check("17.4  a failed test re-opens it immediately")
def _():
    need("CircuitBreaker")
    still_down = Service(fail_first=99)
    b = CircuitBreaker(failure_threshold=2, cooloff_seconds=0.02)

    for _ in range(2):
        b.call(still_down)
    assert b.state == "OPEN"

    time.sleep(0.03)
    outcome = b.call(still_down)
    assert outcome["ok"] is False, "the test call fails, because the service is still down"
    assert b.state == "OPEN", (
        f"a failed test call must go straight back to OPEN, not stay HALF_OPEN or "
        f"close; it is {b.state}. Otherwise every later request becomes another test "
        "against a dead service."
    )

    assert b.call(still_down)["result"] == "circuit open", \
        "and the circuit refuses again without calling out"

    time.sleep(0.03)
    assert b.call(still_down)["ok"] is False, \
        "after another cooloff it tries once more — that is the retry loop working"
    print("     opened after failures, refused without calling out, closed on recovery, "
          "re-opened on a failed test")
''',
    hint=[
        "_allow, in the OPEN branch:",
        "  if time.monotonic() - self.opened_at >= self.cooloff:",
        "      self.state = 'HALF_OPEN'",
        "      return True",
        "  return False",
        "",
        "On a failure:",
        "  self.failures += 1",
        "  if self.state == 'HALF_OPEN' or self.failures >= self.failure_threshold:",
        "      self.state = 'OPEN'; self.opened_at = time.monotonic()",
        "  return {'ok': False, 'result': str(err), 'state': self.state}",
        "",
        "On a success:",
        "  if self.state == 'HALF_OPEN': self.state = 'CLOSED'",
        "  self.failures = 0",
        "  return {'ok': True, 'result': result, 'state': self.state}",
    ],
    notice=r"""
**What to notice.** Three lines carry the whole design:

* `self.state = "HALF_OPEN"` after the cooloff — the circuit is willing to be wrong about
  the outage.
* `if self.state == "HALF_OPEN": self.state = "CLOSED"` on success — the recovery path.
* `self.failures = 0` on success — otherwise old failures accumulate for ever and the
  breaker eventually opens during normal operation.

Leave out any one of them and you have built something that detects an outage and then
causes a permanent one.

### That is the last exercise in the book

Across seventeen notebooks you have built a neural network, an agent loop, tools with real
boundaries, memory, RAG, a vector store, an MCP server, four architectures, a multi-agent
pipeline, a small framework, a safety gateway, an evaluated application, and now the
production machinery around all of it.

The pattern worth taking away is not any single technique. It is that almost every failure
in this book was **quiet** — a filter covering one path, a cached vector from the wrong
model, a partial result that looked complete, a self-grading loop that was pleased with a
wrong answer, a breaker that could not close. None of them raised an exception.

Which is why nearly every exercise ended by making you **measure** something. That habit is
the actual skill.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=17,
    title="Production Deployment",
    intro=r"""
The last notebook, and the one about keeping all of it running: **traces** that tie a
failure to a request, **cost per success** that ties money to outcomes, and a **circuit
breaker** that ties a limit to a refusal.

> Observability earns its place when it changes what the system does. Test each of those
> three by running it — which is what you are about to do.
""",
    minutes=35,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "A **trace** is spans with ids and parents: that is what turns seven log lines "
        "into one explainable request.",
        "Measure durations with a **monotonic clock** — the wall clock can jump backwards.",
        "In production use the **OpenTelemetry SDK**, and keep attribute names in one "
        "place; the GenAI conventions are still changing.",
        "**Cost per success** is all the spending divided by the successes — failures are "
        "overhead on the work that worked.",
        "Break cost down **by task type**: the average hides where the money goes. A task "
        "with no successes costs **infinity**, not zero.",
        "**Refuse a model you cannot price.**",
        "A **circuit breaker** is open, wait, test, close — and one that cannot close is "
        "an outage waiting for a trigger.",
    ],
    next_up="Nothing — you have reached the end of the book. Go and build something, and "
            "measure it.",
)
