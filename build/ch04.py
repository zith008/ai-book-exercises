"""Chapter 4 — The Core Agent Loop: tracing, metrics, parallel tools (4.1-4.3)."""

from lite import Exercise, make_chapter

GIVEN = r'''
import re, time, asyncio

class ScriptedLLM:
    """The same stand-in model as Chapter 3: replays prepared replies."""
    def __init__(self, replies, delay=0.0):
        self.replies, self.delay = list(replies), delay

    def __call__(self, messages):
        if self.delay:
            time.sleep(self.delay)          # pretend the network is slow
        return self.replies.pop(0)


# ---- Tools, but slow on purpose, so timing is visible -----------------------
def slow_weather(city):
    time.sleep(0.4)                         # stands in for a real network call
    return f"Weather in {city.strip().title()}: 18°C, partly cloudy (simulated)"

def slow_search(query):
    time.sleep(0.4)
    return f"Search results for '{query}': found 3 results (simulated)"

def calculate(expression):
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        raise ValueError("only plain arithmetic is allowed")
    return f"Result: {eval(expression)}"

TOOLS = {"weather": slow_weather, "search": slow_search, "calculate": calculate}

SYSTEM_PROMPT = ("You are an agent. Reply with a Thought: line and an Action: line. "
                 "Tools: weather(city), search(query), calculate(expression). "
                 "Finish with Action: finish(answer).")


def parse_actions(reply):
    """Every 'Action: name(args)' line in a reply, as a list of (name, args)."""
    return re.findall(r"Action:\s*(\w+)\((.*?)\)\s*$", reply, re.MULTILINE)


def act(name, args):
    """Run one tool. Never raises — errors come back as text, as in Chapter 3."""
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available: {', '.join(TOOLS)}"
    try:
        return TOOLS[name](args)
    except Exception as err:
        return f"Error running {name}: {err}"


print("tools:", ", ".join(TOOLS), "— weather and search take 0.4s each")
'''

EX1 = Exercise(
    name="Write down every step",
    section="4.1",
    concept=r"""
In Chapter 3 your agent printed as it went. Printing is fine while you watch it. It is
useless the morning after, when a user says "it gave me a weird answer yesterday".

So record each step as it happens. A **trace**: one small dictionary per step, saying what
the model thought, which tool ran, and what came back.

Think of a flight recorder. The plane does not need it to fly. You need it to understand
a flight that already happened.

Each step record holds four things:

| Key | What it is |
|---|---|
| `step` | 1, 2, 3 … |
| `thought` | the model's reply, as text |
| `tool` | the tool name it asked for, or `None` |
| `observation` | what the tool returned |
""",
    todo=r'''
def run_with_trace(llm, goal, max_steps=5):
    """Run the agent and return (answer, trace)."""
    history = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": goal}]
    trace = []

    for step in range(1, max_steps + 1):
        reply = llm(history)
        history.append({"role": "assistant", "content": reply})
        actions = parse_actions(reply)
        name, args = actions[0] if actions else (None, "")

        if name == "finish":
            # TODO 1: record this last step too, with tool="finish" and
            #         observation=None, then return (args, trace)
            raise NotImplementedError

        observation = act(name, args)
        history.append({"role": "user", "content": f"Observation: {observation}"})

        # TODO 2: append one record to trace:
        #         {"step": step, "thought": reply, "tool": name,
        #          "observation": observation}
        raise NotImplementedError

    return None, trace


answer, trace = run_with_trace(ScriptedLLM([
    "Thought: I need the weather.\nAction: weather(Tokyo)",
    "Thought: Now the forecast chatter.\nAction: search(Tokyo rain tomorrow)",
    "Thought: Enough.\nAction: finish(18°C and cloudy, rain unlikely.)",
]), "Will it rain in Tokyo?")

print("answer:", answer, "\n")
for record in trace:
    print(f"step {record['step']}: tool={record['tool']!r} "
          f"observation={str(record['observation'])[:45]!r}")
''',
    solution=r'''
def run_with_trace(llm, goal, max_steps=5):
    """Run the agent and return (answer, trace)."""
    history = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": goal}]
    trace = []

    for step in range(1, max_steps + 1):
        reply = llm(history)
        history.append({"role": "assistant", "content": reply})
        actions = parse_actions(reply)
        name, args = actions[0] if actions else (None, "")

        if name == "finish":
            trace.append({"step": step, "thought": reply,
                          "tool": "finish", "observation": None})
            return args, trace

        observation = act(name, args)
        history.append({"role": "user", "content": f"Observation: {observation}"})

        trace.append({"step": step, "thought": reply,
                      "tool": name, "observation": observation})

    return None, trace


answer, trace = run_with_trace(ScriptedLLM([
    "Thought: I need the weather.\nAction: weather(Tokyo)",
    "Thought: Now the forecast chatter.\nAction: search(Tokyo rain tomorrow)",
    "Thought: Enough.\nAction: finish(18°C and cloudy, rain unlikely.)",
]), "Will it rain in Tokyo?")

print("answer:", answer, "\n")
for record in trace:
    print(f"step {record['step']}: tool={record['tool']!r} "
          f"observation={str(record['observation'])[:45]!r}")
''',
    check=r'''
@check("4.1  every step is recorded, in order")
def _():
    need("run_with_trace", "trace")
    assert len(trace) == 3, (
        f"the trace has {len(trace)} records; that run took 3 steps (weather, search, "
        "finish). Remember to record the finishing step as well."
    )
    assert [r["step"] for r in trace] == [1, 2, 3], \
        f"step numbers are {[r['step'] for r in trace]}, expected [1, 2, 3]"
    assert [r["tool"] for r in trace] == ["weather", "search", "finish"], \
        f"tools recorded: {[r['tool'] for r in trace]}"
    assert "18°C" in trace[0]["observation"], \
        "step 1's observation should be what the weather tool returned"
    assert trace[2]["observation"] is None, \
        "the finishing step calls no tool, so its observation is None"
    assert "Thought:" in trace[0]["thought"], \
        "store the model's whole reply in 'thought' — you will want it when debugging"
    print("     3 steps recorded with thought, tool and observation")
''',
    hint=[
        "On the finish branch:",
        "  trace.append({'step': step, 'thought': reply,",
        "                'tool': 'finish', 'observation': None})",
        "  return args, trace",
        "",
        "At the end of the normal branch:",
        "  trace.append({'step': step, 'thought': reply,",
        "                'tool': name, 'observation': observation})",
    ],
    notice=r"""
**What to notice.** The trace is just a list of dictionaries — no framework, no logging
library. That is enough to answer "what did it do?" for any run you keep.

Keep traces for real runs and you can replay a complaint instead of guessing at it.
""",
)

EX2 = Exercise(
    name="Count things while it runs",
    section="4.2",
    concept=r"""
A trace tells you about **one** run. Metrics tell you about **all** of them.

Four numbers answer most questions you will ever have about a misbehaving agent:

* **steps** — how much thinking it took.
* **tool_calls** — which tools, and how often. An agent calling `weather` nine times is
  stuck in a circle.
* **tool_errors** — how often a tool came back with an error.
* **duration** — how long the user waited.

They are cheap to collect: a dictionary and a timer. Collect them from the first day,
because the day you need them is the day something is already wrong.

> Measure from the start. Steps, tool calls, errors and time are nearly free to record,
> and they answer most of the questions you will have later.
""",
    todo=r'''
def run_with_metrics(llm, goal, max_steps=6):
    """Run the agent and return (answer, metrics)."""
    history = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": goal}]
    metrics = {"steps": 0, "tool_calls": {}, "tool_errors": 0, "duration_s": 0.0}
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        # TODO 1: count this step in metrics["steps"]
        reply = llm(history)
        history.append({"role": "assistant", "content": reply})
        actions = parse_actions(reply)
        name, args = actions[0] if actions else (None, "")

        if name == "finish":
            metrics["duration_s"] = round(time.perf_counter() - started, 2)
            return args, metrics

        observation = act(name, args)

        # TODO 2: add 1 to metrics["tool_calls"][name]. The key may not exist yet, so
        #         use metrics["tool_calls"].get(name, 0) + 1
        # TODO 3: if the observation starts with "Error", count it in
        #         metrics["tool_errors"]
        raise NotImplementedError

        history.append({"role": "user", "content": f"Observation: {observation}"})

    metrics["duration_s"] = round(time.perf_counter() - started, 2)
    return None, metrics


answer, metrics = run_with_metrics(ScriptedLLM([
    "Thought: weather first.\nAction: weather(Tokyo)",
    "Thought: let me do some maths.\nAction: calculate(banana)",   # this one fails
    "Thought: try again.\nAction: calculate(2 + 2)",
    "Thought: done.\nAction: finish(It is 18°C, and 2 + 2 = 4.)",
]), "weather in Tokyo, and what is 2 + 2?")

print("answer:", answer)
print("metrics:", metrics)
''',
    solution=r'''
def run_with_metrics(llm, goal, max_steps=6):
    """Run the agent and return (answer, metrics)."""
    history = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": goal}]
    metrics = {"steps": 0, "tool_calls": {}, "tool_errors": 0, "duration_s": 0.0}
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        metrics["steps"] += 1
        reply = llm(history)
        history.append({"role": "assistant", "content": reply})
        actions = parse_actions(reply)
        name, args = actions[0] if actions else (None, "")

        if name == "finish":
            metrics["duration_s"] = round(time.perf_counter() - started, 2)
            return args, metrics

        observation = act(name, args)

        metrics["tool_calls"][name] = metrics["tool_calls"].get(name, 0) + 1
        if observation.startswith("Error"):
            metrics["tool_errors"] += 1

        history.append({"role": "user", "content": f"Observation: {observation}"})

    metrics["duration_s"] = round(time.perf_counter() - started, 2)
    return None, metrics


answer, metrics = run_with_metrics(ScriptedLLM([
    "Thought: weather first.\nAction: weather(Tokyo)",
    "Thought: let me do some maths.\nAction: calculate(banana)",   # this one fails
    "Thought: try again.\nAction: calculate(2 + 2)",
    "Thought: done.\nAction: finish(It is 18°C, and 2 + 2 = 4.)",
]), "weather in Tokyo, and what is 2 + 2?")

print("answer:", answer)
print("metrics:", metrics)
''',
    check=r'''
@check("4.2  the four numbers are right")
def _():
    need("run_with_metrics", "metrics")
    assert metrics["steps"] == 4, \
        f"that run took 4 steps, metrics says {metrics['steps']}"
    assert metrics["tool_calls"] == {"weather": 1, "calculate": 2}, (
        f"tool_calls is {metrics['tool_calls']}, expected "
        "{'weather': 1, 'calculate': 2} — calculate was called twice, once badly."
    )
    assert metrics["tool_errors"] == 1, (
        f"tool_errors is {metrics['tool_errors']}, expected 1: calculate('banana') "
        "failed. Check the observation for a leading 'Error'."
    )
    assert metrics["duration_s"] > 0, "duration should be measured, not left at 0"
    print(f"     {metrics['steps']} steps, {metrics['tool_errors']} error, "
          f"{metrics['duration_s']}s")


@check("4.2  a circling agent is visible in the numbers")
def _():
    need("run_with_metrics")
    _, m = run_with_metrics(
        ScriptedLLM(["Thought: checking again.\nAction: weather(Tokyo)"] * 8),
        "go round in circles", max_steps=5)
    assert m["tool_calls"].get("weather") == 5, (
        f"tool_calls is {m['tool_calls']} — a loop that calls one tool 5 times should "
        "show up as weather: 5. This is how you spot a stuck agent in production."
    )
    print(f"     stuck run looks like: {m['tool_calls']} in {m['duration_s']}s")
''',
    hint=[
        "metrics['steps'] += 1          <- first line inside the for loop",
        "",
        "After running the tool:",
        "  metrics['tool_calls'][name] = metrics['tool_calls'].get(name, 0) + 1",
        "  if observation.startswith('Error'):",
        "      metrics['tool_errors'] += 1",
        "",
        ".get(name, 0) + 1 is the standard way to count into a dict whose keys",
        "you do not know in advance.",
    ],
    notice=r"""
**What to notice.** `{'weather': 5}` in the second check is a stuck agent, visible at a
glance. You did not need a dashboard or a tracing service to see it — just a dictionary
that counted.
""",
)

EX3 = Exercise(
    name="Run tools at the same time",
    section="4.3",
    concept=r"""
Watch the clock in the last exercise. Each slow tool took 0.4 seconds, one after another.
Three cities means 1.2 seconds of the user staring at a spinner — and *none of those calls
needed the others' results*.

That is the test. **If the calls do not depend on each other, run them together.**

Making breakfast is the analogy: you do not boil the kettle, wait, *then* put the toast
on. You start both and wait once.

In Python, `asyncio` does the waiting:

```python
results = await asyncio.gather(
    asyncio.to_thread(act, "weather", "Tokyo"),
    asyncio.to_thread(act, "weather", "London"),
)
```

* `asyncio.to_thread(fn, *args)` runs an ordinary blocking function without holding
  everything else up.
* `asyncio.gather(...)` starts them all, waits for the slowest, and returns the results
  **in the order you asked for them** — which matters, because the model needs to know
  which answer belongs to which city.

> The saving is usually bigger than it looks: fewer round trips also means fewer model
> calls, and model calls cost more than waiting does.
""",
    todo=r'''
async def run_all(calls):
    """calls is a list of (tool_name, argument). Run them all at the same time.

    Return a list of observation strings, in the same order as `calls`.
    """
    # TODO 1: build one asyncio.to_thread(act, name, args) task per call
    # TODO 2: await asyncio.gather(*tasks) and return the results
    raise NotImplementedError


def run_sequential(calls):
    """The slow way, for comparison. Given to you."""
    return [act(name, args) for name, args in calls]


calls = [("weather", "Tokyo"), ("weather", "London"), ("search", "umbrella sales")]

started = time.perf_counter()
slow_results = run_sequential(calls)
slow_time = time.perf_counter() - started

started = time.perf_counter()
fast_results = asyncio.run(run_all(calls))
fast_time = time.perf_counter() - started

print(f"one after another: {slow_time:.2f}s")
print(f"all at once:       {fast_time:.2f}s   ({slow_time / fast_time:.1f}x faster)")
for (name, args), result in zip(calls, fast_results):
    print(f"  {name}({args}) -> {result[:50]}")
''',
    solution=r'''
async def run_all(calls):
    """calls is a list of (tool_name, argument). Run them all at the same time.

    Return a list of observation strings, in the same order as `calls`.
    """
    tasks = [asyncio.to_thread(act, name, args) for name, args in calls]
    return await asyncio.gather(*tasks)


def run_sequential(calls):
    """The slow way, for comparison. Given to you."""
    return [act(name, args) for name, args in calls]


calls = [("weather", "Tokyo"), ("weather", "London"), ("search", "umbrella sales")]

started = time.perf_counter()
slow_results = run_sequential(calls)
slow_time = time.perf_counter() - started

started = time.perf_counter()
fast_results = asyncio.run(run_all(calls))
fast_time = time.perf_counter() - started

print(f"one after another: {slow_time:.2f}s")
print(f"all at once:       {fast_time:.2f}s   ({slow_time / fast_time:.1f}x faster)")
for (name, args), result in zip(calls, fast_results):
    print(f"  {name}({args}) -> {result[:50]}")
''',
    check=r'''
@check("4.3  parallel calls are faster, and still in order")
def _():
    need("run_all", "fast_results", "slow_time", "fast_time")
    assert len(fast_results) == 3, \
        f"expected 3 results, got {len(fast_results)}"
    assert "Tokyo" in fast_results[0] and "London" in fast_results[1], (
        "the results came back in the wrong order. asyncio.gather keeps the order of "
        "the tasks you pass it — do not sort or re-shuffle them, or the model will "
        "attribute London's weather to Tokyo."
    )
    assert "Search results" in fast_results[2], \
        "the third result should come from the search tool"
    assert fast_time < slow_time * 0.75, (
        f"parallel took {fast_time:.2f}s against {slow_time:.2f}s sequential — barely "
        "faster. Build ALL the tasks first, then await gather once. Awaiting each task "
        "as you create it runs them one at a time again."
    )
    print(f"     {slow_time:.2f}s -> {fast_time:.2f}s, order preserved")


@check("4.3  it still works for a single call")
def _():
    need("run_all")
    one = asyncio.run(run_all([("calculate", "7 * 6")]))
    assert len(one) == 1 and "42" in one[0], f"got {one!r}, expected ['Result: 42']"
''',
    hint=[
        "Two lines:",
        "  tasks = [asyncio.to_thread(act, name, args) for name, args in calls]",
        "  return await asyncio.gather(*tasks)",
        "",
        "The common mistake is awaiting inside the loop:",
        "  for name, args in calls:",
        "      results.append(await asyncio.to_thread(act, name, args))   # still serial!",
        "",
        "Start everything first, then wait once. That is the whole idea.",
    ],
    notice=r"""
**What to notice.** Same tools, same results, a third of the wait. And nothing about the
agent loop changed — parallelism went in at the point where tools are executed.

**The one rule:** only do this when the calls are independent. If step 2 needs step 1's
answer, running them together just gets you a wrong answer faster.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=4,
    title="The Core Agent Loop",
    intro=r"""
Chapter 3 gave you a working agent. This notebook takes the loop at its centre and makes
it **observable** — you will be able to say what a run did, count what it did across many
runs, and make it faster where the work does not depend on itself.

This is the difference between an agent that demos and an agent you can operate.
""",
    minutes=30,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "A **trace** — one record per step — lets you explain a run after it has happened.",
        "Four cheap **metrics** (steps, tool calls, tool errors, duration) answer most "
        "questions about a misbehaving agent.",
        "A tool called five times in a row is a **stuck agent**, and the numbers show it "
        "at a glance.",
        "Independent tool calls should run **together**: build the tasks, then "
        "`await asyncio.gather(*tasks)` once.",
        "`gather` preserves order — which is what keeps London's weather from being "
        "reported as Tokyo's.",
    ],
    next_up="Chapter 5 is about the tools themselves: describing them so the model uses "
            "them correctly, and keeping them from doing damage.",
)
