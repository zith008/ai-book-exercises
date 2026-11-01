"""Chapter 12 — The Six Agentic Architectures: ReAct's cost, plan-first, reflection."""

from lite import Exercise, make_chapter

GIVEN = r'''
import re, time

def estimate_tokens(messages):
    """About 4 characters per token. The API's `usage` is exact; this is for modelling."""
    return sum(len(m["content"]) for m in messages) // 4


class ScriptedLLM:
    """The stand-in model from Chapter 3. Calls are counted, so cost is visible."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def __call__(self, messages):
        self.calls += 1
        return self.replies.pop(0) if self.replies else "Thought: done.\nAction: finish(ok)"


# ---- A task we can check mechanically: clean up a messy list of prices ------
TASK = "Sum the prices in the basket and apply the 10% discount."
BASKET = ["12.50", "7.99", "3.01"]

def sum_prices(_=None):
    return f"Total: {sum(float(p) for p in BASKET):.2f}"

def apply_discount(total_text):
    total = float(re.search(r"[\d.]+", total_text).group())
    return f"Discounted: {total * 0.9:.2f}"

TOOLS = {"sum_prices": sum_prices, "apply_discount": apply_discount}

print("tools:", ", ".join(TOOLS), "| basket:", BASKET)
print("the right answer is", apply_discount(sum_prices()))
'''

EX1 = Exercise(
    name="Why long ReAct loops get expensive fast",
    section="12.1",
    concept=r"""
A **ReAct** loop decides nothing in advance. Think, act, observe, repeat — adapting at every
step. That flexibility is why Chapter 3 started there, and it has a cost that surprises
people the first time they see a bill.

Every step re-sends the **whole history**. So step 1 pays for 1 step of context, step 2 pays
for 2, step 10 pays for 10. Total cost is not proportional to the number of steps — it is
roughly **the square** of it:

$$\text{total tokens} \approx \sum_{i=1}^{n} i \cdot c = c\,\frac{n(n+1)}{2}$$

Ten steps cost about 55 units where you might have guessed 10. Twenty steps cost 210.

Which leads to the three things every loop needs:

* a **budget** — a cap on steps or tokens, so a bad run cannot become a bad invoice
* a **stuck detector** — because the common failure is repeating the same action forever
* a **way to change approach** when stuck, rather than trying harder at the same thing
""",
    todo=r'''
def react_cost(steps, tokens_per_step=100):
    """Total prompt tokens for a ReAct loop of `steps` steps.

    Step 1 sends 1 step's worth of history, step 2 sends 2, and so on.
    """
    # TODO 1: return the total across all the steps, not just the last one.
    raise NotImplementedError


class StuckDetector:
    """Spots a loop repeating itself."""

    def __init__(self, repeats_allowed=2):
        self.repeats_allowed = repeats_allowed
        self.history = []

    def record(self, action):
        """Record an action, e.g. ('sum_prices', ''). Return True if we are stuck."""
        # TODO 2: append the action, then return True if the SAME action has now
        #         happened more than self.repeats_allowed times in a row.
        #         Looking at the last few entries is enough.
        raise NotImplementedError


print(f"{'steps':>6} {'total tokens':>13} {'if it were linear':>18}")
for steps in (1, 5, 10, 20, 40):
    print(f"{steps:>6} {react_cost(steps):>13,} {steps * 100:>18,}")

detector = StuckDetector(repeats_allowed=2)
for action in [("sum_prices", ""), ("sum_prices", ""), ("sum_prices", ""),
               ("apply_discount", "x")]:
    print(f"{str(action):>28} -> stuck: {detector.record(action)}")
''',
    solution=r'''
def react_cost(steps, tokens_per_step=100):
    """Total prompt tokens for a ReAct loop of `steps` steps.

    Step 1 sends 1 step's worth of history, step 2 sends 2, and so on.
    """
    return sum(step * tokens_per_step for step in range(1, steps + 1))


class StuckDetector:
    """Spots a loop repeating itself."""

    def __init__(self, repeats_allowed=2):
        self.repeats_allowed = repeats_allowed
        self.history = []

    def record(self, action):
        """Record an action, e.g. ('sum_prices', ''). Return True if we are stuck."""
        self.history.append(action)
        recent = self.history[-(self.repeats_allowed + 1):]
        return (len(recent) > self.repeats_allowed
                and all(a == action for a in recent))


print(f"{'steps':>6} {'total tokens':>13} {'if it were linear':>18}")
for steps in (1, 5, 10, 20, 40):
    print(f"{steps:>6} {react_cost(steps):>13,} {steps * 100:>18,}")

detector = StuckDetector(repeats_allowed=2)
for action in [("sum_prices", ""), ("sum_prices", ""), ("sum_prices", ""),
               ("apply_discount", "x")]:
    print(f"{str(action):>28} -> stuck: {detector.record(action)}")
''',
    check=r'''
@check("12.1  cost grows with the square of the length")
def _():
    need("react_cost")
    assert react_cost(1) == 100, f"one step costs 100, you returned {react_cost(1)}"
    assert react_cost(2) == 300, (
        f"two steps cost 100 + 200 = 300, you returned {react_cost(2)}. The second "
        "call re-sends the first step's history as well as its own."
    )
    assert react_cost(10) == 5500, \
        f"ten steps cost 100*(10*11/2) = 5500, you returned {react_cost(10)}"
    doubled = react_cost(20) / react_cost(10)
    assert 3.5 < doubled < 4.5, (
        f"doubling the steps multiplied the cost by {doubled:.1f}. Quadratic growth "
        "means roughly 4x — this is the number to quote when someone suggests raising "
        "max_steps to 50."
    )
    print(f"     10 steps: {react_cost(10):,} tokens; 20 steps: {react_cost(20):,}")


@check("12.1  the stuck detector fires on repetition, not on variety")
def _():
    need("StuckDetector")
    d = StuckDetector(repeats_allowed=2)
    assert d.record(("a", "")) is False, "one call is not stuck"
    assert d.record(("a", "")) is False, "two identical calls are allowed here"
    assert d.record(("a", "")) is True, (
        "three identical actions in a row should report stuck with repeats_allowed=2"
    )

    varied = StuckDetector(repeats_allowed=2)
    for action in [("a", ""), ("b", ""), ("a", ""), ("b", ""), ("a", "")]:
        assert varied.record(action) is False, (
            "alternating actions are not 'stuck' by this rule — it looks for the same "
            "action repeated CONSECUTIVELY"
        )

    args_matter = StuckDetector(repeats_allowed=2)
    args_matter.record(("search", "cats"))
    args_matter.record(("search", "cats"))
    assert args_matter.record(("search", "dogs")) is False, (
        "the same tool with different arguments is progress, not repetition — compare "
        "the whole action, arguments included"
    )
    print("     fires on 3 identical calls, ignores variety and changed arguments")
''',
    hint=[
        "react_cost: sum over the steps, not just the last one.",
        "  return sum(step * tokens_per_step for step in range(1, steps + 1))",
        "",
        "StuckDetector.record:",
        "  self.history.append(action)",
        "  recent = self.history[-(self.repeats_allowed + 1):]",
        "  return len(recent) > self.repeats_allowed and all(a == action for a in recent)",
        "",
        "Compare the whole action tuple. ('search', 'cats') twice is repetition;",
        "('search', 'cats') then ('search', 'dogs') is the agent trying something new.",
    ],
    notice=r"""
**What to notice.** Forty steps cost 82,000 tokens where a linear guess says 4,000. This is
why "just raise max_steps" is an expensive suggestion, and why the loop limits from
Chapter 4 are not bureaucracy.

The next exercise is the other answer to the same problem: decide more in advance, so there
are fewer steps to pay for.
""",
)

EX2 = Exercise(
    name="Deciding in advance",
    section="12.2",
    concept=r"""
The six architectures in this chapter differ in **how much they decide in advance**:

| Architecture | Decides in advance | Good for |
|---|---|---|
| **ReAct** | nothing — adapts every step | the next step depends on what you just learned |
| **Plan and Execute** | everything — one plan, then run it | the shape of the work is known |
| **Reflection** | adds looking *back* at its own output | there is a signal to check against |
| **Tree search** | adds looking *sideways* at alternatives | the task is worth many attempts |

**Plan and Execute** calls the model once to produce a plan, then executes the steps —
often with no model call at all per step. For a task whose shape you know, that turns
Exercise 1's quadratic curve into something close to flat.

The price is rigidity: if step 2 fails, the rest of the plan may be nonsense. So a
plan-first agent needs **re-planning** — notice the failure, make a new plan, carry on.

> Plan first when the shape of the work is known and the steps are independent. Keep ReAct
> when the next step depends on what you just learned.
""",
    todo=r'''
def make_plan(task):
    """One model call would produce this. Hard-coded here so the exercise is about
    the executor. Given to you."""
    return [{"tool": "sum_prices", "input": None},
            {"tool": "apply_discount", "input": "$previous"}]


def execute_plan(plan, tools, llm=None):
    """Run a plan. Returns {"result", "steps_run", "model_calls", "failed_at"}.

    "$previous" as an input means "use the previous step's output".
    """
    previous = None
    model_calls = 0

    for index, step in enumerate(plan):
        tool_name = step["tool"]

        # TODO 1: if the tool is not in `tools`, stop and return with failed_at=index,
        #         result=None, steps_run=index. A plan referring to a tool that does
        #         not exist must not look like a success.
        # TODO 2: work out the argument: step["input"], unless it is the string
        #         "$previous", in which case use the previous step's output.
        # TODO 3: run the tool inside try/except. On an exception, return with
        #         failed_at=index and result=None.
        # TODO 4: keep the output as `previous` and carry on.
        raise NotImplementedError

    # TODO 5: all the steps ran — return
    #         {"result": previous, "steps_run": len(plan),
    #          "model_calls": model_calls, "failed_at": None}
    raise NotImplementedError


def plan_and_execute(task, tools):
    """One model call to plan, then execution. Returns the same record."""
    planner = ScriptedLLM(["(a plan)"])
    planner(({"role": "user", "content": task},))      # the one planning call
    outcome = execute_plan(make_plan(task), tools)
    outcome["model_calls"] = outcome["model_calls"] + planner.calls
    return outcome


result = plan_and_execute(TASK, TOOLS)
print("plan and execute:", result)

broken = execute_plan([{"tool": "sum_prices", "input": None},
                       {"tool": "no_such_tool", "input": "$previous"},
                       {"tool": "apply_discount", "input": "$previous"}], TOOLS)
print("a plan with a bad step:", broken)
''',
    solution=r'''
def make_plan(task):
    """One model call would produce this. Hard-coded here so the exercise is about
    the executor. Given to you."""
    return [{"tool": "sum_prices", "input": None},
            {"tool": "apply_discount", "input": "$previous"}]


def execute_plan(plan, tools, llm=None):
    """Run a plan. Returns {"result", "steps_run", "model_calls", "failed_at"}.

    "$previous" as an input means "use the previous step's output".
    """
    previous = None
    model_calls = 0

    for index, step in enumerate(plan):
        tool_name = step["tool"]

        if tool_name not in tools:
            return {"result": None, "steps_run": index,
                    "model_calls": model_calls, "failed_at": index}

        argument = previous if step["input"] == "$previous" else step["input"]

        try:
            previous = tools[tool_name](argument)
        except Exception:
            return {"result": None, "steps_run": index,
                    "model_calls": model_calls, "failed_at": index}

    return {"result": previous, "steps_run": len(plan),
            "model_calls": model_calls, "failed_at": None}


def plan_and_execute(task, tools):
    """One model call to plan, then execution. Returns the same record."""
    planner = ScriptedLLM(["(a plan)"])
    planner(({"role": "user", "content": task},))      # the one planning call
    outcome = execute_plan(make_plan(task), tools)
    outcome["model_calls"] = outcome["model_calls"] + planner.calls
    return outcome


result = plan_and_execute(TASK, TOOLS)
print("plan and execute:", result)

broken = execute_plan([{"tool": "sum_prices", "input": None},
                       {"tool": "no_such_tool", "input": "$previous"},
                       {"tool": "apply_discount", "input": "$previous"}], TOOLS)
print("a plan with a bad step:", broken)
''',
    check=r'''
@check("12.2  a good plan runs to the right answer")
def _():
    need("execute_plan", "result")
    assert result["failed_at"] is None, \
        f"the plan should run cleanly; the record says it failed at {result['failed_at']}"
    assert "21.15" in str(result["result"]), (
        f"the result was {result['result']!r}, expected 21.15 — the discount step must "
        "receive the total from the previous step, not its literal '$previous' string."
    )
    assert result["steps_run"] == 2, f"two steps ran, record says {result['steps_run']}"
    assert result["model_calls"] == 1, (
        f"plan-and-execute made {result['model_calls']} model calls. It should be "
        "exactly one — the planning call. That is the saving over ReAct."
    )


@check("12.2  a broken plan fails honestly")
def _():
    need("execute_plan", "broken")
    assert broken["failed_at"] == 1, (
        f"failed_at is {broken['failed_at']}. The bad tool is step index 1, and the "
        "record must say where it stopped so a re-planner knows what to fix."
    )
    assert broken["result"] is None, (
        "a failed plan must not return a result. Returning step 1's output would make "
        "a half-finished run look finished — the worst possible outcome, because "
        "nothing downstream can tell."
    )
    assert broken["steps_run"] == 1, \
        f"one step completed before the failure, record says {broken['steps_run']}"

    raising = execute_plan([{"tool": "apply_discount", "input": "not a number"}], TOOLS)
    assert raising["failed_at"] == 0 and raising["result"] is None, (
        "a tool that RAISES should be caught and reported the same way, not allowed to "
        f"escape; got {raising}"
    )


@check("12.2  planning first is cheaper than adapting every step")
def _():
    need("execute_plan", "react_cost")
    plan_calls = plan_and_execute(TASK, TOOLS)["model_calls"]
    react_calls = 3            # think-act-observe for each tool, plus a finish
    assert plan_calls < react_calls, \
        f"plan-first used {plan_calls} model calls, ReAct would use about {react_calls}"
    print(f"     plan-first: {plan_calls} model call vs ReAct's ~{react_calls}, "
          f"and ReAct's context cost would be {react_cost(react_calls):,} tokens")
''',
    hint=[
        "Inside the loop:",
        "  if tool_name not in tools:",
        "      return {'result': None, 'steps_run': index,",
        "              'model_calls': model_calls, 'failed_at': index}",
        "",
        "  argument = previous if step['input'] == '$previous' else step['input']",
        "  try:",
        "      previous = tools[tool_name](argument)",
        "  except Exception:",
        "      return {'result': None, 'steps_run': index,",
        "              'model_calls': model_calls, 'failed_at': index}",
        "",
        "After the loop, return the success record with failed_at=None.",
    ],
    notice=r"""
**What to notice.** One model call instead of three or four, and no history re-sent — the
quadratic curve from Exercise 1 never gets started.

And note what the failure record does: `result` is `None` and `failed_at` is `1`. A
half-finished plan that returned step 1's output would look like a success to everything
downstream. **Never let a missing result look like a finished one** — that rule comes back
in Chapter 13.
""",
)

EX3 = Exercise(
    name="Reflection needs a signal from outside",
    section="12.3",
    concept=r"""
**Reflection** means: produce an answer, critique it, improve it. It works, and it has one
condition that is easy to miss.

> **Reflection is only as good as its evaluator.**

If the critic is the same model that wrote the answer, you are asking something to find
mistakes it just made — and models are poor judges of their own output. They are
*confident* about their own output, which is worse than being silent.

Reflection becomes genuinely useful when the signal comes from **outside the model**:

* tests that pass or fail
* a compiler or type checker
* a schema the output must validate against
* a ground-truth label

Those cannot be flattered. Below you will compare a self-grading loop with a test-driven
one on the same task — and see the self-grader stop early, satisfied, while wrong.
""",
    todo=r'''
# Three attempts at the same job, getting better. Attempt 1 is wrong.
ATTEMPTS = [
    "Total: 23.50",                       # forgot the discount
    "Discounted: 23.50",                  # relabelled it, still did not apply it
    "Discounted: 21.15",                  # correct
]

def self_grade(answer):
    """The model judging its own work. It is pleased with everything. Given to you."""
    return True, "Looks correct and well formatted."


def test_signal(answer):
    """An external check: does the answer contain the right number? Given to you."""
    expected = f"{sum(float(p) for p in BASKET) * 0.9:.2f}"
    if expected in answer:
        return True, "matches the expected total"
    return False, f"expected {expected} in the answer, got {answer!r}"


def reflect(attempts, evaluator, max_rounds=5):
    """Try each attempt in turn until the evaluator accepts one.

    Returns {"answer", "rounds", "accepted", "feedback"}.
    """
    # TODO 1: for each attempt (up to max_rounds), ask the evaluator.
    #         The evaluator returns (accepted, feedback).
    # TODO 2: as soon as one is accepted, return
    #         {"answer": attempt, "rounds": how many were tried,
    #          "accepted": True, "feedback": feedback}
    # TODO 3: if nothing was accepted, return the last attempt with accepted=False
    #         and the last feedback.
    raise NotImplementedError


by_self = reflect(ATTEMPTS, self_grade)
by_test = reflect(ATTEMPTS, test_signal)

print("self-grading :", by_self)
print("test-driven  :", by_test)
print("\nthe right answer is", apply_discount(sum_prices()))
''',
    solution=r'''
# Three attempts at the same job, getting better. Attempt 1 is wrong.
ATTEMPTS = [
    "Total: 23.50",                       # forgot the discount
    "Discounted: 23.50",                  # relabelled it, still did not apply it
    "Discounted: 21.15",                  # correct
]

def self_grade(answer):
    """The model judging its own work. It is pleased with everything. Given to you."""
    return True, "Looks correct and well formatted."


def test_signal(answer):
    """An external check: does the answer contain the right number? Given to you."""
    expected = f"{sum(float(p) for p in BASKET) * 0.9:.2f}"
    if expected in answer:
        return True, "matches the expected total"
    return False, f"expected {expected} in the answer, got {answer!r}"


def reflect(attempts, evaluator, max_rounds=5):
    """Try each attempt in turn until the evaluator accepts one.

    Returns {"answer", "rounds", "accepted", "feedback"}.
    """
    feedback = "no attempts were made"
    attempt = None

    for round_number, attempt in enumerate(attempts[:max_rounds], start=1):
        accepted, feedback = evaluator(attempt)
        if accepted:
            return {"answer": attempt, "rounds": round_number,
                    "accepted": True, "feedback": feedback}

    return {"answer": attempt, "rounds": min(len(attempts), max_rounds),
            "accepted": False, "feedback": feedback}


by_self = reflect(ATTEMPTS, self_grade)
by_test = reflect(ATTEMPTS, test_signal)

print("self-grading :", by_self)
print("test-driven  :", by_test)
print("\nthe right answer is", apply_discount(sum_prices()))
''',
    check=r'''
@check("12.3  the reflection loop stops at the first acceptance")
def _():
    need("reflect")
    r = reflect(ATTEMPTS, test_signal)
    assert r["accepted"] is True, f"the third attempt is correct, so it should be accepted"
    assert r["answer"] == "Discounted: 21.15", \
        f"the accepted answer should be the correct one, got {r['answer']!r}"
    assert r["rounds"] == 3, (
        f"it took 3 rounds to reach a correct answer, the record says {r['rounds']} — "
        "count the attempts actually evaluated"
    )

    never = reflect(["wrong", "also wrong"], test_signal)
    assert never["accepted"] is False, "neither attempt is correct, so accept nothing"
    assert never["feedback"] and "expected" in never["feedback"], (
        "return the evaluator's last feedback — that is what the next attempt would be "
        f"built from; yours says {never['feedback']!r}"
    )
    assert reflect(ATTEMPTS, test_signal, max_rounds=2)["accepted"] is False, \
        "max_rounds must be respected: with 2 rounds it never reaches the correct third"


@check("12.3  self-grading accepts a wrong answer; a test does not")
def _():
    need("reflect")
    self_result = reflect(ATTEMPTS, self_grade)
    test_result = reflect(ATTEMPTS, test_signal)

    assert self_result["rounds"] == 1 and self_result["accepted"], \
        "the self-grader is pleased immediately, so it should stop after one round"
    assert "23.50" in self_result["answer"], \
        "...and the answer it settled on is the one that forgot the discount"
    assert not test_signal(self_result["answer"])[0], (
        "(sanity check) the self-graded answer really is wrong — that is the point of "
        "the comparison"
    )
    assert test_result["accepted"] and "21.15" in test_result["answer"], \
        "the external signal keeps going until the answer is actually right"
    assert test_result["rounds"] > self_result["rounds"], (
        "the test-driven loop costs more rounds and gets the right answer. That trade "
        "is the whole argument for an external evaluator."
    )
    print(f"     self-grade stopped at round {self_result['rounds']} with "
          f"{self_result['answer']!r}; test-driven reached {test_result['answer']!r}")
''',
    hint=[
        "reflect:",
        "  feedback = 'no attempts were made'",
        "  attempt = None",
        "  for round_number, attempt in enumerate(attempts[:max_rounds], start=1):",
        "      accepted, feedback = evaluator(attempt)",
        "      if accepted:",
        "          return {'answer': attempt, 'rounds': round_number,",
        "                  'accepted': True, 'feedback': feedback}",
        "",
        "  return {'answer': attempt, 'rounds': min(len(attempts), max_rounds),",
        "          'accepted': False, 'feedback': feedback}",
    ],
    notice=r"""
**What to notice.** The self-grading loop finished **first** and was **wrong**. It reported
success, with a pleasant explanation. If you were watching a dashboard of "reflection
accepted" rates, it would look like your best-performing configuration.

Two cautions the chapter adds:

* **Check what the signal really tests.** `test_signal` here only looks for one number. A
  model could satisfy it by printing that number and nothing else — the test is a proxy,
  and proxies get gamed.
* **Measure on cases the evaluator has not seen.** An evaluator tuned on your examples will
  flatter you on your examples.

> And across all six architectures: **architecture decides how an agent works, context
> decides what it works from.** A better loop around a wrong task definition just reaches
> the wrong answer more efficiently.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=12,
    title="The Six Agentic Architectures",
    intro=r"""
There are only a handful of ways to arrange an agent, and they differ in one thing: **how
much is decided in advance.** ReAct decides nothing and adapts constantly. Plan-and-Execute
decides everything up front. Reflection looks back at its own work; tree search looks
sideways at alternatives.

This notebook measures the trade-offs rather than describing them: what a long ReAct loop
actually costs, what planning first saves, and why reflection without an outside signal is
worse than no reflection at all.
""",
    minutes=30,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "A ReAct loop's cost grows roughly with the **square** of its length, because the "
        "history is re-sent every step.",
        "Every loop needs a **budget**, a **stuck detector**, and a way to change "
        "approach — repetition is the common failure.",
        "**Plan first** when the shape of the work is known: one model call instead of "
        "many, and a plan you can inspect.",
        "**Never let a missing result look like a finished one** — report where a plan "
        "stopped.",
        "**Reflection is only as good as its evaluator.** A model grading itself accepts "
        "wrong answers confidently and early.",
        "Architecture decides *how* an agent works; **context decides what it works "
        "from**.",
    ],
    next_up="Chapter 13 adds more agents — and asks what the coordination costs.",
)
