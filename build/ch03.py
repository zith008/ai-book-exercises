"""Chapter 3 — Building Your First Agent from Scratch (sections 3.1, 3.2, 3.4)."""

from lite import Exercise, make_chapter

GIVEN = r'''
import re

# ---- A stand-in for the model, so everything here runs offline --------------
class ScriptedLLM:
    """Replays fixed replies. A real model would write these itself.

    This is the book's own trick. An agent only needs the model to be
    "a function from messages to text" — so a list of prepared replies works
    perfectly for learning the loop, and costs nothing.
    """
    def __init__(self, replies):
        self.replies = list(replies)

    def __call__(self, messages):
        if not self.replies:
            raise RuntimeError("the scripted model ran out of replies")
        return self.replies.pop(0)


# ---- Three tools. Each one is just a Python function. -----------------------
WEATHER = {"tokyo": "18°C, partly cloudy",
           "london": "12°C, light rain",
           "paris": "15°C, sunny"}

def get_weather(city):
    report = WEATHER.get(city.strip().lower())
    if report is None:
        return f"No weather data for '{city}' (simulated)"
    return f"Weather in {city.strip().title()}: {report} (simulated)"

def search(query):
    return f"Search results for '{query}': found 3 results (simulated)"

def calculate(expression):
    # Only digits and arithmetic get through. Never eval() model text directly.
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        raise ValueError("only plain arithmetic is allowed")
    return f"Result: {eval(expression)}"


# The registry: a tool's name -> (the function, the description the model sees)
TOOLS = {
    "weather":   (get_weather, "weather(city): current weather for a city"),
    "search":    (search,      "search(query): search the web"),
    "calculate": (calculate,   "calculate(expression): do arithmetic"),
}

SYSTEM_PROMPT = """You are an agent. Work in steps. Each reply must be:
Thought: <your reasoning>
Action: <tool>(<argument>)

Tools available:
""" + "\n".join(f"- {desc}" for _, desc in TOOLS.values()) + """
When you have the answer, use Action: finish(<your answer>)"""

print(f"{len(TOOLS)} tools ready:", ", ".join(TOOLS))
'''

EX1 = Exercise(
    name="The agent loop",
    section="3.1",
    concept=r"""
A chatbot answers. An **agent acts**, and then looks at what happened.

Picture a cook following a recipe. Read the step, do it, taste the result, decide the next
step. The tasting is the part that makes it an agent — it reacts to the real result rather
than guessing in advance.

The loop has exactly three moves, repeated:

1. **Think** — ask the model what to do next.
2. **Act** — run the tool it asked for.
3. **Observe** — put the result back into the history, so the model can see it.

The **history** is the agent's memory: a list of messages that grows every step. One
special action, `finish(...)`, ends the loop. And there is a step limit, because a loop
with no limit is a bill with no limit.
""",
    todo=r'''
def parse_action(reply):
    """Find 'Action: name(args)' in the model's reply. Given to you."""
    match = re.search(r"Action:\s*(\w+)\((.*)\)", reply, re.DOTALL)
    return (match.group(1), match.group(2)) if match else (None, None)


def run_agent(llm, goal, max_steps=5):
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    for step in range(1, max_steps + 1):
        # TODO 1 (think):  reply = llm(history), then append it to history as
        #                  {"role": "assistant", "content": reply}
        # TODO 2:          name, args = parse_action(reply)
        # TODO 3:          if the tool is "finish", return (args, history) — we are done
        # TODO 4 (act):    look the tool up in TOOLS and call it with args.
        #                  TOOLS[name] is a (function, description) pair, so the
        #                  function is TOOLS[name][0].
        # TODO 5 (observe): append {"role": "user", "content": f"Observation: {result}"}
        raise NotImplementedError

    return "Stopped: step limit reached.", history


llm = ScriptedLLM([
    "Thought: I need the weather in Tokyo first.\nAction: weather(Tokyo)",
    "Thought: 18°C and partly cloudy, so rain is unlikely.\n"
    "Action: finish(It is 18°C and partly cloudy in Tokyo. An umbrella is optional.)",
])

answer, history = run_agent(llm, "What's the weather in Tokyo? Should I bring an umbrella?")
print("Final answer:", answer)
print(f"\nThe history grew to {len(history)} messages:")
for m in history:
    print(f"  {m['role']:>9}: {m['content'][:60]!r}")
''',
    solution=r'''
def parse_action(reply):
    """Find 'Action: name(args)' in the model's reply. Given to you."""
    match = re.search(r"Action:\s*(\w+)\((.*)\)", reply, re.DOTALL)
    return (match.group(1), match.group(2)) if match else (None, None)


def run_agent(llm, goal, max_steps=5):
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    for step in range(1, max_steps + 1):
        reply = llm(history)                                      # 1. think
        history.append({"role": "assistant", "content": reply})

        name, args = parse_action(reply)
        if name == "finish":
            return args, history

        tool_function = TOOLS[name][0]                             # 2. act
        result = tool_function(args)

        history.append({"role": "user",                            # 3. observe
                        "content": f"Observation: {result}"})

    return "Stopped: step limit reached.", history


llm = ScriptedLLM([
    "Thought: I need the weather in Tokyo first.\nAction: weather(Tokyo)",
    "Thought: 18°C and partly cloudy, so rain is unlikely.\n"
    "Action: finish(It is 18°C and partly cloudy in Tokyo. An umbrella is optional.)",
])

answer, history = run_agent(llm, "What's the weather in Tokyo? Should I bring an umbrella?")
print("Final answer:", answer)
print(f"\nThe history grew to {len(history)} messages:")
for m in history:
    print(f"  {m['role']:>9}: {m['content'][:60]!r}")
''',
    check=r'''
@check("3.1  the loop thinks, acts, observes and finishes")
def _():
    need("run_agent", "answer", "history")
    assert "18°C" in answer or "umbrella" in answer.lower(), (
        f"the final answer was {answer!r}. It should be the text inside finish(...) — "
        "return args when the tool name is 'finish'."
    )
    assert len(history) == 5, (
        f"the history holds {len(history)} messages, expected 5: system, user, "
        "assistant, observation, assistant. Append the model's reply AND the "
        "observation every step."
    )
    assert history[3]["content"].startswith("Observation:"), \
        "the tool result must go back into the history, prefixed 'Observation: '"
    assert "partly cloudy" in history[3]["content"], \
        "the observation should contain what the weather tool actually returned"


@check("3.1  the step limit stops a runaway agent")
def _():
    need("run_agent")
    spinner = ScriptedLLM(["Thought: checking.\nAction: weather(Tokyo)"] * 20)
    out, hist = run_agent(spinner, "loop forever", max_steps=3)
    assert "limit" in out.lower(), (
        f"after 3 steps with no finish, the agent returned {out!r}. It should stop and "
        "say so — an agent without a step limit can run up a real bill."
    )
    print("     stopped cleanly after 3 steps")
''',
    hint=[
        "Inside the for loop, in this order:",
        "  reply = llm(history)",
        "  history.append({'role': 'assistant', 'content': reply})",
        "  name, args = parse_action(reply)",
        "  if name == 'finish':",
        "      return args, history",
        "  result = TOOLS[name][0](args)",
        "  history.append({'role': 'user', 'content': f'Observation: {result}'})",
        "",
        "TOOLS[name] is a PAIR, so TOOLS[name][0] is the function to call.",
    ],
    notice=r"""
**What to notice.** That is the whole of an agent: about ten lines. Everything difficult
about agents is in the details around this loop — the prompt, the parsing, the tool design,
and what happens when something goes wrong. Which is the next two exercises.
""",
)

EX2 = Exercise(
    name="When the model asks for nonsense",
    section="3.2",
    concept=r"""
Your loop works — as long as the model behaves. It will not.

The model writes **text**, and your code has to turn that text into a function call.
Sometimes it asks for a tool that does not exist. Sometimes it writes no `Action:` line at
all. Sometimes it passes `"banana"` to a calculator.

> **Treat the model's output as untrusted input.** Not because the model is malicious, but
> because it is a text generator, and text generators produce text.

Right now your loop crashes on all three cases — `TOOLS["fly"]` raises `KeyError`, and the
whole agent dies.

The fix is not to crash. It is to hand the problem **back to the model as an observation**,
so it can try something else. That one habit is the difference between a demo and something
you would let a user near.
""",
    todo=r'''
def act(name, args):
    """Run one tool safely. Always returns a string for the Observation line."""
    # TODO 1: if name is None (the model wrote no Action: line at all), return a string
    #         reminding it of the format, e.g.
    #         "Error: no Action: line found. Reply with Action: tool(argument)."
    # TODO 2: if name is not in TOOLS, return an error string that LISTS the real tool
    #         names — a useful error tells the model how to fix itself:
    #         f"Error: unknown tool '{name}'. Available: weather, search, calculate"
    # TODO 3: otherwise call the tool inside try/except. If it raises, return
    #         f"Error running {name}: {err}" instead of letting it crash.
    raise NotImplementedError


print(act("weather", "London"))
print(act("fly", "to the moon"))
print(act(None, ""))
print(act("calculate", "banana"))
''',
    solution=r'''
def act(name, args):
    """Run one tool safely. Always returns a string for the Observation line."""
    if name is None:
        return "Error: no Action: line found. Reply with Action: tool(argument)."

    if name not in TOOLS:
        return (f"Error: unknown tool '{name}'. "
                f"Available: {', '.join(TOOLS)}")

    tool_function = TOOLS[name][0]
    try:
        return tool_function(args)
    except Exception as err:
        return f"Error running {name}: {err}"


print(act("weather", "London"))
print(act("fly", "to the moon"))
print(act(None, ""))
print(act("calculate", "banana"))
''',
    check=r'''
@check("3.2  bad tool calls become observations, not crashes")
def _():
    need("act")
    good = act("weather", "London")
    assert "12°C" in good, f"a valid call should still work; got {good!r}"

    unknown = act("fly", "away")
    assert isinstance(unknown, str), "act() must always return a string, never raise"
    assert "fly" in unknown and "weather" in unknown, (
        f"got {unknown!r}. An unknown-tool error should name the bad tool AND list the "
        "real ones, so the model can correct itself on the next step."
    )

    nothing = act(None, "")
    assert isinstance(nothing, str) and "Action" in nothing, \
        "when there is no Action: line, remind the model of the format"

    broken = act("calculate", "banana")
    assert isinstance(broken, str) and "rror" in broken, (
        "calculate('banana') raises ValueError. Catch it and return the message — "
        "that is how the model learns to send a real expression."
    )
    print("     all four cases handled without a crash")


@check("3.2  an agent using act() recovers from its own mistake")
def _():
    need("act", "run_agent")

    def robust_agent(llm, goal, max_steps=5):
        hist = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": goal}]
        for _ in range(max_steps):
            reply = llm(hist)
            hist.append({"role": "assistant", "content": reply})
            name, args = parse_action(reply)
            if name == "finish":
                return args, hist
            hist.append({"role": "user", "content": f"Observation: {act(name, args)}"})
        return "Stopped: step limit reached.", hist

    confused = ScriptedLLM([
        "Thought: I will check the sky.\nAction: lookup_sky(Tokyo)",   # no such tool
        "Thought: Ah, I should use weather.\nAction: weather(Tokyo)",
        "Thought: Got it.\nAction: finish(18°C and partly cloudy.)",
    ])
    out, hist = robust_agent(confused, "weather in Tokyo?")
    assert "18°C" in out, f"the agent should have recovered and finished; got {out!r}"
    assert any("unknown tool" in m["content"] for m in hist), \
        "the error should appear in the history — that is how the model sees it"
    print("     the agent got it wrong, was told why, and fixed itself")
''',
    hint=[
        "Three cases, in this order:",
        "  if name is None: return 'Error: no Action: line found. ...'",
        "  if name not in TOOLS: return f\"Error: unknown tool '{name}'. \"",
        "                               f\"Available: {', '.join(TOOLS)}\"",
        "  try: return TOOLS[name][0](args)",
        "  except Exception as err: return f'Error running {name}: {err}'",
        "",
        "The rule: act() never raises. Every problem comes back as a string the model",
        "can read on its next turn.",
    ],
    notice=r"""
**What to notice.** The agent in the second check made a mistake, was told exactly what was
wrong, and corrected itself — with no new code in the loop. You got that by writing one
good error message.

This is worth remembering: for an agent, **an error message is a prompt**. "Invalid input"
teaches the model nothing. "Unknown tool 'fly'. Available: weather, search, calculate"
teaches it everything it needs.
""",
)

EX3 = Exercise(
    name="A run that reports what happened",
    section="3.4",
    concept=r"""
Your agent now survives mistakes. The last question is the one your future self will ask:
**what actually happened on that run?**

An agent can end in several different ways, and they are not equally good:

| Ending | Meaning |
|---|---|
| `finished` | it answered the question. Good. |
| `step_limit` | it ran out of steps. Probably stuck in a circle. |
| `no_action` | the model stopped following the format. |

If every run just returns a string, all three look identical from the outside, and you
cannot tell a healthy agent from a broken one. So return a small **record** instead: the
answer, how it ended, and how many steps it took.

> Design for failure: cap the steps, report errors back to the model, and make every
> stopping point visible.
""",
    todo=r'''
def run_agent_v2(llm, goal, max_steps=5):
    """Like run_agent, but returns a record of the whole run."""
    history = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": goal}]
    tool_calls = 0

    for step in range(1, max_steps + 1):
        reply = llm(history)
        history.append({"role": "assistant", "content": reply})
        name, args = parse_action(reply)

        if name == "finish":
            # TODO 1: return a dict:
            #   {"answer": args, "ending": "finished", "steps": step,
            #    "tool_calls": tool_calls}
            raise NotImplementedError

        # TODO 2: count this tool call, then run it with act(name, args) and append the
        #         observation to history exactly as you did in Exercise 1.
        raise NotImplementedError

    # TODO 3: the loop finished without anyone calling finish(). Return the same shape
    #         of dict, with answer=None and ending="step_limit".
    raise NotImplementedError


record = run_agent_v2(ScriptedLLM([
    "Thought: Checking.\nAction: weather(Paris)",
    "Thought: Done.\nAction: finish(Paris is 15°C and sunny.)",
]), "weather in Paris?")
print(record)

stuck = run_agent_v2(ScriptedLLM(["Thought: again.\nAction: weather(Paris)"] * 10),
                     "go round in circles", max_steps=4)
print(stuck)
''',
    solution=r'''
def run_agent_v2(llm, goal, max_steps=5):
    """Like run_agent, but returns a record of the whole run."""
    history = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": goal}]
    tool_calls = 0

    for step in range(1, max_steps + 1):
        reply = llm(history)
        history.append({"role": "assistant", "content": reply})
        name, args = parse_action(reply)

        if name == "finish":
            return {"answer": args, "ending": "finished",
                    "steps": step, "tool_calls": tool_calls}

        tool_calls += 1
        observation = act(name, args)
        history.append({"role": "user", "content": f"Observation: {observation}"})

    return {"answer": None, "ending": "step_limit",
            "steps": max_steps, "tool_calls": tool_calls}


record = run_agent_v2(ScriptedLLM([
    "Thought: Checking.\nAction: weather(Paris)",
    "Thought: Done.\nAction: finish(Paris is 15°C and sunny.)",
]), "weather in Paris?")
print(record)

stuck = run_agent_v2(ScriptedLLM(["Thought: again.\nAction: weather(Paris)"] * 10),
                     "go round in circles", max_steps=4)
print(stuck)
''',
    check=r'''
@check("3.4  a successful run is recorded correctly")
def _():
    need("run_agent_v2", "record")
    for field in ("answer", "ending", "steps", "tool_calls"):
        assert field in record, f"the record is missing {field!r}"
    assert record["ending"] == "finished", \
        f"ending was {record['ending']!r}, expected 'finished'"
    assert "15°C" in (record["answer"] or ""), \
        f"answer was {record['answer']!r} — it should be the text from finish(...)"
    assert record["steps"] == 2, f"that run took 2 steps, the record says {record['steps']}"
    assert record["tool_calls"] == 1, \
        f"one tool was called (weather); the record says {record['tool_calls']}"


@check("3.4  a stuck run is honest about it")
def _():
    need("run_agent_v2", "stuck")
    assert stuck["ending"] == "step_limit", (
        f"ending was {stuck['ending']!r}. A run that never called finish() must say so — "
        "otherwise a broken agent looks the same as a working one in your logs."
    )
    assert stuck["answer"] is None, "there is no answer, so do not invent one"
    assert stuck["tool_calls"] == 4, \
        f"4 steps means 4 tool calls; the record says {stuck['tool_calls']}"
    print(f"     finished: {record['steps']} steps / stuck: {stuck['ending']} "
          f"after {stuck['tool_calls']} calls")
''',
    hint=[
        "On finish:",
        "  return {'answer': args, 'ending': 'finished',",
        "          'steps': step, 'tool_calls': tool_calls}",
        "",
        "Otherwise, in the loop body:",
        "  tool_calls += 1",
        "  observation = act(name, args)",
        "  history.append({'role': 'user', 'content': f'Observation: {observation}'})",
        "",
        "After the loop (nobody called finish):",
        "  return {'answer': None, 'ending': 'step_limit',",
        "          'steps': max_steps, 'tool_calls': tool_calls}",
    ],
    notice=r"""
**Why this matters more than it looks.** Once every run returns a record, you can count
endings across a thousand runs: *"8% of our agents hit the step limit"* is a number you can
act on. *"it sometimes doesn't work"* is not.

Chapter 4 takes this loop apart properly — timing, observability, and running several tools
at once.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=3,
    title="Building Your First Agent from Scratch",
    intro=r"""
A chatbot talks. An **agent does things** — it picks a tool, runs it, looks at the result,
and decides what to do next.

In this notebook you write that loop yourself. It is about ten lines of Python, and by the
end you will have one that survives the model asking for a tool that does not exist.
""",
    minutes=30,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "An agent is a **loop**: think, act, observe, repeat. The history carries what has "
        "happened from one step to the next.",
        "A **tool** is just a Python function, plus a description the model can read.",
        "The model's reply is **untrusted text**. Parse it, and never let a bad tool call "
        "crash the loop.",
        "An **error message is a prompt**: say what went wrong and what the valid options "
        "are, and the model fixes itself.",
        "Always cap the steps, and **return a record** of how each run ended — so you can "
        "count the failures instead of guessing at them.",
    ],
    next_up="Chapter 4 opens up the loop itself: observability, timing, and calling "
            "several tools at once.",
)
