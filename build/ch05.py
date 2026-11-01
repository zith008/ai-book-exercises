"""Chapter 5 — Tools: definition, path safety, the function-calling protocol (5.1, 5.3, 5.5)."""

from lite import Exercise, make_chapter

GIVEN = r'''
import json, tempfile
from pathlib import Path

# ---- Three ordinary Python functions, which we will turn into tools ---------
def get_weather(city, units="celsius"):
    temp = {"celsius": "18°C", "fahrenheit": "64°F"}[units]
    return f"Weather in {city}: {temp}, partly cloudy (simulated)"

def convert_currency(amount, from_currency, to_currency):
    rates = {("USD", "GBP"): 0.79, ("GBP", "USD"): 1.27, ("USD", "EUR"): 0.92}
    rate = rates.get((from_currency, to_currency))
    if rate is None:
        return f"No rate for {from_currency}->{to_currency}"
    return f"{amount} {from_currency} = {round(amount * rate, 2)} {to_currency}"


# ---- A workspace for the file exercise, created fresh ------------------------
WORKSPACE = Path(tempfile.mkdtemp(prefix="agent_data_")).resolve()
(WORKSPACE / "notes.txt").write_text("Remember to water the plants.", encoding="utf-8")
(WORKSPACE / "sub").mkdir(exist_ok=True)
(WORKSPACE / "sub" / "deep.txt").write_text("a file in a subfolder", encoding="utf-8")

# A file OUTSIDE the workspace, standing in for something private.
SECRET = WORKSPACE.parent / "private_keys.txt"
SECRET.write_text("sk-do-not-leak-me", encoding="utf-8")

print("workspace:", WORKSPACE)
print("files:", [p.name for p in WORKSPACE.iterdir()])


# ---- A stand-in client that speaks the tool-calling protocol ----------------
class ToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = type("fn", (), {"name": name,
                                        "arguments": json.dumps(arguments)})()

class Reply:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls or []
    def to_message(self):
        msg = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name,
                                               "arguments": c.function.arguments}}
                                 for c in self.tool_calls]
        return msg

class ScriptedToolClient:
    """Replays prepared replies, in the shape a real tool-calling API returns."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages_seen = []
    def create(self, messages=(), **kwargs):
        self.messages_seen.append(list(messages))
        return self.replies.pop(0)

print("stand-in tool-calling client ready")
'''

EX1 = Exercise(
    name="Describing a tool to the model",
    section="5.1",
    concept=r"""
A tool is a Python function the model can ask for. But the model cannot read your code —
it only sees a **description**. If the description is vague, the model guesses, and guesses
arrive at your function as wrong arguments.

So every tool is described in a fixed shape, which the API calls a schema:

```python
{"type": "function",
 "function": {"name": "get_weather",
              "description": "Get the current weather for a city",
              "parameters": {"type": "object",
                             "properties": {"city": {"type": "string",
                                                     "description": "City name"}},
                             "required": ["city"]}}}
```

Think of it as the label on a jar: the name, what is inside, and what you must supply.

**And then you check what comes back.** The model sends arguments as text; your code has to
confirm they match the description before calling anything. One definition does three jobs:
it tells the model, it validates the arguments, and it runs the function.
""",
    todo=r'''
class Tool:
    def __init__(self, name, description, function, parameters):
        self.name = name
        self.description = description
        self.function = function
        # parameters: {"city": {"type": "string", "description": "...",
        #                       "required": True}, ...}
        self.parameters = parameters

    def schema(self):
        """The description the model sees."""
        properties = {
            arg_name: {"type": spec["type"], "description": spec["description"]}
            for arg_name, spec in self.parameters.items()
        }
        required = [n for n, spec in self.parameters.items() if spec.get("required")]
        # TODO 1: return the schema dict shown in the text above, using `properties`
        #         and `required`. The nesting matters: name/description/parameters all
        #         live inside a "function" key.
        raise NotImplementedError

    def call(self, arguments):
        """Check the model's arguments, then run the function."""
        # TODO 2: every required parameter must be present. If any is missing, return
        #         f"Error: missing required argument(s): {...}" — do not raise.
        # TODO 3: reject any argument that is not in self.parameters at all
        #         (a model that invents an argument should be told, not silently obeyed).
        # TODO 4: otherwise return self.function(**arguments)
        raise NotImplementedError


weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a city",
    function=get_weather,
    parameters={
        "city":  {"type": "string", "description": "City name", "required": True},
        "units": {"type": "string", "description": "celsius or fahrenheit",
                  "required": False},
    },
)

print(json.dumps(weather_tool.schema(), indent=2)[:320], "...\n")
print(weather_tool.call({"city": "Tokyo"}))
print(weather_tool.call({}))
print(weather_tool.call({"city": "Tokyo", "colour": "blue"}))
''',
    solution=r'''
class Tool:
    def __init__(self, name, description, function, parameters):
        self.name = name
        self.description = description
        self.function = function
        # parameters: {"city": {"type": "string", "description": "...",
        #                       "required": True}, ...}
        self.parameters = parameters

    def schema(self):
        """The description the model sees."""
        properties = {
            arg_name: {"type": spec["type"], "description": spec["description"]}
            for arg_name, spec in self.parameters.items()
        }
        required = [n for n, spec in self.parameters.items() if spec.get("required")]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def call(self, arguments):
        """Check the model's arguments, then run the function."""
        required = {n for n, spec in self.parameters.items() if spec.get("required")}
        missing = required - set(arguments)
        if missing:
            return f"Error: missing required argument(s): {', '.join(sorted(missing))}"

        unexpected = set(arguments) - set(self.parameters)
        if unexpected:
            return (f"Error: unexpected argument(s): {', '.join(sorted(unexpected))}. "
                    f"Expected: {', '.join(self.parameters)}")

        return self.function(**arguments)


weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a city",
    function=get_weather,
    parameters={
        "city":  {"type": "string", "description": "City name", "required": True},
        "units": {"type": "string", "description": "celsius or fahrenheit",
                  "required": False},
    },
)

print(json.dumps(weather_tool.schema(), indent=2)[:320], "...\n")
print(weather_tool.call({"city": "Tokyo"}))
print(weather_tool.call({}))
print(weather_tool.call({"city": "Tokyo", "colour": "blue"}))
''',
    check=r'''
@check("5.1  the schema is the shape the API expects")
def _():
    need("weather_tool")
    s = weather_tool.schema()
    assert s.get("type") == "function", "the top level needs \"type\": \"function\""
    fn = s.get("function")
    assert isinstance(fn, dict), 'the details go inside a "function" key'
    assert fn["name"] == "get_weather" and fn["description"], \
        "name and description belong inside the function dict"
    params = fn["parameters"]
    assert params["type"] == "object", 'parameters needs "type": "object"'
    assert set(params["properties"]) == {"city", "units"}, \
        f"properties lists {sorted(params['properties'])}, expected city and units"
    assert params["required"] == ["city"], (
        f"required is {params['required']}. Only city is required — if units were "
        "required too, the model would have to guess a value it was not given."
    )
    assert params["properties"]["city"]["description"], \
        "keep each parameter's description: it is how the model knows what to send"


@check("5.1  bad arguments come back as readable errors")
def _():
    need("weather_tool")
    ok = weather_tool.call({"city": "Tokyo"})
    assert "18°C" in ok, f"a valid call should work; got {ok!r}"
    assert "64°F" in weather_tool.call({"city": "Tokyo", "units": "fahrenheit"}), \
        "optional arguments should be passed through when present"

    missing = weather_tool.call({})
    assert isinstance(missing, str) and "city" in missing and "issing" in missing, (
        f"got {missing!r}. A missing required argument should return an error naming "
        "the argument, not raise TypeError."
    )

    extra = weather_tool.call({"city": "Tokyo", "colour": "blue"})
    assert isinstance(extra, str) and "colour" in extra, (
        f"got {extra!r}. An argument the tool never declared should be refused — "
        "otherwise get_weather(colour='blue') raises deep inside your code."
    )
    print("     valid, missing and invented arguments all handled")
''',
    hint=[
        "schema() returns:",
        "  {'type': 'function',",
        "   'function': {'name': self.name, 'description': self.description,",
        "                'parameters': {'type': 'object',",
        "                               'properties': properties,",
        "                               'required': required}}}",
        "",
        "call() checks two sets:",
        "  required = {n for n, s in self.parameters.items() if s.get('required')}",
        "  missing = required - set(arguments)",
        "  unexpected = set(arguments) - set(self.parameters)",
        "then: return self.function(**arguments)",
    ],
    notice=r"""
**What to notice.** The same `parameters` dictionary did all three jobs: it generated the
model's description, it validated the incoming arguments, and it decided what reaches your
function. Write it once, in one place, and the three can never disagree.
""",
)

EX2 = Exercise(
    name="A tool that cannot escape its folder",
    section="5.3",
    concept=r"""
Now a tool that touches the real world: reading files.

Your agent has a workspace folder it is allowed to use. The model asks for a filename. And
one day the model — prompted by something it read on a web page, or just being a text
generator — asks for:

```
../private_keys.txt
```

If you join that onto your workspace path and open it, you have just handed over a file
outside the workspace. This is called **path traversal**, and it is one of the oldest bugs
there is.

The fix is two lines, and the discipline is to put them in **one place every file tool goes
through**:

1. `(base / path).resolve()` — `resolve()` works out the real, final path, following `..`
   and any symbolic links.
2. Check that the result is actually **inside** the workspace. `Path.is_relative_to(base)`
   answers exactly that.

Checking before you touch the disk is the point. A check performed afterwards is not a check.
""",
    todo=r'''
class FileTools:
    def __init__(self, base_path, max_read_chars=200):
        self.base_path = Path(base_path).resolve()
        self.max_read_chars = max_read_chars

    def _resolve(self, path):
        """The ONE place a name becomes a file. Raises PermissionError if it escapes."""
        # TODO 1: join path onto self.base_path and call .resolve() on the result
        # TODO 2: if the resolved path is not inside self.base_path, raise
        #         PermissionError(f"Access denied: {path} is outside the workspace")
        # TODO 3: otherwise return it
        raise NotImplementedError

    def read_file(self, path):
        """Read a file, returning errors as text the model can act on."""
        try:
            text = self._resolve(path).read_text(encoding="utf-8")
        except PermissionError as err:
            return str(err)
        except FileNotFoundError:
            return f"File not found: {path}"
        except (IsADirectoryError, NotADirectoryError):
            return f"Wrong kind of path: {path}"

        if len(text) > self.max_read_chars:
            hidden = len(text) - self.max_read_chars
            return text[:self.max_read_chars] + f"\n... [truncated {hidden} characters]"
        return text


files = FileTools(WORKSPACE)
print("inside:   ", files.read_file("notes.txt"))
print("subfolder:", files.read_file("sub/deep.txt"))
print("escape:   ", files.read_file("../private_keys.txt"))
print("missing:  ", files.read_file("nope.txt"))
''',
    solution=r'''
class FileTools:
    def __init__(self, base_path, max_read_chars=200):
        self.base_path = Path(base_path).resolve()
        self.max_read_chars = max_read_chars

    def _resolve(self, path):
        """The ONE place a name becomes a file. Raises PermissionError if it escapes."""
        full_path = (self.base_path / path).resolve()   # follows .. and symlinks
        if not full_path.is_relative_to(self.base_path):
            raise PermissionError(f"Access denied: {path} is outside the workspace")
        return full_path

    def read_file(self, path):
        """Read a file, returning errors as text the model can act on."""
        try:
            text = self._resolve(path).read_text(encoding="utf-8")
        except PermissionError as err:
            return str(err)
        except FileNotFoundError:
            return f"File not found: {path}"
        except (IsADirectoryError, NotADirectoryError):
            return f"Wrong kind of path: {path}"

        if len(text) > self.max_read_chars:
            hidden = len(text) - self.max_read_chars
            return text[:self.max_read_chars] + f"\n... [truncated {hidden} characters]"
        return text


files = FileTools(WORKSPACE)
print("inside:   ", files.read_file("notes.txt"))
print("subfolder:", files.read_file("sub/deep.txt"))
print("escape:   ", files.read_file("../private_keys.txt"))
print("missing:  ", files.read_file("nope.txt"))
''',
    check=r'''
@check("5.3  files inside the workspace can be read")
def _():
    need("files")
    assert "water the plants" in files.read_file("notes.txt"), \
        "notes.txt is inside the workspace and should read normally"
    assert "subfolder" in files.read_file("sub/deep.txt"), \
        "a file in a subfolder is still inside the workspace"
    assert "not found" in files.read_file("nope.txt").lower(), \
        "a missing file should report that, not crash"


@check("5.3  the workspace cannot be escaped")
def _():
    need("files", "FileTools")
    for attempt in ["../private_keys.txt",
                    "sub/../../private_keys.txt",
                    str(SECRET)]:
        result = files.read_file(attempt)
        assert "sk-do-not-leak-me" not in result, (
            f"reading {attempt!r} returned the secret file. _resolve must call "
            ".resolve() FIRST (so '..' is worked out) and then check "
            "is_relative_to(self.base_path)."
        )
        assert "denied" in result.lower(), (
            f"reading {attempt!r} gave {result[:60]!r}. It should be a clear "
            "'Access denied' message."
        )

    with_absolute = FileTools(WORKSPACE)
    assert isinstance(with_absolute._resolve("sub/deep.txt"), Path), \
        "_resolve should return the resolved Path for a legitimate file"
    print("     three traversal attempts refused, legitimate reads still work")
''',
    hint=[
        "_resolve is two lines and a return:",
        "  full_path = (self.base_path / path).resolve()",
        "  if not full_path.is_relative_to(self.base_path):",
        "      raise PermissionError(f'Access denied: {path} is outside the workspace')",
        "  return full_path",
        "",
        "Order matters. Checking the string for '..' BEFORE resolving is a classic",
        "mistake: symlinks and odd spellings get past a string check, and resolve()",
        "is what turns any of them into the real path.",
    ],
    notice=r"""
**What to notice.** Every file operation — read, write, list, delete — goes through
`_resolve`. One checkpoint, used everywhere, is far safer than a check copied into six
methods, because the seventh method will forget.

> Never let a tool touch a path you have not checked. Resolve first, confirm it is inside
> the workspace, and make that the only door in.
""",
)

EX3 = Exercise(
    name="The tool-calling protocol",
    section="5.5",
    concept=r"""
Until now you parsed `Action: weather(Tokyo)` out of the model's text. Real APIs do this
properly: the model returns **structured tool calls**, each with an `id`, and you send back
one result per call.

It is a strict little protocol, and it is easiest to see as a conversation:

1. You send the messages **and** your tool schemas.
2. The model replies — either with `tool_calls`, or with a final answer.
3. For **every** call in `tool_calls`, you append a message with `role: "tool"` and the
   matching `tool_call_id`.
4. Repeat. When a reply has no `tool_calls`, that is the final answer and the loop ends.

**Every call must get an answer.** Miss one and the API rejects your next request — the
conversation is now malformed, and the error you get back will not obviously say so.

Strict about answering, generous with error messages: a tool that fails should still return
a `role: "tool"` message, containing the error as text.
""",
    todo=r'''
TOOL_OBJECTS = {
    "get_weather": weather_tool,
    "convert_currency": Tool(
        name="convert_currency",
        description="Convert an amount between two currencies",
        function=convert_currency,
        parameters={
            "amount": {"type": "number", "description": "Amount", "required": True},
            "from_currency": {"type": "string", "description": "e.g. USD",
                              "required": True},
            "to_currency": {"type": "string", "description": "e.g. GBP",
                            "required": True},
        }),
}


def run_tool_call(call):
    """Run one structured tool call and return its result as a string. Given to you."""
    name = call.function.name
    arguments = json.loads(call.function.arguments)
    if name not in TOOL_OBJECTS:
        return f"Error: unknown tool '{name}'"
    return str(TOOL_OBJECTS[name].call(arguments))


def run_agent(client, question, max_steps=5):
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}]

    for _ in range(max_steps):
        reply = client.create(
            messages=messages,
            tools=[t.schema() for t in TOOL_OBJECTS.values()],
        )
        messages.append(reply.to_message())

        # TODO 1: if the reply has no tool_calls, it is the final answer —
        #         return reply.content
        # TODO 2: otherwise loop over EVERY call in reply.tool_calls and append
        #         {"role": "tool", "tool_call_id": call.id,
        #          "content": run_tool_call(call)}
        #         Miss one and the next API request is invalid.
        raise NotImplementedError

    return "Stopped: step limit reached."


client = ScriptedToolClient([
    # one reply asking for TWO tools at once
    Reply(tool_calls=[ToolCall("call_1", "get_weather", {"city": "Tokyo"}),
                      ToolCall("call_2", "convert_currency",
                               {"amount": 100, "from_currency": "USD",
                                "to_currency": "GBP"})]),
    Reply(content="Tokyo is 18°C, and 100 USD is about 79 GBP."),
])

print(run_agent(client, "Weather in Tokyo, and what is 100 USD in GBP?"))
''',
    solution=r'''
TOOL_OBJECTS = {
    "get_weather": weather_tool,
    "convert_currency": Tool(
        name="convert_currency",
        description="Convert an amount between two currencies",
        function=convert_currency,
        parameters={
            "amount": {"type": "number", "description": "Amount", "required": True},
            "from_currency": {"type": "string", "description": "e.g. USD",
                              "required": True},
            "to_currency": {"type": "string", "description": "e.g. GBP",
                            "required": True},
        }),
}


def run_tool_call(call):
    """Run one structured tool call and return its result as a string. Given to you."""
    name = call.function.name
    arguments = json.loads(call.function.arguments)
    if name not in TOOL_OBJECTS:
        return f"Error: unknown tool '{name}'"
    return str(TOOL_OBJECTS[name].call(arguments))


def run_agent(client, question, max_steps=5):
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}]

    for _ in range(max_steps):
        reply = client.create(
            messages=messages,
            tools=[t.schema() for t in TOOL_OBJECTS.values()],
        )
        messages.append(reply.to_message())

        if not reply.tool_calls:
            return reply.content

        for call in reply.tool_calls:
            messages.append({"role": "tool",
                             "tool_call_id": call.id,
                             "content": run_tool_call(call)})

    return "Stopped: step limit reached."


client = ScriptedToolClient([
    # one reply asking for TWO tools at once
    Reply(tool_calls=[ToolCall("call_1", "get_weather", {"city": "Tokyo"}),
                      ToolCall("call_2", "convert_currency",
                               {"amount": 100, "from_currency": "USD",
                                "to_currency": "GBP"})]),
    Reply(content="Tokyo is 18°C, and 100 USD is about 79 GBP."),
])

print(run_agent(client, "Weather in Tokyo, and what is 100 USD in GBP?"))
''',
    check=r'''
@check("5.5  every tool call gets exactly one answer")
def _():
    need("run_agent", "ScriptedToolClient")
    c = ScriptedToolClient([
        Reply(tool_calls=[ToolCall("call_1", "get_weather", {"city": "Tokyo"}),
                          ToolCall("call_2", "convert_currency",
                                   {"amount": 100, "from_currency": "USD",
                                    "to_currency": "GBP"})]),
        Reply(content="Final answer."),
    ])
    answer = run_agent(c, "two things at once")
    assert answer == "Final answer.", (
        f"the loop returned {answer!r}. When a reply has no tool_calls, return "
        "reply.content — that is the model's final answer."
    )

    sent = c.messages_seen[-1]          # what the second request contained
    tool_messages = [m for m in sent if m.get("role") == "tool"]
    assert len(tool_messages) == 2, (
        f"the second request carried {len(tool_messages)} tool messages, expected 2. "
        "The model asked for two tools in one reply, and EVERY call needs its own "
        "answer or the API rejects the request."
    )
    assert {m["tool_call_id"] for m in tool_messages} == {"call_1", "call_2"}, (
        "each tool message must carry the tool_call_id of the call it answers — "
        f"yours has {[m.get('tool_call_id') for m in tool_messages]}"
    )
    assert "18°C" in tool_messages[0]["content"], \
        "the first tool message should hold the weather tool's result"
    assert "79" in tool_messages[1]["content"], \
        "the second should hold the currency conversion"
    print("     2 calls in one reply, 2 answers, ids matched")


@check("5.5  a failing tool still gets an answer")
def _():
    need("run_agent")
    c = ScriptedToolClient([
        Reply(tool_calls=[ToolCall("call_9", "get_weather", {})]),   # no city
        Reply(content="I need a city name."),
    ])
    run_agent(c, "weather please")
    tool_messages = [m for m in c.messages_seen[-1] if m.get("role") == "tool"]
    assert len(tool_messages) == 1, \
        "a call that fails still needs a tool message — silence breaks the protocol"
    assert "issing" in tool_messages[0]["content"], (
        f"got {tool_messages[0]['content']!r}. The error from Exercise 1 should travel "
        "back as the tool result, so the model can ask the user for the city."
    )
    print("     the failure travelled back as a tool result, not a crash")
''',
    hint=[
        "Inside the loop, after appending the reply:",
        "  if not reply.tool_calls:",
        "      return reply.content",
        "  for call in reply.tool_calls:",
        "      messages.append({'role': 'tool',",
        "                       'tool_call_id': call.id,",
        "                       'content': run_tool_call(call)})",
        "",
        "Note the loop: one reply can contain several calls, and each needs its own",
        "message with its own id. One answer for two calls is a malformed conversation.",
    ],
    notice=r"""
**What to notice.** This loop has the same shape as Chapter 3's — think, act, observe —
but the parsing problem is gone. The model hands you structured calls, and your job is to
answer every one of them faithfully.

That is why function calling replaced text parsing in practice: fewer ways for the model's
formatting to go wrong, and arguments that arrive as real JSON.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=5,
    title="Tools: Writing Your First Tool Integrations",
    intro=r"""
Tools are what make an agent useful — and what make it dangerous. In this notebook you
write the three pieces that matter: a **description** the model can follow, a file tool
that **cannot escape its folder**, and the **tool-calling protocol** real APIs use.
""",
    minutes=35,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "One tool definition does three jobs: it describes the tool to the model, "
        "validates the arguments, and runs the function.",
        "The model's arguments are **untrusted**: check for missing ones, refuse invented "
        "ones, and return errors it can act on.",
        "Resolve a path **first**, then confirm it is inside the workspace — through a "
        "single checkpoint every file tool uses.",
        "In the tool-calling protocol, **every call gets exactly one answer**, carrying "
        "its `tool_call_id`.",
        "A reply with no `tool_calls` is the final answer, and ends the loop.",
    ],
    next_up="Chapter 6 gives the agent memory — and shows why you test memory by reading "
            "the prompt, not the answer.",
)
