"""Chapter 14 — Frameworks: the features they sell you, built small enough to see.

Design note for the author: this notebook deliberately does NOT install LangGraph,
LlamaIndex or CrewAI. Those libraries move fast, and a notebook pinned to today's
versions stops working within months — which readers blame on the book. Instead the
reader builds the three features frameworks compete on (state + routing, durable
execution, human approval) in about forty lines, so they can then read any framework's
documentation and recognise what it is doing. The chapter's own Ex-67 to Ex-70 scripts
remain the place to exercise the real libraries, with versions pinned.
"""

from lite import Exercise, make_chapter

GIVEN = r'''
import time, json

# ---- Nodes: ordinary functions that take state and return updates -----------
# Every framework in this chapter is, at bottom, a way of running functions like
# these in an order decided by their output.
def fetch(state):
    return {"docs": ["doc-a", "doc-b", "doc-c"], "fetched": True}

def summarise(state):
    return {"summary": f"summary of {len(state['docs'])} documents"}

def check_quality(state):
    return {"score": 0.9 if "summary" in state else 0.0}

def publish(state):
    return {"published": True, "url": "https://example.com/report"}


class Flaky:
    """Fails a given number of times, then succeeds."""
    def __init__(self, failures, delay=0.0):
        self.remaining, self.delay, self.attempts = failures, delay, 0

    def __call__(self, state):
        self.attempts += 1
        if self.delay:
            time.sleep(self.delay)
        if self.remaining > 0:
            self.remaining -= 1
            raise ConnectionError("upstream unavailable")
        return {"flaky_done": True, "attempts": self.attempts}


print("nodes:", ", ".join(["fetch", "summarise", "check_quality", "publish"]))
'''

EX1 = Exercise(
    name="State, nodes and routing",
    section="14.0",
    concept=r"""
Strip the branding off any agent framework and you find the same three decisions:

* **Who holds the state?** One dictionary passed along, or each component keeping its own.
* **Who decides what runs next?** A fixed order, the model, or a routing function.
* **Who handles failure?** You, or the framework.

The dominant answer today is a **graph**: nodes are functions, state is a dictionary, and
each node returns an *update* which is merged into the state. An edge function looks at the
state and names the next node.

That is the whole idea, and it fits in twenty lines. Build it and every framework's
documentation becomes recognisable rather than magical.

> A framework is a set of decisions about control. Pick the one whose decisions match the
> control you need — and test that on a small case before you commit.
""",
    todo=r'''
END = "__end__"


class Graph:
    def __init__(self):
        self.nodes = {}
        self.routes = {}        # node name -> function(state) -> next node name
        self.entry = None

    def add_node(self, name, function):
        self.nodes[name] = function
        return self

    def add_edge(self, name, route):
        """route is a node name, or a function(state) -> node name."""
        self.routes[name] = route
        return self

    def set_entry(self, name):
        self.entry = name
        return self

    def run(self, initial_state, max_steps=20):
        """Run from the entry node until a route returns END.

        Returns (final_state, path). The path is the node names visited, in order.
        """
        state = dict(initial_state)
        path = []
        current = self.entry

        for _ in range(max_steps):
            if current == END:
                return state, path

            # TODO 1: record `current` in path, then call the node function with the
            #         state and MERGE its returned dict into the state
            #         (state.update(...) — a node returns an update, not a whole state).
            # TODO 2: work out the next node. self.routes.get(current) may be a string
            #         (go there) or a callable (call it with the state to get the name).
            #         A node with no route goes to END.
            raise NotImplementedError

        raise RuntimeError(f"graph did not finish within {max_steps} steps")


def quality_gate(state):
    """Route on the score: good enough to publish, or round again?"""
    return "publish" if state["score"] >= 0.8 else "summarise"


graph = (Graph()
         .add_node("fetch", fetch)
         .add_node("summarise", summarise)
         .add_node("check", check_quality)
         .add_node("publish", publish)
         .add_edge("fetch", "summarise")
         .add_edge("summarise", "check")
         .add_edge("check", quality_gate)
         .add_edge("publish", END)
         .set_entry("fetch"))

final, path = graph.run({})
print("path: ", " -> ".join(path))
print("state:", {k: v for k, v in final.items() if k != "docs"})
''',
    solution=r'''
END = "__end__"


class Graph:
    def __init__(self):
        self.nodes = {}
        self.routes = {}        # node name -> function(state) -> next node name
        self.entry = None

    def add_node(self, name, function):
        self.nodes[name] = function
        return self

    def add_edge(self, name, route):
        """route is a node name, or a function(state) -> node name."""
        self.routes[name] = route
        return self

    def set_entry(self, name):
        self.entry = name
        return self

    def run(self, initial_state, max_steps=20):
        """Run from the entry node until a route returns END.

        Returns (final_state, path). The path is the node names visited, in order.
        """
        state = dict(initial_state)
        path = []
        current = self.entry

        for _ in range(max_steps):
            if current == END:
                return state, path

            path.append(current)
            state.update(self.nodes[current](state) or {})

            route = self.routes.get(current, END)
            current = route(state) if callable(route) else route

        raise RuntimeError(f"graph did not finish within {max_steps} steps")


def quality_gate(state):
    """Route on the score: good enough to publish, or round again?"""
    return "publish" if state["score"] >= 0.8 else "summarise"


graph = (Graph()
         .add_node("fetch", fetch)
         .add_node("summarise", summarise)
         .add_node("check", check_quality)
         .add_node("publish", publish)
         .add_edge("fetch", "summarise")
         .add_edge("summarise", "check")
         .add_edge("check", quality_gate)
         .add_edge("publish", END)
         .set_entry("fetch"))

final, path = graph.run({})
print("path: ", " -> ".join(path))
print("state:", {k: v for k, v in final.items() if k != "docs"})
''',
    check=r'''
@check("14.0  the graph runs its nodes and merges their updates")
def _():
    need("Graph", "final", "path")
    assert path == ["fetch", "summarise", "check", "publish"], \
        f"the path was {path}"
    assert final["fetched"] is True and final["published"] is True, (
        "each node's returned dict must be MERGED into the state — a node returns an "
        "update, so state.update(...), not state = ..."
    )
    assert final["summary"] == "summary of 3 documents", (
        "summarise reads state['docs'], which fetch put there. If this is missing, the "
        "state is not carrying between nodes."
    )
    assert "score" in final, "check_quality's output should be in the state too"


@check("14.0  a routing function can send the graph back round")
def _():
    need("Graph", "quality_gate")
    rounds = {"n": 0}

    def poor_then_good(state):
        rounds["n"] += 1
        return {"score": 0.4 if rounds["n"] < 3 else 0.95,
                "summary": f"attempt {rounds['n']}"}

    g = (Graph()
         .add_node("fetch", fetch)
         .add_node("summarise", poor_then_good)
         .add_node("check", lambda s: {})
         .add_node("publish", publish)
         .add_edge("fetch", "summarise")
         .add_edge("summarise", "check")
         .add_edge("check", quality_gate)
         .add_edge("publish", END)
         .set_entry("fetch"))

    state, taken = g.run({})
    assert taken.count("summarise") == 3, (
        f"summarise ran {taken.count('summarise')} times. The quality gate should send "
        "the graph back until the score passes 0.8 — that loop is the point of a "
        "routing function."
    )
    assert state["published"] is True, "and then it should reach publish"
    assert taken[-1] == "publish", f"the last node visited was {taken[-1]!r}"


@check("14.0  a runaway graph is stopped")
def _():
    need("Graph")
    g = (Graph()
         .add_node("spin", lambda s: {"n": s.get("n", 0) + 1})
         .add_edge("spin", "spin")
         .set_entry("spin"))
    try:
        g.run({}, max_steps=10)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "a graph that routes to itself for ever should hit max_steps and raise. "
            "Without that limit a routing bug becomes an unbounded bill."
        )
    print("     3 nodes, a loop that exits on quality, and a step limit that bites")
''',
    hint=[
        "Inside the loop:",
        "  path.append(current)",
        "  state.update(self.nodes[current](state) or {})",
        "",
        "  route = self.routes.get(current, END)",
        "  current = route(state) if callable(route) else route",
        "",
        "The `or {}` matters: a node that returns None should not crash the merge.",
        "`callable(route)` is how one edge type handles both a fixed next node and a",
        "decision function.",
    ],
    notice=r"""
**What to notice.** Forty lines gave you state, nodes, conditional routing, a loop that
exits on a quality score, and a step limit. That is the core of what graph frameworks offer.

What they add on top is worth real money — durable checkpoints, streaming, tracing,
distributed execution, a visual debugger — and the next two exercises build the two you
will actually reach for.
""",
)

EX2 = Exercise(
    name="Durable execution: retries, timeouts, checkpoints",
    section="14.1",
    concept=r"""
This is the feature set frameworks are really selling, and the chapter's summary of it is
exact:

> **A timeout cancels, a retry policy repeats, a handler recovers, a checkpoint remembers,
> an interrupt waits for a person.**

Five different jobs. People reach for "retry" when they needed a checkpoint, and then
wonder why a four-minute pipeline takes twelve minutes to recover.

| Feature | What it does | When you need it |
|---|---|---|
| **retry** | run the same node again | the failure is temporary |
| **timeout** | give up on a node that hangs | an upstream service can stall for ever |
| **checkpoint** | save state after each node | a later failure must not re-run earlier work |
| **resume** | restart from the last checkpoint | the process died or was restarted |

You are adding retry and checkpointing to the graph. Both are about ten lines, and both are
the kind of thing you should know how to write before you let a framework do it for you.
""",
    todo=r'''
class DurableGraph(Graph):
    """A graph with per-node retries and a checkpoint after every node."""

    def __init__(self, max_attempts=3):
        super().__init__()
        self.max_attempts = max_attempts
        self.checkpoints = {}       # node name -> state AFTER that node ran
        self.attempts = {}          # node name -> how many times it was attempted

    def run_node(self, name, state):
        """Run one node, retrying temporary failures. Returns its update."""
        # TODO 1: try up to self.max_attempts times. Count every attempt in
        #         self.attempts[name] (use .get(name, 0) + 1).
        # TODO 2: return the node's update as soon as a call succeeds.
        # TODO 3: on ConnectionError or TimeoutError, try again; if the attempts run
        #         out, re-raise the last error.
        # TODO 4: any other exception is permanent — re-raise it immediately rather
        #         than retrying.
        raise NotImplementedError

    def run(self, initial_state, max_steps=20, resume_from=None):
        """Run the graph, checkpointing after each node.

        resume_from: a node name whose checkpoint should be loaded, continuing after it.
        """
        state = dict(initial_state)
        path = []

        if resume_from is not None:
            # TODO 5: load the saved state from self.checkpoints[resume_from] and
            #         start at the node that route would send us to next, so the
            #         resumed node is NOT run again.
            raise NotImplementedError
        else:
            current = self.entry

        for _ in range(max_steps):
            if current == END:
                return state, path
            path.append(current)
            state.update(self.run_node(current, state) or {})

            # TODO 6: save a COPY of the state as this node's checkpoint. A reference
            #         would keep changing as later nodes run, which makes it useless.
            raise NotImplementedError

            route = self.routes.get(current, END)
            current = route(state) if callable(route) else route

        raise RuntimeError(f"graph did not finish within {max_steps} steps")


flaky_fetch = Flaky(failures=2)

durable = (DurableGraph(max_attempts=3)
           .add_node("fetch", flaky_fetch)
           .add_node("summarise", lambda s: {"summary": "done"})
           .add_node("publish", publish)
           .add_edge("fetch", "summarise")
           .add_edge("summarise", "publish")
           .add_edge("publish", END)
           .set_entry("fetch"))

state, path = durable.run({"docs": ["a", "b"]})
print("path:    ", " -> ".join(path))
print("attempts:", durable.attempts, "<- fetch failed twice and was retried")
print("saved:   ", list(durable.checkpoints))

resumed_state, resumed_path = durable.run({}, resume_from="summarise")
print("\nresuming after 'summarise':", resumed_path)
print("the summary survived:", resumed_state["summary"])
''',
    solution=r'''
class DurableGraph(Graph):
    """A graph with per-node retries and a checkpoint after every node."""

    def __init__(self, max_attempts=3):
        super().__init__()
        self.max_attempts = max_attempts
        self.checkpoints = {}       # node name -> state AFTER that node ran
        self.attempts = {}          # node name -> how many times it was attempted

    def run_node(self, name, state):
        """Run one node, retrying temporary failures. Returns its update."""
        last_error = None
        for _ in range(self.max_attempts):
            self.attempts[name] = self.attempts.get(name, 0) + 1
            try:
                return self.nodes[name](state)
            except (ConnectionError, TimeoutError) as err:
                last_error = err            # temporary: worth another go
            except Exception:
                raise                        # permanent: retrying changes nothing
        raise last_error

    def run(self, initial_state, max_steps=20, resume_from=None):
        """Run the graph, checkpointing after each node.

        resume_from: a node name whose checkpoint should be loaded, continuing after it.
        """
        state = dict(initial_state)
        path = []

        if resume_from is not None:
            state = dict(self.checkpoints[resume_from])
            route = self.routes.get(resume_from, END)
            current = route(state) if callable(route) else route
        else:
            current = self.entry

        for _ in range(max_steps):
            if current == END:
                return state, path
            path.append(current)
            state.update(self.run_node(current, state) or {})

            # A copy, not a reference: later nodes keep mutating `state`.
            self.checkpoints[current] = dict(state)

            route = self.routes.get(current, END)
            current = route(state) if callable(route) else route

        raise RuntimeError(f"graph did not finish within {max_steps} steps")


flaky_fetch = Flaky(failures=2)

durable = (DurableGraph(max_attempts=3)
           .add_node("fetch", flaky_fetch)
           .add_node("summarise", lambda s: {"summary": "done"})
           .add_node("publish", publish)
           .add_edge("fetch", "summarise")
           .add_edge("summarise", "publish")
           .add_edge("publish", END)
           .set_entry("fetch"))

state, path = durable.run({"docs": ["a", "b"]})
print("path:    ", " -> ".join(path))
print("attempts:", durable.attempts, "<- fetch failed twice and was retried")
print("saved:   ", list(durable.checkpoints))

resumed_state, resumed_path = durable.run({}, resume_from="summarise")
print("\nresuming after 'summarise':", resumed_path)
print("the summary survived:", resumed_state["summary"])
''',
    check=r'''
@check("14.1  temporary failures are retried, permanent ones are not")
def _():
    need("DurableGraph")
    g = DurableGraph(max_attempts=3)
    g.add_node("flaky", Flaky(failures=2)).add_edge("flaky", END).set_entry("flaky")
    state, _ = g.run({})
    assert state["flaky_done"] is True, "it succeeds on the third attempt"
    assert g.attempts["flaky"] == 3, \
        f"the node should have been attempted 3 times, record says {g.attempts}"

    permanent = DurableGraph(max_attempts=3)
    permanent.add_node("bad", lambda s: (_ for _ in ()).throw(ValueError("schema")))
    permanent.add_edge("bad", END).set_entry("bad")
    try:
        permanent.run({})
    except ValueError:
        pass
    else:
        raise AssertionError("a permanent error should propagate, not be swallowed")
    assert permanent.attempts["bad"] == 1, (
        f"a ValueError was attempted {permanent.attempts['bad']} times. Retrying a "
        "permanent failure just multiplies the cost."
    )

    never = DurableGraph(max_attempts=2)
    never.add_node("always", Flaky(failures=99)).add_edge("always", END).set_entry("always")
    try:
        never.run({})
    except ConnectionError:
        pass
    else:
        raise AssertionError("when the attempts run out, the last error must be raised")


@check("14.1  every node leaves a checkpoint")
def _():
    need("DurableGraph", "durable")
    need_data(durable.checkpoints, "the DurableGraph cell above")
    assert set(durable.checkpoints) >= {"fetch", "summarise", "publish"}, \
        f"checkpoints exist for {list(durable.checkpoints)}"
    after_fetch = durable.checkpoints["fetch"]
    assert "summary" not in after_fetch, (
        "the checkpoint taken after 'fetch' contains the summary, which a later node "
        "produced. Save dict(state) — a copy — or every checkpoint ends up identical "
        "to the final state and resuming is pointless."
    )
    assert durable.checkpoints["summarise"]["summary"] == "done", \
        "the checkpoint after summarise should contain its output"


@check("14.1  resuming does not redo finished work")
def _():
    need("DurableGraph")
    counts = {"fetch": 0, "summarise": 0, "publish": 0}

    def counting(name, update):
        def node(state):
            counts[name] += 1
            return update
        return node

    g = (DurableGraph()
         .add_node("fetch", counting("fetch", {"docs": ["a"]}))
         .add_node("summarise", counting("summarise", {"summary": "s"}))
         .add_node("publish", counting("publish", {"published": True}))
         .add_edge("fetch", "summarise").add_edge("summarise", "publish")
         .add_edge("publish", END).set_entry("fetch"))

    g.run({})
    assert counts == {"fetch": 1, "summarise": 1, "publish": 1}, \
        f"the first run should visit each node once; got {counts}"

    state, path = g.run({}, resume_from="summarise")
    assert counts["fetch"] == 1 and counts["summarise"] == 1, (
        f"resuming re-ran earlier nodes: {counts}. Load the checkpoint and continue "
        "AFTER that node — re-running it defeats the point."
    )
    assert path == ["publish"], f"only publish should run on resume; got {path}"
    assert state["summary"] == "s", \
        "the resumed state must carry what the earlier nodes produced"
    print("     retry 3x, per-node checkpoints, resume ran 1 node instead of 3")
''',
    hint=[
        "run_node:",
        "  last_error = None",
        "  for _ in range(self.max_attempts):",
        "      self.attempts[name] = self.attempts.get(name, 0) + 1",
        "      try:",
        "          return self.nodes[name](state)",
        "      except (ConnectionError, TimeoutError) as err:",
        "          last_error = err",
        "      except Exception:",
        "          raise",
        "  raise last_error",
        "",
        "resume:",
        "  state = dict(self.checkpoints[resume_from])",
        "  route = self.routes.get(resume_from, END)",
        "  current = route(state) if callable(route) else route",
        "",
        "and the checkpoint itself: self.checkpoints[current] = dict(state)",
    ],
    notice=r"""
**What to notice.** `dict(state)` rather than `state`. Save the reference and every
checkpoint silently becomes the final state — the code looks right, the resume looks like it
works, and it quietly replays from the end. It is exactly the kind of bug a framework has
already found and fixed for you, which is a fair argument for using one.

Note also what is *missing*: a real timeout. Cancelling a function that is blocked inside a
network call needs async nodes or a separate thread, and that is one of the "rules to learn"
the chapter warns about — each feature is real, and each has a condition attached.
""",
)

EX3 = Exercise(
    name="Stopping for a human",
    section="14.4",
    concept=r"""
The last feature, and the one that most often decides whether an agent can ship: an
**interrupt** that stops before something irreversible and waits for a person.

Publishing a report, sending an email, making a payment, deleting a file — a human looks,
then the run continues **from where it stopped**, not from the beginning.

Note that this is the checkpoint from Exercise 2 doing a second job. Pausing for approval
and surviving a crash are the same capability: *state saved, execution resumable*. Which is
why these features are sold together.

Then the chapter's closing question, which is a fair one:

> The cost of a framework is lines of code, load time, and a little time per step. All three
> are small beside a model call. So the real question is whether the features it gives you
> are ones you would otherwise write yourself.

You have now written them. That is the strongest position from which to choose.
""",
    todo=r'''
NEEDS_APPROVAL = {"publish", "send_email", "make_payment"}


class ApprovingGraph(DurableGraph):
    """A graph that pauses before any node needing human approval."""

    def __init__(self, max_attempts=3):
        super().__init__(max_attempts)
        self.pending = None        # {"node", "state"} while waiting for a person

    def run(self, initial_state, max_steps=20, resume_from=None, approved=(),
            start_at=None):
        """Run until finished, or until a node needs approval it has not been given.

        start_at: begin at this node instead of the entry node (used when continuing
        after an approval).

        Returns (status, state, path) where status is "complete" or "waiting".
        """
        state = dict(initial_state)
        path = []

        if resume_from is not None:
            state = dict(self.checkpoints[resume_from])
            route = self.routes.get(resume_from, END)
            current = route(state) if callable(route) else route
        else:
            current = start_at or self.entry

        for _ in range(max_steps):
            if current == END:
                return "complete", state, path

            # TODO 1: if `current` needs approval and is not in `approved`, stop:
            #         save {"node": current, "state": dict(state)} on self.pending
            #         and return ("waiting", state, path). The node must NOT run.
            # TODO 2: otherwise run it as before — append to path, merge the update,
            #         save a checkpoint copy — and follow the route.
            raise NotImplementedError

        raise RuntimeError(f"graph did not finish within {max_steps} steps")

    def approve_and_continue(self, max_steps=20):
        """Continue a waiting run, now that a person has approved the pending node."""
        # TODO 3: take self.pending, clear it, and re-run from its saved state with
        #         that node in `approved` AND start_at set to that node — otherwise the
        #         run begins at the entry node again and redoes finished work.
        #         Return whatever run() returns.
        raise NotImplementedError


approving = (ApprovingGraph()
             .add_node("fetch", fetch)
             .add_node("summarise", summarise)
             .add_node("publish", publish)
             .add_edge("fetch", "summarise")
             .add_edge("summarise", "publish")
             .add_edge("publish", END)
             .set_entry("fetch"))

status, state, path = approving.run({})
print("status:", status, "| ran:", path)
print("waiting on:", approving.pending["node"])
print("published yet?", state.get("published"))

print("\n-- a human looks at the summary and approves --")
status, state, path = approving.approve_and_continue()
print("status:", status, "| ran:", path)
print("published:", state["published"], "at", state["url"])
''',
    solution=r'''
NEEDS_APPROVAL = {"publish", "send_email", "make_payment"}


class ApprovingGraph(DurableGraph):
    """A graph that pauses before any node needing human approval."""

    def __init__(self, max_attempts=3):
        super().__init__(max_attempts)
        self.pending = None        # {"node", "state"} while waiting for a person

    def run(self, initial_state, max_steps=20, resume_from=None, approved=(),
            start_at=None):
        """Run until finished, or until a node needs approval it has not been given.

        start_at: begin at this node instead of the entry node (used when continuing
        after an approval).

        Returns (status, state, path) where status is "complete" or "waiting".
        """
        state = dict(initial_state)
        path = []

        if resume_from is not None:
            state = dict(self.checkpoints[resume_from])
            route = self.routes.get(resume_from, END)
            current = route(state) if callable(route) else route
        else:
            current = start_at or self.entry

        for _ in range(max_steps):
            if current == END:
                return "complete", state, path

            if current in NEEDS_APPROVAL and current not in approved:
                # Stop BEFORE the irreversible step, keeping everything needed to
                # carry on once a person says yes.
                self.pending = {"node": current, "state": dict(state)}
                return "waiting", state, path

            path.append(current)
            state.update(self.run_node(current, state) or {})
            self.checkpoints[current] = dict(state)

            route = self.routes.get(current, END)
            current = route(state) if callable(route) else route

        raise RuntimeError(f"graph did not finish within {max_steps} steps")

    def approve_and_continue(self, max_steps=20):
        """Continue a waiting run, now that a person has approved the pending node."""
        if self.pending is None:
            raise RuntimeError("nothing is waiting for approval")
        pending = self.pending
        self.pending = None
        # start_at matters: without it the run would begin at the entry node and
        # redo every finished step, side effects included.
        return self.run(pending["state"], max_steps=max_steps,
                        approved={pending["node"]}, start_at=pending["node"])


approving = (ApprovingGraph()
             .add_node("fetch", fetch)
             .add_node("summarise", summarise)
             .add_node("publish", publish)
             .add_edge("fetch", "summarise")
             .add_edge("summarise", "publish")
             .add_edge("publish", END)
             .set_entry("fetch"))

status, state, path = approving.run({})
print("status:", status, "| ran:", path)
print("waiting on:", approving.pending["node"])
print("published yet?", state.get("published"))

print("\n-- a human looks at the summary and approves --")
status, state, path = approving.approve_and_continue()
print("status:", status, "| ran:", path)
print("published:", state["published"], "at", state["url"])
''',
    check=r'''
@check("14.4  the run stops before the irreversible step")
def _():
    need("ApprovingGraph")
    g = (ApprovingGraph()
         .add_node("fetch", fetch).add_node("summarise", summarise)
         .add_node("publish", publish)
         .add_edge("fetch", "summarise").add_edge("summarise", "publish")
         .add_edge("publish", END).set_entry("fetch"))

    status, state, path = g.run({})
    assert status == "waiting", f"status was {status!r}, expected 'waiting'"
    assert path == ["fetch", "summarise"], \
        f"the two safe nodes should run; path was {path}"
    assert "published" not in state, (
        "publish RAN. The whole point is to stop before it — check approval before "
        "calling the node, not after."
    )
    assert g.pending and g.pending["node"] == "publish", \
        f"pending should name the node awaiting approval; got {g.pending}"
    assert g.pending["state"]["summary"], \
        "the pending record must carry the state, or there is nothing to resume from"


@check("14.4  approval continues from where it stopped")
def _():
    need("ApprovingGraph")
    ran = []

    def recording(name, update):
        def node(state):
            ran.append(name)
            return update
        return node

    g = (ApprovingGraph()
         .add_node("fetch", recording("fetch", {"docs": ["a"]}))
         .add_node("summarise", recording("summarise", {"summary": "s"}))
         .add_node("publish", recording("publish", {"published": True}))
         .add_edge("fetch", "summarise").add_edge("summarise", "publish")
         .add_edge("publish", END).set_entry("fetch"))

    g.run({})
    assert ran == ["fetch", "summarise"], f"before approval: {ran}"

    status, state, path = g.approve_and_continue()
    assert status == "complete", f"after approval the run should finish; got {status!r}"
    assert ran == ["fetch", "summarise", "publish"], (
        f"nodes ran: {ran}. Approval should run ONLY the approved node and what follows "
        "— re-running fetch and summarise would redo work and could duplicate side "
        "effects."
    )
    assert state["published"] is True and state["summary"] == "s", \
        "the resumed state must carry what was produced before the pause"
    assert g.pending is None, "once approved, nothing is pending any more"


@check("14.4  approval is per node, and not a blanket pass")
def _():
    need("ApprovingGraph")
    g = (ApprovingGraph()
         .add_node("publish", publish)
         .add_node("make_payment", lambda s: {"paid": True})
         .add_edge("publish", "make_payment").add_edge("make_payment", END)
         .set_entry("publish"))

    status, _, _ = g.run({})
    assert status == "waiting" and g.pending["node"] == "publish", \
        "the first approval-needing node should pause the run"

    status, state, path = g.approve_and_continue()
    assert status == "waiting", (
        "approving 'publish' must not also approve 'make_payment'. One approval, one "
        f"node — got status {status!r} and state {state}."
    )
    assert g.pending["node"] == "make_payment", \
        f"the run should now be waiting on make_payment; got {g.pending}"
    assert "paid" not in state, "the payment must not have happened"
    print("     paused before publish, approved it, then paused again before payment")
''',
    hint=[
        "Inside the loop, before running the node:",
        "  if current in NEEDS_APPROVAL and current not in approved:",
        "      self.pending = {'node': current, 'state': dict(state)}",
        "      return 'waiting', state, path",
        "",
        "approve_and_continue:",
        "  pending = self.pending",
        "  self.pending = None",
        "  return self.run(pending['state'], max_steps=max_steps,",
        "                  approved={pending['node']}, start_at=pending['node'])",
        "",
        "Pass only that one node as approved. A blanket pass would wave through every",
        "irreversible step after it.",
    ],
    notice=r"""
**What to notice.** The third check is the one that catches a real mistake: approving
`publish` must not also approve `make_payment`. Approval is per step, because a user who
agreed to one irreversible action did not agree to the next one.

### Choosing a framework, now that you have written one

You have built state and routing, retries, checkpoints, resume and human approval. So the
question stops being "which framework is best?" and becomes answerable:

* **What control do I need?** Who holds state, who decides what runs next, who handles
  failure.
* **Which of these features would I otherwise write?** If it is one or two, forty lines may
  be cheaper than a dependency.
* **Then test it on the smallest case that exercises state, approval and replay** — the
  three things that are painful to retrofit.

A framework's cost is lines, load time and a little latency per step, all small beside a
model call. Its real cost is the control it takes away. Decide that with an experiment, not
a blog post.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=14,
    title="Frameworks: What They Give You",
    intro=r"""
LangGraph, LlamaIndex and CrewAI compete on a short list of features: **state and routing,
durable execution, and human approval.** In this notebook you build all three in about forty
lines — not to replace the frameworks, but so that their documentation reads as recognisable
engineering rather than magic, and so you can tell which of their features you actually need.

> **A note on versions.** This notebook installs nothing. Agent frameworks change fast, and
> a notebook pinned to today's releases breaks within months. The book's own Ex-67 to Ex-70
> scripts are the place to run the real libraries — pin exact versions there.
""",
    minutes=35,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "A framework is a set of decisions about **control**: who holds state, who decides "
        "what runs next, who handles failure.",
        "A **graph** is nodes that return state updates, plus routing functions that read "
        "the state — and a step limit.",
        "Five features, five different jobs: a **timeout** cancels, a **retry** repeats, a "
        "**handler** recovers, a **checkpoint** remembers, an **interrupt** waits for a "
        "person.",
        "Checkpoint a **copy** of the state, and resume **after** the saved node — or "
        "resuming quietly redoes the work.",
        "Approval is **per node**: agreeing to one irreversible step does not authorise "
        "the next.",
        "A framework's cost in lines and latency is small beside a model call. The real "
        "question is whether you would write these features yourself.",
    ],
    next_up="Chapter 15 turns to safety: guardrails, tool authorisation, and injection "
            "end to end.",
)
