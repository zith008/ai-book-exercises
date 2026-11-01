"""Chapter 16 — Real-World Applications: honest evaluation, evidence routing, real governance."""

from lite import Exercise, make_chapter

GIVEN = r'''
import hashlib, json, re, secrets

# ---- A support agent's knowledge, and two sets of tickets ------------------
# The ones it was BUILT from, and the ones it has never seen. Those are different
# questions, and only the second tells you anything.
KNOWN_ANSWERS = {
    "refund": "You can return any item within 30 days for a full refund.",
    "delivery": "Standard delivery takes 3-5 working days; express is next day.",
    "warranty": "Electronics carry a 24 month warranty on manufacturing faults.",
    "password": "Reset your password from the sign-in page using 'forgot password'.",
}

BUILT_FROM = [
    {"text": "How do I get a refund?", "topic": "refund"},
    {"text": "How long is delivery?", "topic": "delivery"},
    {"text": "What is the warranty?", "topic": "warranty"},
    {"text": "I forgot my password", "topic": "password"},
]

HELD_OUT = [
    {"text": "Can I get my money back if the delivery is late?", "topic": "delivery"},
    {"text": "The item arrived broken, can I send it back?", "topic": "refund"},
    {"text": "Is accidental damage covered?", "topic": "warranty"},
    {"text": "Can't sign in to my account", "topic": "password"},
    {"text": "Do you ship to Japan?", "topic": None},          # not in the knowledge
    {"text": "I want to speak to a manager about a complaint", "topic": None},
]


def classify(text):
    """A keyword classifier, as a first version would be. Given to you."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    for topic, keywords in {
        "refund": {"refund", "return", "money", "back"},
        "delivery": {"delivery", "deliver", "shipping", "parcel", "late"},
        "warranty": {"warranty", "guarantee", "covered", "fault"},
        "password": {"password", "login", "sign", "account"},
    }.items():
        if words & keywords:
            return topic
    return None


print(f"{len(BUILT_FROM)} tickets it was built from, {len(HELD_OUT)} it has never seen")
'''

EX1 = Exercise(
    name="Measuring it honestly",
    section="16.1",
    concept=r"""
Every agent demo works. The question is what happens on the next hundred cases, and
answering it needs two disciplines that are easy to skip.

**1. Evaluate on tickets the agent was not built from.** Scoring on your own examples
measures your memory, not the agent. The held-out set is the only number worth quoting.

**2. Count wrong answers separately from escalations.** They are not the same failure:

| Outcome | What it costs |
|---|---|
| **answered correctly** | nothing |
| **escalated** (said "I don't know", passed to a human) | a human's time — the *safe* failure |
| **answered wrongly** | a misled customer, and nobody knows it happened |

A single "accuracy" number hides the difference, and the difference is everything. An agent
that escalates 30% and is never wrong is usually shippable. One that answers everything and
is wrong 10% of the time usually is not.

> And **do not call a ticket resolved until someone says it is.** "The agent produced a
> reply" is not resolution.
""",
    todo=r'''
def respond(ticket_text):
    """Answer a ticket, or escalate. Returns (status, text)."""
    topic = classify(ticket_text)
    if topic is None or topic not in KNOWN_ANSWERS:
        return "escalated", "Passing you to a human colleague."
    return "answered", KNOWN_ANSWERS[topic]


def evaluate(tickets):
    """Score the agent. Returns counts, never a single accuracy number.

    {"total", "correct", "wrong", "escalated", "coverage", "error_rate"}
    """
    counts = {"total": len(tickets), "correct": 0, "wrong": 0, "escalated": 0}

    for ticket in tickets:
        status, text = respond(ticket["text"])

        # TODO 1: if the agent escalated, count an escalation and move on. An
        #         escalation is not a wrong answer.
        # TODO 2: otherwise it answered. It is CORRECT only if the ticket's true
        #         topic is not None and the text equals KNOWN_ANSWERS[that topic].
        #         Anything else is wrong.
        raise NotImplementedError

    # TODO 3: coverage = the fraction of tickets it answered at all (correct + wrong).
    # TODO 4: error_rate = wrong / total. Keep it separate from coverage: these two
    #         numbers trade against each other, and an average of them means nothing.
    raise NotImplementedError


print("scored on the tickets it was BUILT from:")
print(" ", evaluate(BUILT_FROM))
print("\nscored on tickets it has NEVER SEEN:")
print(" ", evaluate(HELD_OUT))


def resolution_rate(outcomes):
    """outcomes: [{"status", "customer_confirmed"}]. What fraction are really resolved?"""
    # TODO 5: a ticket counts as resolved only when it was answered AND
    #         customer_confirmed is True. "The agent replied" is not resolution.
    raise NotImplementedError


outcomes = [{"status": "answered", "customer_confirmed": True},
            {"status": "answered", "customer_confirmed": True},
            {"status": "answered", "customer_confirmed": False},   # replied, not resolved
            {"status": "answered", "customer_confirmed": None},    # nobody asked
            {"status": "escalated", "customer_confirmed": True}]
print(f"\nreplies sent: 4 of 5; actually resolved: {resolution_rate(outcomes):.0%}")
''',
    solution=r'''
def respond(ticket_text):
    """Answer a ticket, or escalate. Returns (status, text)."""
    topic = classify(ticket_text)
    if topic is None or topic not in KNOWN_ANSWERS:
        return "escalated", "Passing you to a human colleague."
    return "answered", KNOWN_ANSWERS[topic]


def evaluate(tickets):
    """Score the agent. Returns counts, never a single accuracy number.

    {"total", "correct", "wrong", "escalated", "coverage", "error_rate"}
    """
    counts = {"total": len(tickets), "correct": 0, "wrong": 0, "escalated": 0}

    for ticket in tickets:
        status, text = respond(ticket["text"])

        if status == "escalated":
            counts["escalated"] += 1
            continue

        true_topic = ticket["topic"]
        if true_topic is not None and text == KNOWN_ANSWERS.get(true_topic):
            counts["correct"] += 1
        else:
            counts["wrong"] += 1

    answered = counts["correct"] + counts["wrong"]
    counts["coverage"] = answered / counts["total"] if counts["total"] else 0.0
    counts["error_rate"] = counts["wrong"] / counts["total"] if counts["total"] else 0.0
    return counts


print("scored on the tickets it was BUILT from:")
print(" ", evaluate(BUILT_FROM))
print("\nscored on tickets it has NEVER SEEN:")
print(" ", evaluate(HELD_OUT))


def resolution_rate(outcomes):
    """outcomes: [{"status", "customer_confirmed"}]. What fraction are really resolved?"""
    resolved = sum(1 for o in outcomes
                   if o["status"] == "answered" and o["customer_confirmed"] is True)
    return resolved / len(outcomes) if outcomes else 0.0


outcomes = [{"status": "answered", "customer_confirmed": True},
            {"status": "answered", "customer_confirmed": True},
            {"status": "answered", "customer_confirmed": False},   # replied, not resolved
            {"status": "answered", "customer_confirmed": None},    # nobody asked
            {"status": "escalated", "customer_confirmed": True}]
print(f"\nreplies sent: 4 of 5; actually resolved: {resolution_rate(outcomes):.0%}")
''',
    check=r'''
@check("16.1  escalations and wrong answers are counted separately")
def _():
    need("evaluate")
    built = evaluate(BUILT_FROM)
    assert built["correct"] == 4 and built["wrong"] == 0, (
        f"on its own examples the agent should get all 4 right; got {built}"
    )

    held = evaluate(HELD_OUT)
    assert held["total"] == 6, f"six held-out tickets, record says {held['total']}"
    assert held["correct"] + held["wrong"] + held["escalated"] == 6, (
        f"every ticket must land in exactly one bucket; got {held}"
    )
    assert held["escalated"] >= 1, (
        "at least one held-out ticket is outside the knowledge base and should be "
        f"escalated, not answered; got {held}"
    )
    assert held["wrong"] >= 1, (
        "the keyword classifier mislabels at least one held-out ticket. If your count "
        "says zero, an escalation is probably being counted as correct — they are "
        f"different outcomes. Got {held}"
    )


@check("16.1  held-out performance is worse than the built-from score")
def _():
    need("evaluate")
    built, held = evaluate(BUILT_FROM), evaluate(HELD_OUT)
    built_rate = built["correct"] / built["total"]
    held_rate = held["correct"] / held["total"]
    assert held_rate < built_rate, (
        f"held-out {held_rate:.0%} vs built-from {built_rate:.0%}. The held-out score "
        "should be lower — if they match, the evaluation is scoring the examples the "
        "agent was built from."
    )
    assert 0 <= held["coverage"] <= 1 and 0 <= held["error_rate"] <= 1, \
        f"coverage and error_rate are fractions; got {held}"
    assert abs(held["coverage"] - (held["correct"] + held["wrong"]) / 6) < 1e-9, \
        "coverage is the fraction of tickets it ANSWERED, right or wrong"
    assert abs(held["error_rate"] - held["wrong"] / 6) < 1e-9, \
        "error_rate is wrong answers over all tickets"
    print(f"     built-from {built_rate:.0%} vs held-out {held_rate:.0%}; "
          f"coverage {held['coverage']:.0%}, errors {held['error_rate']:.0%}")


@check("16.1  a reply is not a resolution")
def _():
    need("resolution_rate")
    rate = resolution_rate(outcomes)
    assert abs(rate - 2 / 5) < 1e-9, (
        f"you reported {rate:.0%}. Four replies were sent, but only two customers "
        "confirmed the problem was solved. Counting replies as resolutions is the "
        "single most common way these systems are oversold."
    )
    assert resolution_rate([]) == 0.0, "an empty list is not 100% resolved"
    assert resolution_rate([{"status": "escalated", "customer_confirmed": True}]) == 0.0, (
        "an escalation resolved by a human is not the agent resolving it — that is the "
        "human's work"
    )
''',
    hint=[
        "In the loop:",
        "  if status == 'escalated':",
        "      counts['escalated'] += 1",
        "      continue",
        "  true_topic = ticket['topic']",
        "  if true_topic is not None and text == KNOWN_ANSWERS.get(true_topic):",
        "      counts['correct'] += 1",
        "  else:",
        "      counts['wrong'] += 1",
        "",
        "Then:",
        "  answered = counts['correct'] + counts['wrong']",
        "  counts['coverage'] = answered / counts['total']",
        "  counts['error_rate'] = counts['wrong'] / counts['total']",
        "",
        "resolution_rate: count only answered AND customer_confirmed is True.",
    ],
    notice=r"""
**What to notice.** The same agent scores 100% on its own examples and much less on tickets
it has not seen. Neither number is a lie — they answer different questions, and only the
second one is about your customers.

Three numbers to report, always: **coverage** (how much it handled), **error rate** (how
often it was wrong), **resolution rate** (how often a human agreed it was solved). One
blended "accuracy" figure hides every trade-off worth discussing.

> And for a handful of fixed answers, use a lookup table. Four known answers do not need a
> language model, and the chapter says so plainly.
""",
)

EX2 = Exercise(
    name="Routing by evidence, not by a hunch",
    section="16.2",
    concept=r"""
You have a cheap model and an expensive one. Which question goes where?

The tempting approach is to guess from the question itself — long questions are hard, or
questions containing "analyse" are hard. **This does not work.** "Is 7919 prime?" is nine
characters and needs real work; "Could you possibly tell me, if it is not too much trouble,
what your opening hours are?" is long and trivial.

The approach that does work is a **cheap-first cascade with a check**:

1. Ask the cheap model.
2. **Check its answer** — against a verifier, a schema, a retrieved source, or its own
   stated confidence.
3. Only if the check fails, escalate to the expensive model.

You are routing on **evidence about this specific answer**, not on a hunch about the
question. And then you measure the pass rate, because that single number tells you whether
the cascade is saving money or just adding a step.

> Count dollars from the usage figures, not from guesses.
""",
    todo=r'''
PRICE_PER_1K = {"cheap": 0.0002, "expensive": 0.003}

QUESTIONS = [
    {"q": "What are your opening hours?", "answer": "9am to 5pm", "cheap_can": True},
    {"q": "How do I reset my password?", "answer": "Use the forgot password link",
     "cheap_can": True},
    {"q": "Reconcile these two conflicting invoices and explain the discrepancy",
     "answer": "Invoice B double-counts shipping", "cheap_can": False},
    {"q": "Is 7919 prime?", "answer": "Yes", "cheap_can": False},
    {"q": "Where is my parcel?", "answer": "In transit", "cheap_can": True},
]


def call_model(model, question):
    """A stand-in. The cheap model gives up on hard questions. Given to you."""
    item = next(q for q in QUESTIONS if q["q"] == question["q"])
    tokens = {"prompt_tokens": len(question["q"]) // 4 + 20, "completion_tokens": 30}
    if model == "cheap" and not item["cheap_can"]:
        return {"text": "I am not sure about this one.", "usage": tokens}
    return {"text": item["answer"], "usage": tokens}


def cost_of(model, usage):
    """Dollars, from the usage figures. Given to you."""
    total = usage["prompt_tokens"] + usage["completion_tokens"]
    return (total / 1000) * PRICE_PER_1K[model]


def is_confident(text):
    """The check: does the answer look like an answer? Given to you."""
    hedges = ("not sure", "i don't know", "cannot determine", "unclear")
    return not any(h in text.lower() for h in hedges)


def answer_with_cascade(question):
    """Cheap model first, escalating only when the check fails.

    Returns {"text", "model_used", "cost", "escalated"}.
    """
    # TODO 1: ask the cheap model and work out its cost with cost_of("cheap", ...).
    # TODO 2: if is_confident(its text), return it with model_used "cheap",
    #         escalated False, and that cost.
    # TODO 3: otherwise call the expensive model and return its answer, with the cost
    #         of BOTH calls added together — the failed cheap attempt was still paid
    #         for — model_used "expensive" and escalated True.
    raise NotImplementedError


def route_by_hunch(question):
    """The approach that does not work, for comparison. Given to you."""
    hard_words = ("reconcile", "analyse", "explain")
    looks_hard = (len(question["q"]) > 60
                  or any(w in question["q"].lower() for w in hard_words))
    model = "expensive" if looks_hard else "cheap"
    reply = call_model(model, question)
    return {"text": reply["text"], "model_used": model,
            "cost": cost_of(model, reply["usage"]), "escalated": model == "expensive"}


cascade = [answer_with_cascade(q) for q in QUESTIONS]
hunch = [route_by_hunch(q) for q in QUESTIONS]

print(f"{'question':>46} {'cascade':>22} {'hunch':>22}")
for item, c, h in zip(QUESTIONS, cascade, hunch):
    print(f"{item['q'][:44]:>46} {c['model_used']:>10} ${c['cost']:.5f} "
          f"{h['model_used']:>10} ${h['cost']:.5f}")

cheap_pass = sum(1 for c in cascade if not c["escalated"]) / len(cascade)
print(f"\ncascade: ${sum(c['cost'] for c in cascade):.5f}, "
      f"cheap model handled {cheap_pass:.0%}")
print(f"hunch:   ${sum(h['cost'] for h in hunch):.5f}, "
      f"and got {sum(1 for i, h in zip(QUESTIONS, hunch) if h['text'] != i['answer'])} "
      "answers wrong")
''',
    solution=r'''
PRICE_PER_1K = {"cheap": 0.0002, "expensive": 0.003}

QUESTIONS = [
    {"q": "What are your opening hours?", "answer": "9am to 5pm", "cheap_can": True},
    {"q": "How do I reset my password?", "answer": "Use the forgot password link",
     "cheap_can": True},
    {"q": "Reconcile these two conflicting invoices and explain the discrepancy",
     "answer": "Invoice B double-counts shipping", "cheap_can": False},
    {"q": "Is 7919 prime?", "answer": "Yes", "cheap_can": False},
    {"q": "Where is my parcel?", "answer": "In transit", "cheap_can": True},
]


def call_model(model, question):
    """A stand-in. The cheap model gives up on hard questions. Given to you."""
    item = next(q for q in QUESTIONS if q["q"] == question["q"])
    tokens = {"prompt_tokens": len(question["q"]) // 4 + 20, "completion_tokens": 30}
    if model == "cheap" and not item["cheap_can"]:
        return {"text": "I am not sure about this one.", "usage": tokens}
    return {"text": item["answer"], "usage": tokens}


def cost_of(model, usage):
    """Dollars, from the usage figures. Given to you."""
    total = usage["prompt_tokens"] + usage["completion_tokens"]
    return (total / 1000) * PRICE_PER_1K[model]


def is_confident(text):
    """The check: does the answer look like an answer? Given to you."""
    hedges = ("not sure", "i don't know", "cannot determine", "unclear")
    return not any(h in text.lower() for h in hedges)


def answer_with_cascade(question):
    """Cheap model first, escalating only when the check fails.

    Returns {"text", "model_used", "cost", "escalated"}.
    """
    first = call_model("cheap", question)
    cheap_cost = cost_of("cheap", first["usage"])

    if is_confident(first["text"]):
        return {"text": first["text"], "model_used": "cheap",
                "cost": cheap_cost, "escalated": False}

    second = call_model("expensive", question)
    # The failed cheap attempt was still billed, so it belongs in the total.
    return {"text": second["text"], "model_used": "expensive",
            "cost": cheap_cost + cost_of("expensive", second["usage"]),
            "escalated": True}


def route_by_hunch(question):
    """The approach that does not work, for comparison. Given to you."""
    hard_words = ("reconcile", "analyse", "explain")
    looks_hard = (len(question["q"]) > 60
                  or any(w in question["q"].lower() for w in hard_words))
    model = "expensive" if looks_hard else "cheap"
    reply = call_model(model, question)
    return {"text": reply["text"], "model_used": model,
            "cost": cost_of(model, reply["usage"]), "escalated": model == "expensive"}


cascade = [answer_with_cascade(q) for q in QUESTIONS]
hunch = [route_by_hunch(q) for q in QUESTIONS]

print(f"{'question':>46} {'cascade':>22} {'hunch':>22}")
for item, c, h in zip(QUESTIONS, cascade, hunch):
    print(f"{item['q'][:44]:>46} {c['model_used']:>10} ${c['cost']:.5f} "
          f"{h['model_used']:>10} ${h['cost']:.5f}")

cheap_pass = sum(1 for c in cascade if not c["escalated"]) / len(cascade)
print(f"\ncascade: ${sum(c['cost'] for c in cascade):.5f}, "
      f"cheap model handled {cheap_pass:.0%}")
print(f"hunch:   ${sum(h['cost'] for h in hunch):.5f}, "
      f"and got {sum(1 for i, h in zip(QUESTIONS, hunch) if h['text'] != i['answer'])} "
      "answers wrong")
''',
    check=r'''
@check("16.2  the cascade escalates only what fails the check")
def _():
    need("answer_with_cascade")
    results = {q["q"]: answer_with_cascade(q) for q in QUESTIONS}

    easy = results["What are your opening hours?"]
    assert easy["model_used"] == "cheap" and easy["escalated"] is False, \
        f"an easy question should be answered by the cheap model; got {easy}"
    assert easy["text"] == "9am to 5pm", f"and answered correctly; got {easy['text']!r}"

    hard = results["Is 7919 prime?"]
    assert hard["escalated"] is True, (
        "the cheap model hedged on this one, so the check should have failed and sent "
        f"it to the expensive model; got {hard}"
    )
    assert hard["text"] == "Yes", f"and the expensive model's answer used; got {hard}"

    for question in QUESTIONS:
        result = results[question["q"]]
        assert result["text"] == question["answer"], (
            f"{question['q'][:40]!r} was answered {result['text']!r}; the cascade must "
            "end with a correct answer for every question"
        )


@check("16.2  escalated questions are charged for both calls")
def _():
    need("answer_with_cascade", "cost_of")
    escalated = answer_with_cascade(QUESTIONS[3])          # "Is 7919 prime?"
    cheap_only = answer_with_cascade(QUESTIONS[0])         # opening hours

    usage = {"prompt_tokens": len(QUESTIONS[3]["q"]) // 4 + 20, "completion_tokens": 30}
    expensive_alone = cost_of("expensive", usage)
    assert escalated["cost"] > expensive_alone, (
        f"the escalated question cost ${escalated['cost']:.5f}, which is no more than "
        f"the expensive call alone (${expensive_alone:.5f}). The cheap attempt that "
        "failed was still billed — a cascade's cost is both calls."
    )
    assert cheap_only["cost"] < expensive_alone, \
        "a question answered by the cheap model alone should cost less"


@check("16.2  guessing from the words gets it wrong")
def _():
    need("answer_with_cascade", "route_by_hunch")
    hunch = {q["q"]: route_by_hunch(q) for q in QUESTIONS}
    cascade = {q["q"]: answer_with_cascade(q) for q in QUESTIONS}

    wrong = [q["q"] for q in QUESTIONS if hunch[q["q"]]["text"] != q["answer"]]
    assert wrong, (
        "(sanity check) the keyword router is supposed to misroute something — "
        "'Is 7919 prime?' is short, so it looks easy"
    )
    assert all(cascade[q["q"]]["text"] == q["answer"] for q in QUESTIONS), \
        "the cascade should get everything right, because it checks"

    cascade_total = sum(cascade[q["q"]]["cost"] for q in QUESTIONS)
    pass_rate = sum(1 for q in QUESTIONS if not cascade[q["q"]]["escalated"]) / len(QUESTIONS)
    assert 0 < pass_rate < 1, (
        f"the cheap-model pass rate is {pass_rate:.0%}. This is the number that tells "
        "you whether a cascade is worth having: near 0 and it is pure overhead, near 1 "
        "and you may not need the expensive model at all."
    )
    print(f"     hunch misrouted {len(wrong)} question(s); cascade got all 5 right at "
          f"${cascade_total:.5f} with a {pass_rate:.0%} cheap pass rate")
''',
    hint=[
        "answer_with_cascade:",
        "  first = call_model('cheap', question)",
        "  cheap_cost = cost_of('cheap', first['usage'])",
        "  if is_confident(first['text']):",
        "      return {'text': first['text'], 'model_used': 'cheap',",
        "              'cost': cheap_cost, 'escalated': False}",
        "",
        "  second = call_model('expensive', question)",
        "  return {'text': second['text'], 'model_used': 'expensive',",
        "          'cost': cheap_cost + cost_of('expensive', second['usage']),",
        "          'escalated': True}",
        "",
        "Do not forget cheap_cost in the escalated total. The attempt that failed was",
        "still a request you paid for.",
    ],
    notice=r"""
**What to notice.** "Is 7919 prime?" is nine characters and the keyword router sent it to the
cheap model, which got it wrong. No amount of tuning the keyword list fixes this, because
the difficulty is not in the words.

The **pass rate** is the number to watch. If the cheap model handles 80% of traffic, the
cascade is saving real money. If it handles 10%, you have added a step and a delay to every
request for very little — and should route by task type instead, which is something your
application knows and the model does not.
""",
)

EX3 = Exercise(
    name="Governance that is actually enforced",
    section="16.3",
    concept=r"""
The agent is good. Now it is an API that other people call, and four things stop being
optional.

**1. Identity comes from a credential, never from a name the caller types.** A request
saying `{"user": "admin"}` is a claim, not an identity. Look the credential up; if it is
not valid, there is no user.

**2. Scopes are enforced on each route**, in code, not described in documentation.

**3. The audit log holds no raw user text.** You need to know *that* a request happened —
who, when, which route, what outcome — not what the customer typed. Logging raw text copies
personal data into a second system that nobody remembered to protect.

**4. Errors do not leak.** "No user matching token sk-abc123" tells an attacker their token
reached your database. Return a flat refusal and keep the detail in your own logs.
""",
    todo=r'''
# Credentials: token -> who they are and what they may do
CREDENTIALS = {
    "tok_alice": {"user": "alice", "scopes": {"ask", "history"}},
    "tok_bob":   {"user": "bob",   "scopes": {"ask"}},
    "tok_admin": {"user": "admin", "scopes": {"ask", "history", "admin"}},
}

ROUTES = {                      # route -> the scope it requires
    "/ask": "ask",
    "/history": "history",
    "/admin/stats": "admin",
}


class API:
    def __init__(self):
        self.audit = []

    def record(self, route, user, outcome):
        """Write one audit entry. NO raw user text."""
        # TODO 1: append {"route", "user", "outcome", "request_id"} — a random
        #         request_id from secrets.token_hex(8). Do not accept or store the
        #         question itself: you need to know THAT a request happened, not what
        #         was in it.
        raise NotImplementedError

    def handle(self, route, token, body):
        """Handle a request. Returns (status_code, response)."""
        # TODO 2: look the token up in CREDENTIALS. If it is missing or unknown,
        #         record the attempt (user None, outcome "unauthenticated") and return
        #         (401, {"error": "unauthorized"}) — exactly that message, whatever
        #         went wrong. Never echo the token or say whether it exists.
        # TODO 3: ignore any "user" field in the body. Identity is the credential's
        #         user, never what the caller claims.
        # TODO 4: unknown route -> record it and return (404, {"error": "not found"}).
        # TODO 5: if the route's required scope is not in the caller's scopes, record
        #         "forbidden" and return (403, {"error": "forbidden"}).
        # TODO 6: otherwise record "ok" and return
        #         (200, {"user": the credential's user, "route": route}).
        raise NotImplementedError


api = API()

requests = [
    ("/ask", "tok_bob", {"question": "where is my parcel?"}),
    ("/history", "tok_bob", {}),                                  # no history scope
    ("/history", "tok_alice", {}),
    ("/admin/stats", "tok_alice", {}),                            # not an admin
    ("/admin/stats", "tok_admin", {}),
    ("/ask", "tok_nonsense", {"question": "hello"}),              # bad token
    ("/ask", None, {"question": "hello"}),                        # no token
    ("/ask", "tok_bob", {"question": "hi", "user": "admin"}),     # claims to be admin
    ("/nope", "tok_bob", {}),
]

for route, token, body in requests:
    status, response = api.handle(route, token, body)
    print(f"{status} {route:>14} {str(token):>14} -> {response}")

print("\naudit log:")
for entry in api.audit:
    print(" ", entry)
''',
    solution=r'''
# Credentials: token -> who they are and what they may do
CREDENTIALS = {
    "tok_alice": {"user": "alice", "scopes": {"ask", "history"}},
    "tok_bob":   {"user": "bob",   "scopes": {"ask"}},
    "tok_admin": {"user": "admin", "scopes": {"ask", "history", "admin"}},
}

ROUTES = {                      # route -> the scope it requires
    "/ask": "ask",
    "/history": "history",
    "/admin/stats": "admin",
}


class API:
    def __init__(self):
        self.audit = []

    def record(self, route, user, outcome):
        """Write one audit entry. NO raw user text."""
        self.audit.append({"route": route, "user": user, "outcome": outcome,
                           "request_id": secrets.token_hex(8)})

    def handle(self, route, token, body):
        """Handle a request. Returns (status_code, response)."""
        credential = CREDENTIALS.get(token) if token else None
        if credential is None:
            # One message for every authentication failure: a different one for
            # "unknown token" would confirm which tokens exist.
            self.record(route, None, "unauthenticated")
            return 401, {"error": "unauthorized"}

        user = credential["user"]          # from the credential, never from the body

        if route not in ROUTES:
            self.record(route, user, "not_found")
            return 404, {"error": "not found"}

        if ROUTES[route] not in credential["scopes"]:
            self.record(route, user, "forbidden")
            return 403, {"error": "forbidden"}

        self.record(route, user, "ok")
        return 200, {"user": user, "route": route}


api = API()

requests = [
    ("/ask", "tok_bob", {"question": "where is my parcel?"}),
    ("/history", "tok_bob", {}),                                  # no history scope
    ("/history", "tok_alice", {}),
    ("/admin/stats", "tok_alice", {}),                            # not an admin
    ("/admin/stats", "tok_admin", {}),
    ("/ask", "tok_nonsense", {"question": "hello"}),              # bad token
    ("/ask", None, {"question": "hello"}),                        # no token
    ("/ask", "tok_bob", {"question": "hi", "user": "admin"}),     # claims to be admin
    ("/nope", "tok_bob", {}),
]

for route, token, body in requests:
    status, response = api.handle(route, token, body)
    print(f"{status} {route:>14} {str(token):>14} -> {response}")

print("\naudit log:")
for entry in api.audit:
    print(" ", entry)
''',
    check=r'''
@check("16.3  identity comes from the credential")
def _():
    need("API")
    a = API()
    status, response = a.handle("/ask", "tok_bob", {"question": "hi", "user": "admin"})
    assert status == 200, f"bob has the ask scope, so this should succeed; got {status}"
    assert response["user"] == "bob", (
        f"the response says the user is {response['user']!r}. The body claimed 'admin' "
        "and must be ignored — a name in a request is a claim, not an identity."
    )

    assert a.handle("/admin/stats", "tok_bob", {"user": "admin"})[0] == 403, \
        "claiming to be an admin in the body must not grant the admin scope"
    assert a.handle("/admin/stats", "tok_admin", {})[0] == 200, \
        "the real admin token should work"


@check("16.3  scopes are enforced per route")
def _():
    need("API")
    a = API()
    assert a.handle("/ask", "tok_bob", {})[0] == 200, "bob may ask"
    assert a.handle("/history", "tok_bob", {})[0] == 403, (
        "bob has no history scope, so /history must be 403 — enforced on the route, "
        "not written down in documentation"
    )
    assert a.handle("/history", "tok_alice", {})[0] == 200, "alice may read history"
    assert a.handle("/nope", "tok_alice", {})[0] == 404, "an unknown route is 404"


@check("16.3  failures do not leak, and nor does the audit log")
def _():
    need("API")
    a = API()
    for bad_token in ("tok_nonsense", None, "", "tok_alice "):
        status, response = a.handle("/ask", bad_token, {"question": "secret question"})
        assert status == 401, f"{bad_token!r} should be 401, got {status}"
        assert response == {"error": "unauthorized"}, (
            f"the response was {response}. Every authentication failure returns the "
            "same flat message — a different one for 'unknown token' tells an attacker "
            "which tokens are real."
        )
        if bad_token:
            assert str(bad_token) not in json.dumps(response), \
                "never echo the token back"

    a.handle("/ask", "tok_bob", {"question": "my card number is 4111 1111 1111 1111"})
    dumped = json.dumps(a.audit)
    assert "4111" not in dumped and "card number" not in dumped, (
        "the audit log contains the user's text. Log that a request happened — route, "
        "user, outcome, id — and not what was in it, or you have copied personal data "
        "into a second system."
    )
    assert "secret question" not in dumped, "and not for failed requests either"
    assert all({"route", "user", "outcome", "request_id"} <= set(e) for e in a.audit), \
        "each entry still needs route, user, outcome and a request id"
    assert any(e["outcome"] == "unauthenticated" for e in a.audit), \
        "failed authentication attempts belong in the audit log too"
    ids = [e["request_id"] for e in a.audit]
    assert len(set(ids)) == len(ids), "request ids must be unique"
    print(f"     {len(a.audit)} audit entries, no user text, uniform 401s")
''',
    hint=[
        "handle:",
        "  credential = CREDENTIALS.get(token) if token else None",
        "  if credential is None:",
        "      self.record(route, None, 'unauthenticated')",
        "      return 401, {'error': 'unauthorized'}",
        "",
        "  user = credential['user']        # NOT body.get('user')",
        "  if route not in ROUTES:",
        "      self.record(route, user, 'not_found'); return 404, {'error': 'not found'}",
        "  if ROUTES[route] not in credential['scopes']:",
        "      self.record(route, user, 'forbidden'); return 403, {'error': 'forbidden'}",
        "  self.record(route, user, 'ok')",
        "  return 200, {'user': user, 'route': route}",
        "",
        "record takes no question text at all — that is the easiest way to be sure none",
        "ends up in the log.",
    ],
    notice=r"""
**What to notice.** `record()` has no parameter for the request body. That is deliberate: the
simplest way to guarantee user text never reaches your audit log is to give the logging
function no way to receive it.

Same with the uniform 401. Four different authentication failures, one message — because
the useful detail belongs in your own logs, not in a response to whoever is probing.

> Identity comes from a credential. Scopes are enforced on each route. The audit log holds
> no raw user text. Errors do not leak.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=16,
    title="Real-World Applications",
    intro=r"""
Three things separate an agent that demos well from one you can put in front of customers:
an **honest evaluation**, **routing that is based on evidence**, and **governance that is
enforced in code**.

This notebook builds all three — and the first exercise will show the same agent scoring
100% and then much less, depending only on which tickets you score it against.

> Treat industry statistics as prompts for questions, not as evidence. The evidence for your
> project is your own evaluation, on data your agent has not seen.
""",
    minutes=40,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "Evaluate on cases the agent was **not built from** — anything else measures your "
        "memory.",
        "Count **wrong answers separately from escalations**: one costs a human's time, "
        "the other costs a customer.",
        "**A reply is not a resolution.** Do not call a ticket resolved until someone "
        "says it is.",
        "Route by **evidence about the answer** (a check on the cheap model's output), "
        "never by guessing difficulty from the question's words.",
        "A cascade's cost includes **the failed cheap attempt**, and its pass rate tells "
        "you whether it is worth having.",
        "**Identity comes from a credential**, scopes are enforced per route, the audit "
        "log holds no raw user text, and errors do not leak.",
    ],
    next_up="Chapter 17 is the last one: tracing, cost per success, and a circuit breaker "
            "that can close again.",
)
