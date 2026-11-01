"""Chapter 15 — Safety: a tool gateway, tamper-proof approval, cost controls."""

from lite import Exercise, make_chapter

GIVEN = r'''
import hashlib, json, re, secrets, time

# ---- The tool registry. RISK LIVES HERE, not in the call. -------------------
# This is the whole design: the caller says what it wants to do, and the registry
# says how dangerous that is. A caller that could declare its own risk level would
# simply declare everything "low".
REGISTRY = {
    "search":        {"risk": "low",    "scopes": {"read"}},
    "read_file":     {"risk": "low",    "scopes": {"read"}},
    "send_email":    {"risk": "high",   "scopes": {"write", "email"}},
    "make_payment":  {"risk": "critical", "scopes": {"write", "money"}},
    "delete_file":   {"risk": "high",   "scopes": {"write"}},
}

SCHEMAS = {
    "search":       {"query": str},
    "read_file":    {"path": str},
    "send_email":   {"to": str, "subject": str, "body": str},
    "make_payment": {"amount": float, "currency": str, "to_account": str},
    "delete_file":  {"path": str},
}

def search(query): return f"3 results for {query!r}"
def read_file(path): return f"contents of {path}"
def send_email(to, subject, body): return f"email sent to {to}"
def make_payment(amount, currency, to_account): return f"paid {amount} {currency}"
def delete_file(path): return f"deleted {path}"

TOOLS = {"search": search, "read_file": read_file, "send_email": send_email,
         "make_payment": make_payment, "delete_file": delete_file}

print("tools:", ", ".join(f"{n} ({d['risk']})" for n, d in REGISTRY.items()))
'''

EX1 = Exercise(
    name="A gateway the model cannot talk its way past",
    section="15.2",
    concept=r"""
Here is the question to ask of every safety control you build:

> **If the model is completely fooled, does this control still hold?**

A prompt that says "never send email to external addresses" does not hold — it is text,
and the model can be talked out of text. A function that checks the recipient **in code**
before calling the email API does hold, for everything it covers.

That function is a **gateway**: one door that every tool call goes through. Four rules, each
from a real failure:

1. **Check arguments in code.** Types, required fields, limits. Not in the prompt.
2. **Take risk from the registry, not from the call.** If the caller declares its own risk
   level, it will declare `"low"`.
3. **Treat a limit as a refusal.** Quietly clamping a £10,000 payment to £1,000 means
   *making a payment nobody authorised*. Refuse it.
4. **Fail closed.** If the check itself raises, the answer is no. A validator that throws an
   exception you do not handle is not a control — it is a crash with a tool call behind it.
""",
    todo=r'''
LIMITS = {"make_payment": {"amount": 1000.0}}
ALLOWED_RISK = {"low", "medium"}        # anything higher needs approval (Exercise 2)


class Gateway:
    """Every tool call goes through here. Nothing else may touch TOOLS."""

    def __init__(self, scopes, allowed_risk=ALLOWED_RISK):
        self.scopes = set(scopes)       # what this caller is permitted to do
        self.allowed_risk = allowed_risk
        self.denials = []

    def deny(self, reason):
        self.denials.append(reason)
        return {"ok": False, "error": reason, "result": None}

    def call(self, name, arguments):
        """Returns {"ok", "error", "result"}. Never raises."""
        try:
            # TODO 1: unknown tool -> self.deny(f"unknown tool: {name!r}")
            # TODO 2: scopes: every scope in REGISTRY[name]["scopes"] must be in
            #         self.scopes, or deny with a reason naming what is missing.
            # TODO 3: risk comes from REGISTRY[name]["risk"] — NOT from arguments.
            #         If it is not in self.allowed_risk, deny
            #         f"{name} is {risk} risk and needs approval".
            # TODO 4: arguments must match SCHEMAS[name]: every field present, and
            #         isinstance(value, expected_type). Deny on a missing or wrongly
            #         typed field. (Accept an int where a float is wanted.)
            # TODO 5: limits: for each limit in LIMITS.get(name, {}), if the argument
            #         exceeds it, DENY — do not clamp the value and proceed.
            # TODO 6: all clear -> run TOOLS[name](**arguments) and return
            #         {"ok": True, "error": None, "result": ...}
            raise NotImplementedError
        except Exception as err:
            # TODO 7: fail closed. Any unexpected error becomes a denial, never a
            #         call that goes ahead.
            raise NotImplementedError


gateway = Gateway(scopes={"read", "write", "email", "money"})

attempts = [
    ("search", {"query": "vector databases"}),
    ("read_file", {"path": "notes.txt"}),
    ("send_email", {"to": "a@b.com", "subject": "hi", "body": "hello"}),
    ("make_payment", {"amount": 50.0, "currency": "GBP", "to_account": "123"}),
    ("search", {}),                                       # missing argument
    ("search", {"query": 42}),                            # wrong type
    ("launch_rocket", {}),                                # not a tool
]

for name, arguments in attempts:
    outcome = gateway.call(name, arguments)
    status = "OK " if outcome["ok"] else "NO "
    print(f"{status} {name:>14}  {outcome['result'] or outcome['error']}")
''',
    solution=r'''
LIMITS = {"make_payment": {"amount": 1000.0}}
ALLOWED_RISK = {"low", "medium"}        # anything higher needs approval (Exercise 2)


class Gateway:
    """Every tool call goes through here. Nothing else may touch TOOLS."""

    def __init__(self, scopes, allowed_risk=ALLOWED_RISK):
        self.scopes = set(scopes)       # what this caller is permitted to do
        self.allowed_risk = allowed_risk
        self.denials = []

    def deny(self, reason):
        self.denials.append(reason)
        return {"ok": False, "error": reason, "result": None}

    def call(self, name, arguments):
        """Returns {"ok", "error", "result"}. Never raises."""
        try:
            if name not in REGISTRY or name not in TOOLS:
                return self.deny(f"unknown tool: {name!r}")

            entry = REGISTRY[name]

            missing_scopes = entry["scopes"] - self.scopes
            if missing_scopes:
                return self.deny(f"{name} needs scope(s) "
                                 f"{', '.join(sorted(missing_scopes))}")

            risk = entry["risk"]                   # from the registry, never the call
            if risk not in self.allowed_risk:
                return self.deny(f"{name} is {risk} risk and needs approval")

            schema = SCHEMAS[name]
            for field, expected in schema.items():
                if field not in arguments:
                    return self.deny(f"{name} is missing required argument {field!r}")
                value = arguments[field]
                if expected is float and isinstance(value, int) and not isinstance(value, bool):
                    continue
                if not isinstance(value, expected):
                    return self.deny(f"{name}: {field!r} must be "
                                     f"{expected.__name__}, got "
                                     f"{type(value).__name__}")

            for field, maximum in LIMITS.get(name, {}).items():
                if field in arguments and arguments[field] > maximum:
                    # Refuse, do not clamp: a clamped payment is a payment nobody
                    # authorised.
                    return self.deny(f"{name}: {field} {arguments[field]} exceeds the "
                                     f"limit of {maximum}")

            return {"ok": True, "error": None, "result": TOOLS[name](**arguments)}

        except Exception as err:
            # Fail closed: if the checking itself goes wrong, the answer is no.
            return self.deny(f"{name} refused after an internal error: "
                             f"{type(err).__name__}")


gateway = Gateway(scopes={"read", "write", "email", "money"})

attempts = [
    ("search", {"query": "vector databases"}),
    ("read_file", {"path": "notes.txt"}),
    ("send_email", {"to": "a@b.com", "subject": "hi", "body": "hello"}),
    ("make_payment", {"amount": 50.0, "currency": "GBP", "to_account": "123"}),
    ("search", {}),                                       # missing argument
    ("search", {"query": 42}),                            # wrong type
    ("launch_rocket", {}),                                # not a tool
]

for name, arguments in attempts:
    outcome = gateway.call(name, arguments)
    status = "OK " if outcome["ok"] else "NO "
    print(f"{status} {name:>14}  {outcome['result'] or outcome['error']}")
''',
    check=r'''
@check("15.2  arguments are checked in code")
def _():
    need("Gateway")
    g = Gateway(scopes={"read", "write", "email"})
    assert g.call("search", {"query": "x"})["ok"] is True, \
        "a valid low-risk call should go through"
    assert g.call("search", {})["ok"] is False, \
        "a missing required argument must be refused"
    assert g.call("search", {"query": 42})["ok"] is False, (
        "query must be a string. Checking types in code is the point — a prompt asking "
        "the model nicely for a string is not a control."
    )
    assert g.call("launch_rocket", {})["ok"] is False, \
        "a tool that does not exist must be refused"
    bad = g.call("search", {"query": 42})
    assert "str" in bad["error"], \
        f"say what was expected, so the model can correct itself: {bad['error']!r}"


@check("15.2  risk and scope come from the registry")
def _():
    need("Gateway")
    read_only = Gateway(scopes={"read"})
    denied = read_only.call("send_email", {"to": "a@b.com", "subject": "s", "body": "b"})
    assert denied["ok"] is False, "a read-only caller cannot send email"
    assert "scope" in denied["error"], f"explain which scope is missing: {denied['error']}"

    full = Gateway(scopes={"read", "write", "email", "money"})
    payment = full.call("make_payment",
                        {"amount": 10.0, "currency": "GBP", "to_account": "1"})
    assert payment["ok"] is False and "critical" in payment["error"], (
        "make_payment is critical risk in the registry, so it needs approval even for "
        f"a caller with every scope; got {payment}"
    )

    lying = full.call("make_payment", {"amount": 10.0, "currency": "GBP",
                                       "to_account": "1", "risk": "low"})
    assert lying["ok"] is False, (
        "the call tried to declare its own risk level and must not be believed. Risk "
        "is read from REGISTRY, never from the arguments."
    )


@check("15.2  a limit is a refusal, not a quiet adjustment")
def _():
    need("Gateway")
    g = Gateway(scopes={"read", "write", "money"},
                allowed_risk={"low", "medium", "critical"})

    ok = g.call("make_payment", {"amount": 50.0, "currency": "GBP", "to_account": "1"})
    assert ok["ok"] is True, f"a payment inside the limit should go through: {ok}"

    over = g.call("make_payment",
                  {"amount": 10000.0, "currency": "GBP", "to_account": "1"})
    assert over["ok"] is False, (
        "a payment over the limit must be REFUSED. Clamping it to 1000 and proceeding "
        "means making a payment that nobody authorised — the worst of both options."
    )
    assert "limit" in over["error"], f"say why: {over['error']!r}"
    assert over["result"] is None, "and nothing should have been paid"


@check("15.2  the gateway fails closed")
def _():
    need("Gateway")
    g = Gateway(scopes={"read"})
    original = TOOLS["search"]
    try:
        TOOLS["search"] = lambda query: (_ for _ in ()).throw(
            RuntimeError("the tool itself exploded"))
        outcome = g.call("search", {"query": "x"})
        assert isinstance(outcome, dict), (
            "the gateway raised instead of returning. A validator that throws an "
            "exception nobody handles is not a control — it is a crash."
        )
        assert outcome["ok"] is False, "an unexpected error must become a denial"
    finally:
        TOOLS["search"] = original

    broken = Gateway(scopes={"read"})
    broken.scopes = None            # something upstream went wrong
    outcome = broken.call("search", {"query": "x"})
    assert outcome["ok"] is False, \
        "with its own configuration broken, the gateway must refuse, not proceed"
    print("     bad arguments, wrong scope, excess amount and internal errors all denied")
''',
    hint=[
        "The order of the checks: exists -> scopes -> risk -> schema -> limits -> run.",
        "",
        "  missing_scopes = REGISTRY[name]['scopes'] - self.scopes",
        "  if missing_scopes: return self.deny(...)",
        "",
        "  risk = REGISTRY[name]['risk']        # never arguments.get('risk')",
        "  if risk not in self.allowed_risk: return self.deny(...)",
        "",
        "  for field, expected in SCHEMAS[name].items():",
        "      if field not in arguments: return self.deny(...)",
        "      if not isinstance(arguments[field], expected): return self.deny(...)",
        "",
        "  for field, maximum in LIMITS.get(name, {}).items():",
        "      if field in arguments and arguments[field] > maximum:",
        "          return self.deny(...)",
        "",
        "And wrap the lot in try/except, returning a denial from the except.",
    ],
    notice=r"""
**What to notice.** The call that tried to declare `"risk": "low"` for a payment was ignored,
because risk is not something a caller gets to assert. That pattern — **authority comes from
your configuration, not from the request** — is the most reusable idea in the chapter.

And "fail closed" is tested here with the gateway's own configuration broken. Controls fail
too. The question is which way they fail.
""",
)

EX2 = Exercise(
    name="An approval that cannot be reused or edited",
    section="15.4",
    concept=r"""
Some actions should wait for a person. But "a human approved it" is only worth something if
the approval is tied to **exactly** what gets done. Three bindings, and each one closes a
real attack:

| Binding | Without it |
|---|---|
| **to the exact parameters** | approve a £50 payment, execute a £50,000 one |
| **to a person who is not the requester** | the agent approves its own request |
| **to a single use** | one approval replayed for a hundred payments |

The parameter binding is a **hash**: take a copy of the arguments, hash it, and store the
hash with the approval. At execution, hash the arguments again and compare. Change one
character and the hashes differ.

> Approving without reading is the failure that remains, and no code fixes that. Display the
> values clearly, and never let the requester be the approver.
""",
    todo=r'''
def fingerprint(name, arguments):
    """A stable hash of a tool call. Given to you.

    sort_keys matters: {"a": 1, "b": 2} and {"b": 2, "a": 1} are the same call and
    must produce the same hash.
    """
    payload = json.dumps({"tool": name, "arguments": arguments}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class ApprovalStore:
    def __init__(self):
        self.requests = {}      # token -> record

    def request(self, name, arguments, requested_by):
        """Record a pending approval and return its token."""
        token = secrets.token_urlsafe(16)
        self.requests[token] = {
            "tool": name,
            "arguments": dict(arguments),           # a COPY, so later edits do not show
            "fingerprint": fingerprint(name, arguments),
            "requested_by": requested_by,
            "approved_by": None,
            "used": False,
        }
        return token

    def approve(self, token, approver):
        """A person approves. Raises on a bad approval."""
        record = self.requests.get(token)
        if record is None:
            raise KeyError("no such approval request")

        # TODO 1: the approver must not be the requester. Raise
        #         PermissionError("the requester cannot approve their own request")
        # TODO 2: an already-used approval cannot be approved again ->
        #         PermissionError("this approval has already been used")
        # TODO 3: record the approver and return the record.
        raise NotImplementedError

    def execute(self, token, name, arguments, gateway):
        """Run the approved call — if everything still matches."""
        record = self.requests.get(token)
        if record is None:
            return {"ok": False, "error": "no such approval", "result": None}

        # TODO 4: refuse unless the request has been approved by someone.
        # TODO 5: refuse if it has already been used (single use).
        # TODO 6: refuse unless fingerprint(name, arguments) equals the stored
        #         fingerprint. This is the binding that stops an approved 50 becoming
        #         50000 on the way to execution.
        # TODO 7: mark it used, then run the call through the gateway with the
        #         critical risk allowed, and return the gateway's outcome.
        raise NotImplementedError


store = ApprovalStore()
payment = {"amount": 50.0, "currency": "GBP", "to_account": "12345"}
token = store.request("make_payment", payment, requested_by="agent")

print("the agent tries to approve its own request:")
try:
    store.approve(token, approver="agent")
except PermissionError as err:
    print("  refused:", err)

store.approve(token, approver="alice")
print("\napproved by alice. Now the agent edits the amount:")
tampered = dict(payment, amount=50000.0)
print(" ", store.execute(token, "make_payment", tampered, gateway)["error"])

print("\nexecuting exactly what was approved:")
print(" ", store.execute(token, "make_payment", payment, gateway))

print("\nreplaying the same approval:")
print(" ", store.execute(token, "make_payment", payment, gateway)["error"])
''',
    solution=r'''
def fingerprint(name, arguments):
    """A stable hash of a tool call. Given to you.

    sort_keys matters: {"a": 1, "b": 2} and {"b": 2, "a": 1} are the same call and
    must produce the same hash.
    """
    payload = json.dumps({"tool": name, "arguments": arguments}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class ApprovalStore:
    def __init__(self):
        self.requests = {}      # token -> record

    def request(self, name, arguments, requested_by):
        """Record a pending approval and return its token."""
        token = secrets.token_urlsafe(16)
        self.requests[token] = {
            "tool": name,
            "arguments": dict(arguments),           # a COPY, so later edits do not show
            "fingerprint": fingerprint(name, arguments),
            "requested_by": requested_by,
            "approved_by": None,
            "used": False,
        }
        return token

    def approve(self, token, approver):
        """A person approves. Raises on a bad approval."""
        record = self.requests.get(token)
        if record is None:
            raise KeyError("no such approval request")

        if approver == record["requested_by"]:
            raise PermissionError("the requester cannot approve their own request")

        if record["used"]:
            raise PermissionError("this approval has already been used")

        record["approved_by"] = approver
        return record

    def execute(self, token, name, arguments, gateway):
        """Run the approved call — if everything still matches."""
        record = self.requests.get(token)
        if record is None:
            return {"ok": False, "error": "no such approval", "result": None}

        if record["approved_by"] is None:
            return {"ok": False, "error": "not approved", "result": None}

        if record["used"]:
            return {"ok": False, "error": "approval already used", "result": None}

        if fingerprint(name, arguments) != record["fingerprint"]:
            return {"ok": False,
                    "error": "arguments do not match what was approved",
                    "result": None}

        record["used"] = True
        allowed = Gateway(scopes=gateway.scopes,
                          allowed_risk=gateway.allowed_risk | {"high", "critical"})
        return allowed.call(name, arguments)


store = ApprovalStore()
payment = {"amount": 50.0, "currency": "GBP", "to_account": "12345"}
token = store.request("make_payment", payment, requested_by="agent")

print("the agent tries to approve its own request:")
try:
    store.approve(token, approver="agent")
except PermissionError as err:
    print("  refused:", err)

store.approve(token, approver="alice")
print("\napproved by alice. Now the agent edits the amount:")
tampered = dict(payment, amount=50000.0)
print(" ", store.execute(token, "make_payment", tampered, gateway)["error"])

print("\nexecuting exactly what was approved:")
print(" ", store.execute(token, "make_payment", payment, gateway))

print("\nreplaying the same approval:")
print(" ", store.execute(token, "make_payment", payment, gateway)["error"])
''',
    check=r'''
@check("15.4  the requester cannot approve their own request")
def _():
    need("ApprovalStore")
    s = ApprovalStore()
    token = s.request("send_email", {"to": "a@b.com", "subject": "s", "body": "b"},
                      requested_by="agent")
    try:
        s.approve(token, approver="agent")
    except PermissionError:
        pass
    else:
        raise AssertionError(
            "the agent approved its own request. An approval by the requester is not "
            "an approval — it is a formality the agent can perform on demand."
        )
    record = s.approve(token, approver="alice")
    assert record["approved_by"] == "alice", "a different person may approve"


@check("15.4  the approval is bound to the exact parameters")
def _():
    need("ApprovalStore", "gateway")
    s = ApprovalStore()
    payment = {"amount": 50.0, "currency": "GBP", "to_account": "12345"}
    token = s.request("make_payment", payment, requested_by="agent")
    s.approve(token, approver="alice")

    for tampered in (dict(payment, amount=50000.0),
                     dict(payment, to_account="99999"),
                     dict(payment, currency="USD")):
        outcome = s.execute(token, "make_payment", tampered, gateway)
        assert outcome["ok"] is False, (
            f"executing {tampered} went through on an approval for {payment}. Compare "
            "the fingerprint of what is being executed with the one stored at request "
            "time — this is the attack the hash exists to stop."
        )
        assert "match" in outcome["error"], f"explain why: {outcome['error']!r}"

    good = s.execute(token, "make_payment", payment, gateway)
    assert good["ok"] is True, \
        f"the exact approved call must still work: {good}"
    assert "50" in str(good["result"]), f"and actually run: {good['result']!r}"


@check("15.4  an approval is good for exactly one use")
def _():
    need("ApprovalStore", "gateway")
    s = ApprovalStore()
    payment = {"amount": 20.0, "currency": "GBP", "to_account": "1"}
    token = s.request("make_payment", payment, requested_by="agent")
    s.approve(token, approver="bob")

    assert s.execute(token, "make_payment", payment, gateway)["ok"] is True, \
        "the first use should succeed"
    second = s.execute(token, "make_payment", payment, gateway)
    assert second["ok"] is False, (
        "the same approval ran twice. Mark it used BEFORE executing — otherwise one "
        "approval of a 20 GBP payment authorises an unlimited number of them."
    )
    assert "used" in second["error"], f"say so: {second['error']!r}"

    unapproved = ApprovalStore()
    t2 = unapproved.request("make_payment", payment, requested_by="agent")
    assert unapproved.execute(t2, "make_payment", payment, gateway)["ok"] is False, \
        "a request nobody approved must not execute"
    print("     self-approval, tampering and replay all refused")
''',
    hint=[
        "approve:",
        "  if approver == record['requested_by']:",
        "      raise PermissionError('the requester cannot approve their own request')",
        "  if record['used']:",
        "      raise PermissionError('this approval has already been used')",
        "  record['approved_by'] = approver",
        "  return record",
        "",
        "execute, in this order:",
        "  not approved        -> refuse",
        "  already used        -> refuse",
        "  fingerprint differs -> refuse",
        "  then record['used'] = True BEFORE running the call,",
        "  and run it through a Gateway whose allowed_risk includes 'critical'.",
    ],
    notice=r"""
**What to notice.** `record["used"] = True` goes *before* the call, not after. Set it
afterwards and a call that raises half-way leaves the approval usable again — and now you
have an approval that may already have had an effect.

The three bindings together are what makes a human approval mean something. Without them,
"a human approved it" is a sentence in a log file.
""",
)

EX3 = Exercise(
    name="Spending money you have reserved",
    section="15.5",
    concept=r"""
An agent in a loop spends money. The naive budget check is:

```python
if spent + estimate > budget:      # check
    refuse()
result = call_model()              # ...then spend
spent += actual_cost
```

It has a hole, and the hole is the gap between the two lines. Ten requests arrive at once,
all ten check against the same `spent`, all ten pass, and all ten spend. The budget is
exceeded by a factor of ten and every individual check was correct.

The fix is to **reserve first**: deduct the estimate *as you check*, in one step, so the
money is gone from the budget before the call starts. Then settle up afterwards with the
real cost.

Two details worth having:

* **Estimate from characters, not words.** Roughly four characters per token. Counting
  words underestimates code, non-English text and anything with punctuation.
* **Count everything**, including failed calls. A call that errors after generating 500
  tokens still cost 500 tokens.
""",
    todo=r'''
PRICE_PER_1K = {"small": 0.0002, "large": 0.003}      # placeholder prices


def estimate_tokens(text):
    """About 4 characters per token. Given to you."""
    return max(1, len(text) // 4)


class Budget:
    def __init__(self, limit_usd):
        self.limit = limit_usd
        self.reserved = 0.0         # committed but not yet settled
        self.spent = 0.0            # actually spent
        self.refusals = 0

    @property
    def available(self):
        """What is left to commit. Given to you."""
        return self.limit - self.spent - self.reserved

    def reserve(self, prompt, model, max_output_tokens=500):
        """Check and commit in ONE step. Returns a reservation, or None if refused."""
        cost = ((estimate_tokens(prompt) + max_output_tokens) / 1000) * PRICE_PER_1K[model]

        # TODO 1: if cost is more than self.available, count a refusal and return None.
        # TODO 2: otherwise add cost to self.reserved and return a reservation dict:
        #         {"model": model, "estimate": cost}. Reserving as part of the check is
        #         what closes the gap: the money is gone before the call starts.
        raise NotImplementedError

    def settle(self, reservation, prompt_tokens, completion_tokens, failed=False):
        """Release the reservation and record what was really spent."""
        actual = ((prompt_tokens + completion_tokens) / 1000) \
            * PRICE_PER_1K[reservation["model"]]

        # TODO 3: subtract the estimate from self.reserved (it is no longer pending).
        # TODO 4: add `actual` to self.spent — INCLUDING when failed is True. A call
        #         that errored after generating tokens still cost money.
        # TODO 5: return actual.
        raise NotImplementedError


budget = Budget(limit_usd=0.01)
prompt = "Summarise this document. " * 20

print(f"budget ${budget.limit}, available ${budget.available:.5f}\n")

# Ten requests arrive at once. All ten check, then all ten spend.
reservations = [budget.reserve(prompt, "large") for _ in range(10)]
granted = [r for r in reservations if r]
print(f"granted {len(granted)} of 10 reservations, refused {budget.refusals}")
print(f"reserved ${budget.reserved:.5f}, available ${budget.available:.5f}")

for reservation in granted:
    budget.settle(reservation, prompt_tokens=120, completion_tokens=200)
print(f"\nafter settling: spent ${budget.spent:.5f} of ${budget.limit} "
      f"(reserved ${budget.reserved:.5f})")
''',
    solution=r'''
PRICE_PER_1K = {"small": 0.0002, "large": 0.003}      # placeholder prices


def estimate_tokens(text):
    """About 4 characters per token. Given to you."""
    return max(1, len(text) // 4)


class Budget:
    def __init__(self, limit_usd):
        self.limit = limit_usd
        self.reserved = 0.0         # committed but not yet settled
        self.spent = 0.0            # actually spent
        self.refusals = 0

    @property
    def available(self):
        """What is left to commit. Given to you."""
        return self.limit - self.spent - self.reserved

    def reserve(self, prompt, model, max_output_tokens=500):
        """Check and commit in ONE step. Returns a reservation, or None if refused."""
        cost = ((estimate_tokens(prompt) + max_output_tokens) / 1000) * PRICE_PER_1K[model]

        if cost > self.available:
            self.refusals += 1
            return None

        self.reserved += cost          # committed now, not after the call returns
        return {"model": model, "estimate": cost}

    def settle(self, reservation, prompt_tokens, completion_tokens, failed=False):
        """Release the reservation and record what was really spent."""
        actual = ((prompt_tokens + completion_tokens) / 1000) \
            * PRICE_PER_1K[reservation["model"]]

        self.reserved -= reservation["estimate"]
        # Failed calls still generated tokens, and those tokens were billed.
        self.spent += actual
        return actual


budget = Budget(limit_usd=0.01)
prompt = "Summarise this document. " * 20

print(f"budget ${budget.limit}, available ${budget.available:.5f}\n")

# Ten requests arrive at once. All ten check, then all ten spend.
reservations = [budget.reserve(prompt, "large") for _ in range(10)]
granted = [r for r in reservations if r]
print(f"granted {len(granted)} of 10 reservations, refused {budget.refusals}")
print(f"reserved ${budget.reserved:.5f}, available ${budget.available:.5f}")

for reservation in granted:
    budget.settle(reservation, prompt_tokens=120, completion_tokens=200)
print(f"\nafter settling: spent ${budget.spent:.5f} of ${budget.limit} "
      f"(reserved ${budget.reserved:.5f})")
''',
    check=r'''
@check("15.5  reserving and checking are one step")
def _():
    need("Budget")
    b = Budget(limit_usd=0.01)
    prompt = "x" * 400

    first = b.reserve(prompt, "large")
    assert first is not None, "the first reservation should be granted"
    assert b.reserved > 0, (
        "nothing was reserved. Deduct the estimate as part of the check — if reserving "
        "waits until after the call, concurrent requests all pass the same check."
    )
    assert b.available < b.limit, "available must fall as soon as money is committed"

    granted = [r for r in (b.reserve(prompt, "large") for _ in range(20)) if r]
    total_committed = sum(r["estimate"] for r in granted) + first["estimate"]
    assert total_committed <= b.limit + 1e-12, (
        f"${total_committed:.5f} was committed against a ${b.limit} budget. Every "
        "reservation must see the previous ones."
    )
    assert b.refusals > 0, "once the budget is committed, further requests are refused"


@check("15.5  settling records the real cost, failures included")
def _():
    need("Budget")
    b = Budget(limit_usd=1.0)
    reservation = b.reserve("x" * 400, "small", max_output_tokens=500)
    before_reserved = b.reserved

    actual = b.settle(reservation, prompt_tokens=100, completion_tokens=50)
    assert abs(b.reserved) < 1e-12, (
        f"reserved is {b.reserved:.6f} after settling. Release the reservation, or the "
        "budget leaks: money stays committed for calls that already finished."
    )
    assert abs(b.spent - actual) < 1e-12, "spent should be the actual cost"
    assert actual < reservation["estimate"], \
        "the call used fewer tokens than reserved, so it should cost less"

    failed = b.reserve("x" * 400, "small")
    cost = b.settle(failed, prompt_tokens=100, completion_tokens=500, failed=True)
    assert cost > 0 and b.spent > actual, (
        "a failed call was recorded as free. It generated 500 completion tokens before "
        "failing, and those were billed — count everything."
    )


@check("15.5  the limit actually holds")
def _():
    need("Budget")
    b = Budget(limit_usd=0.005)
    prompt = "x" * 2000

    for _ in range(50):
        reservation = b.reserve(prompt, "large")
        if reservation is None:
            continue
        b.settle(reservation, prompt_tokens=500, completion_tokens=500)

    assert b.spent <= b.limit + 1e-9, (
        f"${b.spent:.5f} was spent against a ${b.limit} limit. The reservation has to "
        "be made before the call, and settled with the real cost after."
    )
    assert b.refusals > 0, "some requests should have been refused"
    assert b.spent > 0, "and some should have gone through"
    print(f"     spent ${b.spent:.5f} of ${b.limit}, refused {b.refusals} requests")
''',
    hint=[
        "reserve:",
        "  if cost > self.available:",
        "      self.refusals += 1",
        "      return None",
        "  self.reserved += cost",
        "  return {'model': model, 'estimate': cost}",
        "",
        "settle:",
        "  self.reserved -= reservation['estimate']",
        "  self.spent += actual",
        "  return actual",
        "",
        "Note what `available` already does for you: limit - spent - reserved. That is",
        "why reserving inside the check closes the gap.",
    ],
    notice=r"""
**What to notice.** Ten simultaneous requests, one budget, and the limit held — because the
money was committed during the check rather than after the call.

This is the same shape of bug as the permission leak in Chapter 9 and the unlabelled partial
result in Chapter 13: the code is individually correct at every line, and wrong as a
sequence. Those are the bugs worth learning to see.

> Reserve before you spend, count everything, estimate from characters, and make the check
> and the commitment one step.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=15,
    title="Safety: Implementing Guardrails",
    intro=r"""
One question organises this whole notebook:

> **If the model is completely fooled, does this control still hold?**

A prompt does not. A filter mostly does not. Code that checks arguments, an approval bound
to a hash, and a budget that reserves before it spends — those do. You will write all three.
""",
    minutes=40,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "Ask of every control: **if the model is fully fooled, does this still hold?** "
        "Prompts and filters do not; code does.",
        "A **gateway** is one door every tool call passes through: arguments checked in "
        "code, risk read from the registry, never from the call.",
        "**A limit is a refusal**, not a quiet adjustment — a clamped payment is a payment "
        "nobody authorised.",
        "**Fail closed.** A validator that raises an unhandled exception is not a control.",
        "A human approval needs three bindings: to the **exact parameters** (a hash), to "
        "**someone other than the requester**, and to a **single use**.",
        "**Reserve before you spend.** Check and commit in one step, and count failed "
        "calls — they were billed too.",
    ],
    next_up="Chapter 16 builds three real applications, and measures them on data they "
            "were not built from.",
)
