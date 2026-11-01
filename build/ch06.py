"""Chapter 6 — Memory: working memory, what reaches the prompt, meaning-based recall."""

from lite import Exercise, make_chapter

GIVEN = r'''
import re, math, json
from collections import Counter

def estimate_tokens(text):
    """Rough estimate: about 4 characters per token. Good enough for budgeting."""
    return max(1, len(text) // 4)


# ---- A stand-in embedder, so meaning-based search works with no API key -----
# A real embedder is a neural network. This one is a bag of words with a small
# hand-made thesaurus, which is enough to show what embeddings buy you.
THESAURUS = {
    "dog": {"dog", "puppy", "retriever", "pet", "animal"},
    "pet": {"pet", "dog", "cat", "animal", "retriever", "puppy"},
    "called": {"called", "name", "named", "names"},
    "name": {"name", "called", "named", "names"},
    "food": {"food", "eat", "eats", "meal", "diet", "allergy", "allergic"},
    "allergic": {"allergic", "allergy", "food", "eat", "peanuts", "nuts"},
    "allergy": {"allergy", "allergic", "food", "eat", "peanuts", "nuts"},
    "eat": {"eat", "food", "allergy", "allergic", "peanuts", "meal", "diet"},
    "peanuts": {"peanuts", "nuts", "food", "allergy", "allergic", "eat"},
    "retriever": {"retriever", "dog", "pet", "puppy", "animal"},
    "named": {"named", "name", "called", "names"},
    "city": {"city", "live", "lives", "living", "based", "london", "tokyo"},
    "live": {"live", "lives", "living", "city", "based", "home"},
    "work": {"work", "works", "job", "engineer", "developer", "career"},
    "job": {"job", "work", "works", "engineer", "developer", "career"},
}

# Words that carry no meaning of their own. A real embedder learns to downweight
# these; here we simply drop them, so "what is my dog called" is about dogs.
STOPWORDS = {"a", "an", "the", "is", "are", "am", "i", "my", "me", "what", "whats",
             "s", "do", "does", "there", "anything", "should", "not", "of", "to",
             "in", "on", "and", "any", "it", "be", "you", "for", "by", "way"}

def _expand(word):
    return THESAURUS.get(word, {word})

def embed(text):
    """Turn text into a vector: a count of every meaningful word, plus related words."""
    bag = Counter()
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in STOPWORDS:
            continue
        for related in _expand(word):
            bag[related] += 1
    return bag

def distance(a, b):
    """0 means identical meaning, 1 means nothing in common (cosine distance)."""
    shared = sum(a[k] * b[k] for k in a if k in b)
    size_a = math.sqrt(sum(v * v for v in a.values()))
    size_b = math.sqrt(sum(v * v for v in b.values()))
    if not size_a or not size_b:
        return 1.0
    return 1 - shared / (size_a * size_b)


# ---- The conversation we will remember things from --------------------------
CONVERSATION = [
    ("user", "Hi, I'm Sam. I live in London."),
    ("assistant", "Nice to meet you, Sam."),
    ("user", "I have a golden retriever named Max."),
    ("assistant", "Max sounds lovely."),
    ("user", "I'm allergic to peanuts, by the way."),
    ("assistant", "Noted — I'll keep that in mind."),
]

print("embedder ready. distance('my dog', 'golden retriever named Max') =",
      round(distance(embed("my dog"), embed("golden retriever named Max")), 2))
'''

EX1 = Exercise(
    name="Two kinds of memory",
    section="6.1",
    concept=r"""
A model remembers **nothing** between calls. Every conversation starts like meeting someone
with amnesia. So "memory" always means the same three jobs: save something outside the
model, find the right piece later, put it back in the prompt.

The book describes four kinds. Two of them do most of the work:

| Kind | Human parallel | In an agent | Lifetime |
|---|---|---|---|
| **Working** | what you are holding in mind right now | the recent conversation turns | one conversation |
| **Semantic** | facts you simply know | facts about the user, stored as key/value | forever |

Working memory is a **sliding window** — keep the last few turns, drop the rest.
Semantic memory is a **dictionary** that outlives the conversation.

And one rule that is not optional: **never store secrets.** An agent that helpfully
remembers `password: hunter2` has turned a convenience into a breach. Refuse at the point
of writing, because once it is saved it will end up in a prompt.
""",
    todo=r'''
SECRET_WORDS = ("password", "passwd", "secret", "token", "api_key", "apikey", "pin", "ssn")
SECRET_VALUES = [re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),      # API-key look-alikes
                 re.compile(r"\b(?:\d[ -]?){13,19}\b")]       # card-number look-alikes


class Memory:
    def __init__(self, window=4):
        self.window = window      # how many recent turns working memory keeps
        self.turns = []           # working memory
        self.facts = {}           # semantic memory

    def add_turn(self, role, content):
        """Working memory: keep only the most recent `window` turns."""
        # TODO 1: append {"role": role, "content": content} to self.turns, then keep
        #         only the last self.window entries (a list slice does this).
        raise NotImplementedError

    def learn_fact(self, key, value):
        """Semantic memory: store a fact, unless it looks like a secret."""
        # TODO 2: if any word in SECRET_WORDS appears in key.lower(), raise
        #         ValueError(f"'{key}' looks like a secret and was not stored")
        # TODO 3: if any pattern in SECRET_VALUES matches str(value), raise
        #         ValueError("that value looks like a key or card number; not stored")
        # TODO 4: otherwise save it: self.facts[key] = value
        raise NotImplementedError


memory = Memory(window=4)
for role, content in CONVERSATION:
    memory.add_turn(role, content)

memory.learn_fact("name", "Sam")
memory.learn_fact("city", "London")
memory.learn_fact("pet", "a golden retriever named Max")
memory.learn_fact("food_allergy", "peanuts")

print(f"working memory holds {len(memory.turns)} of {len(CONVERSATION)} turns:")
for t in memory.turns:
    print(f"  {t['role']:>9}: {t['content']}")
print("\nfacts:", memory.facts)

for bad_key, bad_value in [("password", "hunter2"), ("note", "my card is 4111 1111 1111 1111")]:
    try:
        memory.learn_fact(bad_key, bad_value)
        print(f"\n!! stored {bad_key} — that should not have happened")
    except ValueError as err:
        print(f"\nrefused: {err}")
''',
    solution=r'''
SECRET_WORDS = ("password", "passwd", "secret", "token", "api_key", "apikey", "pin", "ssn")
SECRET_VALUES = [re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),      # API-key look-alikes
                 re.compile(r"\b(?:\d[ -]?){13,19}\b")]       # card-number look-alikes


class Memory:
    def __init__(self, window=4):
        self.window = window      # how many recent turns working memory keeps
        self.turns = []           # working memory
        self.facts = {}           # semantic memory

    def add_turn(self, role, content):
        """Working memory: keep only the most recent `window` turns."""
        self.turns.append({"role": role, "content": content})
        self.turns = self.turns[-self.window:]

    def learn_fact(self, key, value):
        """Semantic memory: store a fact, unless it looks like a secret."""
        if any(word in key.lower() for word in SECRET_WORDS):
            raise ValueError(f"'{key}' looks like a secret and was not stored")
        if any(pattern.search(str(value)) for pattern in SECRET_VALUES):
            raise ValueError("that value looks like a key or card number; not stored")
        self.facts[key] = value


memory = Memory(window=4)
for role, content in CONVERSATION:
    memory.add_turn(role, content)

memory.learn_fact("name", "Sam")
memory.learn_fact("city", "London")
memory.learn_fact("pet", "a golden retriever named Max")
memory.learn_fact("food_allergy", "peanuts")

print(f"working memory holds {len(memory.turns)} of {len(CONVERSATION)} turns:")
for t in memory.turns:
    print(f"  {t['role']:>9}: {t['content']}")
print("\nfacts:", memory.facts)

for bad_key, bad_value in [("password", "hunter2"), ("note", "my card is 4111 1111 1111 1111")]:
    try:
        memory.learn_fact(bad_key, bad_value)
        print(f"\n!! stored {bad_key} — that should not have happened")
    except ValueError as err:
        print(f"\nrefused: {err}")
''',
    check=r'''
@check("6.1  working memory slides, semantic memory keeps")
def _():
    need("Memory", "memory")
    need_data(memory.turns and memory.facts, "the Memory cell above")
    assert len(memory.turns) == 4, (
        f"working memory holds {len(memory.turns)} turns; the window is 4. Slice the "
        "list after appending: self.turns = self.turns[-self.window:]"
    )
    assert memory.turns[-1]["content"].startswith("Noted"), \
        "the window must keep the NEWEST turns and drop the oldest"
    assert "Sam" not in json.dumps(memory.turns), (
        "the introduction has scrolled out of working memory — which is exactly why "
        "facts are stored separately"
    )
    assert memory.facts.get("name") == "Sam" and memory.facts.get("city") == "London", \
        f"facts are {memory.facts}; name and city should both be there"
    print(f"     4 turns in the window, {len(memory.facts)} facts kept outside it")


@check("6.1  secrets are refused at the point of writing")
def _():
    need("Memory")
    m = Memory()
    for key, value in [("password", "hunter2"), ("api_key", "abc"),
                       ("user_pin", "1234")]:
        try:
            m.learn_fact(key, value)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"learn_fact({key!r}, ...) was allowed. Check the key against "
                "SECRET_WORDS and raise ValueError — an agent must not remember "
                "credentials, however helpfully it was asked to."
            )

    try:
        m.learn_fact("note", "card 4111 1111 1111 1111")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "a card-number-shaped value was stored. The key looked innocent, so the "
            "VALUE has to be checked too — that is what SECRET_VALUES is for."
        )

    m.learn_fact("city", "London")
    assert m.facts == {"city": "London"}, \
        "ordinary facts must still be stored normally"
    print("     3 secret keys and 1 secret value refused, ordinary facts fine")
''',
    hint=[
        "add_turn:",
        "  self.turns.append({'role': role, 'content': content})",
        "  self.turns = self.turns[-self.window:]",
        "",
        "learn_fact:",
        "  if any(w in key.lower() for w in SECRET_WORDS):",
        "      raise ValueError(f\"'{key}' looks like a secret and was not stored\")",
        "  if any(p.search(str(value)) for p in SECRET_VALUES):",
        "      raise ValueError('that value looks like a key or card number; not stored')",
        "  self.facts[key] = value",
    ],
    notice=r"""
**What to notice.** "Sam" left working memory after six turns, but survived as a fact. Two
kinds of memory, two lifetimes — and knowing which is which is most of designing memory.

The secret check is a safety net, not a guarantee; it cannot catch everything. It is still
worth having, because the alternative is a password sitting in a JSON file waiting to be
pasted into a prompt.
""",
)

EX2 = Exercise(
    name="What actually reaches the prompt",
    section="6.2",
    concept=r"""
Here is the sentence to remember from this chapter:

> **Memory is not what is saved. It is what reaches the prompt.**

You can store a thousand facts. The model sees only what you put in the messages — and the
context window is finite, so you are choosing. A memory system is really an **assembler**:
given a question and a budget, build the best prompt you can afford.

A sensible order, and the reasoning behind it:

1. **The system instructions.** Always.
2. **The profile** — while there are only a handful of facts, include *all* of them. It is
   cheap and it cannot miss. Searching is only necessary once there are too many to fit.
3. **The recent turns**, newest first if you have to cut.

Each piece costs tokens. Add pieces while there is budget; stop when there is not.
""",
    todo=r'''
SYSTEM = "You are a helpful assistant with memory of this user."


def build_prompt(memory, question, budget_tokens=120):
    """Assemble the messages to send, staying inside the token budget."""
    messages = [{"role": "system", "content": SYSTEM}]
    used = estimate_tokens(SYSTEM)

    # ---- the profile: every known fact, as one short block ----
    if memory.facts:
        profile = "What you know about the user:\n" + "\n".join(
            f"- {key.replace('_', ' ')}: {value}" for key, value in memory.facts.items())
        # TODO 1: if used + estimate_tokens(profile) fits in budget_tokens, append it as
        #         a {"role": "system", "content": profile} message and add its cost
        #         to `used`.
        raise NotImplementedError

    # ---- the recent turns: newest first, while there is room ----
    chosen = []
    for turn in reversed(memory.turns):
        # TODO 2: work out this turn's cost. If it fits, add it to the FRONT of `chosen`
        #         (so the original order is kept) and add the cost to `used`.
        #         If it does not fit, stop looking — break.
        raise NotImplementedError

    messages.extend(chosen)
    messages.append({"role": "user", "content": question})
    return messages


prompt = build_prompt(memory, "What should I avoid at dinner?", budget_tokens=120)
for m in prompt:
    print(f"{m['role']:>9}: {m['content'][:70]}")
print(f"\nestimated tokens: {sum(estimate_tokens(m['content']) for m in prompt)}")

tiny = build_prompt(memory, "What should I avoid at dinner?", budget_tokens=45)
print(f"\nwith a 45-token budget: {len(tiny)} messages instead of {len(prompt)}")
''',
    solution=r'''
SYSTEM = "You are a helpful assistant with memory of this user."


def build_prompt(memory, question, budget_tokens=120):
    """Assemble the messages to send, staying inside the token budget."""
    messages = [{"role": "system", "content": SYSTEM}]
    used = estimate_tokens(SYSTEM)

    # ---- the profile: every known fact, as one short block ----
    if memory.facts:
        profile = "What you know about the user:\n" + "\n".join(
            f"- {key.replace('_', ' ')}: {value}" for key, value in memory.facts.items())
        cost = estimate_tokens(profile)
        if used + cost <= budget_tokens:
            messages.append({"role": "system", "content": profile})
            used += cost

    # ---- the recent turns: newest first, while there is room ----
    chosen = []
    for turn in reversed(memory.turns):
        cost = estimate_tokens(turn["content"])
        if used + cost > budget_tokens:
            break
        chosen.insert(0, turn)          # front, so the conversation stays in order
        used += cost

    messages.extend(chosen)
    messages.append({"role": "user", "content": question})
    return messages


prompt = build_prompt(memory, "What should I avoid at dinner?", budget_tokens=120)
for m in prompt:
    print(f"{m['role']:>9}: {m['content'][:70]}")
print(f"\nestimated tokens: {sum(estimate_tokens(m['content']) for m in prompt)}")

tiny = build_prompt(memory, "What should I avoid at dinner?", budget_tokens=45)
print(f"\nwith a 45-token budget: {len(tiny)} messages instead of {len(prompt)}")
''',
    check=r'''
@check("6.2  the prompt carries the facts the question needs")
def _():
    need("build_prompt", "prompt")
    need_data(len(prompt) > 2, "the build_prompt cell above")
    text = json.dumps(prompt)
    assert "peanuts" in text, (
        "the prompt does not mention peanuts, so the model cannot possibly warn Sam "
        "about dinner. The fact was saved — but saving is not remembering."
    )
    assert prompt[0]["role"] == "system" and prompt[-1]["role"] == "user", \
        "the instructions come first and the question last"
    assert prompt[-1]["content"] == "What should I avoid at dinner?", \
        "the user's actual question must be the final message"
    assert any("What you know about the user" in m["content"] for m in prompt), \
        "include the profile block when it fits in the budget"


@check("6.2  the budget is respected, newest turns kept")
def _():
    need("build_prompt", "memory")
    need_data(memory.facts, "the Memory cell above")
    for budget in (30, 45, 70, 200):
        p = build_prompt(memory, "and now?", budget_tokens=budget)
        spent = sum(estimate_tokens(m["content"]) for m in p
                    if m["content"] != "and now?")
        assert spent <= budget, (
            f"with a {budget}-token budget the prompt used {spent}. Add a piece only "
            "when used + cost still fits."
        )

    small = build_prompt(memory, "and now?", budget_tokens=45)
    big = build_prompt(memory, "and now?", budget_tokens=200)
    assert len(small) < len(big), \
        "a smaller budget should produce a shorter prompt"

    turns_in_big = [m for m in big if m["content"] in
                    [t["content"] for t in memory.turns]]
    assert turns_in_big == [t for t in memory.turns if t in turns_in_big], (
        "the conversation turns are out of order. Walk them newest-first to decide what "
        "fits, but insert at the front so the final prompt reads in order."
    )
    print("     budgets 30/45/70/200 all respected, order preserved")
''',
    hint=[
        "The profile:",
        "  cost = estimate_tokens(profile)",
        "  if used + cost <= budget_tokens:",
        "      messages.append({'role': 'system', 'content': profile})",
        "      used += cost",
        "",
        "The turns:",
        "  cost = estimate_tokens(turn['content'])",
        "  if used + cost > budget_tokens:",
        "      break",
        "  chosen.insert(0, turn)",
        "  used += cost",
        "",
        "insert(0, ...) is the trick: you decide newest-first, but the model reads",
        "oldest-first.",
    ],
    notice=r"""
**What to notice.** Shrink the budget and the prompt shrinks — the facts go last, the old
turns go first. You just wrote a policy, and policies have consequences you can measure.

This is also how you **test** memory, which is the next exercise. A model can produce a
plausible answer with no memory at all, so the answer proves nothing. The prompt is the
evidence.
""",
)

EX3 = Exercise(
    name="Finding memories by meaning",
    section="6.4",
    concept=r"""
Once there are hundreds of facts, you cannot put them all in the prompt. You have to
**search** — and this is where keyword search quietly fails.

The user asks: *"What's my dog called?"*

The stored fact says: *"pet: a golden retriever named Max"*

No shared word except "my". Keyword search returns nothing, and your agent — which knows
the answer — says it does not know.

**Embeddings** fix this by turning text into numbers that are close together when the
*meaning* is close, however the words differ. You then measure **distance**: 0 is identical
meaning, 1 is nothing in common.

And one detail that matters in practice: a search always returns *something*, however
irrelevant. So set a **cutoff** and discard anything beyond it. Without one, a question
about dinner cheerfully drags in the user's job title, and the model treats it as relevant
because you put it in the prompt.
""",
    todo=r'''
def recall_by_keyword(memory, question):
    """Keyword search: a fact matches if it shares a word with the question."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    return [key for key, value in memory.facts.items()
            if words & set(re.findall(r"[a-z]+", f"{key} {value}".lower()))]


def recall_by_meaning(memory, question, cutoff=0.6, limit=2):
    """Return the keys of the facts closest in meaning to the question.

    Closest first, nothing beyond `cutoff`, at most `limit` of them.
    """
    question_vector = embed(question)

    # TODO 1: for each fact, build the text f"{key} {value}" (with underscores in the
    #         key replaced by spaces), embed it, and measure distance() to
    #         question_vector. Collect (distance, key) pairs.
    # TODO 2: drop every pair whose distance is greater than cutoff.
    # TODO 3: sort by distance, smallest first, and return at most `limit` keys.
    raise NotImplementedError


question = "What's my dog called?"
print("question:", question)
print("keyword search found:", recall_by_keyword(memory, question))
print("meaning search found:", recall_by_meaning(memory, question))

print("\nunrelated question:", "What is the capital of Peru?")
print("meaning search found:", recall_by_meaning(memory, "What is the capital of Peru?"))
''',
    solution=r'''
def recall_by_keyword(memory, question):
    """Keyword search: a fact matches if it shares a word with the question."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    return [key for key, value in memory.facts.items()
            if words & set(re.findall(r"[a-z]+", f"{key} {value}".lower()))]


def recall_by_meaning(memory, question, cutoff=0.6, limit=2):
    """Return the keys of the facts closest in meaning to the question.

    Closest first, nothing beyond `cutoff`, at most `limit` of them.
    """
    question_vector = embed(question)

    scored = []
    for key, value in memory.facts.items():
        text = f"{key.replace('_', ' ')} {value}"
        scored.append((distance(question_vector, embed(text)), key))

    close_enough = [pair for pair in scored if pair[0] <= cutoff]
    close_enough.sort()
    return [key for _, key in close_enough[:limit]]


question = "What's my dog called?"
print("question:", question)
print("keyword search found:", recall_by_keyword(memory, question))
print("meaning search found:", recall_by_meaning(memory, question))

print("\nunrelated question:", "What is the capital of Peru?")
print("meaning search found:", recall_by_meaning(memory, "What is the capital of Peru?"))
''',
    check=r'''
@check("6.4  meaning search finds what keywords miss")
def _():
    need("recall_by_meaning")
    found = recall_by_meaning(memory, "What's my dog called?")
    assert "pet" in found, (
        f"recall returned {found}. The fact 'pet: a golden retriever named Max' shares "
        "almost no words with the question, but it is the closest in MEANING — that is "
        "the whole point of embeddings."
    )
    keyword_result = recall_by_keyword(memory, "What's my dog called?")
    assert "pet" not in keyword_result, \
        "(sanity check) keyword search is not supposed to find this one"

    dinner = recall_by_meaning(memory, "Is there anything I should not eat?")
    assert "food_allergy" in dinner, \
        f"a question about eating should recall the food allergy; got {dinner}"


@check("6.4  the cutoff keeps irrelevant memories out")
def _():
    need("recall_by_meaning")
    nonsense = recall_by_meaning(memory, "What is the capital of Peru?")
    assert nonsense == [], (
        f"an unrelated question recalled {nonsense}. Search always ranks something "
        "first, so discard everything beyond the cutoff — otherwise irrelevant facts "
        "land in the prompt and the model treats them as relevant."
    )

    limited = recall_by_meaning(memory, "my dog and my city and my food",
                               cutoff=1.0, limit=2)
    assert len(limited) <= 2, f"limit=2 should return at most 2 keys, got {limited}"

    strict = recall_by_meaning(memory, "What's my dog called?", cutoff=0.0)
    assert strict == [], "a cutoff of 0 should admit nothing but an exact match"
    print("     relevant questions hit, unrelated ones return nothing")
''',
    hint=[
        "Three steps:",
        "  scored = []",
        "  for key, value in memory.facts.items():",
        "      text = f\"{key.replace('_', ' ')} {value}\"",
        "      scored.append((distance(question_vector, embed(text)), key))",
        "",
        "  close = [p for p in scored if p[0] <= cutoff]",
        "  close.sort()                      # (distance, key) sorts by distance",
        "  return [key for _, key in close[:limit]]",
        "",
        "Remember: SMALL distance means similar. Sorting ascending puts the best first.",
    ],
    notice=r"""
**What to notice.** Three things, in order of how often they bite people:

* **Keyword search misses paraphrases.** "dog" and "golden retriever" are the same thing to
  a user and different strings to a computer.
* **The cutoff is not optional.** Search returns a ranked list, always — including when
  nothing is relevant.
* **Write memories as complete statements.** "pet: a golden retriever named Max" is
  findable. "Max" alone is not, because it carries no meaning to match against.

And the test above is the real lesson of §6.3: it checks **what was recalled**, not what the
model said. A model can answer plausibly with no memory at all, so only the prompt is
evidence.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=6,
    title="Memory: Implementing Persistent Agent Memory",
    intro=r"""
A model remembers nothing between calls. Everything you have ever experienced as an AI
"remembering" you was engineering, not the model — and in this notebook you write it: a
sliding window of recent turns, facts that outlive the conversation, a prompt assembled to
a budget, and recall that works by **meaning** rather than matching words.
""",
    minutes=35,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "Memory is three jobs: **save** outside the model, **find** the right piece, "
        "**put it back** in the prompt.",
        "Working memory is a sliding window; semantic memory is a store that outlives the "
        "conversation. Different lifetimes, different code.",
        "**Never remember secrets** — check the key and the value at the point of writing.",
        "**Memory is not what is saved, it is what reaches the prompt.** Assemble to a "
        "budget: instructions, profile, then recent turns.",
        "Embeddings find memories by meaning, which keywords cannot — but set a distance "
        "**cutoff**, or irrelevant memories arrive dressed as relevant ones.",
        "Test memory by reading the **prompt**, not the answer. A model can sound right "
        "with no memory at all.",
    ],
    next_up="Chapter 7 applies the same retrieval idea to documents instead of memories: "
            "RAG.",
)
