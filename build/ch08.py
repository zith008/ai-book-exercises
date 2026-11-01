"""Chapter 8 — Where RAG Is Heading: proposed memories, graphs, hybrid retrieval."""

from kit import DOCUMENTS, EMBEDDER
from lite import Exercise, make_chapter

GIVEN = EMBEDDER + DOCUMENTS + r'''
# ---- Chunks, as built in Chapter 7 (given here so this notebook stands alone) ----
def split_document(text, source="handbook", max_chars=220):
    chunks, section = [], "(none)"
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            section = block.lstrip("#").strip()
            continue
        chunks.append({"text": f"{section}: {block}", "source": source,
                       "section": section})
    return chunks

CHUNKS = split_document(HANDBOOK)


# ---- What a model "proposed" after a conversation ---------------------------
# In production an LLM writes these after each turn. Here they are fixed, so the
# exercise is about your code's judgement rather than the model's.
PROPOSED = [
    {"key": "city", "value": "London", "confidence": 0.95},
    {"key": "city", "value": "London", "confidence": 0.90},   # duplicate
    {"key": "city", "value": "Berlin", "confidence": 0.60},   # contradiction
    {"key": "password", "value": "hunter2", "confidence": 0.99},   # must be refused
    {"key": "", "value": "something", "confidence": 0.99},         # malformed
    {"key": "newsletter", "value": "", "confidence": 0.80},        # empty value
    {"key": "diet", "value": "vegetarian", "confidence": 0.30},    # too unsure
    {"key": "job", "value": "software engineer", "confidence": 0.88},
]

print(f"{len(CHUNKS)} chunks, {len(PROPOSED)} proposed memories to judge")
'''

EX1 = Exercise(
    name="The model proposes, your code decides",
    section="8.2",
    concept=r"""
Chapter 6 remembered things when the user asked. The other common design is automatic:
after every turn, a model decides what was worth remembering.

That is more useful and much riskier, because the model is now **writing to your database
without being asked**. The mistakes are predictable:

* the same fact proposed twice
* a new fact that **contradicts** a stored one (the user said London; now it says Berlin)
* something that should never be stored at all, like a password
* a malformed proposal — empty key, empty value
* a guess the model itself is not confident about

So the rule is: **let the model propose, and let your code decide.** Every proposal goes
through a gate that can accept, update, or reject it — and records why.

Think of a junior assistant who writes notes about your clients. Useful. You still read the
notes before they go in the file.
""",
    todo=r'''
SECRET_WORDS = ("password", "passwd", "secret", "token", "api_key", "apikey", "pin", "ssn")


class MemoryGate:
    def __init__(self, min_confidence=0.5):
        self.facts = {}        # key -> value
        self.rejected = []     # (key, reason) for everything refused
        self.conflicts = []    # (key, old_value, new_value) needing a human

    def consider(self, proposal):
        """Decide what to do with one proposed memory. Returns the decision as a string:
        'stored', 'duplicate', 'conflict', or 'rejected'.
        """
        key = (proposal.get("key") or "").strip()
        value = str(proposal.get("value") or "").strip()
        confidence = proposal.get("confidence", 0)

        # TODO 1: reject a malformed proposal — empty key or empty value. Record
        #         (key, "malformed") in self.rejected and return "rejected".
        # TODO 2: reject anything whose key contains a word from SECRET_WORDS, with
        #         reason "secret".
        # TODO 3: reject anything below self.min_confidence, with reason "low confidence".
        # TODO 4: if this key is already stored with the SAME value, return "duplicate"
        #         and change nothing.
        # TODO 5: if this key is already stored with a DIFFERENT value, record
        #         (key, old_value, value) in self.conflicts and return "conflict".
        #         Do NOT overwrite: a contradiction is for a human to resolve.
        # TODO 6: otherwise store it and return "stored".
        raise NotImplementedError


gate = MemoryGate(min_confidence=0.5)
for proposal in PROPOSED:
    decision = gate.consider(proposal)
    print(f"{decision:>10}  {proposal['key']!r} = {proposal['value']!r} "
          f"(confidence {proposal['confidence']})")

print("\nstored:   ", gate.facts)
print("rejected: ", gate.rejected)
print("conflicts:", gate.conflicts)
''',
    solution=r'''
SECRET_WORDS = ("password", "passwd", "secret", "token", "api_key", "apikey", "pin", "ssn")


class MemoryGate:
    def __init__(self, min_confidence=0.5):
        self.facts = {}        # key -> value
        self.rejected = []     # (key, reason) for everything refused
        self.conflicts = []    # (key, old_value, new_value) needing a human
        self.min_confidence = min_confidence

    def consider(self, proposal):
        """Decide what to do with one proposed memory. Returns the decision as a string:
        'stored', 'duplicate', 'conflict', or 'rejected'.
        """
        key = (proposal.get("key") or "").strip()
        value = str(proposal.get("value") or "").strip()
        confidence = proposal.get("confidence", 0)

        if not key or not value:
            self.rejected.append((key, "malformed"))
            return "rejected"

        if any(word in key.lower() for word in SECRET_WORDS):
            self.rejected.append((key, "secret"))
            return "rejected"

        if confidence < self.min_confidence:
            self.rejected.append((key, "low confidence"))
            return "rejected"

        if key in self.facts:
            if self.facts[key] == value:
                return "duplicate"
            self.conflicts.append((key, self.facts[key], value))
            return "conflict"

        self.facts[key] = value
        return "stored"


gate = MemoryGate(min_confidence=0.5)
for proposal in PROPOSED:
    decision = gate.consider(proposal)
    print(f"{decision:>10}  {proposal['key']!r} = {proposal['value']!r} "
          f"(confidence {proposal['confidence']})")

print("\nstored:   ", gate.facts)
print("rejected: ", gate.rejected)
print("conflicts:", gate.conflicts)
''',
    check=r'''
@check("8.2  every kind of bad proposal is caught")
def _():
    need("MemoryGate", "gate")
    need_data(gate.facts or gate.rejected, "the MemoryGate cell above")
    assert gate.facts == {"city": "London", "job": "software engineer"}, (
        f"stored facts are {gate.facts}. Only the two good proposals should be there: "
        "city=London and job=software engineer."
    )
    reasons = dict(gate.rejected)
    assert reasons.get("password") == "secret", (
        "the password proposal must be rejected with reason 'secret' — the model was "
        "confident, which is exactly why your code has to be the one deciding."
    )
    assert reasons.get("diet") == "low confidence", \
        f"a 0.30-confidence proposal should be rejected; rejections were {gate.rejected}"
    assert any(r == "malformed" for r in reasons.values()), \
        "the empty-key and empty-value proposals are malformed and must be rejected"
    assert len(gate.rejected) == 4, \
        f"4 proposals should be rejected, you rejected {len(gate.rejected)}"


@check("8.2  duplicates are ignored and contradictions are escalated")
def _():
    need("MemoryGate")
    g = MemoryGate()
    assert g.consider({"key": "city", "value": "London", "confidence": 0.9}) == "stored"
    assert g.consider({"key": "city", "value": "London", "confidence": 0.9}) == "duplicate", \
        "the same fact proposed twice is a duplicate, not a second store"
    assert g.consider({"key": "city", "value": "Berlin", "confidence": 0.9}) == "conflict", \
        "a different value for a stored key is a conflict"
    assert g.facts["city"] == "London", (
        "the stored value was overwritten by the contradiction. Last-write-wins means "
        "one confident mistake silently replaces something the user actually told you — "
        "record the conflict and leave the original alone."
    )
    assert g.conflicts == [("city", "London", "Berlin")], \
        f"conflicts should record key, old and new; yours are {g.conflicts}"
    print("     duplicate ignored, contradiction escalated, original preserved")
''',
    hint=[
        "Six gates, in this order (order matters — reject before comparing):",
        "  if not key or not value:            -> rejected, 'malformed'",
        "  if any(w in key.lower() for w in SECRET_WORDS): -> rejected, 'secret'",
        "  if confidence < self.min_confidence: -> rejected, 'low confidence'",
        "  if key in self.facts:",
        "      if self.facts[key] == value: return 'duplicate'",
        "      self.conflicts.append((key, self.facts[key], value))",
        "      return 'conflict'",
        "  self.facts[key] = value; return 'stored'",
        "",
        "Note __init__ needs self.min_confidence = min_confidence — it is missing.",
    ],
    notice=r"""
**What to notice.** The password proposal arrived with 0.99 confidence. The model was *sure*.
Confidence is not permission, and that is the whole argument for a gate.

The conflict is the subtle one. Overwriting would have been one line shorter and would have
quietly replaced a fact the user gave you with one the model inferred. Recording it instead
means a human can look.
""",
)

EX2 = Exercise(
    name="Questions that need two hops",
    section="8.3",
    concept=r"""
Some questions cannot be answered by any single chunk, however good your search is:

> *"Who manages the person who wrote the returns policy?"*

Chunk-based search looks for text similar to the question. But the answer lives in a
**chain**: policy → author → manager. No one paragraph contains it.

A **knowledge graph** makes the relationships explicit:

```
alice --wrote--> returns_policy
bob --manages--> alice
```

Now the question becomes a **walk**: start at `returns_policy`, follow `wrote` backwards to
`alice`, follow `manages` backwards to `bob`. Two hops, and the answer is exact rather than
approximate.

Two practical points the chapter insists on:

* **Keep the source of every edge.** A walk should end in evidence, not just a name — "bob,
  according to the team page".
* It costs a model call per chunk to extract the edges, so this is for cases where
  relationships genuinely matter.
""",
    todo=r'''
EDGES = [
    # (subject, relation, object, where we learned it)
    ("alice",  "wrote",   "returns_policy", "doc:policies#authors"),
    ("bob",    "manages", "alice",          "doc:team#structure"),
    ("carol",  "manages", "bob",            "doc:team#structure"),
    ("alice",  "wrote",   "shipping_policy", "doc:policies#authors"),
    ("dave",   "wrote",   "warranty_policy", "doc:policies#authors"),
    ("bob",    "manages", "dave",           "doc:team#structure"),
]


class Graph:
    def __init__(self, edges):
        self.edges = list(edges)

    def find(self, subject=None, relation=None, obj=None):
        """Every edge matching the parts you specify. None means 'anything'."""
        return [e for e in self.edges
                if (subject is None or e[0] == subject)
                and (relation is None or e[1] == relation)
                and (obj is None or e[2] == obj)]

    def who_wrote(self, document):
        """The person who wrote this document, and the source of that claim."""
        # TODO 1: find the edge (?, "wrote", document) and return
        #         (subject, source). Return (None, None) if there is no such edge.
        raise NotImplementedError

    def who_manages(self, person):
        """The person who manages this person, and the source."""
        # TODO 2: find the edge (?, "manages", person) and return (subject, source).
        raise NotImplementedError

    def manager_of_author(self, document):
        """Two hops: document -> who wrote it -> who manages them.

        Return a dict: {"answer": name or None, "path": [...], "evidence": [...]}
        where path is the names visited in order and evidence is the sources used.
        """
        # TODO 3: use who_wrote, then who_manages. Build up `path` and `evidence` as
        #         you go, so the answer arrives with its workings. If either hop
        #         fails, return answer None with whatever you managed to collect.
        raise NotImplementedError


graph = Graph(EDGES)
result = graph.manager_of_author("returns_policy")
print(result)
print("\nand for a document nobody wrote:")
print(graph.manager_of_author("privacy_policy"))
''',
    solution=r'''
EDGES = [
    # (subject, relation, object, where we learned it)
    ("alice",  "wrote",   "returns_policy", "doc:policies#authors"),
    ("bob",    "manages", "alice",          "doc:team#structure"),
    ("carol",  "manages", "bob",            "doc:team#structure"),
    ("alice",  "wrote",   "shipping_policy", "doc:policies#authors"),
    ("dave",   "wrote",   "warranty_policy", "doc:policies#authors"),
    ("bob",    "manages", "dave",           "doc:team#structure"),
]


class Graph:
    def __init__(self, edges):
        self.edges = list(edges)

    def find(self, subject=None, relation=None, obj=None):
        """Every edge matching the parts you specify. None means 'anything'."""
        return [e for e in self.edges
                if (subject is None or e[0] == subject)
                and (relation is None or e[1] == relation)
                and (obj is None or e[2] == obj)]

    def who_wrote(self, document):
        """The person who wrote this document, and the source of that claim."""
        matches = self.find(relation="wrote", obj=document)
        if not matches:
            return None, None
        subject, _, _, source = matches[0]
        return subject, source

    def who_manages(self, person):
        """The person who manages this person, and the source."""
        matches = self.find(relation="manages", obj=person)
        if not matches:
            return None, None
        subject, _, _, source = matches[0]
        return subject, source

    def manager_of_author(self, document):
        """Two hops: document -> who wrote it -> who manages them.

        Return a dict: {"answer": name or None, "path": [...], "evidence": [...]}
        where path is the names visited in order and evidence is the sources used.
        """
        path, evidence = [document], []

        author, author_source = self.who_wrote(document)
        if author is None:
            return {"answer": None, "path": path, "evidence": evidence}
        path.append(author)
        evidence.append(author_source)

        manager, manager_source = self.who_manages(author)
        if manager is None:
            return {"answer": None, "path": path, "evidence": evidence}
        path.append(manager)
        evidence.append(manager_source)

        return {"answer": manager, "path": path, "evidence": evidence}


graph = Graph(EDGES)
result = graph.manager_of_author("returns_policy")
print(result)
print("\nand for a document nobody wrote:")
print(graph.manager_of_author("privacy_policy"))
''',
    check=r'''
@check("8.3  one hop at a time")
def _():
    need("Graph", "graph")
    author, source = graph.who_wrote("returns_policy")
    assert author == "alice", f"who_wrote returned {author!r}, expected 'alice'"
    assert source == "doc:policies#authors", (
        f"the source was {source!r}. Return the fourth part of the edge — an answer "
        "without its evidence cannot be checked by anyone."
    )
    assert graph.who_manages("alice")[0] == "bob", \
        "bob manages alice, per the edges"
    assert graph.who_wrote("privacy_policy") == (None, None), \
        "a document nobody wrote should give (None, None), not raise"


@check("8.3  two hops, with the path and the evidence")
def _():
    need("graph")
    r = graph.manager_of_author("returns_policy")
    assert r["answer"] == "bob", (
        f"the answer was {r['answer']!r}. returns_policy -> alice wrote it -> bob "
        "manages alice, so the answer is bob. No single chunk contains this."
    )
    assert r["path"] == ["returns_policy", "alice", "bob"], (
        f"path is {r['path']}; it should record every node visited, in order — that is "
        "what lets a user see how the answer was reached."
    )
    assert len(r["evidence"]) == 2 and all(r["evidence"]), \
        f"evidence is {r['evidence']}; one source per hop"

    deeper = graph.manager_of_author("warranty_policy")
    assert deeper["answer"] == "bob", \
        f"warranty_policy -> dave -> bob, but you got {deeper['answer']!r}"

    missing = graph.manager_of_author("privacy_policy")
    assert missing["answer"] is None, "an unanswerable walk returns None, not a guess"
    assert missing["path"] == ["privacy_policy"], \
        f"path is {missing['path']} — it should stop where the walk stopped"
    print("     2-hop answer 'bob' with path and 2 pieces of evidence")
''',
    hint=[
        "who_wrote:",
        "  matches = self.find(relation='wrote', obj=document)",
        "  if not matches: return None, None",
        "  subject, _, _, source = matches[0]",
        "  return subject, source",
        "",
        "manager_of_author, hop by hop:",
        "  path, evidence = [document], []",
        "  author, src = self.who_wrote(document)",
        "  if author is None: return {'answer': None, 'path': path, 'evidence': evidence}",
        "  path.append(author); evidence.append(src)",
        "  ... same again with who_manages(author) ...",
    ],
    notice=r"""
**What to notice.** The answer came with its workings: `returns_policy → alice → bob`, plus
the two documents those claims came from. A user can check it. A vector search returning
"here are three paragraphs that looked similar" cannot offer that.

**And the catch.** This graph was handed to you. In reality a model extracts the edges from
your documents, which costs a call per chunk and makes mistakes — above all in deciding
that "Alice Smith", "alice" and "A. Smith" are one person. A graph is only as good as that
merging.
""",
)

EX3 = Exercise(
    name="Keywords and meaning together",
    section="8.4",
    concept=r"""
Two kinds of search, failing in opposite ways:

* **Keyword search** nails exact strings — an error code, a product SKU, `ERR_4042` — and
  is helpless with paraphrase.
* **Meaning search** handles paraphrase and misses the exact string, because `ERR_4042`
  looks much like `ERR_4043` to an embedder.

So combine them. The naive way is to average the two scores — but the scores are on
different scales, and averaging incomparable numbers produces nonsense that looks like a
number.

The reliable trick is **reciprocal rank fusion**: throw the scores away and use only the
*positions*.

$$\text{score}(d) = \sum_{\text{each list}} \frac{1}{k + \text{rank}(d)}$$

with $k$ usually 60. A document ranked 1st by either method scores well; one ranked
decently by *both* scores best of all. No rescaling, nothing to tune, and it cannot be
broken by one method's scores being on a strange scale.
""",
    todo=r'''
# A document set with both kinds of trap in it
DOCS = [
    "Error ERR_4042 means the payment gateway timed out. Retry after 30 seconds.",
    "Error ERR_4043 means the payment gateway refused the card.",
    "If a customer cannot pay, the checkout will show a message and no order is created.",
    "Refunds are issued to the original payment method within 5 working days.",
    "The warehouse processes returns each morning before dispatching new orders.",
]


def keyword_search(query, docs, top_k=5):
    """Rank by how many query words appear in the document. Given to you."""
    words = set(_words(query))
    scored = [(len(words & set(_words(d))), i) for i, d in enumerate(docs)]
    scored = [(score, i) for score, i in scored if score > 0]
    scored.sort(key=lambda pair: -pair[0])
    return [i for _, i in scored[:top_k]]


def meaning_search(query, docs, top_k=5):
    """Rank by embedding distance. Given to you."""
    query_vector = embed(query)
    scored = [(distance(query_vector, embed(d)), i) for i, d in enumerate(docs)]
    scored.sort(key=lambda pair: pair[0])
    return [i for _, i in scored[:top_k]]


def fuse(ranked_lists, k=60, top_k=3):
    """Reciprocal rank fusion of several ranked lists of document indexes.

    Each list is in rank order, best first. Return the fused indexes, best first.
    """
    # TODO 1: build a score for every document index that appears in any list.
    #         For each list, for each position, add 1 / (k + rank) where rank starts
    #         at 1 for the first item. Use a dict of index -> running score.
    # TODO 2: sort by score, HIGHEST first (a bigger fused score is better), and
    #         return at most top_k indexes.
    raise NotImplementedError


query = "ERR_4042 payment problem"
by_keyword = keyword_search(query, DOCS)
by_meaning = meaning_search(query, DOCS)
fused = fuse([by_keyword, by_meaning])

print("query:", query)
print("\nkeyword order:", by_keyword)
print("meaning order:", by_meaning)
print("fused order:  ", fused)
print("\ntop fused result:", DOCS[fused[0]])
''',
    solution=r'''
# A document set with both kinds of trap in it
DOCS = [
    "Error ERR_4042 means the payment gateway timed out. Retry after 30 seconds.",
    "Error ERR_4043 means the payment gateway refused the card.",
    "If a customer cannot pay, the checkout will show a message and no order is created.",
    "Refunds are issued to the original payment method within 5 working days.",
    "The warehouse processes returns each morning before dispatching new orders.",
]


def keyword_search(query, docs, top_k=5):
    """Rank by how many query words appear in the document. Given to you."""
    words = set(_words(query))
    scored = [(len(words & set(_words(d))), i) for i, d in enumerate(docs)]
    scored = [(score, i) for score, i in scored if score > 0]
    scored.sort(key=lambda pair: -pair[0])
    return [i for _, i in scored[:top_k]]


def meaning_search(query, docs, top_k=5):
    """Rank by embedding distance. Given to you."""
    query_vector = embed(query)
    scored = [(distance(query_vector, embed(d)), i) for i, d in enumerate(docs)]
    scored.sort(key=lambda pair: pair[0])
    return [i for _, i in scored[:top_k]]


def fuse(ranked_lists, k=60, top_k=3):
    """Reciprocal rank fusion of several ranked lists of document indexes.

    Each list is in rank order, best first. Return the fused indexes, best first.
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, index in enumerate(ranked, start=1):
            scores[index] = scores.get(index, 0.0) + 1 / (k + rank)

    ordered = sorted(scores.items(), key=lambda pair: -pair[1])
    return [index for index, _ in ordered[:top_k]]


query = "ERR_4042 payment problem"
by_keyword = keyword_search(query, DOCS)
by_meaning = meaning_search(query, DOCS)
fused = fuse([by_keyword, by_meaning])

print("query:", query)
print("\nkeyword order:", by_keyword)
print("meaning order:", by_meaning)
print("fused order:  ", fused)
print("\ntop fused result:", DOCS[fused[0]])
''',
    check=r'''
@check("8.4  rank fusion combines the two orders")
def _():
    need("fuse")
    # Document 2 is ranked first by one method and second by the other, so it should
    # beat a document that one method loves and the other does not see at all.
    fused = fuse([[7, 2, 1], [2, 9, 1]], k=60, top_k=3)
    assert fused[0] == 2, (
        f"fused order was {fused}. Document 2 is 2nd and 1st across the two lists, so "
        "it should win: 1/61 + 1/62 beats 1/61 alone. Agreement between methods is "
        "exactly what fusion rewards."
    )
    assert set(fuse([[7, 2, 1], [2, 9, 1]], top_k=10)) == {1, 2, 7, 9}, \
        "every document appearing in any list should get a fused score"

    only_one = fuse([[5], []], top_k=3)
    assert only_one == [5], f"a single non-empty list should pass through; got {only_one}"
    assert fuse([[], []]) == [], "no results in, no results out"

    assert fuse([[1, 2, 3]], top_k=2) == [1, 2], \
        "with one list, fusion must preserve that list's order (1/61 > 1/62 > 1/63)"


@check("8.4  fusion finds what one method alone would miss")
def _():
    need("fuse", "DOCS", "keyword_search", "meaning_search")
    exact = "ERR_4042"
    k_list = keyword_search(exact, DOCS)
    m_list = meaning_search(exact, DOCS)
    fused = fuse([k_list, m_list], top_k=2)
    assert 0 in fused, (
        f"searching for the exact code {exact!r} did not surface document 0 in the top "
        f"2 (got {fused}). Keyword search finds it immediately; fusion must not bury it."
    )

    vague = "customer could not pay at checkout"
    fused_vague = fuse([keyword_search(vague, DOCS), meaning_search(vague, DOCS)],
                       top_k=3)
    assert 2 in fused_vague, (
        f"the paraphrased question returned {fused_vague}; document 2 is the one that "
        "answers it"
    )
    print(f"     exact code -> {fused}, paraphrase -> {fused_vague}")
''',
    hint=[
        "fuse is a loop and a sort:",
        "  scores = {}",
        "  for ranked in ranked_lists:",
        "      for rank, index in enumerate(ranked, start=1):",
        "          scores[index] = scores.get(index, 0.0) + 1 / (k + rank)",
        "",
        "  ordered = sorted(scores.items(), key=lambda pair: -pair[1])",
        "  return [index for index, _ in ordered[:top_k]]",
        "",
        "enumerate(..., start=1) matters: rank 0 would make the first item's score",
        "1/60 instead of 1/61 — harmless here, but the formula says rank starts at 1.",
    ],
    notice=r"""
**What to notice.** Fusion never looked at a single score — only at positions. That is what
makes it safe to combine methods whose numbers mean different things.

**And the honest caveat from the chapter:** whether hybrid search helps *your* system depends
on *your* data. Measure it on your own questions, with a test set like Chapter 7's. "Hybrid
is the new default" is a claim, not a measurement.

> These techniques are **layers, not rivals**. Start from the symptom — "it cannot find
> exact codes", "it cannot follow relationships", "it keeps rediscovering the same facts" —
> and apply the simplest fix that addresses it.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=8,
    title="Where RAG Is Heading",
    intro=r"""
Classic RAG — split, embed, search, answer — works, and has limits. It rediscovers the same
knowledge on every question, it cannot follow a chain of relationships, and it misses exact
error codes.

This notebook addresses three of those directly: memories the model **proposes** and your
code **approves**, a graph you can **walk** for multi-step questions, and **rank fusion** to
combine keyword and meaning search without comparing incomparable scores.
""",
    minutes=35,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "When a model writes to your memory, **it proposes and your code decides**: "
        "validate, catch duplicates, escalate contradictions, never store secrets.",
        "**Confidence is not permission** — the worst proposal in the set arrived at 0.99.",
        "A **graph** turns a multi-step question into a walk, and the walk should end in "
        "evidence: keep the source of every edge.",
        "Graphs depend on good extraction and good entity merging, and cost a model call "
        "per chunk.",
        "**Reciprocal rank fusion** combines ranked lists using positions, not scores, so "
        "nothing needs rescaling.",
        "These are **layers, not rivals**: start from the symptom, measure, use the "
        "simplest fix that works.",
    ],
    next_up="Chapter 9 puts the vectors in a real database — and asks what the database "
            "is quietly trading away.",
)
