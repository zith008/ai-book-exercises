"""Chapter 9 — Vector Databases: recall, one contract, and hybrid search without a leak."""

from kit import EMBEDDER
from lite import Exercise, make_chapter

GIVEN = EMBEDDER + r'''
import random

# ---- A document set with owners and visibility ------------------------------
DOCS = [
    # id, text, team, visibility
    ("d1", "Returns policy: items may be returned within 30 days for a full refund.",
     "support", "public"),
    ("d2", "Shipping: express delivery costs 9.99 GBP and arrives next working day.",
     "support", "public"),
    ("d3", "Warranty: electronics carry a 24 month warranty for manufacturing faults.",
     "support", "public"),
    ("d4", "Salary bands for the support team are reviewed every March.",
     "hr", "confidential"),
    ("d5", "Disciplinary process: a written warning precedes any dismissal.",
     "hr", "confidential"),
    ("d6", "Refund abuse: accounts with repeated refund claims are flagged for review.",
     "support", "internal"),
    ("d7", "Payment gateway credentials rotate every 90 days.",
     "platform", "confidential"),
    ("d8", "Account closure removes order history permanently and cannot be undone.",
     "support", "public"),
]

RECORDS = [{"id": i, "text": t, "team": team, "visibility": v}
           for i, t, team, v in DOCS]
VECTORS = {r["id"]: embed(r["text"]) for r in RECORDS}


# ---- An exact store: compares against everything, always right, never fast --
class ExactStore:
    def search(self, query, top_k=3, records=None):
        records = RECORDS if records is None else records
        q = embed(query)
        scored = [(distance(q, VECTORS[r["id"]]), r["id"]) for r in records]
        scored.sort(key=lambda pair: pair[0])
        return [doc_id for _, doc_id in scored[:top_k]]


# ---- An "approximate" store: the kind of index a real database uses ---------
# It groups vectors into buckets and searches only the nearest few buckets.
# That is the speed-for-accuracy trade every approximate index makes.
class ApproximateStore:
    def __init__(self, buckets=4, probe=1, seed=0):
        self.probe = probe                   # how many buckets to look inside
        rng = random.Random(seed)
        ids = [r["id"] for r in RECORDS]
        rng.shuffle(ids)
        self.buckets = [ids[i::buckets] for i in range(buckets)]
        self.centroids = [
            sum((VECTORS[i] for i in bucket), start=np.zeros(EMBED_DIM)) / max(len(bucket), 1)
            for bucket in self.buckets
        ]

    def search(self, query, top_k=3):
        q = embed(query)
        order = sorted(range(len(self.buckets)),
                       key=lambda b: distance(q, self.centroids[b]))
        candidates = [i for b in order[:self.probe] for i in self.buckets[b]]
        scored = sorted((distance(q, VECTORS[i]), i) for i in candidates)
        return [doc_id for _, doc_id in scored[:top_k]]


QUERIES = ["how long do I have to return an item",
           "express delivery cost",
           "warranty on electronics",
           "closing my account",
           "refund abuse flagging"]

print(f"{len(RECORDS)} documents indexed; {len(QUERIES)} test queries")
'''

EX1 = Exercise(
    name="What an approximate index costs you",
    section="9.1",
    concept=r"""
A vector database does one thing: given a query vector, find the nearest stored vectors.

The honest way is to compare against **every** vector. That is exact, and slow — fine for
10,000 documents, hopeless for 10 million.

So real databases use an **approximate index**. They group vectors and search only the
promising groups. It is dramatically faster, and it sometimes **misses the right answer** —
not because of a bug, but by design.

The measure of that miss is **recall**: of the results exact search would have given you,
what fraction did the approximate index find?

$$\text{recall@k} = \frac{|\text{approximate top-k} \cap \text{exact top-k}|}{k}$$

Recall of 1.0 means you lost nothing. 0.6 means four in ten of your best results never
reached the user — and nothing anywhere will warn you, because the search still returned
results and they still looked plausible.

> Measure recall against exact search on your own data before you trust anyone's defaults.
""",
    todo=r'''
exact = ExactStore()


def recall_at_k(approximate_store, queries, k=3):
    """The average fraction of exact results the approximate store also found."""
    # TODO 1: for each query, get exact.search(query, top_k=k) and
    #         approximate_store.search(query, top_k=k).
    # TODO 2: the overlap is len(set(approx) & set(exact_result)) / k.
    # TODO 3: return the mean over all the queries.
    raise NotImplementedError


print(f"{'probe':>6} {'recall@3':>10}   (how many buckets each search looks inside)")
for probe in (1, 2, 3, 4):
    store = ApproximateStore(buckets=4, probe=probe)
    print(f"{probe:>6} {recall_at_k(store, QUERIES):>10.2f}")

one_bucket = ApproximateStore(buckets=4, probe=1)
print("\nexample query:", QUERIES[0])
print("  exact says:      ", exact.search(QUERIES[0]))
print("  approximate says:", one_bucket.search(QUERIES[0]))
''',
    solution=r'''
exact = ExactStore()


def recall_at_k(approximate_store, queries, k=3):
    """The average fraction of exact results the approximate store also found."""
    scores = []
    for query in queries:
        truth = set(exact.search(query, top_k=k))
        found = set(approximate_store.search(query, top_k=k))
        scores.append(len(found & truth) / k)
    return sum(scores) / len(scores)


print(f"{'probe':>6} {'recall@3':>10}   (how many buckets each search looks inside)")
for probe in (1, 2, 3, 4):
    store = ApproximateStore(buckets=4, probe=probe)
    print(f"{probe:>6} {recall_at_k(store, QUERIES):>10.2f}")

one_bucket = ApproximateStore(buckets=4, probe=1)
print("\nexample query:", QUERIES[0])
print("  exact says:      ", exact.search(QUERIES[0]))
print("  approximate says:", one_bucket.search(QUERIES[0]))
''',
    check=r'''
@check("9.1  recall is measured correctly")
def _():
    need("recall_at_k", "exact")

    class Perfect:
        def search(self, query, top_k=3):
            return exact.search(query, top_k=top_k)

    class Useless:
        def search(self, query, top_k=3):
            return []

    assert abs(recall_at_k(Perfect(), QUERIES) - 1.0) < 1e-9, (
        "a store that returns exactly the exact results must score 1.0; you got "
        f"{recall_at_k(Perfect(), QUERIES):.3f}"
    )
    assert recall_at_k(Useless(), QUERIES) == 0.0, \
        "a store that returns nothing must score 0.0"

    class Half:
        def search(self, query, top_k=3):
            return exact.search(query, top_k=top_k)[:1] + ["nonsense", "garbage"]

    assert abs(recall_at_k(Half(), QUERIES, k=3) - 1 / 3) < 1e-9, (
        "a store finding 1 of the 3 right answers should score 0.33 — divide the "
        f"overlap by k, not by the number of results. You got "
        f"{recall_at_k(Half(), QUERIES, k=3):.3f}"
    )


@check("9.1  searching more buckets raises recall")
def _():
    need("recall_at_k")
    narrow = recall_at_k(ApproximateStore(buckets=4, probe=1), QUERIES)
    wide = recall_at_k(ApproximateStore(buckets=4, probe=4), QUERIES)
    assert wide >= narrow, \
        f"probing every bucket ({wide:.2f}) should be at least as good as one ({narrow:.2f})"
    assert wide > 0.99, (
        f"probing ALL 4 buckets gave recall {wide:.2f}. Looking everywhere is the same "
        "as exact search, so it should be 1.0 — if it is not, the recall calculation "
        "is off."
    )
    assert narrow < 0.99, (
        f"probing one bucket of four gave recall {narrow:.2f}. It should lose something: "
        "that loss is the price of the speed, and the reason to measure it."
    )
    print(f"     recall {narrow:.2f} with 1 bucket, {wide:.2f} with all 4 "
          "— that gap is the trade")
''',
    hint=[
        "recall_at_k:",
        "  scores = []",
        "  for query in queries:",
        "      truth = set(exact.search(query, top_k=k))",
        "      found = set(approximate_store.search(query, top_k=k))",
        "      scores.append(len(found & truth) / k)",
        "  return sum(scores) / len(scores)",
        "",
        "Divide by k, not by len(found). A store returning one correct result out of",
        "one has not achieved perfect recall — it missed k-1 others.",
    ],
    notice=r"""
**What to notice.** The approximate search returned confident, sensible-looking results and
was missing the best match. There was no error, no warning, no slow query to notice.

This is the quiet failure mode of vector databases, and the only defence is the number you
just computed — measured on **your** documents and **your** queries, because recall depends
on both.
""",
)

EX2 = Exercise(
    name="One contract, any database",
    section="9.2",
    concept=r"""
You will change vector database at least once: pricing changes, scale changes, or you move
from a local prototype to a hosted service.

That is cheap if every store sits behind **one small contract**, and expensive if your
application code calls each vendor's API directly. The difference is where you put the
translation.

Here the contract is three methods:

```python
add(records)          # store documents and their vectors
query(text, top_k, where=None)   # search, optionally filtered
count()               # how many documents are stored
```

The awkward part is always **filters**, because every vendor spells them differently:

| Our contract | Store A | Store B |
|---|---|---|
| `{"team": "support"}` | `{"team": {"$eq": "support"}}` | `"team == 'support'"` |

Translate in **one place** — inside the adapter. Then the same tests run against both
stores, and a vendor's odd syntax never leaks into your application.
""",
    todo=r'''
# ---- Two "vendors", with deliberately different filter syntax --------------
class VendorA:
    """Filters look like {"team": {"$eq": "support"}}."""
    def __init__(self):
        self.items = {}
    def upsert(self, items):
        for item in items:
            self.items[item["id"]] = item
    def knn(self, vector, k, filter=None):
        matches = list(self.items.values())
        if filter:
            for field, condition in filter.items():
                wanted = condition["$eq"]
                matches = [m for m in matches if m["metadata"].get(field) == wanted]
        scored = sorted((distance(vector, m["values"]), m["id"]) for m in matches)
        return [doc_id for _, doc_id in scored[:k]]


class VendorB:
    """Filters are a string expression, e.g. "team == 'support'"."""
    def __init__(self):
        self.rows = []
    def insert(self, rows):
        self.rows.extend(rows)
    def nearest(self, vector, limit, expression=None):
        rows = self.rows
        if expression:
            field, _, raw = expression.split(" ", 2)
            wanted = raw.strip().strip("'")
            rows = [r for r in rows if r["meta"].get(field) == wanted]
        scored = sorted((distance(vector, r["vec"]), r["key"]) for r in rows)
        return [key for _, key in scored[:limit]]


class StoreA:
    """Adapter: our contract, implemented with VendorA."""
    def __init__(self):
        self.backend = VendorA()

    def add(self, records):
        self.backend.upsert([
            {"id": r["id"], "values": embed(r["text"]),
             "metadata": {k: v for k, v in r.items() if k != "text"}}
            for r in records])

    def query(self, text, top_k=3, where=None):
        return self.backend.knn(embed(text), top_k,
                                filter={f: {"$eq": v} for f, v in (where or {}).items()}
                                or None)

    def count(self):
        return len(self.backend.items)


class StoreB:
    """Adapter: the SAME contract, implemented with VendorB. Your turn."""
    def __init__(self):
        self.backend = VendorB()

    def add(self, records):
        # TODO 1: VendorB wants rows shaped {"key", "vec", "meta"}. Build one row per
        #         record — key is the id, vec is embed(record["text"]), meta is every
        #         field except "text" — and call self.backend.insert(rows).
        raise NotImplementedError

    def query(self, text, top_k=3, where=None):
        # TODO 2: translate `where` into VendorB's string syntax. {"team": "support"}
        #         becomes "team == 'support'". If `where` is empty, pass None.
        #         Then call self.backend.nearest(embed(text), top_k, expression).
        #         (One field is enough — note in a comment that several would need
        #          more work, which is the point of keeping it in one place.)
        raise NotImplementedError

    def count(self):
        # TODO 3: how many rows VendorB holds
        raise NotImplementedError


def run_contract_tests(store):
    """The same tests, against any store that honours the contract."""
    store.add(RECORDS)
    results = {}
    results["count"] = store.count()
    results["unfiltered"] = store.query("return an item", top_k=3)
    results["filtered"] = store.query("return an item", top_k=3,
                                      where={"team": "support"})
    results["hr_only"] = store.query("return an item", top_k=3, where={"team": "hr"})
    return results


a, b = run_contract_tests(StoreA()), run_contract_tests(StoreB())
for key in a:
    same = "same" if a[key] == b[key] else "DIFFERENT"
    print(f"{key:>12}: {same:>9}  A={a[key]}  B={b[key]}")
''',
    solution=r'''
# ---- Two "vendors", with deliberately different filter syntax --------------
class VendorA:
    """Filters look like {"team": {"$eq": "support"}}."""
    def __init__(self):
        self.items = {}
    def upsert(self, items):
        for item in items:
            self.items[item["id"]] = item
    def knn(self, vector, k, filter=None):
        matches = list(self.items.values())
        if filter:
            for field, condition in filter.items():
                wanted = condition["$eq"]
                matches = [m for m in matches if m["metadata"].get(field) == wanted]
        scored = sorted((distance(vector, m["values"]), m["id"]) for m in matches)
        return [doc_id for _, doc_id in scored[:k]]


class VendorB:
    """Filters are a string expression, e.g. "team == 'support'"."""
    def __init__(self):
        self.rows = []
    def insert(self, rows):
        self.rows.extend(rows)
    def nearest(self, vector, limit, expression=None):
        rows = self.rows
        if expression:
            field, _, raw = expression.split(" ", 2)
            wanted = raw.strip().strip("'")
            rows = [r for r in rows if r["meta"].get(field) == wanted]
        scored = sorted((distance(vector, r["vec"]), r["key"]) for r in rows)
        return [key for _, key in scored[:limit]]


class StoreA:
    """Adapter: our contract, implemented with VendorA."""
    def __init__(self):
        self.backend = VendorA()

    def add(self, records):
        self.backend.upsert([
            {"id": r["id"], "values": embed(r["text"]),
             "metadata": {k: v for k, v in r.items() if k != "text"}}
            for r in records])

    def query(self, text, top_k=3, where=None):
        return self.backend.knn(embed(text), top_k,
                                filter={f: {"$eq": v} for f, v in (where or {}).items()}
                                or None)

    def count(self):
        return len(self.backend.items)


class StoreB:
    """Adapter: the SAME contract, implemented with VendorB. Your turn."""
    def __init__(self):
        self.backend = VendorB()

    def add(self, records):
        self.backend.insert([
            {"key": r["id"], "vec": embed(r["text"]),
             "meta": {k: v for k, v in r.items() if k != "text"}}
            for r in records])

    def query(self, text, top_k=3, where=None):
        # Every vendor's filter dialect is translated HERE and nowhere else. Several
        # fields would need an "and" expression; keeping it in one place is what makes
        # that a ten-line change rather than a hunt through the application.
        expression = None
        if where:
            field, value = next(iter(where.items()))
            expression = f"{field} == '{value}'"
        return self.backend.nearest(embed(text), top_k, expression)

    def count(self):
        return len(self.backend.rows)


def run_contract_tests(store):
    """The same tests, against any store that honours the contract."""
    store.add(RECORDS)
    results = {}
    results["count"] = store.count()
    results["unfiltered"] = store.query("return an item", top_k=3)
    results["filtered"] = store.query("return an item", top_k=3,
                                      where={"team": "support"})
    results["hr_only"] = store.query("return an item", top_k=3, where={"team": "hr"})
    return results


a, b = run_contract_tests(StoreA()), run_contract_tests(StoreB())
for key in a:
    same = "same" if a[key] == b[key] else "DIFFERENT"
    print(f"{key:>12}: {same:>9}  A={a[key]}  B={b[key]}")
''',
    check=r'''
@check("9.2  both stores behave identically through the contract")
def _():
    need("StoreA", "StoreB", "run_contract_tests")
    a = run_contract_tests(StoreA())
    b = run_contract_tests(StoreB())
    assert a["count"] == b["count"] == len(RECORDS), \
        f"counts differ: A={a['count']}, B={b['count']}, expected {len(RECORDS)}"
    for key in ("unfiltered", "filtered", "hr_only"):
        assert a[key] == b[key], (
            f"{key}: StoreA returned {a[key]} and StoreB returned {b[key]}. Running the "
            "same tests against both stores is the point of the contract — a difference "
            "here is exactly the vendor behaviour change you want to catch."
        )
    print(f"     same results for all 3 queries: {a['filtered']}")


@check("9.2  the filter is translated, not ignored")
def _():
    need("StoreB")
    store = StoreB()
    store.add(RECORDS)

    hr = store.query("anything at all", top_k=5, where={"team": "hr"})
    assert set(hr) <= {"d4", "d5"}, (
        f"filtering on team=hr returned {hr}. Only d4 and d5 are HR documents — if "
        "other ids appear, the filter was dropped on the way to the vendor."
    )
    assert hr, "filtering on team=hr should return the two HR documents, not nothing"

    confidential = store.query("anything", top_k=5, where={"visibility": "confidential"})
    assert set(confidential) <= {"d4", "d5", "d7"}, \
        f"filtering on visibility=confidential returned {confidential}"

    everything = store.query("anything", top_k=8)
    assert len(everything) == len(RECORDS), \
        f"an unfiltered query should see all {len(RECORDS)} documents, got {len(everything)}"
    print("     filters translated correctly for 2 different fields")
''',
    hint=[
        "add():",
        "  self.backend.insert([",
        "      {'key': r['id'], 'vec': embed(r['text']),",
        "       'meta': {k: v for k, v in r.items() if k != 'text'}}",
        "      for r in records])",
        "",
        "query():",
        "  expression = None",
        "  if where:",
        "      field, value = next(iter(where.items()))",
        "      expression = f\"{field} == '{value}'\"",
        "  return self.backend.nearest(embed(text), top_k, expression)",
        "",
        "count(): return len(self.backend.rows)",
    ],
    notice=r"""
**What to notice.** Two completely different vendor APIs, one set of tests, identical
results. Swapping database now means writing one adapter — not editing your application.

**Be honest about what a fake proves, though.** These two "vendors" are dictionaries. They
cannot show you a real database's latency, its index settings, its consistency delays, or
the ways it fails under load. A fake proves your *contract* is sound. Only the real thing
proves the *store* is.
""",
)

EX3 = Exercise(
    name="A filter that leaks",
    section="9.3",
    concept=r"""
This exercise is a bug hunt, and the bug is one that ships.

You have hybrid search from Chapter 8: a keyword path and a meaning path, fused. You also
have permissions — HR documents must not reach a support agent.

So someone adds the permission filter. To one of the two paths.

> **A filter that applies to one search path and not the other is not a filter.**

The function below has exactly that flaw. A support user searching for "salary" will see an
HR document, because the keyword path never checked. And it will not look like a security
hole — it will look like a relevant result.

The fix is an order-of-operations change, and it is worth saying out loud:

**Decide what this person may see *before* you rank anything.**

Not after. Not per path. Once, at the start, on the candidate set that every path then
draws from.
""",
    todo=r'''
VISIBLE_TO = {
    "support_agent": {"public", "internal"},
    "hr_manager":    {"public", "internal", "confidential"},
    "guest":         {"public"},
}


def keyword_hits(query, records):
    """Record ids sharing at least one word with the query, best first."""
    words = set(_words(query))
    scored = [(len(words & set(_words(r["text"]))), r["id"]) for r in records]
    return [doc_id for score, doc_id in sorted(scored, key=lambda p: -p[0]) if score > 0]


def meaning_hits(query, records):
    """Record ids by embedding distance, closest first."""
    q = embed(query)
    return [doc_id for _, doc_id in
            sorted((distance(q, VECTORS[r["id"]]), r["id"]) for r in records)]


def fuse(ranked_lists, k=60, top_k=3):
    """Reciprocal rank fusion, as in Chapter 8. Given to you."""
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda p: -p[1])[:top_k]]


def leaky_search(query, role, top_k=3):
    """The version with the bug. Read it, then run the check to see it leak."""
    allowed = VISIBLE_TO[role]
    by_keyword = keyword_hits(query, RECORDS)                      # never filtered!
    by_meaning = meaning_hits(query, [r for r in RECORDS
                                      if r["visibility"] in allowed])
    return fuse([by_keyword, by_meaning], top_k=top_k)


def safe_search(query, role, top_k=3):
    """Your version. Decide what this role may see BEFORE ranking anything."""
    # TODO 1: look up the visibilities this role is allowed to see.
    # TODO 2: build the candidate records ONCE — every record whose visibility is
    #         allowed — and use that same list for BOTH paths.
    # TODO 3: fuse the two ranked lists and return the top_k ids.
    raise NotImplementedError


print("query: 'salary bands for the support team'\n")
for role in ("support_agent", "hr_manager", "guest"):
    leaky = leaky_search("salary bands for the support team", role)
    safe = safe_search("salary bands for the support team", role)
    print(f"{role:>13}  leaky={leaky}  safe={safe}")
''',
    solution=r'''
VISIBLE_TO = {
    "support_agent": {"public", "internal"},
    "hr_manager":    {"public", "internal", "confidential"},
    "guest":         {"public"},
}


def keyword_hits(query, records):
    """Record ids sharing at least one word with the query, best first."""
    words = set(_words(query))
    scored = [(len(words & set(_words(r["text"]))), r["id"]) for r in records]
    return [doc_id for score, doc_id in sorted(scored, key=lambda p: -p[0]) if score > 0]


def meaning_hits(query, records):
    """Record ids by embedding distance, closest first."""
    q = embed(query)
    return [doc_id for _, doc_id in
            sorted((distance(q, VECTORS[r["id"]]), r["id"]) for r in records)]


def fuse(ranked_lists, k=60, top_k=3):
    """Reciprocal rank fusion, as in Chapter 8. Given to you."""
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda p: -p[1])[:top_k]]


def leaky_search(query, role, top_k=3):
    """The version with the bug. Read it, then run the check to see it leak."""
    allowed = VISIBLE_TO[role]
    by_keyword = keyword_hits(query, RECORDS)                      # never filtered!
    by_meaning = meaning_hits(query, [r for r in RECORDS
                                      if r["visibility"] in allowed])
    return fuse([by_keyword, by_meaning], top_k=top_k)


def safe_search(query, role, top_k=3):
    """Your version. Decide what this role may see BEFORE ranking anything."""
    allowed = VISIBLE_TO[role]

    # ONE candidate set, decided before any ranking happens. Both paths draw from it,
    # so neither can surface something this role may not see.
    candidates = [r for r in RECORDS if r["visibility"] in allowed]

    by_keyword = keyword_hits(query, candidates)
    by_meaning = meaning_hits(query, candidates)
    return fuse([by_keyword, by_meaning], top_k=top_k)


print("query: 'salary bands for the support team'\n")
for role in ("support_agent", "hr_manager", "guest"):
    leaky = leaky_search("salary bands for the support team", role)
    safe = safe_search("salary bands for the support team", role)
    print(f"{role:>13}  leaky={leaky}  safe={safe}")
''',
    check=r'''
@check("9.3  the leak is real (this check proves the bug exists)")
def _():
    need("leaky_search")
    leaked = leaky_search("salary bands for the support team", "support_agent", top_k=3)
    assert "d4" in leaked, (
        "the leaky version did not leak here, which means the exercise cannot "
        "demonstrate its point — check you have not already fixed leaky_search."
    )
    print(f"     leaky_search let a support agent see {leaked} — d4 is HR confidential")


@check("9.3  no role ever sees beyond its permissions")
def _():
    need("safe_search", "VISIBLE_TO")
    by_id = {r["id"]: r for r in RECORDS}

    probes = ["salary bands for the support team",
              "disciplinary process dismissal",
              "payment gateway credentials rotate",
              "refund abuse flagged accounts",
              "return an item for a refund"]

    for role, allowed in VISIBLE_TO.items():
        for query in probes:
            for doc_id in safe_search(query, role, top_k=8):
                visibility = by_id[doc_id]["visibility"]
                assert visibility in allowed, (
                    f"role {role!r} searching {query!r} received {doc_id} "
                    f"({visibility}), which it may not see. Filter the candidate set "
                    "ONCE, before ranking, and use it for both paths."
                )

    hr = safe_search("salary bands", "hr_manager", top_k=3)
    assert "d4" in hr, (
        f"an HR manager searching for salary bands got {hr} — the fix must not break "
        "legitimate access. Over-filtering is a bug too."
    )
    support = safe_search("return an item for a refund", "support_agent", top_k=3)
    assert "d1" in support, \
        f"a support agent should still find the returns policy; got {support}"
    print("     3 roles x 5 queries: no leak, and legitimate results still arrive")
''',
    hint=[
        "safe_search is three lines:",
        "  allowed = VISIBLE_TO[role]",
        "  candidates = [r for r in RECORDS if r['visibility'] in allowed]",
        "  return fuse([keyword_hits(query, candidates),",
        "               meaning_hits(query, candidates)], top_k=top_k)",
        "",
        "Compare with leaky_search: it passes RECORDS to one path and a filtered list",
        "to the other. The fix is to build the candidate set once, first.",
    ],
    notice=r"""
**What to notice.** The leaky version was not careless-looking code. Someone added a filter,
tested it, saw it work — on the path they were looking at.

Three habits come out of this:

* **Filter once, before ranking.** One candidate set, every path draws from it.
* **Test access control by trying to break it.** The passing check above probes three roles
  against five queries. It is cheap and it is the only kind of test that finds this.
* **Over-filtering is also a bug.** The check insists HR managers still find HR documents.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=9,
    title="Vector Databases",
    intro=r"""
A vector database finds the nearest stored vectors to a query. Everything interesting is in
what it trades away to do that quickly, and in what your code assumes about it.

Three exercises, each about a failure that is **quiet** — no error, no crash, just wrong
results that look right: an index that misses answers, a vendor swap that changes
behaviour, and a permission filter that only covers half the search.
""",
    minutes=35,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "An approximate index trades accuracy for speed. **Recall** measures what you "
        "lost, and nothing warns you without it.",
        "Measure recall against exact search on **your own** data and queries — it "
        "depends on both.",
        "Put every store behind **one small contract**, translate each vendor's filter "
        "dialect in one place, and run the same tests against all of them.",
        "A fake store proves your contract, not the database. Latency, index settings and "
        "failure under load need the real thing.",
        "**Decide what a person may see before you rank anything.** A filter on one search "
        "path and not another is not a filter.",
        "Test access control by **trying to break it** — and check that legitimate access "
        "still works.",
    ],
    next_up="Chapter 10 goes one level down, to the embeddings themselves: comparing "
            "models, shrinking vectors, and caching them safely.",
)
