"""Chapter 10 — Embeddings: one space per setup, shrinking vectors, caching safely."""

from kit import EMBEDDER
from lite import Exercise, make_chapter

GIVEN = EMBEDDER + r'''
# ---- A dense embedder, which is what a real model gives you ----------------
# The stand-in above produces a big sparse vector. Real models return a short dense
# one (384, 768, 1536 numbers). We get there by projecting the sparse vector through
# a fixed random matrix — a genuine technique, and it makes the vectors behave like
# real ones, so shortening and quantising below are honest experiments.
_PROJECTIONS = {}

def dense_embed(text, model="small-v1", dim=128):
    """Embed text into `dim` dense numbers. Different `model` = a different space."""
    key = (model, dim)
    if key not in _PROJECTIONS:
        seed = int(hashlib.md5(model.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        _PROJECTIONS[key] = rng.standard_normal((EMBED_DIM, dim)) / np.sqrt(dim)
    vector = embed(text) @ _PROJECTIONS[key]
    length = np.linalg.norm(vector)
    return vector / length if length else vector


SENTENCES = [
    "Customers may return any item within 30 days for a full refund.",
    "Refunds are issued to the original payment method within five working days.",
    "Express delivery arrives the next working day and costs 9.99 GBP.",
    "Standard delivery takes three to five working days and costs 4.99 GBP.",
    "Electronics carry a 24 month warranty covering manufacturing faults.",
    "To make a warranty claim, email support with your order number.",
    "You can reset your password from the sign-in page.",
    "Closing your account removes your order history permanently.",
]

QUERIES = ["how do I return an item for a refund",
           "cost of express delivery",
           "warranty length on electronics",
           "reset my password",
           "closing my account"]

print(f"{len(SENTENCES)} sentences, {len(QUERIES)} queries")
print("a 128-number embedding looks like:",
      np.round(dense_embed(SENTENCES[0])[:6], 3), "...")
'''

EX1 = Exercise(
    name="One space per setup",
    section="10.0",
    concept=r"""
An embedding is **a position in a space made by one particular setup** — one model, one
vector length, one way of prefixing the text.

Vectors from two different setups are not comparable. Not "less accurate" — *meaningless*.
It is like comparing a grid reference from one map to a grid reference from another: the
numbers are fine, the comparison is not.

And here is what makes this dangerous rather than merely academic:

> **A mix-up raises no error.** `distance()` happily takes two vectors of the same length
> from different models and returns a plausible number. Your retrieval quietly gets worse
> and nothing anywhere complains.

This happens for real when a team upgrades their embedding model and re-embeds the
queries but not the stored documents.

The defence is to **record the setup next to the data**, and refuse comparisons across
setups. You are writing that refusal.
""",
    todo=r'''
class EmbeddingStore:
    """Vectors, plus the setup that produced them."""

    def __init__(self, model="small-v1", dim=128, prefix=""):
        self.model, self.dim, self.prefix = model, dim, prefix
        self.texts, self.vectors = [], []

    @property
    def space(self):
        """A label identifying this exact setup."""
        # TODO 1: return a tuple of the three things that define the space:
        #         (self.model, self.dim, self.prefix)
        raise NotImplementedError

    def add(self, texts):
        for text in texts:
            self.texts.append(text)
            self.vectors.append(dense_embed(self.prefix + text,
                                            model=self.model, dim=self.dim))

    def search(self, query, top_k=3, query_vector=None, query_space=None):
        """Search with a query string, or with a vector that states its own space.

        A vector from a different space must be refused.
        """
        # TODO 2: if query_vector was given, check query_space == self.space.
        #         If it differs, raise ValueError with a message naming BOTH spaces —
        #         a silent wrong answer is the thing you are preventing.
        # TODO 3: otherwise embed the query with THIS store's setup (remember the
        #         prefix) and search as usual, returning [(distance, text)] closest
        #         first, at most top_k.
        raise NotImplementedError


store = EmbeddingStore(model="small-v1", dim=128)
store.add(SENTENCES)

print("search with the right setup:")
for d, text in store.search("how do I return an item for a refund"):
    print(f"  {d:.3f}  {text[:55]}")

# A vector from a different model, the same length. Nothing about it looks wrong.
other_vector = dense_embed("how do I return an item for a refund",
                           model="big-v2", dim=128)
print("\nsearch with a vector from another model:")
try:
    store.search("ignored", query_vector=other_vector,
                 query_space=("big-v2", 128, ""))
    print("  !! it was allowed — that is the silent failure")
except ValueError as err:
    print(f"  refused: {err}")
''',
    solution=r'''
class EmbeddingStore:
    """Vectors, plus the setup that produced them."""

    def __init__(self, model="small-v1", dim=128, prefix=""):
        self.model, self.dim, self.prefix = model, dim, prefix
        self.texts, self.vectors = [], []

    @property
    def space(self):
        """A label identifying this exact setup."""
        return (self.model, self.dim, self.prefix)

    def add(self, texts):
        for text in texts:
            self.texts.append(text)
            self.vectors.append(dense_embed(self.prefix + text,
                                            model=self.model, dim=self.dim))

    def search(self, query, top_k=3, query_vector=None, query_space=None):
        """Search with a query string, or with a vector that states its own space.

        A vector from a different space must be refused.
        """
        if query_vector is not None:
            if query_space != self.space:
                raise ValueError(
                    f"vector is from space {query_space}, this store holds "
                    f"{self.space} — these are not comparable"
                )
            vector = query_vector
        else:
            vector = dense_embed(self.prefix + query,
                                 model=self.model, dim=self.dim)

        scored = [(distance(vector, v), text)
                  for v, text in zip(self.vectors, self.texts)]
        scored.sort(key=lambda pair: pair[0])
        return scored[:top_k]


store = EmbeddingStore(model="small-v1", dim=128)
store.add(SENTENCES)

print("search with the right setup:")
for d, text in store.search("how do I return an item for a refund"):
    print(f"  {d:.3f}  {text[:55]}")

# A vector from a different model, the same length. Nothing about it looks wrong.
other_vector = dense_embed("how do I return an item for a refund",
                           model="big-v2", dim=128)
print("\nsearch with a vector from another model:")
try:
    store.search("ignored", query_vector=other_vector,
                 query_space=("big-v2", 128, ""))
    print("  !! it was allowed — that is the silent failure")
except ValueError as err:
    print(f"  refused: {err}")
''',
    check=r'''
@check("10.0  the store searches correctly within its own space")
def _():
    need("EmbeddingStore", "store")
    assert store.space == ("small-v1", 128, ""), \
        f"space is {store.space}; it should name the model, the length and the prefix"
    results = store.search("how do I return an item for a refund", top_k=3)
    assert len(results) == 3 and isinstance(results[0][1], str), \
        "search should return (distance, text) pairs"
    assert "return any item" in results[0][1], (
        f"the closest sentence was {results[0][1][:40]!r} — the returns sentence should "
        "rank first"
    )
    distances = [d for d, _ in results]
    assert distances == sorted(distances), "results must be closest-first"


@check("10.0  a vector from another space is refused, not used")
def _():
    need("EmbeddingStore", "store")
    for wrong_space in [("big-v2", 128, ""),        # different model
                        ("small-v1", 128, "query: "),   # different prefix
                        ("small-v1", 64, "")]:          # different length
        wrong = dense_embed("anything", model=wrong_space[0], dim=128)
        try:
            store.search("ignored", query_vector=wrong, query_space=wrong_space)
        except ValueError as err:
            assert str(wrong_space) in str(err) or wrong_space[0] in str(err), (
                "the error should name the spaces involved, so whoever hits it can see "
                f"what went wrong; yours says {err!r}"
            )
        else:
            raise AssertionError(
                f"a vector from {wrong_space} was accepted. Nothing about it looks "
                "wrong — same length, plausible numbers, plausible distances. That is "
                "exactly why the store has to check."
            )

    right = dense_embed("reset my password", model="small-v1", dim=128)
    ok = store.search("ignored", query_vector=right, query_space=("small-v1", 128, ""))
    assert "password" in ok[0][1], \
        "a vector from the SAME space must still work — do not refuse everything"
    print("     3 mismatched spaces refused, matching space accepted")
''',
    hint=[
        "space is one line:",
        "  return (self.model, self.dim, self.prefix)",
        "",
        "search():",
        "  if query_vector is not None:",
        "      if query_space != self.space:",
        "          raise ValueError(f'vector is from space {query_space}, this store '",
        "                           f'holds {self.space} — these are not comparable')",
        "      vector = query_vector",
        "  else:",
        "      vector = dense_embed(self.prefix + query, model=self.model, dim=self.dim)",
        "",
        "Then score, sort by distance, return scored[:top_k].",
    ],
    notice=r"""
**What to notice.** The rejected vector had the right length and contained perfectly
reasonable numbers. Had you compared it, you would have got distances — wrong ones — and
retrieval would have degraded without a single error in your logs.

Three things define the space, and all three catch real mistakes: the **model**, the
**vector length**, and the **prefix**. That last one surprises people: several embedding
models expect `"query: "` before a question and `"passage: "` before a document, and
getting it wrong costs accuracy silently.
""",
)

EX2 = Exercise(
    name="Making vectors smaller",
    section="10.2",
    concept=r"""
Vectors are most of what a vector database stores. A million documents at 1536 dimensions
in 32-bit floats is about 6 GB — before any index. There are two standard ways to shrink
that, and both cost a little accuracy:

**Shortening** (truncation): keep the first *n* numbers and throw the rest away, then
re-normalise. Halve the length, halve the memory. Some models are *trained* for this
(so-called Matryoshka embeddings) and barely suffer; others degrade noticeably.

**Quantising**: store each number as one byte instead of four. Map the range
$[-1, 1]$ onto the integers $[-127, 127]$:

```python
quantised = np.round(vector * 127).astype(np.int8)
```

A quarter of the memory, and a small rounding error in every number.

The point of this exercise is the **measurement**, not the technique. Judge the cost by
**retrieval** — does the right document still come back? — and never by eyeballing the
distance between one pair of sentences.
""",
    todo=r'''
def shorten(vector, dim):
    """Keep the first `dim` numbers and re-normalise to unit length."""
    # TODO 1: slice the vector, then divide by its norm (guard against a zero norm).
    raise NotImplementedError


def quantise(vector):
    """Store as int8: multiply by 127, round, cast to np.int8."""
    # TODO 2: return np.round(vector * 127).astype(np.int8)
    raise NotImplementedError


def dequantise(vector_int8):
    """Back to floats for comparison: divide by 127."""
    return np.asarray(vector_int8, dtype=float) / 127


def retrieval_accuracy(transform, dim=128):
    """What fraction of queries still return the same best sentence after `transform`?

    transform takes a vector and returns a comparable vector.
    """
    full = [dense_embed(s, dim=dim) for s in SENTENCES]
    small = [transform(v) for v in full]

    correct = 0
    for query in QUERIES:
        q_full, q_small = dense_embed(query, dim=dim), transform(dense_embed(query, dim=dim))
        best_full = min(range(len(full)), key=lambda i: distance(q_full, full[i]))
        best_small = min(range(len(small)), key=lambda i: distance(q_small, small[i]))
        correct += (best_full == best_small)
    return correct / len(QUERIES)


bytes_full = 128 * 4           # 128 numbers, 4 bytes each (float32)

print(f"{'setup':>22} {'bytes/vector':>13} {'of original':>12} {'top-1 kept':>11}")
print(f"{'full 128-dim float32':>22} {bytes_full:>13} {'100%':>12} "
      f"{retrieval_accuracy(lambda v: v):>11.0%}")
print(f"{'shortened to 64':>22} {64 * 4:>13} {'50%':>12} "
      f"{retrieval_accuracy(lambda v: shorten(v, 64)):>11.0%}")
print(f"{'shortened to 32':>22} {32 * 4:>13} {'25%':>12} "
      f"{retrieval_accuracy(lambda v: shorten(v, 32)):>11.0%}")
print(f"{'quantised to int8':>22} {128 * 1:>13} {'25%':>12} "
      f"{retrieval_accuracy(lambda v: dequantise(quantise(v))):>11.0%}")
print(f"{'short 64 + quantised':>22} {64 * 1:>13} {'12.5%':>12} "
      f"{retrieval_accuracy(lambda v: dequantise(quantise(shorten(v, 64)))):>11.0%}")
''',
    solution=r'''
def shorten(vector, dim):
    """Keep the first `dim` numbers and re-normalise to unit length."""
    short = np.asarray(vector, dtype=float)[:dim]
    length = np.linalg.norm(short)
    return short / length if length else short


def quantise(vector):
    """Store as int8: multiply by 127, round, cast to np.int8."""
    return np.round(np.asarray(vector, dtype=float) * 127).astype(np.int8)


def dequantise(vector_int8):
    """Back to floats for comparison: divide by 127."""
    return np.asarray(vector_int8, dtype=float) / 127


def retrieval_accuracy(transform, dim=128):
    """What fraction of queries still return the same best sentence after `transform`?

    transform takes a vector and returns a comparable vector.
    """
    full = [dense_embed(s, dim=dim) for s in SENTENCES]
    small = [transform(v) for v in full]

    correct = 0
    for query in QUERIES:
        q_full, q_small = dense_embed(query, dim=dim), transform(dense_embed(query, dim=dim))
        best_full = min(range(len(full)), key=lambda i: distance(q_full, full[i]))
        best_small = min(range(len(small)), key=lambda i: distance(q_small, small[i]))
        correct += (best_full == best_small)
    return correct / len(QUERIES)


bytes_full = 128 * 4           # 128 numbers, 4 bytes each (float32)

print(f"{'setup':>22} {'bytes/vector':>13} {'of original':>12} {'top-1 kept':>11}")
print(f"{'full 128-dim float32':>22} {bytes_full:>13} {'100%':>12} "
      f"{retrieval_accuracy(lambda v: v):>11.0%}")
print(f"{'shortened to 64':>22} {64 * 4:>13} {'50%':>12} "
      f"{retrieval_accuracy(lambda v: shorten(v, 64)):>11.0%}")
print(f"{'shortened to 32':>22} {32 * 4:>13} {'25%':>12} "
      f"{retrieval_accuracy(lambda v: shorten(v, 32)):>11.0%}")
print(f"{'quantised to int8':>22} {128 * 1:>13} {'25%':>12} "
      f"{retrieval_accuracy(lambda v: dequantise(quantise(v))):>11.0%}")
print(f"{'short 64 + quantised':>22} {64 * 1:>13} {'12.5%':>12} "
      f"{retrieval_accuracy(lambda v: dequantise(quantise(shorten(v, 64)))):>11.0%}")
''',
    check=r'''
@check("10.2  shortening keeps unit length")
def _():
    need("shorten")
    v = dense_embed(SENTENCES[0])
    short = shorten(v, 64)
    assert len(short) == 64, f"shorten(v, 64) returned {len(short)} numbers"
    assert abs(np.linalg.norm(short) - 1.0) < 1e-9, (
        f"the shortened vector has length {np.linalg.norm(short):.3f}. Re-normalise "
        "after slicing, or distances are no longer on the same scale as the rest of "
        "your store."
    )
    assert np.allclose(short[:5] / short[0], v[:5] / v[0]), \
        "shortening keeps the FIRST dim numbers — do not reorder or resample them"
    assert len(shorten(v, 128)) == 128, "shortening to the full length should work too"


@check("10.2  quantising is reversible to within rounding")
def _():
    need("quantise", "dequantise")
    v = dense_embed(SENTENCES[2])
    q = quantise(v)
    assert q.dtype == np.int8, (
        f"quantise returned dtype {q.dtype}. It must be np.int8 — that is where the "
        "4x memory saving comes from."
    )
    assert q.nbytes == len(v), f"128 int8 values should be 128 bytes, got {q.nbytes}"
    back = dequantise(q)
    assert np.abs(back - v).max() < 0.01, (
        f"the largest error after a round trip is {np.abs(back - v).max():.4f}, which "
        "is too big. Multiply by 127 before rounding, and divide by 127 coming back."
    )
    assert distance(back, v) < 1e-3, \
        "a quantised vector should still point in almost exactly the same direction"


@check("10.2  the cost is measured by retrieval, not by eye")
def _():
    need("retrieval_accuracy", "shorten", "quantise", "dequantise")
    full = retrieval_accuracy(lambda v: v)
    assert full == 1.0, (
        f"with no transformation at all, accuracy should be 100%, not {full:.0%} — "
        "the measurement itself is wrong"
    )
    quantised = retrieval_accuracy(lambda v: dequantise(quantise(v)))
    assert quantised >= 0.8, (
        f"quantising dropped top-1 accuracy to {quantised:.0%}. int8 should cost very "
        "little — check the round trip."
    )
    tiny = retrieval_accuracy(lambda v: shorten(v, 8))
    assert tiny < 1.0, (
        f"cutting a 128-number vector down to 8 still scored {tiny:.0%}. Something that "
        "aggressive should lose results — if it does not, the measurement is not "
        "actually comparing before and after."
    )
    print(f"     full {full:.0%}, int8 {quantised:.0%}, 8 numbers {tiny:.0%}")
''',
    hint=[
        "shorten:",
        "  short = np.asarray(vector, dtype=float)[:dim]",
        "  length = np.linalg.norm(short)",
        "  return short / length if length else short",
        "",
        "quantise:",
        "  return np.round(np.asarray(vector, dtype=float) * 127).astype(np.int8)",
        "",
        "The re-normalisation is the part people forget. Slicing a unit vector leaves",
        "something shorter than 1, and then its distances are not comparable with",
        "anything else in the store.",
    ],
    notice=r"""
**What to notice.** Quantising cut memory to a quarter and cost almost nothing. Shortening
to 8 numbers broke retrieval. In between there is a curve, and where your model sits on it
is a property of **that model** — so test each one.

> Memory savings are proportional and predictable. The accuracy cost is not: it depends on
> the model, so measure it on retrieval, with your own queries.
""",
)

EX3 = Exercise(
    name="A cache that cannot lie to you",
    section="10.3",
    concept=r"""
Embedding the same text twice is pure waste — the result is deterministic, and you paid for
it the first time. So everyone caches. Two rules, and both are about trust.

**1. Key the cache by everything that makes the vector what it is.**

Not just the text. The model, the vector length, the prefix — exactly the "space" from
Exercise 1. Key on text alone and the day you change model, your cache cheerfully serves
vectors from the old one, forever.

**2. Do not trust what you read back.**

A cache can be corrupted, truncated, written by an older version of your code, or shared
between processes that disagree. Check the shape of what comes out, and treat anything
unexpected as a miss.

> **A wrong vector from a cache looks exactly like a right one.** There is no exception
> raised, no log line, no slow query. Only worse answers.
""",
    todo=r'''
class EmbeddingCache:
    def __init__(self, model="small-v1", dim=128, prefix=""):
        self.model, self.dim, self.prefix = model, dim, prefix
        self.store = {}
        self.hits = self.misses = self.rejected = 0

    def key(self, text):
        """The cache key for this text under this setup."""
        # TODO 1: include everything that defines the vector: the text, the model,
        #         the dimension and the prefix. A tuple is fine.
        raise NotImplementedError

    def get(self, text):
        """Embed `text`, using the cache when it can be trusted."""
        key = self.key(text)
        cached = self.store.get(key)

        # TODO 2: if there is nothing cached, count a miss, embed with
        #         dense_embed(self.prefix + text, model=self.model, dim=self.dim),
        #         store it and return it.
        # TODO 3: if something IS cached, validate it before use: it must be an array
        #         of exactly self.dim numbers, all finite (np.isfinite(...).all()).
        #         If it fails, count self.rejected, treat it as a miss (re-embed,
        #         overwrite the bad entry) and return the fresh vector.
        # TODO 4: if it passes, count a hit and return it.
        raise NotImplementedError


cache = EmbeddingCache()
for text in SENTENCES + SENTENCES:          # every sentence twice
    cache.get(text)
print(f"hits {cache.hits}, misses {cache.misses}, rejected {cache.rejected}")

# Someone corrupts an entry: the wrong length, and a NaN.
cache.store[cache.key(SENTENCES[0])] = np.zeros(64)
cache.store[cache.key(SENTENCES[1])] = np.full(128, np.nan)
first, second = cache.get(SENTENCES[0]), cache.get(SENTENCES[1])
print(f"after corruption: rejected {cache.rejected}, "
      f"lengths back = {len(first)}, {len(second)}")

# A different model must not reuse the first model's vectors.
other = EmbeddingCache(model="big-v2")
other.store.update(cache.store)             # same dictionary contents
other.get(SENTENCES[0])
print(f"different model: hits {other.hits}, misses {other.misses} "
      "<- a miss is correct here")
''',
    solution=r'''
class EmbeddingCache:
    def __init__(self, model="small-v1", dim=128, prefix=""):
        self.model, self.dim, self.prefix = model, dim, prefix
        self.store = {}
        self.hits = self.misses = self.rejected = 0

    def key(self, text):
        """The cache key for this text under this setup."""
        return (text, self.model, self.dim, self.prefix)

    def _fresh(self, text, key):
        vector = dense_embed(self.prefix + text, model=self.model, dim=self.dim)
        self.store[key] = vector
        return vector

    def _is_valid(self, vector):
        return (isinstance(vector, np.ndarray)
                and vector.shape == (self.dim,)
                and bool(np.isfinite(vector).all()))

    def get(self, text):
        """Embed `text`, using the cache when it can be trusted."""
        key = self.key(text)
        cached = self.store.get(key)

        if cached is None:
            self.misses += 1
            return self._fresh(text, key)

        if not self._is_valid(cached):
            # Never use a vector that does not look right. A bad one raises no error,
            # it just makes every answer slightly worse.
            self.rejected += 1
            self.misses += 1
            return self._fresh(text, key)

        self.hits += 1
        return cached


cache = EmbeddingCache()
for text in SENTENCES + SENTENCES:          # every sentence twice
    cache.get(text)
print(f"hits {cache.hits}, misses {cache.misses}, rejected {cache.rejected}")

# Someone corrupts an entry: the wrong length, and a NaN.
cache.store[cache.key(SENTENCES[0])] = np.zeros(64)
cache.store[cache.key(SENTENCES[1])] = np.full(128, np.nan)
first, second = cache.get(SENTENCES[0]), cache.get(SENTENCES[1])
print(f"after corruption: rejected {cache.rejected}, "
      f"lengths back = {len(first)}, {len(second)}")

# A different model must not reuse the first model's vectors.
other = EmbeddingCache(model="big-v2")
other.store.update(cache.store)             # same dictionary contents
other.get(SENTENCES[0])
print(f"different model: hits {other.hits}, misses {other.misses} "
      "<- a miss is correct here")
''',
    check=r'''
@check("10.3  the cache saves work when it should")
def _():
    need("EmbeddingCache")
    c = EmbeddingCache()
    for text in SENTENCES:
        c.get(text)
    assert (c.hits, c.misses) == (0, len(SENTENCES)), \
        f"first pass should be all misses; got {c.hits} hits, {c.misses} misses"
    for text in SENTENCES:
        c.get(text)
    assert c.hits == len(SENTENCES), \
        f"the second pass should be all hits; got {c.hits}"
    assert np.allclose(c.get(SENTENCES[0]), dense_embed(SENTENCES[0])), \
        "a cached vector must equal the one dense_embed would produce"


@check("10.3  the key covers the whole setup")
def _():
    need("EmbeddingCache")
    base = EmbeddingCache(model="small-v1", dim=128, prefix="")
    base.get(SENTENCES[0])
    shared = dict(base.store)

    for changed in (EmbeddingCache(model="big-v2", dim=128, prefix=""),
                    EmbeddingCache(model="small-v1", dim=64, prefix=""),
                    EmbeddingCache(model="small-v1", dim=128, prefix="query: ")):
        changed.store.update(shared)
        changed.get(SENTENCES[0])
        assert changed.hits == 0, (
            f"a cache for {changed.model}/{changed.dim}/{changed.prefix!r} reused an "
            "entry made by a different setup. Key on the text AND the model AND the "
            "dimension AND the prefix — otherwise changing model silently serves old "
            "vectors forever."
        )
    print("     model, dimension and prefix all change the key")


@check("10.3  a bad cached vector is refused, not used")
def _():
    need("EmbeddingCache")
    c = EmbeddingCache()
    c.get(SENTENCES[0])

    for bad in (np.zeros(64),                    # wrong length
                np.full(128, np.nan),            # not finite
                np.full(128, np.inf),
                "not an array at all"):
        c.store[c.key(SENTENCES[0])] = bad
        before = c.rejected
        got = c.get(SENTENCES[0])
        assert c.rejected == before + 1, (
            f"the cached value {type(bad).__name__} of size "
            f"{getattr(bad, 'shape', 'n/a')} was accepted. Validate before use: an "
            "array of exactly dim finite numbers, or treat it as a miss."
        )
        assert isinstance(got, np.ndarray) and got.shape == (128,), \
            "after rejecting a bad entry, return a freshly computed vector"
        assert np.isfinite(got).all(), "the replacement must be a usable vector"
        assert np.allclose(c.store[c.key(SENTENCES[0])], got), \
            "overwrite the bad entry, or the next read hits it again"
    print(f"     4 kinds of corrupt entry caught, {c.rejected} rejections recorded")
''',
    hint=[
        "key():",
        "  return (text, self.model, self.dim, self.prefix)",
        "",
        "get(), three branches:",
        "  if cached is None:          miss  -> embed, store, return",
        "  if not valid(cached):       reject + miss -> embed, OVERWRITE, return",
        "  otherwise:                  hit   -> return cached",
        "",
        "valid means:",
        "  isinstance(v, np.ndarray) and v.shape == (self.dim,)",
        "  and bool(np.isfinite(v).all())",
    ],
    notice=r"""
**What to notice.** Both failures in this exercise are invisible without the checks you
wrote. A cache keyed on text alone keeps working after a model change — it just returns the
wrong vectors. A corrupted entry produces distances, not errors.

That is the theme of this whole chapter: **embedding mistakes do not crash, they degrade.**
Record the setup, refuse what does not match, and count what you reject — so there is at
least one number that would have told you.
""",
)

CELLS, TITLE, SOLUTION_BANNER = make_chapter(
    number=10,
    title="Embeddings: Practical Implementation Guide",
    intro=r"""
Embeddings are the layer underneath Chapters 7 to 9 — the step that turns text into the
numbers everything else searches.

They have one unusual property, and this notebook is organised around it: **embedding
mistakes do not raise errors.** They quietly make every answer a little worse. So all three
exercises are about making the invisible visible.
""",
    minutes=35,
    given=GIVEN,
    exercises=[EX1, EX2, EX3],
    learned=[
        "An embedding is a position in a space made by **one setup**: model, vector "
        "length, prefix. Vectors from different setups are not comparable.",
        "**A mix-up raises no error** — so record the setup next to the data and refuse "
        "comparisons across spaces.",
        "**Shortening** and **quantising** cut memory proportionally and cost accuracy by "
        "an amount that depends on the model.",
        "Re-normalise after shortening, or distances stop being comparable.",
        "Judge compression by **retrieval**, with your own queries — not by one pair of "
        "sentences.",
        "Key a cache by **everything that makes the vector**, and validate what you read "
        "back: a wrong cached vector looks exactly like a right one.",
    ],
    next_up="Chapter 11 moves to MCP — the protocol that lets an agent use tools it did "
            "not ship with.",
)
