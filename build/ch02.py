"""Chapter 2 — LLM Mechanics: API Calls, Tokens, and Temperature.

Maps to sections 2.1-2.5 of the Developer Guide (scripts Ex-07 to Ex-11).

Design note: every exercise here must be completable and checkable with *no API key
and no spend*. The setup cell therefore installs a stand-in client that mimics the
OpenAI response shape. A reader with a key gets real calls; a reader without one gets
the same exercises and the same checks.
"""

from harness import HARNESS
from nbtools import code, ex, md

TITLE = "Chapter_02_LLM_Mechanics"

CELLS = [

md(r"""
# Chapter 2 — LLM Mechanics: API Calls, Tokens, and Temperature
### Exercise notebook · *Developer Guide: Building with AI*

In Chapter 1 you trained a network yourself. From here on you use models someone else
trained, through an API — and the handful of mechanics in this notebook are the ones that
decide whether what you build is reliable and affordable.

| Exercise | Book section | What you write |
|---|---|---|
| 1 | 2.1 | Your first API call, reading `usage` and `finish_reason` |
| 2 | 2.2 | Counting tokens, and seeing what a token really is |
| 3 | 2.3 | Temperature, implemented from scratch |
| 4 | 2.4 | A sliding-window conversation manager |
| 5 | 2.5 | A chatbot — and why its cost grows every turn |
| 6 | summary | Combine 4 and 5; measure the difference |

**Time:** about 60 minutes.

---

### 💳 You do not need an API key to finish this notebook

Every exercise runs in one of two modes, and the setup cell picks for you:

| Mode | When | What happens |
|---|---|---|
| **LIVE** | you provide a key | real calls to `gpt-4o-mini`; a few tenths of a cent in total |
| **MOCK** | no key found | a stand-in client with the same response shape; free, instant, offline |

Every check passes in both modes, because the checks test **your code**, not the model's
wording. Tokens, temperature and context management are all genuinely learnable in MOCK
mode — only the model's prose is fake.

**To run LIVE in Colab** (optional): click the 🔑 key icon in the left sidebar, add a
secret named `OPENAI_API_KEY`, enable notebook access, then re-run the setup cell.
Never paste a key into a notebook cell — a notebook you share carries its cell contents
with it, and a leaked key can be spent by whoever finds it.

### How to use this notebook

Run the setup cells first, go in order, and after each exercise run the check cell under
it. `# TODO` marks your work; delete the `raise NotImplementedError` when you are done.
Call `progress()` at any time, and `hint(...)` when you have been stuck for ten minutes.
"""),

code(HARNESS, form=True),

code(r"""
%pip install -q openai tiktoken
""", form=False),

code(r"""
#@title Client setup — finds your key, or installs a stand-in (double-click to read)
import os, random, textwrap

import tiktoken

ENC = tiktoken.encoding_for_model("gpt-4")        # the cl100k_base vocabulary
MODEL = "gpt-4o-mini"


def _find_key():
    try:                                          # Colab secrets (the 🔑 sidebar icon)
        from google.colab import userdata
        key = userdata.get("OPENAI_API_KEY")
        if key:
            return key.strip(), "Colab secret"
    except Exception:
        pass
    key = os.environ.get("OPENAI_API_KEY")        # local .env / shell export
    return (key.strip(), "environment variable") if key else (None, None)


# ---- the stand-in -----------------------------------------------------------
# It mimics just enough of the OpenAI response shape for this chapter:
#   response.choices[0].message.content / .finish_reason
#   response.usage.prompt_tokens / .completion_tokens / .total_tokens
class _Message:
    def __init__(self, content): self.content = content
class _Choice:
    def __init__(self, content, finish_reason):
        self.message, self.finish_reason = _Message(content), finish_reason
class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens, self.completion_tokens, self.total_tokens = p, c, p + c
class _Response:
    def __init__(self, content, finish_reason, p, c):
        self.choices, self.usage = [_Choice(content, finish_reason)], _Usage(p, c)
        self.model = MODEL

_CANNED = {
    "name": ["I'm an AI assistant. You can call me Assistant.",
             "I don't have a name, but Assistant works.",
             "You can call me Assistant — I'm a language model."],
    "neural": ["A neural network is a stack of simple maths functions whose weights are "
               "adjusted until its predictions match the data.",
               "Neural networks learn patterns by nudging weighted connections until "
               "their output matches the examples they were shown."],
    "sky": ["blue", "clear", "grey"],
    "default": ["That depends on what you are trying to do — tell me a little more.",
                "Here is a short answer: it depends on the specifics of your case.",
                "I can help with that. What outcome are you aiming for?"],
}

class _MockCompletions:
    def create(self, model=MODEL, messages=(), temperature=0.7, max_tokens=None, **kw):
        messages = list(messages)
        prompt_text = "".join(m.get("content", "") for m in messages)
        last_user = next((m["content"] for m in reversed(messages)
                          if m.get("role") == "user"), "")

        # Pick a bucket of canned replies, then let temperature decide how much it varies.
        low = last_user.lower()
        bucket = next((k for k in ("name", "neural", "sky") if k in low), "default")
        options = _CANNED[bucket]
        rng = random.Random(0 if temperature <= 0 else None)
        reply = options[0] if temperature <= 0.01 else rng.choice(options)

        # An earlier question is answerable only if it is still in `messages` —
        # exactly like the real thing, which is the point of section 2.5.
        if "first question" in low or "what did i" in low:
            earlier = [m["content"] for m in messages[:-1] if m.get("role") == "user"]
            reply = (f'Your first question was: "{earlier[0]}"' if earlier
                     else "I don't have an earlier question from you in this conversation.")
        if any(m.get("role") == "system" and "pirate" in m.get("content", "").lower()
               for m in messages):
            reply = "Arr! " + reply

        finish_reason = "stop"
        completion = ENC.encode(reply)
        if max_tokens is not None and len(completion) > max_tokens:
            completion = completion[:max_tokens]
            reply = ENC.decode(completion)
            finish_reason = "length"             # cut off, exactly as the API reports it

        prompt_tokens = len(ENC.encode(prompt_text)) + 3 * len(messages)   # chat overhead
        return _Response(reply, finish_reason, prompt_tokens, len(completion))

class _MockChat:
    def __init__(self): self.completions = _MockCompletions()
class MockOpenAI:
    '''A stand-in with the same call signature as openai.OpenAI.'''
    def __init__(self, *a, **kw): self.chat = _MockChat()


def get_client():
    '''Return (client, mode). mode is "LIVE" or "MOCK".'''
    key, source = _find_key()
    if key:
        from openai import OpenAI
        print(f"LIVE mode — key loaded from your {source}. Calls cost real money "
              "(a few tenths of a cent for this notebook).")
        return OpenAI(api_key=key), "LIVE"
    print(textwrap.dedent('''\
        MOCK mode — no OPENAI_API_KEY found, so a stand-in client is in use.
        Every exercise and every check works. Replies are canned; token counts,
        finish_reason and the conversation mechanics are real.
        To go live: add OPENAI_API_KEY as a Colab secret (🔑 in the sidebar) and re-run.'''))
    return MockOpenAI(), "MOCK"


client, MODE = get_client()
"""),

# ------------------------------------------------------------------ 2.1
md(r"""
---
## Exercise 1 — Your first API call  ·  *book §2.1 (Ex-07.py)*

An API call sends a **list of messages** and gets one reply back. That is the building
block under every chatbot and every agent in this book.

Three roles make up the list:

| Role | Purpose |
|---|---|
| `system` | instructions that set behaviour — "You are a helpful assistant" |
| `user` | what the person is asking |
| `assistant` | the model's *earlier* replies, which **you** add back in to continue a conversation |

Two fields of the response matter more than beginners expect, and reading them is a habit
worth forming now:

* **`usage`** — the exact token counts you are billed for.
* **`finish_reason`** — `"stop"` means the model finished its thought; `"length"` means
  `max_tokens` cut it off mid-sentence. A reply that ends abruptly is almost always this.

> **Key idea:** each call is completely independent. The model remembers nothing between
> calls, so everything it needs must be in the messages you send.
"""),

ex(r"""
def ask(client, question, system="You are a helpful assistant.",
        temperature=0.7, max_tokens=50):
    '''Send one question and return a dict of the parts that matter.'''
    # TODO 1: call client.chat.completions.create(...) with
    #           model=MODEL
    #           messages=[{"role": "system", ...}, {"role": "user", ...}]
    #           temperature=temperature, max_tokens=max_tokens
    # TODO 2: pull out choice = response.choices[0]
    # TODO 3: return this exact dict:
    #           {"text": ..., "finish_reason": ...,
    #            "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
    #         The token counts live on response.usage.
    raise NotImplementedError


result = ask(client, "Explain neural networks in one sentence.")
print(result["text"])
print(f"\nPrompt tokens:     {result['prompt_tokens']}")
print(f"Completion tokens: {result['completion_tokens']}")
print(f"Total tokens:      {result['total_tokens']}")
print(f"Finish reason:     {result['finish_reason']}")
""", r"""
def ask(client, question, system="You are a helpful assistant.",
        temperature=0.7, max_tokens=50):
    '''Send one question and return a dict of the parts that matter.'''
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    return {
        "text": choice.message.content,
        "finish_reason": choice.finish_reason,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }


result = ask(client, "Explain neural networks in one sentence.")
print(result["text"])
print(f"\nPrompt tokens:     {result['prompt_tokens']}")
print(f"Completion tokens: {result['completion_tokens']}")
print(f"Total tokens:      {result['total_tokens']}")
print(f"Finish reason:     {result['finish_reason']}")
"""),

code(r"""
@check("2.1  ask() returns the text, the usage and the finish reason")
def _():
    need("ask", "result")
    needed = {"text", "finish_reason", "prompt_tokens", "completion_tokens", "total_tokens"}
    missing = needed - set(result)
    assert not missing, f"the returned dict is missing: {sorted(missing)}"
    assert isinstance(result["text"], str) and result["text"], \
        "text is empty — the reply lives at response.choices[0].message.content"
    assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"], \
        "total_tokens should equal prompt + completion; read all three from response.usage"
    assert result["prompt_tokens"] > 0, "prompt_tokens should count the messages you sent"


@check("2.1  a short max_tokens is reported as finish_reason='length'")
def _():
    need("ask")
    cut = ask(client, "Write a detailed paragraph about the history of computing.",
              max_tokens=5)
    assert cut["finish_reason"] == "length", (
        f"finish_reason was {cut['finish_reason']!r}. With max_tokens=5 the model is cut "
        "off, and the API says so. Make sure you pass max_tokens through to create()."
    )
    assert cut["completion_tokens"] <= 5, \
        f"the reply used {cut['completion_tokens']} tokens despite max_tokens=5"
    print(f"     truncated reply: {cut['text']!r}")
"""),

code(r"""
hint("The call itself:",
     "  response = client.chat.completions.create(",
     "      model=MODEL,",
     "      messages=[{'role': 'system', 'content': system},",
     "                {'role': 'user',   'content': question}],",
     "      temperature=temperature, max_tokens=max_tokens)",
     "",
     "Then: choice = response.choices[0]",
     "      choice.message.content   <- the text",
     "      choice.finish_reason     <- 'stop' or 'length'",
     "      response.usage.prompt_tokens / .completion_tokens / .total_tokens")
"""),

md(r"""
**When it goes wrong** (the real API, LIVE mode):

| Symptom | Likely cause |
|---|---|
| `OPENAI_API_KEY is not set` | the secret is missing, or notebook access is not enabled for it |
| an authentication error | the key is wrong, revoked, or has quotes or spaces around it |
| a rate-limit error | too many requests too fast, or no credit left on the account |
| the reply stops mid-sentence | `finish_reason` is `"length"` — raise `max_tokens` |
"""),

# ------------------------------------------------------------------ 2.2
md(r"""
---
## Exercise 2 — Tokens  ·  *book §2.2 (Ex-08.py)*

A **token** is the unit a model reads, writes and bills in. It is roughly three-quarters
of an English word: common short words are usually one token, longer or rarer words get
split into pieces.

This exercise runs completely offline — `tiktoken` is OpenAI's tokenizer, and no model is
involved. Seeing the split for yourself is what makes cost and context stop being abstract.
"""),

ex(r"""
def count_tokens(text, model="gpt-4"):
    '''How many tokens does this text become?'''
    # TODO: encode the text with tiktoken.encoding_for_model(model) and return the length.
    #       (ENC is already the gpt-4 encoding, but take `model` so this works for others.)
    raise NotImplementedError


def token_pieces(text, model="gpt-4"):
    '''Return the text split the way the model actually sees it, as a list of strings.'''
    # TODO: encode the text, then decode each token id ON ITS OWN, so you get the pieces.
    #       enc.decode([single_id]) gives one piece back as a string.
    raise NotImplementedError


text = "Large language models are fascinating."
print(f"Text:        {text}")
print(f"Token count: {count_tokens(text)}")
print(f"Pieces:      {token_pieces(text)}")

print()
for word in ["helping", "extraordinary", "Tokyo", "東京"]:
    pieces = token_pieces(word)
    print(f"{word!r} -> {len(pieces)} token(s): {pieces}")
""", r"""
def count_tokens(text, model="gpt-4"):
    '''How many tokens does this text become?'''
    return len(tiktoken.encoding_for_model(model).encode(text))


def token_pieces(text, model="gpt-4"):
    '''Return the text split the way the model actually sees it, as a list of strings.'''
    enc = tiktoken.encoding_for_model(model)
    return [enc.decode([token_id]) for token_id in enc.encode(text)]


text = "Large language models are fascinating."
print(f"Text:        {text}")
print(f"Token count: {count_tokens(text)}")
print(f"Pieces:      {token_pieces(text)}")

print()
for word in ["helping", "extraordinary", "Tokyo", "東京"]:
    pieces = token_pieces(word)
    print(f"{word!r} -> {len(pieces)} token(s): {pieces}")
"""),

code(r"""
@check("2.2  token counting and splitting")
def _():
    need("count_tokens", "token_pieces")
    assert count_tokens("Large language models are fascinating.") == 6, (
        "this sentence is 6 tokens in the gpt-4 vocabulary, you got "
        f"{count_tokens('Large language models are fascinating.')}. "
        "Are you counting characters or words by mistake?"
    )
    assert count_tokens("") == 0, "empty text is zero tokens"

    pieces = token_pieces("Large language models are fascinating.")
    assert "".join(pieces) == "Large language models are fascinating.", \
        "joining the pieces back together must reproduce the original text exactly"
    assert pieces[1] == " language", (
        f"the second piece should be ' language' WITH its leading space, you got "
        f"{pieces[1]!r}. Spaces belong to tokens — that is why the same word costs a "
        "different amount at the start of a sentence than in the middle."
    )
    assert len(token_pieces("helping")) == 2, "'helping' splits into 'help' + 'ing'"
    print(f"     'extraordinary' -> {token_pieces('extraordinary')}")
"""),

ex(r"""
# Now use it. A real cost question: does the language you write in change the bill?
english = ("Artificial intelligence is changing how software is built, and developers "
           "who understand it will build better products.")
other = ("人工知能はソフトウェアの作り方を変えており、それを理解する開発者は"
         "より良い製品を作るでしょう。")

# TODO: print the token count of each, and the ratio between them.
#       Roughly how many tokens per word is the English text?
raise NotImplementedError
""", r"""
# Now use it. A real cost question: does the language you write in change the bill?
english = ("Artificial intelligence is changing how software is built, and developers "
           "who understand it will build better products.")
other = ("人工知能はソフトウェアの作り方を変えており、それを理解する開発者は"
         "より良い製品を作るでしょう。")

en_tokens, other_tokens = count_tokens(english), count_tokens(other)
print(f"English:  {en_tokens} tokens for {len(english.split())} words "
      f"({en_tokens / len(english.split()):.2f} tokens per word)")
print(f"Japanese: {other_tokens} tokens")
print(f"Ratio:    {other_tokens / en_tokens:.2f}x the tokens for the same meaning")
print("\nNon-English text usually costs more tokens per unit of meaning. If your users "
      "are not writing in English, that is a line item, not a detail.")
"""),

md(r"""
**What to notice.**

* **Encoding and decoding are inverses.** The model only ever sees the integer IDs.
* **Spaces belong to tokens.** `' language'` carries its leading space.
* **Token IDs belong to a tokenizer.** A different model has a different vocabulary, so
  the same text gives different IDs and sometimes a different count.
* **`東京` is two characters but three tokens**, and printing one of those tokens alone
  shows a `�` because a single character's bytes were split across tokens. That is expected.

**Two limits of counting ahead of time.** The chat format adds a few tokens of overhead per
message, so summing message text slightly underestimates the real total. And `tiktoken` is
for *estimating before you send* — after a call, the exact figures are in `response.usage`.
"""),

# ------------------------------------------------------------------ 2.3
md(r"""
---
## Exercise 3 — Temperature  ·  *book §2.3 (Ex-09.py)*

The model does not output a word. It outputs a **probability for every token in its
vocabulary** — say 40% `blue`, 15% `clear`, 9% `grey`, and a long tail of unlikely
alternatives. Temperature decides how strictly it commits to its own favourite.

It works by dividing the raw scores (logits) by $T$ before the softmax:

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

* **Low $T$** (say 0.2) — divide by a small number, the gaps between scores stretch, the
  favourite dominates. Predictable. Right for extraction, classification, code.
* **$T = 1$** — the model's honest distribution, long shots included.
* **High $T$** (say 1.5+) — the gaps flatten, unlikely words get a real chance. Creative,
  and eventually incoherent.

Temperature does **not** change what the model believes is most likely. It changes how
strictly it commits to that belief. And it cannot be exactly 0 in this formula — you
cannot divide by zero — so $T = 0$ is special-cased to "always take the top choice".
"""),

ex(r"""
import numpy as np

words = ["blue", "clear", "grey", "green", "purple"]
logits = np.log([0.40, 0.15, 0.09, 0.05, 0.02])     # the model's raw scores


def softmax_with_temperature(logits, temperature):
    '''Turn scores into probabilities. Lower temperature = sharper peak.'''
    # TODO 1: scaled = logits / temperature
    # TODO 2: exps = np.exp(scaled - scaled.max())
    #         Subtracting the max changes nothing mathematically and stops np.exp from
    #         overflowing on large scores. Always do it.
    # TODO 3: return exps / exps.sum()      (so the probabilities add to 1)
    raise NotImplementedError


temps = [0.2, 0.5, 1.0, 1.5]
print(f"{'':>8}" + "".join(f"{w:>9}" for w in words))
for t in temps:
    probs = softmax_with_temperature(logits, t)
    print(f"T = {t:<4}" + "".join(f"{p:9.1%}" for p in probs))

# Sample 1000 words at each temperature to see how often the favourite actually wins
rng = np.random.default_rng(0)
print()
for t in temps:
    probs = softmax_with_temperature(logits, t)
    picks = rng.choice(len(words), size=1000, p=probs)
    print(f"T = {t:<4} picked '{words[0]}' {np.mean(picks == 0):.1%} of the time")
""", r"""
import numpy as np

words = ["blue", "clear", "grey", "green", "purple"]
logits = np.log([0.40, 0.15, 0.09, 0.05, 0.02])     # the model's raw scores


def softmax_with_temperature(logits, temperature):
    '''Turn scores into probabilities. Lower temperature = sharper peak.'''
    scaled = logits / temperature
    exps = np.exp(scaled - scaled.max())            # subtract the max for stability
    return exps / exps.sum()


temps = [0.2, 0.5, 1.0, 1.5]
print(f"{'':>8}" + "".join(f"{w:>9}" for w in words))
for t in temps:
    probs = softmax_with_temperature(logits, t)
    print(f"T = {t:<4}" + "".join(f"{p:9.1%}" for p in probs))

# Sample 1000 words at each temperature to see how often the favourite actually wins
rng = np.random.default_rng(0)
print()
for t in temps:
    probs = softmax_with_temperature(logits, t)
    picks = rng.choice(len(words), size=1000, p=probs)
    print(f"T = {t:<4} picked '{words[0]}' {np.mean(picks == 0):.1%} of the time")
"""),

code(r"""
@check("2.3  softmax with temperature")
def _():
    need("softmax_with_temperature")
    for t in (0.1, 0.5, 1.0, 2.0):
        probs = softmax_with_temperature(logits, t)
        assert abs(probs.sum() - 1) < 1e-9, \
            f"at T={t} the probabilities sum to {probs.sum():.4f}, not 1 — divide by exps.sum()"
        assert (probs >= 0).all(), "probabilities cannot be negative"

    at1 = softmax_with_temperature(logits, 1.0)
    assert np.allclose(at1, [0.40, 0.15, 0.09, 0.05, 0.02] / np.sum([0.40, 0.15, 0.09, 0.05, 0.02]),
                       atol=1e-6), \
        "at T=1 the probabilities should match the original distribution"

    cold, hot = softmax_with_temperature(logits, 0.2), softmax_with_temperature(logits, 1.5)
    assert cold[0] > at1[0] > hot[0], (
        "lower temperature must make the top choice MORE likely and higher temperature "
        f"less: you got {cold[0]:.3f} / {at1[0]:.3f} / {hot[0]:.3f} at T=0.2 / 1.0 / 1.5. "
        "Check you divide the logits by T rather than multiplying."
    )
    assert cold.argmax() == hot.argmax() == 0, \
        "temperature must not change WHICH word is most likely, only by how much"

    big = softmax_with_temperature(np.array([800.0, 1.0, 2.0]), 0.5)
    assert np.isfinite(big).all(), \
        "overflow on large logits — subtract scaled.max() before np.exp"
    print(f"     T=0.2 favourite {cold[0]:.1%} → T=1.5 favourite {hot[0]:.1%}")
"""),

code(r"""
# A picture of the same scores at four temperatures.
import matplotlib.pyplot as plt

x = np.arange(len(words))
plt.figure(figsize=(7, 4))
for i, t in enumerate(temps):
    plt.bar(x + i * 0.2 - 0.3, softmax_with_temperature(logits, t), width=0.2, label=f"T = {t}")
plt.xticks(x, words); plt.ylabel("probability")
plt.title("The same scores, reshaped by temperature")
plt.legend(); plt.show()
"""),

md(r"""
> **Try it:** add `0.1` and `3.0` to `temps` and re-run. At 0.1 the favourite takes almost
> everything; at 3.0 the distribution is nearly flat and `purple` becomes a live option.
> Then ask yourself why the formula cannot take $T = 0$ at all.

**In practice:** near 0 for anything with a right answer — extraction, classification,
code, routing. 0.7 to 1.0 for open-ended writing. And note that a low temperature buys
*consistency*, not *correctness*: a confidently wrong model is still wrong at $T = 0$.
"""),

# ------------------------------------------------------------------ 2.4
md(r"""
---
## Exercise 4 — The context window  ·  *book §2.4 (Ex-10.py)*

The **context window** is everything the model can see at once. Inside it, a fact can
influence the answer. Outside it, the fact does not exist, however relevant it is.

A long conversation eventually exceeds the window, and gets slower and dearer well before
that. The simplest policy is a **sliding window**: when the conversation goes over a token
budget, drop the oldest message — but *never* the system prompt, which holds the
instructions.

Index 0 is the system prompt, so index 1 is always the oldest message you are allowed to
drop. That single detail is the exercise.
"""),

ex(r"""
class ConversationManager:
    '''Keeps a conversation inside a token budget by dropping the oldest messages.'''

    def __init__(self, max_tokens=4000, model="gpt-4o-mini"):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)   # load the tokenizer ONCE
        self.messages = []
        self.total_tokens = 0

    def count(self, content):
        return len(self.encoding.encode(content))

    def add_message(self, role, content, verbose=True):
        '''Add a message, evicting old ones while we are over budget.'''
        # TODO 1: append {"role": role, "content": content} to self.messages
        # TODO 2: add this message's token count to self.total_tokens
        # TODO 3: WHILE we are over max_tokens and more than one message remains:
        #           pop the message at index 1 (index 0 is the system prompt - keep it),
        #           subtract its tokens from self.total_tokens,
        #           and if verbose, print f"  [evicted {removed['role']!r}: ...]"
        raise NotImplementedError

    def get_messages(self):
        return self.messages


conv = ConversationManager(max_tokens=20)          # tiny budget, so you can watch it work
conv.add_message("system", "You are a helpful assistant.")
conv.add_message("user", "Hello! This is a short message.")
conv.add_message("assistant", "Hi there!")
print(f"Tokens used: {conv.total_tokens} / {conv.max_tokens}")

print("\nAdding a longer message...")
conv.add_message("user", "This is another message that's slightly longer.")
print(f"Tokens used: {conv.total_tokens} / {conv.max_tokens}")

print("\nMessages the model would see:")
for message in conv.get_messages():
    print(f"  {message['role']:>9}: {message['content']}")
""", r"""
class ConversationManager:
    '''Keeps a conversation inside a token budget by dropping the oldest messages.'''

    def __init__(self, max_tokens=4000, model="gpt-4o-mini"):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)   # load the tokenizer ONCE
        self.messages = []
        self.total_tokens = 0

    def count(self, content):
        return len(self.encoding.encode(content))

    def add_message(self, role, content, verbose=True):
        '''Add a message, evicting old ones while we are over budget.'''
        self.messages.append({"role": role, "content": content})
        self.total_tokens += self.count(content)

        # Index 0 is the system prompt, so index 1 is the oldest droppable message.
        while self.total_tokens > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(1)
            self.total_tokens -= self.count(removed["content"])
            if verbose:
                print(f"  [evicted {removed['role']!r}: {removed['content']!r}]")

    def get_messages(self):
        return self.messages


conv = ConversationManager(max_tokens=20)          # tiny budget, so you can watch it work
conv.add_message("system", "You are a helpful assistant.")
conv.add_message("user", "Hello! This is a short message.")
conv.add_message("assistant", "Hi there!")
print(f"Tokens used: {conv.total_tokens} / {conv.max_tokens}")

print("\nAdding a longer message...")
conv.add_message("user", "This is another message that's slightly longer.")
print(f"Tokens used: {conv.total_tokens} / {conv.max_tokens}")

print("\nMessages the model would see:")
for message in conv.get_messages():
    print(f"  {message['role']:>9}: {message['content']}")
"""),

code(r"""
@check("2.4  the sliding window evicts, and keeps the system prompt")
def _():
    need("ConversationManager")
    c = ConversationManager(max_tokens=20)
    c.add_message("system", "You are a helpful assistant.", verbose=False)
    for i in range(12):
        c.add_message("user", f"Message number {i} with some extra words.", verbose=False)

    assert c.total_tokens <= c.max_tokens, (
        f"{c.total_tokens} tokens held against a budget of {c.max_tokens} — the while "
        "loop is not evicting. Is it an `if` instead of a `while`?"
    )
    assert c.messages[0]["role"] == "system", (
        "the system prompt was dropped. Pop index 1, never index 0 — losing the system "
        "prompt silently changes the model's behaviour mid-conversation."
    )
    assert len(c.messages) < 13, "nothing was evicted at all"
    assert c.messages[-1]["content"].endswith("extra words."), \
        "the newest message must survive — evict from the front, not the back"

    exact = sum(c.count(m["content"]) for m in c.messages)
    assert exact == c.total_tokens, (
        f"total_tokens says {c.total_tokens} but the messages actually hold {exact}. "
        "Subtract the evicted message's tokens when you pop it."
    )
    print(f"     12 messages added, {len(c.messages)} kept, {c.total_tokens}/20 tokens, "
          "system prompt intact")
"""),

code(r"""
hint("add_message, in order:",
     "  self.messages.append({'role': role, 'content': content})",
     "  self.total_tokens += self.count(content)",
     "  while self.total_tokens > self.max_tokens and len(self.messages) > 1:",
     "      removed = self.messages.pop(1)",
     "      self.total_tokens -= self.count(removed['content'])",
     "",
     "`while`, not `if`: one new message may push out several old ones.",
     "pop(1), not pop(0): index 0 is the system prompt and must stay.")
"""),

md(r"""
**Now criticise what you just built** — the chapter does, and the criticism is the useful
part:

* **Orphaned replies.** In the output above, the assistant's "Hi there!" survives after the
  question it answered was dropped. Real systems evict a question and its answer together.
* **Oldest is not least important.** The first message often holds the user's actual
  requirement. Dropping by age alone can throw away the point of the conversation.
* **Overhead is ignored.** This counts message text only, not the per-message overhead of
  the chat format.
* **Better policies exist.** Summarise old turns into one short message; *pin* facts that
  may never be dropped; or retrieve only the relevant older messages on demand — which is
  Chapter 7's RAG, arriving for exactly this reason.

And remember the **"lost in the middle"** effect: information at the very start or very end
of a long context tends to carry more weight than material buried in the middle. A big
window full of text does not guarantee the model uses it all equally well.

> **Key idea:** context management is a decision about what deserves a seat in a limited
> window. A sliding window is the simplest policy, not the best one.
"""),

# ------------------------------------------------------------------ 2.5
md(r"""
---
## Exercise 5 — A chatbot  ·  *book §2.5 (Ex-11.py)*

Now put it together. A chatbot is a **loop around a stateless API**. Its "memory" is a
Python list that you resend in full on every turn. The model keeps nothing.

That single fact explains the whole cost curve: turn 10 pays for turns 1 through 9 again.
Watch `total_tokens` climb faster than the number of turns.
"""),

ex(r"""
class SimpleChatbot:
    def __init__(self, client, model=MODEL, temperature=0.7,
                 system_prompt="You are a helpful assistant."):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.reset()

    def reset(self):
        '''Start over: only the system prompt, and a zeroed token counter.'''
        # TODO 1: self.conversation = a list holding ONLY the system message
        # TODO 2: self.total_tokens = 0
        raise NotImplementedError

    def chat(self, user_input):
        # TODO 3: append the user's message to self.conversation
        # TODO 4: send the WHOLE conversation to the API (that is the memory)
        # TODO 5: take the reply text, append it as a {"role": "assistant", ...} message
        #         — forget this and the bot will not remember its own answers
        # TODO 6: add response.usage.total_tokens to self.total_tokens
        # TODO 7: return the reply
        raise NotImplementedError


bot = SimpleChatbot(client)
print("You: Hi! What's your name?")
print("Bot:", bot.chat("Hi! What's your name?"))
print("\nYou: What was my first question?")
print("Bot:", bot.chat("What was my first question?"))
print(f"\nMessages stored: {len(bot.conversation)}, tokens used: {bot.total_tokens}")

bot.reset()
print("\n--- after reset ---")
print("You: What was my first question?")
print("Bot:", bot.chat("What was my first question?"))
""", r"""
class SimpleChatbot:
    def __init__(self, client, model=MODEL, temperature=0.7,
                 system_prompt="You are a helpful assistant."):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.reset()

    def reset(self):
        '''Start over: only the system prompt, and a zeroed token counter.'''
        self.conversation = [{"role": "system", "content": self.system_prompt}]
        self.total_tokens = 0

    def chat(self, user_input):
        self.conversation.append({"role": "user", "content": user_input})

        # The model has no memory: the whole conversation goes every time.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            temperature=self.temperature,
        )
        reply = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": reply})
        self.total_tokens += response.usage.total_tokens
        return reply


bot = SimpleChatbot(client)
print("You: Hi! What's your name?")
print("Bot:", bot.chat("Hi! What's your name?"))
print("\nYou: What was my first question?")
print("Bot:", bot.chat("What was my first question?"))
print(f"\nMessages stored: {len(bot.conversation)}, tokens used: {bot.total_tokens}")

bot.reset()
print("\n--- after reset ---")
print("You: What was my first question?")
print("Bot:", bot.chat("What was my first question?"))
"""),

code(r"""
@check("2.5  the chatbot keeps the conversation, and reset clears it")
def _():
    need("SimpleChatbot")
    b = SimpleChatbot(client)
    assert len(b.conversation) == 1 and b.conversation[0]["role"] == "system", \
        "a fresh bot holds exactly one message: the system prompt"
    assert b.total_tokens == 0, "reset() must zero the token counter too"

    b.chat("Remember the number 7.")
    assert len(b.conversation) == 3, (
        f"after one turn the list should hold system + user + assistant = 3 messages, "
        f"yours holds {len(b.conversation)}. Append the model's reply as well as the "
        "user's message, or the bot forgets what it just said."
    )
    assert b.conversation[2]["role"] == "assistant", \
        "the third message must be the assistant's reply"
    assert b.total_tokens > 0, "add response.usage.total_tokens to the running total"

    b.chat("And now?")
    assert len(b.conversation) == 5, "each turn adds two messages"

    before = b.total_tokens
    b.reset()
    assert len(b.conversation) == 1 and b.total_tokens == 0, \
        "reset() should put the bot back to a single system message and zero tokens"
    print(f"     2 turns -> 5 messages, {before} tokens, reset back to 1 message")


@check("2.5  cost really does grow with every turn")
def _():
    need("SimpleChatbot")
    b = SimpleChatbot(client)
    per_turn = []
    for i in range(4):
        before = b.total_tokens
        b.chat(f"Question {i}: tell me something about software engineering.")
        per_turn.append(b.total_tokens - before)
    assert per_turn[-1] > per_turn[0], (
        f"turn costs were {per_turn} — later turns should cost MORE, because the whole "
        "history is resent. If they are flat, the conversation is not accumulating."
    )
    print(f"     tokens per turn: {per_turn}  <- this is why long chats get expensive")
"""),

code(r"""
hint("reset():",
     "  self.conversation = [{'role': 'system', 'content': self.system_prompt}]",
     "  self.total_tokens = 0",
     "",
     "chat():",
     "  self.conversation.append({'role': 'user', 'content': user_input})",
     "  response = self.client.chat.completions.create(",
     "      model=self.model, messages=self.conversation, temperature=self.temperature)",
     "  reply = response.choices[0].message.content",
     "  self.conversation.append({'role': 'assistant', 'content': reply})",
     "  self.total_tokens += response.usage.total_tokens",
     "  return reply",
     "",
     "The most common bug: forgetting to append the assistant's reply. The bot then",
     "cannot see its own previous answers and repeats itself.")
"""),

# ------------------------------------------------------------------ 6
md(r"""
---
## Exercise 6 — Put 4 and 5 together  ·  *book: "Try it yourself", first item*

The chatbot grows without limit. The conversation manager trims. Combine them: a bot that
stays inside a token budget, whatever the user does.

This is the chapter's own closing exercise, and it is the first piece of real engineering
in the book — two components you wrote, joined so that one fixes the other's flaw.
"""),

ex(r"""
class BudgetedChatbot(SimpleChatbot):
    '''A chatbot that trims its history before every call.'''

    def __init__(self, client, max_tokens=200, **kwargs):
        self.max_tokens = max_tokens
        super().__init__(client, **kwargs)

    def reset(self):
        # TODO 1: build a ConversationManager(max_tokens=self.max_tokens) and keep it
        #         on self.manager, then seed it with the system prompt.
        # TODO 2: self.total_tokens = 0
        raise NotImplementedError

    @property
    def conversation(self):
        # TODO 3: the conversation IS the manager's message list. Return it so that the
        #         checks from Exercise 5 still apply to this subclass.
        raise NotImplementedError

    def chat(self, user_input):
        # TODO 4: add the user's message through the MANAGER (so it can evict), send
        #         self.conversation to the API, add the reply through the manager too,
        #         accumulate usage, and return the reply.
        raise NotImplementedError


budgeted = BudgetedChatbot(client, max_tokens=120)
for i in range(6):
    budgeted.chat(f"Turn {i}: explain one idea about building software, briefly.")
    print(f"turn {i}: {len(budgeted.conversation)} messages, "
          f"{budgeted.manager.total_tokens} tokens in context, "
          f"{budgeted.total_tokens} billed so far")
""", r"""
class BudgetedChatbot(SimpleChatbot):
    '''A chatbot that trims its history before every call.'''

    def __init__(self, client, max_tokens=200, **kwargs):
        self.max_tokens = max_tokens
        super().__init__(client, **kwargs)

    def reset(self):
        self.manager = ConversationManager(max_tokens=self.max_tokens)
        self.manager.add_message("system", self.system_prompt, verbose=False)
        self.total_tokens = 0

    @property
    def conversation(self):
        return self.manager.messages

    def chat(self, user_input):
        self.manager.add_message("user", user_input, verbose=False)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            temperature=self.temperature,
        )
        reply = response.choices[0].message.content
        self.manager.add_message("assistant", reply, verbose=False)
        self.total_tokens += response.usage.total_tokens
        return reply


budgeted = BudgetedChatbot(client, max_tokens=120)
for i in range(6):
    budgeted.chat(f"Turn {i}: explain one idea about building software, briefly.")
    print(f"turn {i}: {len(budgeted.conversation)} messages, "
          f"{budgeted.manager.total_tokens} tokens in context, "
          f"{budgeted.total_tokens} billed so far")
"""),

code(r"""
@check("2.6  the budgeted bot stays inside its budget")
def _():
    need("BudgetedChatbot")
    b = BudgetedChatbot(client, max_tokens=120)
    for i in range(8):
        b.chat(f"Turn {i}: a longer question with plenty of words in it, please answer.")

    assert b.manager.total_tokens <= 120, (
        f"context holds {b.manager.total_tokens} tokens against a 120 budget — "
        "the manager is not being used for every message"
    )
    assert b.conversation[0]["role"] == "system", "the system prompt must still be first"
    assert len(b.conversation) < 17, "nothing was ever evicted over 8 turns"
    print(f"     8 turns, {len(b.conversation)} messages kept, "
          f"{b.manager.total_tokens}/120 tokens in context")


@check("2.6  trimming is cheaper than not trimming")
def _():
    need("BudgetedChatbot", "SimpleChatbot")
    plain = SimpleChatbot(client)
    thrifty = BudgetedChatbot(client, max_tokens=120)
    turns = [f"Turn {i}: please say something about software design." for i in range(8)]
    for t in turns:
        plain.chat(t)
        thrifty.chat(t)
    assert thrifty.total_tokens < plain.total_tokens, (
        f"budgeted bot billed {thrifty.total_tokens}, plain bot {plain.total_tokens}. "
        "Trimming should cost less over 8 turns."
    )
    saved = 1 - thrifty.total_tokens / plain.total_tokens
    print(f"     {plain.total_tokens} tokens unbudgeted vs {thrifty.total_tokens} "
          f"budgeted — {saved:.0%} cheaper over 8 turns")
"""),

md(r"""
**And now the trade-off you just bought.** The budgeted bot is cheaper *and* it has
forgotten things. Ask it about turn 1 after turn 8 and it cannot answer. Nothing here is
free — you chose which of two costs to pay, which is what engineering a context window is.
"""),

# ------------------------------------------------------------------ experiments
md(r"""
---
## Experiments  ·  *book: "Try it yourself"*

Open-ended, ungraded. These are cheap in MOCK mode and cost a fraction of a cent LIVE.
"""),

ex(r"""
# A — does a persona survive a long conversation?
# TODO: build a SimpleChatbot with system_prompt="You are a pirate." Have five turns of
#       ordinary conversation and look at how piratical the last reply still is.
#       Then reset() and compare the first reply again.
raise NotImplementedError
""", r"""
# A — does a persona survive a long conversation?
pirate = SimpleChatbot(client, system_prompt="You are a pirate. Always speak as one.")
for i in range(5):
    reply = pirate.chat(f"Turn {i}: tell me briefly about writing good software.")
    print(f"turn {i}: {reply[:90]}")
pirate.reset()
print("\nafter reset:", pirate.chat("Tell me briefly about writing good software.")[:90])
print("\nA persona lives in the system prompt, which never gets evicted here — but with a "
      "sliding window that drops index 1 repeatedly, the EXAMPLES of the persona go, "
      "and the voice usually drifts.")
"""),

ex(r"""
# B — a realistic cost estimate
# TODO: write estimate_cost(prompt_tokens, completion_tokens) using SEPARATE input and
#       output rates (output is normally several times dearer), then use it to price a
#       1,000-conversation day at the per-turn token counts you measured in Exercise 5.
#       Treat the rates as placeholders: look up today's real prices before quoting anyone.
raise NotImplementedError
""", r"""
# B — a realistic cost estimate
# Placeholder rates, in dollars per 1,000 tokens. Real prices differ by model and change;
# look them up before you quote a figure to anyone.
IN_RATE, OUT_RATE = 0.00015, 0.00060


def estimate_cost(prompt_tokens, completion_tokens):
    return (prompt_tokens / 1000) * IN_RATE + (completion_tokens / 1000) * OUT_RATE


per_conversation_in, per_conversation_out = 2400, 600     # measured over ~8 turns
one = estimate_cost(per_conversation_in, per_conversation_out)
print(f"one conversation:      ${one:.5f}")
print(f"1,000 a day:           ${one * 1000:.2f}")
print(f"1,000 a day for a year ${one * 1000 * 365:,.0f}")
print("\nOutput tokens are the expensive ones, and resending history inflates the input "
      "side. That is the whole business case for Exercise 6.")
"""),

md(r"""
### Write down what you found

| Question | Your answer |
|---|---|
| Tokens per word, English vs another language |  |
| What temperature would you use for data extraction, and why |  |
| What the sliding window lost that mattered |  |
| Your estimated cost per 1,000 conversations |  |
"""),

# ------------------------------------------------------------------ wrap up
md(r"""
---
## Where you stand
"""),

code(r"""
progress()
print(f"\nmode: {MODE}")
"""),

md(r"""
### What you learned

* An LLM is a **next-token predictor**: a probability for every token, one picked, repeat.
* An **API call** sends messages and returns a reply. Check `usage` and `finish_reason`
  every time, and keep your key out of your code and out of Git.
* **Tokens** are the unit of reading, writing and billing. Count them with the tokenizer
  that matches your model; `response.usage` is the truth after the fact.
* **Temperature** reshapes the choice among likely tokens without changing which is most
  likely. Near 0 for factual work, 0.7–1.0 for creative work.
* The **context window** is limited working memory, and deciding what gets a seat in it is
  most of practical prompt engineering.
* A **chatbot is a loop around a stateless API**. The memory is a list you keep, and the
  cost of keeping it grows every turn.

### Before you move on

- [ ] Every check is ✅.
- [ ] You can say what `finish_reason == "length"` means without looking it up.
- [ ] You know roughly what 1,000 conversations a day would cost you.
- [ ] Your API key is in a secret or an environment variable — never in a cell.

**Next:** Chapter 3 gives the model the ability to *act*, not just answer — the step from
chatbot to agent.
"""),
]

SOLUTION_BANNER = r"""
# ⚠️ Solutions — Chapter 2

The worked key for *Chapter 2 — LLM Mechanics*. Every cell is complete and the notebook
runs top to bottom in both LIVE and MOCK mode.

Use it to check an exercise you have already attempted, or to unblock one specific line.
Reading solutions you have not attempted feels like learning and is not.
"""
