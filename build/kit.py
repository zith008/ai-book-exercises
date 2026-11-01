"""Shared stand-ins for the retrieval chapters (7-10).

One embedder, one distance function, one small document set — used by every notebook
from 7 to 10, so a reader who has done chapter 7 recognises the tools in chapter 10,
and so the chapters cannot quietly disagree with each other.

Why a stand-in at all: real embeddings cost money per call and need a key. This one is
a hashing bag-of-words, which is a genuine vector space — close meanings land close
when words overlap — so chunking, cutoffs, recall, truncation and quantisation all
behave the way they do in production. What it cannot do is understand paraphrase, and
each notebook says so where it matters.
"""

EMBEDDER = r'''
import re, hashlib
import numpy as np

EMBED_DIM = 16384
STOPWORDS = {"a", "an", "the", "is", "are", "to", "of", "and", "in", "on", "for",
             "it", "that", "this", "with", "as", "by", "be", "or", "from", "at",
             "you", "your", "we", "i", "what", "how", "do", "does", "can", "if"}


def _words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def embed(text, model="stand-in-v1", dim=EMBED_DIM):
    """Turn text into a unit vector: one slot per word of a large vocabulary.

    Each word is hashed to its own slot and adds 1 there, then the vector is scaled to
    length 1. Texts sharing words point in similar directions; texts sharing nothing are
    at right angles, which is a distance of exactly 1.

    A real embedder is a neural network, and understands that "refund window" and
    "return within 30 days" mean the same thing. This one only sees shared words. The
    mechanics of searching, cutoffs and recall are identical; the matching is cruder.
    """
    vector = np.zeros(dim)
    salt = model.encode()
    for word in _words(text):
        position = int(hashlib.md5(salt + word.encode()).hexdigest(), 16) % dim
        vector[position] += 1.0
    length = np.linalg.norm(vector)
    return vector / length if length else vector


def distance(a, b):
    """Cosine distance: 0.0 is identical direction, 1.0 is nothing in common."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    size = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 if size == 0 else float(1 - np.dot(a, b) / size)
'''

DOCUMENTS = r'''
# A small handbook, written the way real documents are: headings, short sections,
# one fact per paragraph. Everything in chapters 7 to 10 is retrieved from this.
HANDBOOK = """# Returns policy

Customers may return any item within 30 days of delivery for a full refund.
Items must be unused and in their original packaging.

Refunds are issued to the original payment method within 5 working days of
the item arriving back at our warehouse.

# Shipping

Standard delivery takes 3 to 5 working days and costs 4.99 GBP.
Orders over 50 GBP are delivered free of charge.

Express delivery arrives the next working day if ordered before 2pm, and
costs 9.99 GBP.

# Warranty

Electronics carry a 24 month warranty covering manufacturing faults.
Accidental damage is not covered by the warranty.

To make a warranty claim, email support with your order number and a photo
of the fault. Claims are assessed within 10 working days.

# Account

You can reset your password from the sign-in page using the link marked
'forgot password'. The reset link expires after one hour.

Closing your account removes your order history permanently and cannot be
undone."""
'''
