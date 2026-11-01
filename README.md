# Developer Guide — Exercise Notebooks

Hands-on notebooks for *Developer Guide: Building with AI*. Each one takes a chapter of the
book and turns it into code **you** write, with checks that tell you whether you got it
right.

Open any notebook in Google Colab — nothing to install, and **no API key needed**.

| # | Chapter | Notebook | Time |
|---|---|---|---|
| 1 | Building a Neural Network from Scratch | [`Chapter_01`](notebooks/Chapter_01_Neural_Network_From_Scratch.ipynb) | 60–90 min |
| 2 | LLM Mechanics: API Calls, Tokens, Temperature | [`Chapter_02`](notebooks/Chapter_02_LLM_Mechanics.ipynb) | 60 min |
| 3 | Building Your First Agent from Scratch | [`Chapter_03`](notebooks/Chapter_03_Building_Your_First_Agent_from_Scratch.ipynb) | 30 min |
| 4 | The Core Agent Loop | [`Chapter_04`](notebooks/Chapter_04_The_Core_Agent_Loop.ipynb) | 30 min |
| 5 | Tools: Writing Your First Tool Integrations | [`Chapter_05`](notebooks/Chapter_05_Tools_Writing_Your_First_Tool_Integrations.ipynb) | 35 min |
| 6 | Memory: Implementing Persistent Agent Memory | [`Chapter_06`](notebooks/Chapter_06_Memory_Implementing_Persistent_Agent_Memory.ipynb) | 35 min |
| 7 | RAG in Practice | [`Chapter_07`](notebooks/Chapter_07_RAG_in_Practice.ipynb) | 35 min |
| 8 | Where RAG Is Heading | [`Chapter_08`](notebooks/Chapter_08_Where_RAG_Is_Heading.ipynb) | 35 min |
| 9 | Vector Databases | [`Chapter_09`](notebooks/Chapter_09_Vector_Databases.ipynb) | 35 min |
| 10 | Embeddings: Practical Implementation Guide | [`Chapter_10`](notebooks/Chapter_10_Embeddings_Practical_Implementation_Guide.ipynb) | 35 min |
| 11 | MCP: Building Your First MCP Server | [`Chapter_11`](notebooks/Chapter_11_MCP_Building_Your_First_MCP_Server.ipynb) | 35 min |
| 12 | The Six Agentic Architectures | [`Chapter_12`](notebooks/Chapter_12_The_Six_Agentic_Architectures.ipynb) | 30 min |
| 13 | Multi-Agent Systems | [`Chapter_13`](notebooks/Chapter_13_Multi-Agent_Systems.ipynb) | 35 min |
| 14 | Frameworks: What They Give You | [`Chapter_14`](notebooks/Chapter_14_Frameworks_What_They_Give_You.ipynb) | 35 min |
| 15 | Safety: Implementing Guardrails | [`Chapter_15`](notebooks/Chapter_15_Safety_Implementing_Guardrails.ipynb) | 40 min |
| 16 | Real-World Applications | [`Chapter_16`](notebooks/Chapter_16_Real-World_Applications.ipynb) | 40 min |
| 17 | Production Deployment | [`Chapter_17`](notebooks/Chapter_17_Production_Deployment.ipynb) | 35 min |

Solutions sit beside each notebook as `*_SOLUTIONS.ipynb`. **130 self-checks** across the
seventeen notebooks.

## How to work through a notebook

1. **Open it in Colab** and run the setup cell at the top first.
2. **Go in order.** Later exercises use what you built earlier.
3. Each exercise is a cell with `# TODO` comments ending in `raise NotImplementedError`.
   Write your code, delete that line, run the cell.
4. **Run the check cell underneath.** It prints ✅, or a message telling you what is wrong.
   ⬜ means you have not attempted it yet; ❌ is a hint, not a grade.
5. Stuck for ten minutes? Run the `hint(...)` cell. Hints exist to be used.
6. `progress()` shows where you are at any time.
7. When you finish, `Runtime → Run all` on a fresh runtime. If something breaks then, you
   had state lying around that a real reader would not have.

**Do not read the solutions before attempting the exercise.** Reading a solution feels like
learning and is not. Attempt, fail, read the check message, try again — *then* compare.

## Do I need an API key?

**No.** Every notebook runs free and offline:

* Chapters 3 to 17 use the same **stand-in model** the book's own scripts use — a scripted
  reply, a fake client, a deterministic embedder. The mechanics are real; only the model's
  prose is canned.
* Chapter 2 looks for a key and, if it finds none, swaps in a stand-in with the same
  response shape. Every exercise and check still passes. Token counts, `finish_reason` and
  the cost arithmetic are genuine.

To run Chapter 2 against the real API in Colab: click the 🔑 icon in the left sidebar, add a
secret called `OPENAI_API_KEY`, switch on notebook access, re-run the setup cell. Running
locally, export it in your shell or put it in a `.env` file.

**Never paste a key into a notebook cell.** A shared notebook carries its cells with it, and
a leaked key gets spent by whoever finds it.

## Running locally instead of Colab

```bash
pip install numpy scikit-learn matplotlib torch openai tiktoken jupyter
jupyter notebook
```

Chapters 3 to 17 need only `numpy` (and `matplotlib` for one plot).

## For the author

See [`AUTHOR_GUIDE.md`](AUTHOR_GUIDE.md) — the design rules behind these notebooks, how the
build works, and how to publish them so readers reach them in one click.
