"""The self-check harness that is pasted into every notebook's setup cell.

It is deliberately plain code with no dependencies, so a reader can read it,
and so the notebook keeps working if this repository disappears.
"""

HARNESS = r'''
#@title Setup — run this cell first (double-click to read the code)
import sys, time

_RESULTS = {}   # exercise label -> "pass" | "fail" | "todo" | "error"

def check(label):
    """Decorator that runs a block of assertions and reports the result.

    Use it like this:

        @check("1.1  single neuron")
        def _():
            assert my_thing() == 42, "a message the reader can act on"
    """
    def run(fn):
        started = time.time()
        try:
            fn()
        except NotImplementedError as err:
            _RESULTS[label] = "todo"
            reason = str(err) or ("not attempted yet (write your code, then delete "
                                  "the `raise NotImplementedError`)")
            print(f"⬜  {label}  —  {reason}")
        except AssertionError as err:
            _RESULTS[label] = "fail"
            print(f"❌  {label}\n     {err}")
        except Exception as err:                     # noqa: BLE001 - readers see the type
            _RESULTS[label] = "error"
            print(f"⚠️   {label}  —  {type(err).__name__}: {err}")
            print("     The code crashed before the check could finish. "
                  "Scroll up to the traceback or re-run your cell on its own.")
        else:
            _RESULTS[label] = "pass"
            print(f"✅  {label}   ({time.time() - started:.1f}s)")
        return fn
    return run

def progress():
    """Print a summary of every check run so far in this session."""
    if not _RESULTS:
        print("No checks have run yet. Run the exercise cells above.")
        return
    icon = {"pass": "✅", "fail": "❌", "todo": "⬜", "error": "⚠️ "}
    done = sum(1 for v in _RESULTS.values() if v == "pass")
    width = max(len(k) for k in _RESULTS)
    print(f"{done}/{len(_RESULTS)} checks passing\n" + "-" * (width + 6))
    for label, state in _RESULTS.items():
        print(f"{icon[state]}  {label}")
    if done < len(_RESULTS):
        print("\nAnything not green is a hint, not a grade. "
              "Read the message, change one thing, re-run the cell.")

def need(*names):
    """Assert that the exercise cell above has been run.

    Gives a readable message instead of a bare NameError when a reader runs a
    check cell before the exercise it belongs to.
    """
    missing = [n for n in names if n not in globals()]
    if missing:
        raise NotImplementedError(
            f"{', '.join(missing)} is not defined yet — run the exercise cell "
            "directly above this one (and make sure it finished without an error)."
        )

def need_data(ok, what="the exercise above"):
    """Like need(), but for state that exists and is empty until the exercise runs."""
    if not ok:
        raise NotImplementedError(
            f"nothing to check yet — run {what} and make sure it finished without "
            "an error, then run this cell again."
        )

def hint(*lines):
    """Print a hint. Readers call this only when they want it."""
    print("\n".join(lines))

print(f"Python {sys.version.split()[0]} — setup complete. "
      "Run the cells in order; call progress() at any time.")
'''
