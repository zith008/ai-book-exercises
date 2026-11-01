"""Chapter 1 — Building a Neural Network from Scratch.

Maps to sections 1.1-1.6 of the Developer Guide (scripts Ex-01 to Ex-06).
"""

from harness import HARNESS
from nbtools import code, ex, md

TITLE = "Chapter_01_Neural_Network_From_Scratch"

CELLS = [

md(r"""
# Chapter 1 — Build a Neural Network from Scratch
### Exercise notebook · *Developer Guide: Building with AI*

By the end of this notebook you will have written, from nothing but NumPy, a network
that separates two interlocking crescents of data — and then rebuilt the same network
in PyTorch in about fifteen lines.

You are not reading code here. You are writing it. Every exercise is a small gap in
working code, and every exercise has a check cell underneath it that tells you
whether what you wrote is right.

| Exercise | Book section | What you write |
|---|---|---|
| 1 | 1.1 | A single neuron's forward pass |
| 2 | 1.2 | A layer's backward pass (the chain rule) |
| 3 | 1.2–1.3 | The training loop: forward, loss, backward, update |
| 4 | 1.4 | A numerical gradient check |
| 5 | 1.5 | The same network in PyTorch |
| 6 | 1.6 | Debugging challenge: find the shape bug |
| 7 | summary | Experiments — no right answer, just evidence |

**Time:** about 60–90 minutes. **Prerequisites:** you can read Python functions and classes.
No calculus needed — where a derivative appears, the formula is given to you.

---

### How to use this notebook

1. **Run the setup cell below first.** Nothing else works until you do.
2. **Go in order.** Later exercises import what you built earlier.
3. Each exercise cell contains `# TODO` and ends with `raise NotImplementedError`.
   Write your code, **delete that line**, and run the cell.
4. Run the **check cell** underneath. It prints ✅, or it prints a message telling you
   what went wrong. A ❌ is information, not a verdict.
5. **Stuck for more than ten minutes?** Run the `hint(...)` cell under the exercise.
   Hints are there to be used; getting unstuck matters more than being unaided.
6. Call `progress()` whenever you want to see where you are.

> **Colab tip:** `Runtime → Run all` after you finish, to prove the whole notebook works
> top to bottom. If a later cell fails on a fresh run, something earlier depended on
> state you created by hand.
"""),

code(HARNESS, form=True),

code(r"""
# Everything this notebook needs is already installed in Colab.
# Running locally? pip install numpy scikit-learn matplotlib torch
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(0)   # same random weights every run, so your results match the book's
print("numpy", np.__version__)
"""),

# ------------------------------------------------------------------ 1.1
md(r"""
---
## Exercise 1 — One neuron  ·  *book §1.1 (Ex-01.py)*

A neuron takes several numbers in and produces **one** number between 0 and 1.
It does exactly two things:

1. **A weighted sum.** Every input gets a weight saying how much it matters, and a
   bias shifts the total up or down regardless of the inputs:

   $$z = x_1 w_1 + x_2 w_2 + \dots + b$$

   Think of a hiring decision: years of experience and interview score each get a
   weight, and the bias is how generous the hiring bar is overall.

2. **A squash.** The sigmoid function turns any number into something between 0 and 1,
   which you can read as a probability:

   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

   Large positive $z$ → close to 1. Large negative $z$ → close to 0. $z = 0$ → exactly 0.5.

Your job: write both.
"""),

ex(r"""
def sigmoid(z):
    '''Squash any number (or array) into the range 0 to 1.'''
    # TODO: return 1 / (1 + e**-z).  np.exp(x) computes e**x.
    raise NotImplementedError


class Neuron:
    def __init__(self, n_inputs):
        # One small random weight per input, and a bias of zero.
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0

    def forward(self, inputs):
        '''Compute the weighted sum, then squash it. Save both.'''
        # TODO: set self.z to the weighted sum plus the bias.
        #       np.dot(a, b) multiplies two vectors and adds up the results.
        # TODO: set self.output to sigmoid(self.z) and return it.
        #       (backward() needs both values later, which is why we store them.)
        raise NotImplementedError


neuron = Neuron(3)
x = np.array([0.5, -0.3, 0.8])
print(f"Output: {neuron.forward(x):.4f}")
""", r"""
def sigmoid(z):
    '''Squash any number (or array) into the range 0 to 1.'''
    return 1 / (1 + np.exp(-z))


class Neuron:
    def __init__(self, n_inputs):
        # One small random weight per input, and a bias of zero.
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0

    def forward(self, inputs):
        '''Compute the weighted sum, then squash it. Save both.'''
        self.z = np.dot(inputs, self.weights) + self.bias
        self.output = sigmoid(self.z)
        return self.output


neuron = Neuron(3)
x = np.array([0.5, -0.3, 0.8])
print(f"Output: {neuron.forward(x):.4f}")
"""),

code(r"""
@check("1.1  sigmoid and the neuron's forward pass")
def _():
    assert abs(sigmoid(0) - 0.5) < 1e-12, \
        f"sigmoid(0) should be exactly 0.5, you returned {sigmoid(0)}"
    assert sigmoid(20) > 0.999, "a large positive z should give an output close to 1"
    assert sigmoid(-20) < 0.001, "a large negative z should give an output close to 0"
    assert np.allclose(sigmoid(np.array([0.0, 0.0])), 0.5), \
        "sigmoid must work on a whole array, not just one number — use np.exp, not math.exp"

    n = Neuron(2)
    n.weights = np.array([2.0, -1.0])
    n.bias = 0.5
    got = n.forward(np.array([1.0, 1.0]))          # z = 2 - 1 + 0.5 = 1.5
    assert abs(n.z - 1.5) < 1e-12, f"z should be 1.5 here, you computed {n.z}"
    assert abs(got - sigmoid(1.5)) < 1e-12, "forward() should return sigmoid(z)"
    assert hasattr(n, "output"), "store the result on self.output as well as returning it"
"""),

code(r"""
hint("The weighted sum is one call:  np.dot(inputs, self.weights) + self.bias",
     "Store it as self.z, then self.output = sigmoid(self.z), then return self.output.",
     "",
     "Why store them? backward() needs z and output to apply the chain rule.",
     "Recomputing them would work but waste the forward pass you already did.")
"""),

md(r"""
**What to notice.** With weights around ±0.1 and inputs around ±1, $z$ lands close to 0,
so the output is close to 0.5 — the network starts out undecided, which is what you want
before it has learned anything. Re-run the cell: the number changes, because the weights
are random.

**One neuron can only draw a straight line.** That is the whole reason for the next exercise.
""" ),

# ------------------------------------------------------------------ 1.2
md(r"""
---
## Exercise 2 — A layer, and the backward pass  ·  *book §1.2 (Ex-02.py)*

A layer is many neurons at once. Instead of a vector of weights, it holds a **matrix**:
one column per neuron. One matrix multiply then computes every neuron, for every sample,
in a single line — which is why `forward` below is so short.

```
inputs (batch, n_inputs) @ weights (n_inputs, n_outputs) + bias  ->  z (batch, n_outputs)
```

`forward` is given to you. You write `backward`, which is where learning actually happens.

### What backward is for

Training needs to know, for every weight: *if I nudge this weight up slightly, does the
loss get better or worse, and by how much?* That number is the **gradient**.

Backward receives `gradient` — how much the loss changes with this layer's **output** —
and turns it into three things, using the chain rule:

| You compute | Formula | Why |
|---|---|---|
| `dz` | `gradient * output * (1 - output)` | `output * (1 - output)` is the derivative of sigmoid: how sensitive the output was to `z` |
| `self.dweights` | `inputs.T @ dz / len(dz)` | how much each weight is to blame, averaged over the batch |
| `self.dbias` | `dz.mean(axis=0)` | the same for each bias |
| *return* | `dz @ weights.T` | the gradient to hand to the layer **before** this one |

Each line is one multiplication by a local derivative. That is all backpropagation is.
"""),

ex(r"""
class Layer:
    '''A layer of neurons: Exercise 1's Neuron, but n_outputs of them at once.'''

    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.5   # one column per neuron
        self.bias = np.zeros(n_outputs)

    def forward(self, inputs):
        self.inputs = inputs            # this layer's own input; backward needs it
        self.z = inputs @ self.weights + self.bias
        self.output = sigmoid(self.z)
        return self.output

    def backward(self, gradient):
        '''gradient is d(loss)/d(output), shape (batch, n_outputs).'''
        # TODO 1: dz = incoming gradient * the sigmoid derivative (see the table above)
        # TODO 2: self.dweights = this layer's inputs, transposed, @ dz, averaged over
        #         the batch.  `arr.T` transposes; len(dz) is the batch size.
        # TODO 3: self.dbias = the mean of dz down the batch axis (axis=0)
        # TODO 4: return the gradient for the previous layer: dz @ weights transposed
        raise NotImplementedError


layer = Layer(2, 3)
out = layer.forward(np.array([[0.5, -0.8]]))
print("forward output:", out.round(4))
print("back to previous layer:", layer.backward(np.ones_like(out)).round(4))
print("dweights shape:", layer.dweights.shape, "(should match weights:", layer.weights.shape, ")")
""", r"""
class Layer:
    '''A layer of neurons: Exercise 1's Neuron, but n_outputs of them at once.'''

    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.5   # one column per neuron
        self.bias = np.zeros(n_outputs)

    def forward(self, inputs):
        self.inputs = inputs            # this layer's own input; backward needs it
        self.z = inputs @ self.weights + self.bias
        self.output = sigmoid(self.z)
        return self.output

    def backward(self, gradient):
        '''gradient is d(loss)/d(output), shape (batch, n_outputs).'''
        dz = gradient * self.output * (1 - self.output)
        self.dweights = self.inputs.T @ dz / len(dz)
        self.dbias = dz.mean(axis=0)
        return dz @ self.weights.T


layer = Layer(2, 3)
out = layer.forward(np.array([[0.5, -0.8]]))
print("forward output:", out.round(4))
print("back to previous layer:", layer.backward(np.ones_like(out)).round(4))
print("dweights shape:", layer.dweights.shape, "(should match weights:", layer.weights.shape, ")")
"""),

code(r"""
@check("1.2  Layer.backward shapes")
def _():
    need("Layer")
    lay = Layer(4, 3)
    x = np.random.randn(5, 4)               # batch of 5
    out = lay.forward(x)
    back = lay.backward(np.ones_like(out))

    assert lay.dweights.shape == lay.weights.shape, (
        f"dweights is {lay.dweights.shape} but weights is {lay.weights.shape}. "
        "Every weight needs exactly one gradient, or the update cannot subtract them."
    )
    assert lay.dbias.shape == lay.bias.shape, \
        f"dbias is {lay.dbias.shape}, expected {lay.bias.shape} — did you average over axis=0?"
    assert back.shape == x.shape, (
        f"backward returned {back.shape}, but the previous layer sent {x.shape}. "
        "The returned gradient must match this layer's input."
    )


@check("1.2  Layer.backward is mathematically correct (numerical gradient check)")
def _():
    np.random.seed(7)
    lay = Layer(3, 2)
    x = np.random.randn(4, 3)
    target = np.random.rand(4, 2)

    def loss():
        return np.mean((lay.forward(x) - target) ** 2)

    lay.forward(x)
    lay.backward(2 * (lay.output - target) / lay.output.shape[1])

    # Nudge one weight up and down and see how the loss really moves.
    eps, i, j = 1e-6, 1, 0
    lay.weights[i, j] += eps; up = loss()
    lay.weights[i, j] -= 2 * eps; down = loss()
    lay.weights[i, j] += eps
    numeric = (up - down) / (2 * eps)
    analytic = lay.dweights[i, j]

    assert abs(numeric - analytic) < 1e-5, (
        f"backprop says {analytic:.8f}, nudging the weight says {numeric:.8f}. "
        "They should agree to about 6 decimal places. Check the sigmoid derivative "
        "term and the / len(dz) average."
    )
    print(f"     backprop {analytic:.8f}  ==  numerical {numeric:.8f}")
"""),

code(r"""
hint("Line by line:",
     "  dz = gradient * self.output * (1 - self.output)",
     "  self.dweights = self.inputs.T @ dz / len(dz)",
     "  self.dbias = dz.mean(axis=0)",
     "  return dz @ self.weights.T",
     "",
     "If a shape error appears, print self.inputs.shape, dz.shape and self.weights.shape.",
     "In matrix multiplication the inner dimensions must match: (a, b) @ (b, c) -> (a, c).")
"""),

# ------------------------------------------------------------------ 1.3
md(r"""
---
## Exercise 3 — The training loop  ·  *book §1.2–1.3 (Ex-02.py, Ex-03.py)*

Everything so far has been one step. Training is those steps, repeated. Each pass over
the data (an **epoch**) is the same four moves:

1. **Forward** — run the data through every layer to get predictions.
2. **Loss** — measure how wrong they are: `mean((output - y) ** 2)`, the mean squared error.
3. **Backward** — start from the loss gradient `2 * (output - y)` and send it back through
   the layers **in reverse order**. Why reverse? The last layer's gradient depends only on
   the loss; the layer before it needs the last layer's answer. Going backwards reuses each
   result instead of recomputing it.
4. **Update** — `weights -= learning_rate * dweights`. Every parameter takes a small step
   in the direction that lowers the loss. That is gradient descent.

Think of it as walking downhill in fog: you cannot see the valley, but you can feel which
way the ground slopes, so you take a step that way and feel again.

The data is the **two moons** set: two interlocking crescents. One neuron could never
separate them — it can only draw a straight line. A hidden layer can learn the curve.
"""),

ex(r"""
class NeuralNetwork:
    def __init__(self, layer_sizes):
        '''layer_sizes=[2, 8, 1] means 2 inputs, 8 hidden neurons, 1 output.'''
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int).ravel()

    def train(self, X, y, epochs=5000, learning_rate=1.0, verbose=True):
        y = y.reshape(-1, 1)                    # column shape, to match the output
        for epoch in range(epochs):
            # TODO 1: output = predictions for X   (one call)
            # TODO 2: loss   = mean squared error between output and y
            # TODO 3: gradient = 2 * (output - y), then pass it through every layer in
            #         REVERSE order, reassigning gradient each time:
            #             for layer in reversed(self.layers): gradient = layer.backward(...)
            # TODO 4: for every layer, subtract learning_rate * its gradient from its
            #         weights and from its bias
            raise NotImplementedError

            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return loss


np.random.seed(0)
X, y = make_moons(n_samples=400, noise=0.15, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

net = NeuralNetwork([2, 8, 1])
net.train(X_train, y_train)
print(f"Train accuracy: {accuracy_score(y_train, net.predict(X_train)):.3f}")
print(f"Test accuracy:  {accuracy_score(y_test, net.predict(X_test)):.3f}")
""", r"""
class NeuralNetwork:
    def __init__(self, layer_sizes):
        '''layer_sizes=[2, 8, 1] means 2 inputs, 8 hidden neurons, 1 output.'''
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int).ravel()

    def train(self, X, y, epochs=5000, learning_rate=1.0, verbose=True):
        y = y.reshape(-1, 1)                    # column shape, to match the output
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)

            gradient = 2 * (output - y)
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient)

            for layer in self.layers:
                layer.weights -= learning_rate * layer.dweights
                layer.bias -= learning_rate * layer.dbias

            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return loss


np.random.seed(0)
X, y = make_moons(n_samples=400, noise=0.15, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

net = NeuralNetwork([2, 8, 1])
net.train(X_train, y_train)
print(f"Train accuracy: {accuracy_score(y_train, net.predict(X_train)):.3f}")
print(f"Test accuracy:  {accuracy_score(y_test, net.predict(X_test)):.3f}")
"""),

code(r"""
@check("1.3  the loss actually falls")
def _():
    need("NeuralNetwork")
    np.random.seed(0)
    small = NeuralNetwork([2, 8, 1])
    Xs, ys = make_moons(n_samples=200, noise=0.15, random_state=1)
    first = np.mean((small.forward(Xs) - ys.reshape(-1, 1)) ** 2)
    small.train(Xs, ys, epochs=400, learning_rate=1.0, verbose=False)
    last = np.mean((small.forward(Xs) - ys.reshape(-1, 1)) ** 2)
    assert last < first, (
        f"loss went from {first:.4f} to {last:.4f} — it should go DOWN. "
        "A loss that rises usually means the update ADDS the gradient instead of "
        "subtracting it, or the learning rate is far too large."
    )
    print(f"     loss {first:.4f} -> {last:.4f} after 400 epochs")


@check("1.3  the network generalises to unseen data")
def _():
    need("net")
    train_acc = accuracy_score(y_train, net.predict(X_train))
    test_acc = accuracy_score(y_test, net.predict(X_test))
    assert train_acc > 0.9, (
        f"training accuracy is only {train_acc:.3f}. The network has not learned the "
        "shape yet — check the four steps are all present and in order."
    )
    assert test_acc > 0.85, (
        f"test accuracy is {test_acc:.3f}. Training looks fine but held-out data does not, "
        "which is worth investigating rather than ignoring."
    )
    print(f"     train {train_acc:.3f} / test {test_acc:.3f} "
          "— the book reports about 0.99 / 0.96")
"""),

code(r"""
hint("The four steps, in order, inside the epoch loop:",
     "  output = self.forward(X)",
     "  loss = np.mean((output - y) ** 2)",
     "  gradient = 2 * (output - y)",
     "  for layer in reversed(self.layers): gradient = layer.backward(gradient)",
     "  for layer in self.layers:",
     "      layer.weights -= learning_rate * layer.dweights",
     "      layer.bias    -= learning_rate * layer.dbias",
     "",
     "Two traps: reassign `gradient` each time round the backward loop (it is a relay",
     "race, not a broadcast), and update AFTER all the backward calls, not during.")
"""),

md(r"""
### See what you built

Nothing in this notebook is more convincing than the picture. The cell below colours
every point of the plane by what your network predicts there.
"""),

code(r"""
def plot_boundary(predict_fn, X, y, title):
    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.asarray(predict_fn(grid)).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, alpha=0.25, levels=1)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=25)
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.show()


plot_boundary(net.predict, X_test, y_test, "Your network's decision boundary (test data)")
"""),

md(r"""
The boundary **bends** to follow the gap between the two moons. A single neuron could
only ever have drawn a straight line across this picture. The bend is what the hidden
layer bought you.
"""),

# ------------------------------------------------------------------ 1.4
md(r"""
---
## Exercise 4 — Prove backprop is right  ·  *book §1.4 (Ex-04.py)*

Backpropagation is easy to get subtly wrong, and a subtly wrong gradient still trains —
just badly. So professionals check it, with a trick that needs no calculus at all:

> Nudge one weight up by a tiny amount $\varepsilon$ and record the loss.
> Nudge it down by $\varepsilon$ and record the loss.
> The slope is $\dfrac{\text{loss}_{up} - \text{loss}_{down}}{2\varepsilon}$.

That is the gradient, measured instead of derived. If your backward pass disagrees with
it, your backward pass is wrong. This is called a **gradient check**, and it is the single
most useful debugging tool in this chapter.

Write it, and remember to put the weight back where you found it.
"""),

ex(r"""
def numerical_gradient(net, inputs, target, layer, i, j, eps=1e-5):
    '''Estimate d(loss)/d(weight[i, j]) by nudging that one weight up and down.'''

    def loss():
        return np.mean((net.forward(inputs) - target) ** 2)

    # TODO 1: add eps to layer.weights[i, j], then measure the loss -> up
    # TODO 2: subtract 2 * eps (so the weight is now eps BELOW where it started),
    #         then measure the loss -> down
    # TODO 3: restore the weight by adding eps back. This matters: leaving it modified
    #         would silently corrupt every later experiment.
    # TODO 4: return the slope, (up - down) / (2 * eps)
    raise NotImplementedError


np.random.seed(0)
demo = NeuralNetwork([2, 3, 1])
x1 = np.array([[0.5, -0.8]])
y1 = np.array([[1.0]])

demo.forward(x1)                                 # one backward pass by hand
g = 2 * (demo.layers[-1].output - y1)
for layer in reversed(demo.layers):
    g = layer.backward(g)

analytic = demo.layers[0].dweights[0, 0]
numeric = numerical_gradient(demo, x1, y1, demo.layers[0], 0, 0)
print(f"backprop  = {analytic:.8f}")
print(f"numerical = {numeric:.8f}")
""", r"""
def numerical_gradient(net, inputs, target, layer, i, j, eps=1e-5):
    '''Estimate d(loss)/d(weight[i, j]) by nudging that one weight up and down.'''

    def loss():
        return np.mean((net.forward(inputs) - target) ** 2)

    layer.weights[i, j] += eps
    up = loss()
    layer.weights[i, j] -= 2 * eps
    down = loss()
    layer.weights[i, j] += eps                   # put it back exactly as it was
    return (up - down) / (2 * eps)


np.random.seed(0)
demo = NeuralNetwork([2, 3, 1])
x1 = np.array([[0.5, -0.8]])
y1 = np.array([[1.0]])

demo.forward(x1)                                 # one backward pass by hand
g = 2 * (demo.layers[-1].output - y1)
for layer in reversed(demo.layers):
    g = layer.backward(g)

analytic = demo.layers[0].dweights[0, 0]
numeric = numerical_gradient(demo, x1, y1, demo.layers[0], 0, 0)
print(f"backprop  = {analytic:.8f}")
print(f"numerical = {numeric:.8f}")
"""),

code(r"""
@check("1.4  gradient check agrees with backprop")
def _():
    need("numerical_gradient", "demo", "analytic")
    numeric = numerical_gradient(demo, x1, y1, demo.layers[0], 0, 0)
    assert abs(analytic - numeric) < 1e-6, (
        f"backprop says {analytic:.8f}, your numerical estimate says {numeric:.8f}. "
        "If the numbers are close but not close enough, check you used 2 * eps in the "
        "denominator. If one is zero, the weight was probably not restored."
    )
    before = demo.layers[0].weights.copy()
    numerical_gradient(demo, x1, y1, demo.layers[0], 0, 0)
    assert np.allclose(before, demo.layers[0].weights), (
        "the weight was left modified after the check. Restore it before returning — "
        "a diagnostic must never change the thing it measures."
    )
    print(f"     agreement to {abs(analytic - numeric):.1e}, weights left untouched")
"""),

md(r"""
### Watch one step of learning

Run this. It prints the five steps of one training step on a single example, and the
loss before and after. The loss should drop by roughly a fifth.
"""),

code(r"""
def backward_pass_explained(net, inputs, target, learning_rate=0.5):
    output = net.forward(inputs)
    print(f"1. Forward pass:  output = {output.ravel()}")

    error = output - target
    print(f"2. Error:         output - target = {error.ravel()}")

    gradient = 2 * error
    print(f"3. Loss gradient: 2 * error = {gradient.ravel()}")

    for i, layer in reversed(list(enumerate(net.layers))):
        gradient = layer.backward(gradient)
        print(f"4. Layer {i}: dweights shape {layer.dweights.shape}, "
              f"largest |dweight| = {np.abs(layer.dweights).max():.5f}")

    for layer in net.layers:
        layer.weights -= learning_rate * layer.dweights
        layer.bias -= learning_rate * layer.dbias
    print("5. Weights updated")


before = np.mean((demo.forward(x1) - y1) ** 2)
backward_pass_explained(demo, x1, y1)
after = np.mean((demo.forward(x1) - y1) ** 2)
print(f"\nLoss before: {before:.5f}, after one step: {after:.5f}")
"""),

md(r"""
**Two things to see in that output.**

The error is **negative** because the output (about 0.53) is below the target (1.0), so the
step pushes the output up.

And the earlier layer's gradients are much smaller than the later layer's — in the book's
run, 0.021 against 0.153. That shrinking is the beginning of the **vanishing gradient**
problem, and it is why the deeper network in Exercise 7 will disappoint you.

> **Try it:** change `learning_rate` to `5.0`, then to `0.001`, and re-run. One overshoots,
> one barely moves. Neither is "wrong" — the learning rate is a dial you tune.
"""),

# ------------------------------------------------------------------ 1.5
md(r"""
---
## Exercise 5 — The same network in PyTorch  ·  *book §1.5 (Ex-05.py)*

You have now written by hand everything a deep-learning library does for you. Here is
what it costs in PyTorch — and the point of this exercise is how little is left:

| You wrote | PyTorch | Who does it now |
|---|---|---|
| `Layer.forward` | `nn.Linear(in, out)` | library |
| `Layer.backward` | `loss.backward()` | library (it tracked every operation) |
| the update loop | `optimizer.step()` | library |
| the four-step loop | still yours | **you** |

The loop is the same four steps. Only one is new: `optimizer.zero_grad()`, because
PyTorch *accumulates* gradients, so you must clear last step's before computing this one's.
Forget it and your gradients quietly pile up — a classic bug worth meeting once on purpose.
"""),

ex(r"""
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO 1: three linear layers, matching the shape [2, 8, 8, 1]:
        #         self.fc1 = nn.Linear(2, 8), then 8 -> 8, then 8 -> 1
        self.sigmoid = nn.Sigmoid()
        raise NotImplementedError

    def forward(self, x):
        # TODO 2: pass x through fc1, fc2, fc3, applying self.sigmoid after each
        raise NotImplementedError


X2, y2 = make_moons(n_samples=200, noise=0.1, random_state=42)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)

X_tr_t = torch.tensor(X2_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y2_tr, dtype=torch.float32).reshape(-1, 1)   # float, column shape
X_te_t = torch.tensor(X2_te, dtype=torch.float32)

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(2000):
    # TODO 3: clear the gradients left over from the previous step
    # TODO 4: loss = criterion(model(X_tr_t), y_tr_t)
    # TODO 5: loss.backward()      <- the whole of Exercise 2, done for you
    # TODO 6: optimizer.step()     <- the whole of Exercise 3's update, done for you
    raise NotImplementedError

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():                            # no gradients needed to predict
    torch_preds = (model(X_te_t) > 0.5).int().numpy().ravel()
print(f"Accuracy: {accuracy_score(y2_te, torch_preds):.4f}")
""", r"""
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


X2, y2 = make_moons(n_samples=200, noise=0.1, random_state=42)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)

X_tr_t = torch.tensor(X2_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y2_tr, dtype=torch.float32).reshape(-1, 1)   # float, column shape
X_te_t = torch.tensor(X2_te, dtype=torch.float32)

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(2000):
    optimizer.zero_grad()
    loss = criterion(model(X_tr_t), y_tr_t)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():                            # no gradients needed to predict
    torch_preds = (model(X_te_t) > 0.5).int().numpy().ravel()
print(f"Accuracy: {accuracy_score(y2_te, torch_preds):.4f}")
"""),

code(r"""
@check("1.5  PyTorch model and training loop")
def _():
    need("model", "torch_preds", "loss")
    linear = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linear) == 3, f"expected 3 nn.Linear layers, found {len(linear)}"
    assert (linear[0].in_features, linear[-1].out_features) == (2, 1), (
        "the network must take 2 inputs and produce 1 output — "
        f"yours takes {linear[0].in_features} and produces {linear[-1].out_features}"
    )

    acc = accuracy_score(y2_te, torch_preds)
    assert acc > 0.80, (
        f"accuracy is {acc:.3f}. If it sits near 0.5 the model is guessing: check that "
        "zero_grad(), backward() and step() are all inside the loop, in that order."
    )
    assert loss.item() < 0.2, \
        f"final loss {loss.item():.4f} is high — did optimizer.step() run every epoch?"
    print(f"     accuracy {acc:.3f} with ~15 lines instead of ~60")
"""),

code(r"""
hint("__init__: self.fc1 = nn.Linear(2, 8); self.fc2 = nn.Linear(8, 8); self.fc3 = nn.Linear(8, 1)",
     "forward:  x = self.sigmoid(self.fc1(x)) ... and so on, returning x",
     "",
     "The loop, in this exact order:",
     "  optimizer.zero_grad()",
     "  loss = criterion(model(X_tr_t), y_tr_t)",
     "  loss.backward()",
     "  optimizer.step()",
     "",
     "Order matters. step() before backward() would update using stale gradients.")
"""),

md(r"""
> **Try it:** swap `optim.Adam(..., lr=0.01)` for `optim.SGD(model.parameters(), lr=0.5)`
> and re-run. SGD is plain gradient descent — the thing you wrote in Exercise 3. Adam adapts
> its step size per parameter. The accuracy gap you see is what that adaptation is worth.
"""),

# ------------------------------------------------------------------ 1.6
md(r"""
---
## Exercise 6 — Debugging challenge  ·  *book §1.6 (Ex-06.py)*

This is the most realistic exercise in the notebook. The code below is a real first
attempt at this chapter's network. It looks right. It crashes.

```
ValueError: shapes (1,) and (8,) not aligned
```

Your job is not to rewrite it. It is to **read the traceback, find the cause, and make the
smallest change that fixes it** — which is what the job actually consists of.

Three things are wrong, and they are connected:

1. The "layers" are single neurons, so `[2, 8, 8, 1]` built a chain of three neurons
   rather than layers of 8 — and the final `1` was never used at all.
2. Every layer receives the *network's* input in `backward`, instead of the input that
   layer itself saw.
3. So the shapes do not line up, and NumPy says so.

Run it first and read the error. Then fix it.

> **The lesson of this exercise:** the broken forward pass produced plausible-looking
> numbers. Only training exposed the problem. Running code beats reading code.
"""),

ex(r"""
# ---- the broken version: run it, read the error, then fix it ----------------
class BrokenNeuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs) * 0.5
        self.bias = 0.0

    def forward(self, inputs):
        self.z = inputs @ self.weights + self.bias
        self.output = sigmoid(self.z)
        return self.output

    def backward(self, inputs, gradient):
        sigmoid_grad = self.output * (1 - self.output)
        self.dweights = np.outer(inputs, gradient * sigmoid_grad)
        self.dbias = gradient * sigmoid_grad
        return np.dot(gradient * sigmoid_grad, self.weights)


class BrokenNetwork:
    def __init__(self, layer_sizes):
        # BUG 1: one neuron per layer, and the last size is thrown away
        self.layers = [BrokenNeuron(layer_sizes[i]) for i in range(len(layer_sizes) - 1)]

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def train(self, X, y, epochs=2000, lr=0.5, verbose=True):
        y = y.reshape(-1, 1)
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            gradient = 2 * (output - y)
            for layer in reversed(self.layers):
                # BUG 2: every layer is handed the network's input, not its own
                gradient = layer.backward(X, gradient)
            for layer in self.layers:
                layer.weights -= lr * layer.dweights
                layer.bias -= lr * layer.dbias
            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        return loss


# TODO: write `fixed_network(layer_sizes)` returning a network that trains correctly.
#       You already wrote the fix — Layer and NeuralNetwork from Exercises 2 and 3 are
#       the corrected design. Reuse them rather than patching the broken classes.
def fixed_network(layer_sizes):
    raise NotImplementedError


np.random.seed(42)
X3, y3 = make_moons(n_samples=200, noise=0.1, random_state=42)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42)

fixed = fixed_network([2, 8, 8, 1])
fixed.train(X3_tr, y3_tr, epochs=2000, learning_rate=0.5)
print(f"Accuracy: {accuracy_score(y3_te, fixed.predict(X3_te)):.4f}")
""", r"""
# ---- the broken version is kept for reference; the fix is below -------------
class BrokenNeuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs) * 0.5
        self.bias = 0.0

    def forward(self, inputs):
        self.z = inputs @ self.weights + self.bias
        self.output = sigmoid(self.z)
        return self.output

    def backward(self, inputs, gradient):
        sigmoid_grad = self.output * (1 - self.output)
        self.dweights = np.outer(inputs, gradient * sigmoid_grad)
        self.dbias = gradient * sigmoid_grad
        return np.dot(gradient * sigmoid_grad, self.weights)


class BrokenNetwork:
    def __init__(self, layer_sizes):
        # BUG 1: one neuron per layer, and the last size is thrown away
        self.layers = [BrokenNeuron(layer_sizes[i]) for i in range(len(layer_sizes) - 1)]

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def train(self, X, y, epochs=2000, lr=0.5, verbose=True):
        y = y.reshape(-1, 1)
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            gradient = 2 * (output - y)
            for layer in reversed(self.layers):
                # BUG 2: every layer is handed the network's input, not its own
                gradient = layer.backward(X, gradient)
            for layer in self.layers:
                layer.weights -= lr * layer.dweights
                layer.bias -= lr * layer.dbias
            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        return loss


# The fix is the design you already built:
#   * Layer(n_inputs, n_outputs) holds a weight MATRIX, so one layer is many neurons.
#   * Each layer stores its own input in forward(), so backward() needs only the gradient.
#   * Everything is matrix maths, so the shapes line up and a whole batch runs at once.
def fixed_network(layer_sizes):
    return NeuralNetwork(layer_sizes)


np.random.seed(42)
X3, y3 = make_moons(n_samples=200, noise=0.1, random_state=42)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42)

fixed = fixed_network([2, 8, 8, 1])
fixed.train(X3_tr, y3_tr, epochs=2000, learning_rate=0.5)
print(f"Accuracy: {accuracy_score(y3_te, fixed.predict(X3_te)):.4f}")
"""),

code(r"""
@check("1.6  the fixed network trains and has the right architecture")
def _():
    need("fixed_network", "fixed")
    net6 = fixed_network([2, 8, 8, 1])
    assert len(net6.layers) == 3, (
        f"a [2, 8, 8, 1] network needs 3 layers (2->8, 8->8, 8->1); yours has "
        f"{len(net6.layers)}. That was bug 1: the last size was being dropped."
    )
    assert net6.layers[0].weights.shape == (2, 8), (
        f"the first layer's weights are {net6.layers[0].weights.shape}, expected (2, 8). "
        "A layer needs a weight MATRIX — one column per neuron — not a vector."
    )
    acc = accuracy_score(y3_te, fixed.predict(X3_te))
    assert acc > 0.8, f"accuracy {acc:.3f} — the fixed network should reach roughly 0.875"
    print(f"     accuracy {acc:.4f} (the book reports 0.8750 with these exact settings)")
"""),

md(r"""
**Why 87.5% and not 96%?** This network has *two* hidden layers and does worse than the
single-hidden-layer network in Exercise 3. The loss stops improving around epoch 1000 and
the boundary is almost straight — the network has **stalled**, not finished.

The likely reason is the one you saw in Exercise 4: squared error combined with sigmoid
produces very small gradients, and they shrink further with each layer you go back
through. The test set is also only 40 points, so each mistake costs 2.5%.

Deeper is not automatically better. That is a real and often expensive lesson.
"""),

# ------------------------------------------------------------------ experiments
md(r"""
---
## Exercise 7 — Experiments  ·  *book: "Try it yourself"*

No checks here. These are the questions the chapter leaves open, and the answers are
whatever your runs show. Fill in the table at the end from your own output — writing the
number down is what turns a run into knowledge.
"""),

ex(r"""
# Experiment A — how much does hidden-layer size matter?
# TODO: for each size, train a [2, size, 1] network on the data below and record the
#       test accuracy. Keep epochs and learning rate fixed so the comparison is fair.
for size in [2, 4, 8, 32]:
    raise NotImplementedError
""", r"""
# Experiment A — how much does hidden-layer size matter?
np.random.seed(0)
Xa, ya = make_moons(n_samples=400, noise=0.15, random_state=0)
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.25, random_state=0)

for size in [2, 4, 8, 32]:
    np.random.seed(0)                            # same start, so only `size` differs
    trial = NeuralNetwork([2, size, 1])
    final_loss = trial.train(Xa_tr, ya_tr, epochs=2000, learning_rate=1.0, verbose=False)
    acc = accuracy_score(ya_te, trial.predict(Xa_te))
    print(f"hidden={size:>3}   final loss {final_loss:.4f}   test accuracy {acc:.3f}")
"""),

ex(r"""
# Experiment B — the learning rate
# TODO: train [2, 8, 1] for 2000 epochs at each learning rate and print the final loss.
#       Expect one to be too slow and one to misbehave. Both failures are informative.
for lr in [0.001, 0.1, 1.0, 10.0]:
    raise NotImplementedError
""", r"""
# Experiment B — the learning rate
for lr in [0.001, 0.1, 1.0, 10.0]:
    np.random.seed(0)
    trial = NeuralNetwork([2, 8, 1])
    final_loss = trial.train(Xa_tr, ya_tr, epochs=2000, learning_rate=lr, verbose=False)
    acc = accuracy_score(ya_te, trial.predict(Xa_te))
    print(f"lr={lr:<7} final loss {final_loss:.4f}   test accuracy {acc:.3f}")
"""),

ex(r"""
# Experiment C — does the stalled deep network recover with more time?
# TODO: train [2, 8, 8, 1] for 10,000 epochs (book §1.3 suggests exactly this), then
#       plot the boundary with plot_boundary(...). Does the straight line start to curve?
raise NotImplementedError
""", r"""
# Experiment C — does the stalled deep network recover with more time?
np.random.seed(42)
deep = NeuralNetwork([2, 8, 8, 1])
deep.train(X3_tr, y3_tr, epochs=10000, learning_rate=0.5, verbose=False)
print(f"after 10,000 epochs: test accuracy "
      f"{accuracy_score(y3_te, deep.predict(X3_te)):.4f}")
plot_boundary(deep.predict, X3_te, y3_te, "[2, 8, 8, 1] after 10,000 epochs")
"""),

md(r"""
### Write down what you found

Double-click this cell and fill it in. If you cannot explain a row, that is the part of
the chapter to re-read.

| Experiment | What I changed | What happened | Why I think that is |
|---|---|---|---|
| A | hidden size 2 → 32 |  |  |
| B | learning rate 0.001 → 10 |  |  |
| C | 2,000 → 10,000 epochs |  |  |

**A harder question, for the brave.** The chapter says squared error learns slowly with
sigmoid, and that classifiers normally use **cross-entropy** loss instead:

$$L = -\frac{1}{n}\sum \big[y \log(p) + (1 - y)\log(1 - p)\big]$$

Its gradient with respect to the output is `(p - y) / (p * (1 - p) * n)`, which cancels the
sigmoid derivative in your backward pass — so the vanishing-gradient problem largely goes
away. Swap it into `train` and see whether the deep network stops stalling. (Add a small
`1e-9` inside the logs to avoid `log(0)`.)
"""),

# ------------------------------------------------------------------ wrap up
md(r"""
---
## Where you stand
"""),

code(r"""
progress()
"""),

md(r"""
### What you built

* A **neuron**: a weighted vote, squashed into a 0-to-1 verdict. The weights are the
  learned part.
* A **layer**: many neurons as one weight matrix, so one matrix multiply handles the whole
  batch. Each layer remembers its own input, because the weight gradient depends on it.
* **Training**: forward, loss, backward, update — repeated. Nothing more.
* **Backpropagation**: the chain rule, applied from the last layer to the first. Verifiable
  with a numerical gradient check, which you should reach for whenever a network trains
  suspiciously badly.
* **PyTorch**: the same loop, with the backward pass and the update handed to the library.
* **Shapes**: the source of most first-attempt bugs. Print `.shape` early and often.

### Before you move on

- [ ] Every check above is ✅ (or you know exactly why one is not).
- [ ] `Runtime → Run all` works top to bottom on a fresh runtime.
- [ ] You could explain, without looking, why the backward pass goes *backwards*.
- [ ] You filled in the experiments table in your own words.

**Next:** Chapter 2 moves from a network you trained to models someone else trained —
API calls, tokens, temperature and the context window.
"""),
]

SOLUTION_BANNER = r"""
# ⚠️ Solutions — Chapter 1

This is the worked key for *Chapter 1 — Build a Neural Network from Scratch*.

Every cell is complete and the whole notebook runs top to bottom. Use it to check your
work **after** you have attempted an exercise, or to get unstuck on one specific line.
Reading a solution you have not attempted feels like learning and is not: the struggle is
where the understanding comes from.
"""
