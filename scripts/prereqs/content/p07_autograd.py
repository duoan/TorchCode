"""Prerequisite 7: Autograd essentials.

Content for templates/00_p07_autograd.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P7",
    "file": "00_p07_autograd",
    "task_id": "p07_autograd",
    "title": "Autograd Essentials",
    "intro": r'''
# 🔵 Prerequisite 7: Autograd Essentials

Every training-related problem (optimizers, gradient clipping, gradient accumulation, BatchNorm's running stats, DPO/PPO/GRPO losses) depends on knowing **what autograd tracks**, **where gradients end up**, and **how to stop them**.

The classic bugs are silent: gradients that accumulate across steps, a parameter update that gets recorded in the graph, or a gradient leaking into something that should be constant (a target, an old policy, an advantage).

### What you'll learn
1. `requires_grad`, leaf tensors, and where `.grad` appears
2. Gradients **accumulate**: zero them between steps
3. `torch.no_grad()` for updates and initialization
4. `detach()` as a stop-gradient
5. `torch.autograd.grad` for a gradient without touching `.grad`
6. Inspecting gradients: `inference_mode`, tensor hooks, `gradcheck`, anomaly detection
7. Many ways to do it: stopping gradients, and their differences

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `grad_of(f, x)` | checking gradients by hand, #40 Linear Regression |
| 2 | `sgd_step_(params, lr)` | #29 Adam, #40 Linear Regression |
| 3 | `stop_gradient_mse(pred, target)` | #37 DPO, #38 GRPO, #39 PPO, #41 OPD |
| 4 | `accumulate_grads(w, batches)` | #31 Gradient Accumulation |
| 5 | `straight_through_round(x)` | quantization-aware training (#36) |

### Rules
- **No Python loops** over tensor elements. (Looping over a *list* of parameters or micro-batches is fine.)
- Do **not** use `torch.optim`.
''',
    "lesson": [
        ("md", r'''
## 1. Leaf tensors and `.grad`

A tensor you create with `requires_grad=True` is a **leaf**. Every operation on it records a node in a graph. Calling `.backward()` on a **scalar** result fills `.grad` on the **leaves**.
Intermediate (non-leaf) tensors don't keep `.grad` unless you ask with `.retain_grad()`.
'''),
        ("code", r'''
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (w ** 2)              # intermediate
loss = y.sum()
loss.backward()
print(w.is_leaf, y.is_leaf)        # True False
print(w.grad)                      # tensor([2., 4., 6.]) = d(sum w^2)/dw
print(y.grad)                      # None (with a warning): non-leaf
'''),
        ("md", r'''
## 2. ⚠️ Gradients accumulate

`.backward()` **adds** to `.grad`. It doesn't overwrite it. That's what makes gradient accumulation (#31) work, and it's also why every training loop calls `zero_grad()` before each step.
'''),
        ("code", r'''
w = torch.tensor([1.0, 2.0], requires_grad=True)
(w * 3).sum().backward()
print(w.grad)              # tensor([3., 3.])
(w * 3).sum().backward()
print(w.grad)              # tensor([6., 6.]): added, not replaced
w.grad = None              # or w.grad.zero_()
'''),
        ("md", r'''
## 3. `torch.no_grad()` for updates

Updating a parameter in place (`w -= lr * w.grad`) must **not** be recorded in the graph. PyTorch refuses in-place ops on a leaf that requires grad unless you wrap them in `torch.no_grad()`.
Weight initialization (`weight.normal_(...)` in #20) and running-stat updates (#07) use the same pattern.
'''),
        ("code", r'''
w = torch.tensor([1.0, 2.0], requires_grad=True)
(w ** 2).sum().backward()
try:
    w -= 0.1 * w.grad
except RuntimeError as e:
    print("RuntimeError:", str(e)[:70])
with torch.no_grad():
    w -= 0.1 * w.grad
print(w, w.is_leaf, w.requires_grad)   # still a trainable leaf
'''),
        ("md", r'''
## 4. `detach()` as a stop-gradient

`x.detach()` returns the same values, cut off from the graph. Anything computed from it is treated as a **constant** during backward.
Use it whenever a quantity must not be trained through: regression targets, PPO's old log-probs, GRPO's advantages, a teacher's logits, BatchNorm's running stats.

A useful trick: `x + (f(x) - x).detach()` has the *value* of `f(x)` but the *gradient* of `x`. This is the **straight-through estimator**.
'''),
        ("code", r'''
x = torch.tensor([2.0], requires_grad=True)
a = (x * x).sum()                  # d/dx = 2x = 4
b = (x * x.detach()).sum()         # second factor is a constant: d/dx = x = 2
print(torch.autograd.grad(a, x)[0], torch.autograd.grad(b, x)[0])   # tensor([4.]) tensor([2.])
'''),
        ("md", r'''
## 5. `torch.autograd.grad`

`torch.autograd.grad(output, inputs)` returns the gradients as a tuple **without** writing to `.grad`. It's handy for checking a gradient you derived by hand, or for gradients of gradients.
'''),
        ("code", r'''
x = torch.tensor([1.0, -2.0], requires_grad=True)
(g,) = torch.autograd.grad((x ** 3).sum(), x)
print(g, x.grad)          # tensor([3., 12.]) None: .grad untouched
'''),
        ("md", r'''
## 6. Inspecting gradients

- **`torch.inference_mode()`** is a faster, stricter `no_grad()` for pure inference: tensors created inside can **never** be used in autograd later. Use `no_grad()` for things like parameter updates, `inference_mode()` for serving and evaluation.
- **`t.register_hook(fn)`** calls `fn(grad)` when the gradient of `t` is computed. Use it to print, log, or even modify a gradient in flight (it works on intermediate tensors too).
- **`torch.autograd.gradcheck(fn, inputs)`** compares your gradients against finite differences. Use `float64` inputs, or the check is too noisy.
- **`torch.autograd.detect_anomaly()`** makes backward raise an error at the **first** operation that produced `NaN`, with a traceback to the forward line that created it.
'''),
        ("code", r'''
with torch.inference_mode():
    y = torch.ones(2) * 3
print(y.is_inference())                        # True: can't be fed into autograd later

x = torch.tensor([1.0, 2.0], requires_grad=True)
h = x * 3
h.register_hook(lambda g: print("grad reaching h:", g))
(h ** 2).sum().backward()                      # prints the gradient flowing into h
print(x.grad)

f = lambda t: (t.sin() * t).sum()
print(torch.autograd.gradcheck(f, (torch.randn(4, dtype=torch.float64, requires_grad=True),)))   # True

z = torch.tensor([0.0], requires_grad=True)
try:
    with torch.autograd.detect_anomaly():
        torch.sqrt(z - 1).sum().backward()      # sqrt(-1) = nan
except RuntimeError as e:
    print("anomaly:", str(e)[:70])
'''),
        ("md", r'''
## 7. Many ways to do it: stopping gradients

| Form | What it does | Use for |
|------|--------------|---------|
| `x.detach()` | same values, cut out of the graph; **shares memory** with `x` | constants inside a loss (targets, old log-probs) |
| `with torch.no_grad():` | nothing inside is recorded | parameter updates, running stats, evaluation |
| `with torch.inference_mode():` | like `no_grad`, faster, results can never enter autograd | pure inference |
| `p.requires_grad_(False)` | stops computing gradients **for a leaf** from now on | freezing parameters (P8) |
| `x.detach().clone()` | a cut-off **copy** | when you'll modify the copy in place |

`x.data` also exists in old code. Avoid it: autograd doesn't track changes made through it, so it can silently corrupt gradients.
'''),
        ("code", r'''
x = torch.tensor([1.0, 2.0], requires_grad=True)
d = x.detach()
print(d.data_ptr() == x.data_ptr())          # True: shares memory
d2 = x.detach().clone()
print(d2.data_ptr() == x.data_ptr())         # False: an independent copy
with torch.no_grad():
    y = x * 2
print(y.requires_grad, (x * 2).requires_grad)  # False True
'''),
    ],
    "exercises_intro": r'''
---
# ✏️ Exercises

Implement the five functions below. Each one has a debug cell you can use before you submit.
''',
    "exercises": [
        {
            "spec": r'''
### Exercise 1: `grad_of(f, x)`

Return the gradient of the scalar `f(x)` with respect to `x`.

- `f`: a function from a tensor to a **scalar** tensor
- `x`: a tensor, which may or may not have `requires_grad`
- returns: a tensor with the shape of `x`
- must **not** change `x` or `x.grad`

*Comparing autograd with a hand-derived gradient is how you check manual-gradient code like #40's gradient descent.*
''',
            "stub": r'''
def grad_of(f, x: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def grad_of(f, x: torch.Tensor) -> torch.Tensor:
    x_ = x.detach().requires_grad_(True)     # fresh leaf; leaves x untouched
    (g,) = torch.autograd.grad(f(x_), x_)
    return g
''',
            "debug": r'''
x = torch.tensor([1.0, -2.0, 3.0])
print(grad_of(lambda t: (t ** 2).sum(), x))   # expect tensor([ 2., -4.,  6.])
''',
        },
        {
            "spec": r'''
### Exercise 2: `sgd_step_(params, lr)`

One step of plain gradient descent, then clear the gradients, as `optimizer.step()` followed by `optimizer.zero_grad()` would.

- `params`: a list of leaf tensors with `requires_grad=True`; some may have `.grad is None` (skip those)
- `lr`: float
- update each parameter **in place**: `p ← p - lr * p.grad`
- afterwards every parameter must still be a leaf with `requires_grad=True`, and its `.grad` must be `None` or all zeros
- returns: nothing

*#29 Adam is the same loop with a smarter update.*
''',
            "stub": r'''
def sgd_step_(params: list[torch.Tensor], lr: float) -> None:
    pass  # Replace this
''',
            "solution": r'''
def sgd_step_(params: list[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            p -= lr * p.grad
            p.grad = None
''',
            "debug": r'''
w = torch.tensor([1.0, 2.0], requires_grad=True)
(w ** 2).sum().backward()          # grad = 2w = [2, 4]
sgd_step_([w], lr=0.1)
print(w, w.grad)                   # expect tensor([0.8000, 1.6000], requires_grad=True) None
''',
        },
        {
            "spec": r'''
### Exercise 3: `stop_gradient_mse(pred, target)`

Mean squared error where gradients flow into `pred` **only**. `target` is treated as a constant, even if it requires grad.

- `pred`, `target`: same shape
- returns: a scalar tensor, `mean((pred - target) ** 2)`

*PPO, GRPO, DPO and OPD all detach something: the old log-probs, the advantages, the reference model, the teacher.*
''',
            "stub": r'''
def stop_gradient_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def stop_gradient_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target.detach()) ** 2).mean()
''',
            "debug": r'''
pred = torch.tensor([1.0, 2.0], requires_grad=True)
target = torch.tensor([0.0, 0.0], requires_grad=True)
stop_gradient_mse(pred, target).backward()
print(pred.grad, target.grad)      # expect tensor([1., 2.]) None
''',
        },
        {
            "spec": r'''
### Exercise 4: `accumulate_grads(w, batches)`

Gradient accumulation: compute the gradient of the **average** loss over several equally sized micro-batches, calling `.backward()` once per micro-batch.

- `w`: leaf tensor `(D,)` with `requires_grad=True`; it may already hold an old `.grad`
- `batches`: a list of `(x, y)` pairs, `x`: `(n, D)`, `y`: `(n,)`
- loss of one micro-batch: `mean((x @ w - y) ** 2)`
- returns: `w.grad`, which must equal the gradient of the mean of the micro-batch losses

*Two things to get right: clear the old gradient first, and scale each micro-batch loss. This is #31.*
''',
            "stub": r'''
def accumulate_grads(w: torch.Tensor, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def accumulate_grads(w: torch.Tensor, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    w.grad = None
    for x, y in batches:
        loss = ((x @ w - y) ** 2).mean() / len(batches)
        loss.backward()
    return w.grad
''',
            "debug": r'''
torch.manual_seed(0)
w = torch.randn(3, requires_grad=True)
X, Y = torch.randn(8, 3), torch.randn(8)
batches = [(X[:4], Y[:4]), (X[4:], Y[4:])]
g = accumulate_grads(w, batches).clone()
full = grad_of(lambda t: ((X @ t - Y) ** 2).mean(), w)
print(g, full)          # expect the same values (uses your grad_of from Exercise 1)
''',
        },
        {
            "spec": r'''
### Exercise 5: `straight_through_round(x)`

Round to the nearest integer in the **forward** pass, but let the gradient pass through **unchanged** in the backward pass, as if rounding were the identity.

- `x`: float tensor
- returns: same shape, values equal to `torch.round(x)`, and `d out / d x = 1` everywhere

*`torch.round` alone has zero gradient almost everywhere, so nothing could learn through it. Quantization-aware training (#36) uses this trick.*
''',
            "stub": r'''
def straight_through_round(x: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def straight_through_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()
''',
            "debug": r'''
x = torch.tensor([0.2, 1.7, -2.6], requires_grad=True)
out = straight_through_round(x)
out.sum().backward()
print(out, x.grad)      # expect tensor([ 0.,  2., -3.]) tensor([1., 1., 1.])
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (7, "Updating running stats from `batch_mean.detach()` so they never enter the graph"),
        (20, "Filling a weight in place under `torch.no_grad()`"),
        (42, "Updating per-channel running stats in place without entering the graph"),
        (21, "Reading and rescaling `p.grad` for every parameter"),
        (29, "`sgd_step_`: the in-place update loop Adam builds on"),
        (31, "`accumulate_grads`: scale each micro-batch loss and let `.backward()` accumulate"),
        (37, "`stop_gradient_mse`-style constants: the reference model's log-probs"),
        (38, "Detaching the advantages before weighting the log-probs"),
        (39, "Detaching the old log-probs and the advantages in the ratio"),
        (40, "`grad_of` to check the hand-derived gradient of the gradient-descent method"),
        (41, "Detaching the teacher's logits"),
    ],
}
