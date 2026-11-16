"""Bonus 5: Vectorizing with torch.func.

Content for templates/00_b05_torch_func.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "B5",
    "file": "00_b05_torch_func",
    "task_id": "b05_torch_func",
    "title": "Vectorizing with torch.func",
    "covers": "`vmap` (in_dims, nesting), functional `grad`, `functional_call`, per-sample gradients, `jacrev` / `hessian`, model ensembles with `stack_module_state`",
    "imports": "import copy\nimport torch\nfrom torch import nn\nfrom torch.func import vmap, grad, functional_call, jacrev, hessian, stack_module_state",
    "intro": r'''
# ⭐ Bonus 5: Vectorizing with `torch.func`

Some computations are easy to write for **one** example but awkward to batch by hand: a per-example gradient, a pairwise comparison, an ensemble of models. `torch.func` (formerly *functorch*) offers **function transforms**: `vmap` adds a batch dimension to a function written for one example, `grad` turns a function into its gradient function, and they compose.

### What you'll learn
1. `vmap`: batch a single-example function; `in_dims` and nesting
2. Functional `grad`: gradients as functions, not `.grad` side effects
3. `functional_call`: run an `nn.Module` with parameters you pass in
4. Per-sample gradients with `vmap(grad(...))`, the core of differentially private SGD
5. Jacobians, Hessians, and model ensembles

### Exercises
| # | Function | Use it on |
|---|----------|-----------|
| 1 | `batched_dot(a, b)` | #05 attention written for one example |
| 2 | `pairwise_fn(f, X, Y)` | similarity / distance matrices |
| 3 | `per_sample_grads(model, loss_fn, x, y)` | #21 Gradient Clipping, per sample |
| 4 | `jacobian_of(f, x)` | #40 Linear Regression |
| 5 | `ensemble_forward(models, x)` | #28 MoE experts |

### Rules
- Use the `torch.func` transforms. In particular, no Python loops over examples, pairs, or models (Exercise 5 may loop to **build** the ensemble, not to run it).
''',
    "lesson": [
        ("md", r'''
## 1. `vmap`: write it for one example, run it for a batch

`vmap(f)` returns a function that maps `f` over a batch dimension (dim 0 by default) **without a Python loop**. It's typically as fast as hand-batched code.
- `in_dims=(0, None)`: batch over the first argument, broadcast the second one unchanged
- `out_dims`: where the batch dimension goes in the output
- Inside `f`, write code for a **single** example; data-dependent Python control flow (`if x.sum() > 0:`) doesn't work under `vmap`, so use `torch.where` instead.
'''),
        ("code", r'''
def attention_one(q, K, V):                       # q: (d,), K: (S, d), V: (S, d): ONE query
    w = torch.softmax(K @ q / q.shape[0] ** 0.5, dim=0)
    return w @ V

S, d = 5, 4
Q, K, V = torch.randn(3, d), torch.randn(S, d), torch.randn(S, d)
out = vmap(attention_one, in_dims=(0, None, None))(Q, K, V)    # batch over queries only
ref = torch.softmax(Q @ K.T / d ** 0.5, dim=-1) @ V
print(out.shape, torch.allclose(out, ref, atol=1e-6))           # (3, 4) True
'''),
        ("md", r'''
## 2. Functional `grad`

`grad(f)` returns a **new function** that computes the gradient of the scalar `f` with respect to its first argument (or `argnums=`). Nothing is written to `.grad`, and transforms compose: `grad(grad(f))` is a second derivative, `vmap(grad(f))` gives gradients for a whole batch.
'''),
        ("code", r'''
f = lambda x: (x ** 3).sum()
print(grad(f)(torch.tensor([1.0, 2.0])))               # 3x^2 = [3, 12]
g1 = lambda x: torch.sin(x)
print(grad(grad(g1))(torch.tensor(0.5)), -torch.sin(torch.tensor(0.5)))   # second derivative
'''),
        ("md", r'''
## 3. `functional_call`: modules as pure functions

To take gradients with respect to a module's **parameters** functionally, the parameters must be an input. `functional_call(model, params, args)` runs `model.forward(*args)` using the tensors in the `params` dict instead of the module's own.
'''),
        ("code", r'''
model = nn.Linear(3, 1)
params = {k: v.detach() for k, v in model.named_parameters()}
x = torch.randn(4, 3)
print(torch.allclose(functional_call(model, params, (x,)), model(x)))   # True
zeros = {k: torch.zeros_like(v) for k, v in params.items()}
print(functional_call(model, zeros, (x,)).flatten())                    # all zeros: our params were used
'''),
        ("md", r'''
## 4. Per-sample gradients

`loss.backward()` gives the gradient of the **average** loss. Some methods need the gradient of **each example separately**: differentially private SGD clips every example's gradient before averaging, and some data-selection methods rank examples by gradient norm. With a Python loop that's one backward pass per example. With `torch.func` it's one vectorized call:

```python
def loss_one(params, x, y):           # loss for ONE example
    out = functional_call(model, params, (x.unsqueeze(0),))
    return loss_fn(out, y.unsqueeze(0))
per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)   # dict: name -> (B, *param.shape)
```
'''),
        ("code", r'''
model = nn.Linear(3, 1)
params = {k: v.detach() for k, v in model.named_parameters()}
X, Y = torch.randn(5, 3), torch.randn(5, 1)

def loss_one(p, x, y):
    return nn.functional.mse_loss(functional_call(model, p, (x.unsqueeze(0),)), y.unsqueeze(0))

per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)
print({k: tuple(v.shape) for k, v in per_sample.items()})        # one gradient per example
print(torch.allclose(per_sample["weight"].mean(0), grad(lambda p: nn.functional.mse_loss(functional_call(model, p, (X,)), Y))(params)["weight"], atol=1e-6))
'''),
        ("md", r'''
## 5. Jacobians, Hessians, ensembles

- `jacrev(f)(x)` / `jacfwd(f)(x)`: the full Jacobian, shape `(*out.shape, *x.shape)`. `jacrev` is better for few outputs, `jacfwd` for few inputs. `hessian(f)(x)` gives second derivatives.
- **Ensembles:** `stack_module_state(models)` stacks the parameters of several identical models along a new dim 0; `vmap` over that dim runs all models in **one** batched call (the experts of #28 MoE, or bootstrap ensembles).
- **When not to bother:** if ordinary batched tensor code is easy to write, write that. `torch.compile` is the separate tool for making code faster.
'''),
        ("code", r'''
f = lambda x: torch.stack([x[0] * x[1], x[0] ** 2])
print(jacrev(f)(torch.tensor([2.0, 3.0])))                        # [[3, 2], [4, 0]]

X = torch.randn(10, 3)
mse = lambda w: ((X @ w) ** 2).mean()
print(torch.allclose(hessian(mse)(torch.randn(3)), 2 * X.T @ X / 10, atol=1e-5))   # Hessian of MSE = 2 XᵀX / N
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
### Exercise 1: `batched_dot(a, b)`

Row-wise dot products, by vmapping `torch.dot` (which only accepts 1-D tensors).

- `a`, `b`: `(B, D)`
- returns: `(B,)` with `out[i] = dot(a[i], b[i])`
''',
            "stub": r'''
from torch.func import vmap

def batched_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
from torch.func import vmap

def batched_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return vmap(torch.dot)(a, b)
''',
            "debug": r'''
a, b = torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[5., 6.], [7., 8.]])
print(batched_dot(a, b))    # expect tensor([17., 53.])
''',
        },
        {
            "spec": r'''
### Exercise 2: `pairwise_fn(f, X, Y)`

Apply a function of **two vectors** to every pair of rows, with nested `vmap`.

- `f`: a function `(x: (D,), y: (D,)) -> scalar tensor`
- `X`: `(N, D)`, `Y`: `(M, D)`
- returns: `(N, M)` with `out[i, j] = f(X[i], Y[j])`

*Hint: the inner `vmap` maps over `Y` with `X`'s row fixed (`in_dims=(None, 0)`), the outer one over `X`.*
''',
            "stub": r'''
from torch.func import vmap

def pairwise_fn(f, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
from torch.func import vmap

def pairwise_fn(f, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    over_y = vmap(f, in_dims=(None, 0))           # one x against all ys -> (M,)
    return vmap(over_y, in_dims=(0, None))(X, Y)  # all xs -> (N, M)
''',
            "debug": r'''
sq_dist = lambda x, y: ((x - y) ** 2).sum()
X, Y = torch.tensor([[0., 0.], [1., 1.]]), torch.tensor([[0., 0.], [3., 4.], [1., 0.]])
print(pairwise_fn(sq_dist, X, Y))    # expect [[0, 25, 1], [2, 13, 1]] (P2's pairwise_sq_dists)
''',
        },
        {
            "spec": r'''
### Exercise 3: `per_sample_grads(model, loss_fn, x, y)`

The gradient of the loss of **each example separately**, for every parameter, without a Python loop.

- `model`: an `nn.Module`; `loss_fn(pred, target)`: returns a scalar (e.g. `nn.functional.mse_loss`)
- `x`: `(B, ...)`, `y`: `(B, ...)`; example `i` is `(x[i], y[i])`, fed to the model as a batch of one
- returns: a dict `{param_name: tensor of shape (B, *param.shape)}` for every parameter in `model.named_parameters()`
- the model's own `.grad`s must not change
''',
            "stub": r'''
from torch.func import functional_call, grad, vmap

def per_sample_grads(model: nn.Module, loss_fn, x: torch.Tensor, y: torch.Tensor) -> dict:
    pass  # Replace this
''',
            "solution": r'''
from torch.func import functional_call, grad, vmap

def per_sample_grads(model: nn.Module, loss_fn, x: torch.Tensor, y: torch.Tensor) -> dict:
    params = {name: p.detach() for name, p in model.named_parameters()}

    def loss_one(p, xi, yi):
        out = functional_call(model, p, (xi.unsqueeze(0),))
        return loss_fn(out, yi.unsqueeze(0))

    return vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)
''',
            "debug": r'''
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
x, y = torch.randn(6, 3), torch.randn(6, 1)
g = per_sample_grads(model, nn.functional.mse_loss, x, y)
print({k: tuple(v.shape) for k, v in g.items()})      # expect a leading 6 on every shape
print(g["0.weight"].flatten(1).norm(dim=1))           # one gradient norm per example (for DP-SGD clipping)
''',
        },
        {
            "spec": r'''
### Exercise 4: `jacobian_of(f, x)`

The full Jacobian of a vector function, using `torch.func.jacrev`.

- `f`: a function of one tensor `x` returning a tensor
- returns: shape `(*f(x).shape, *x.shape)` with `out[i..., j...] = ∂f(x)[i...] / ∂x[j...]`
''',
            "stub": r'''
from torch.func import jacrev

def jacobian_of(f, x: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
from torch.func import jacrev

def jacobian_of(f, x: torch.Tensor) -> torch.Tensor:
    return jacrev(f)(x)
''',
            "debug": r'''
f = lambda x: torch.stack([x[0] * x[1], x[0] ** 2, x[1].sin()])
print(jacobian_of(f, torch.tensor([2.0, 0.0])))
# expect [[0, 2], [4, 0], [0, 1]]
''',
        },
        {
            "spec": r'''
### Exercise 5: `ensemble_forward(models, x)`

Run several identically-shaped models on the same input in **one** vectorized call.

- `models`: a list of `E` modules with the same architecture (different weights)
- `x`: the input for all of them
- returns: `(E, *model(x).shape)`, equal to `torch.stack([m(x) for m in models])`
- use `stack_module_state` + `functional_call` + `vmap`; don't run the models in a Python loop

*`stack_module_state` returns `(params, buffers)`. You need one "template" module to call: a `copy.deepcopy(models[0])` moved to the `"meta"` device works and holds no real memory.*
''',
            "stub": r'''
import copy
from torch.func import functional_call, stack_module_state, vmap

def ensemble_forward(models: list[nn.Module], x: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
import copy
from torch.func import functional_call, stack_module_state, vmap

def ensemble_forward(models: list[nn.Module], x: torch.Tensor) -> torch.Tensor:
    params, buffers = stack_module_state(models)          # each leaf: (E, *shape)
    base = copy.deepcopy(models[0]).to("meta")            # a stateless template

    def run_one(p, b, inp):
        return functional_call(base, (p, b), (inp,))

    return vmap(run_one, in_dims=(0, 0, None))(params, buffers, x)
''',
            "debug": r'''
torch.manual_seed(0)
models = [nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)) for _ in range(5)]
x = torch.randn(7, 3)
out = ensemble_forward(models, x)
print(out.shape, torch.allclose(out, torch.stack([m(x) for m in models]), atol=1e-6))   # expect (5, 7, 2) True
''',
        },
    ],
    # Problem number -> how to use this notebook on it ("Where to use it" section)
    "unlocks": [
        (5, "Write attention for a single query, then `vmap` it over queries and batches"),
        (21, "Clip each example's gradient (from `per_sample_grads`) before averaging: DP-SGD"),
        (28, "Run all experts as one stacked ensemble with `ensemble_forward`"),
        (40, "Check your closed form: the `hessian` of the MSE is `2 XᵀX / N`"),
    ],
}
