"""Bonus 5: Vectorizing with torch.func (multi-exercise task)."""

# Counts low-level (dispatcher) ops while a function runs. A vmap'd function runs the
# same number of ops for any batch size; a Python loop runs ~B times more.
_COUNTER = """
import torch
from torch.utils._python_dispatch import TorchDispatchMode
class _OpCount(TorchDispatchMode):
    def __init__(self): super().__init__(); self.n = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))
def _ops(fn):
    with _OpCount() as c:
        fn()
    return c.n
"""

TASK = {
    "title": "Bonus 5: Vectorizing with torch.func",
    "difficulty": "Bonus",
    "order": 5,
    "function_name": "batched_dot",
    "function_names": [
        "batched_dot",
        "pairwise_fn",
        "per_sample_grads",
        "jacobian_of",
        "ensemble_forward",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `batched_dot`: `vmap(torch.dot)(a, b)`.\n"
        "- `pairwise_fn`: `vmap(vmap(f, in_dims=(None, 0)), in_dims=(0, None))(X, Y)`.\n"
        "- `per_sample_grads`: detach the params into a dict, write `loss_one(p, xi, yi)` with `functional_call` on `xi.unsqueeze(0)`, then `vmap(grad(loss_one), in_dims=(None, 0, 0))`.\n"
        "- `jacobian_of`: `jacrev(f)(x)`.\n"
        "- `ensemble_forward`: `params, buffers = stack_module_state(models)`; template = `copy.deepcopy(models[0]).to('meta')`; `vmap` a `functional_call` over dim 0 of params and buffers, with `in_dims=(0, 0, None)`."
    ),
    "tests": [
        {
            "name": "batched_dot",
            "code": """
import torch
torch.manual_seed(0)
a, b = torch.randn(7, 5), torch.randn(7, 5)
out = batched_dot(a, b)
assert isinstance(out, torch.Tensor) and out.shape == (7,), f'Expected shape (7,), got {getattr(out, "shape", out)}'
assert torch.allclose(out, (a * b).sum(-1), atol=1e-6), 'Wrong values'
""",
        },
        {
            "name": "pairwise_fn: arbitrary pair functions",
            "code": """
import torch
torch.manual_seed(0)
X, Y = torch.randn(4, 3), torch.randn(6, 3)
cos = lambda x, y: (x * y).sum() / (x.norm() * y.norm())
out = pairwise_fn(cos, X, Y)
assert isinstance(out, torch.Tensor) and out.shape == (4, 6), f'Expected (N, M) = (4, 6), got {getattr(out, "shape", out)}'
ref = torch.stack([torch.stack([cos(x, y) for y in Y]) for x in X])
assert torch.allclose(out, ref, atol=1e-6), 'out[i, j] should be f(X[i], Y[j])'
asym = pairwise_fn(lambda x, y: x[0] - 2 * y[0], X, Y)
assert torch.allclose(asym, X[:, :1] - 2 * Y[:, 0][None, :], atol=1e-6), 'Rows must index X and columns Y (the order of the arguments matters)'
""",
        },
        {
            "name": "per_sample_grads: matches a per-example loop",
            "code": """
import torch
from torch import nn
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
x, y = torch.randn(6, 3), torch.randn(6, 1)
g = per_sample_grads(model, nn.functional.mse_loss, x, y)
assert isinstance(g, dict) and set(g) == {n for n, _ in model.named_parameters()}, 'Return a dict with one entry per parameter name'
assert all(p.grad is None for p in model.parameters()), 'Do not write to the model\\'s .grad'
for i in range(6):
    model.zero_grad(set_to_none=True)
    nn.functional.mse_loss(model(x[i:i+1]), y[i:i+1]).backward()
    for n, p in model.named_parameters():
        assert g[n].shape == (6, *p.shape), f'{n}: expected shape {(6, *p.shape)}, got {tuple(g[n].shape)}'
        assert torch.allclose(g[n][i], p.grad, atol=1e-5), f'{n}: gradient of example {i} differs from a per-example backward'
""",
        },
        {
            "name": "per_sample_grads: vectorized, not a loop over examples",
            "code": _COUNTER + """
from torch import nn
model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
res = per_sample_grads(model, nn.functional.mse_loss, torch.randn(4, 3), torch.randn(4, 1))
assert isinstance(res, dict) and res and all(v.shape[0] == 4 for v in res.values()), 'per_sample_grads must return a dict of (B, ...) gradients'
run = lambda B: _ops(lambda: per_sample_grads(model, nn.functional.mse_loss, torch.randn(B, 3), torch.randn(B, 1)))
n2, n16 = run(2), run(16)
assert n16 <= n2 * 1.5 + 10, f'{n2} ops for 2 examples but {n16} for 16: looks like a Python loop over examples. Use vmap(grad(...))'
""",
        },
        {
            "name": "jacobian_of",
            "code": """
import torch
torch.manual_seed(0)
W = torch.randn(2, 3, 4)
f = lambda x: torch.tanh(W @ x)                  # x: (4,) -> (2, 3)
x = torch.randn(4)
J = jacobian_of(f, x)
assert isinstance(J, torch.Tensor) and J.shape == (2, 3, 4), f'Expected shape (*out.shape, *x.shape) = (2, 3, 4), got {getattr(J, "shape", J)}'
ref = torch.autograd.functional.jacobian(f, x)
assert torch.allclose(J, ref, atol=1e-6), 'Wrong Jacobian'
""",
        },
        {
            "name": "ensemble_forward: matches running each model, with buffers",
            "code": """
import torch
from torch import nn
class Shift(nn.Module):
    def __init__(self, s):
        super().__init__(); self.register_buffer('s', torch.tensor(float(s)))
    def forward(self, x): return x + self.s
torch.manual_seed(0)
models = [nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), Shift(i)) for i in range(5)]
x = torch.randn(7, 3)
out = ensemble_forward(models, x)
ref = torch.stack([m(x) for m in models])
assert isinstance(out, torch.Tensor) and out.shape == (5, 7, 2), f'Expected (E, B, out) = (5, 7, 2), got {getattr(out, "shape", out)}'
assert torch.allclose(out, ref, atol=1e-5), 'Outputs differ from running each model (are buffers passed to functional_call?)'
""",
        },
        {
            "name": "ensemble_forward: vectorized, not a loop over models",
            "code": _COUNTER + """
from torch import nn
x = torch.randn(4, 3)
def run(E):
    models = [nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2)) for _ in range(E)]
    with torch.no_grad():
        return _ops(lambda: ensemble_forward(models, x))
probe = ensemble_forward([nn.Linear(3, 2) for _ in range(3)], x)
assert isinstance(probe, torch.Tensor) and probe.shape == (3, 4, 2), 'ensemble_forward must return the stacked outputs'
n2, n16 = run(2), run(16)
assert n16 <= n2 * 1.5 + 10, f'{n2} ops for 2 models but {n16} for 16: looks like a Python loop over models. Use stack_module_state + vmap'
""",
        },
    ],
}
