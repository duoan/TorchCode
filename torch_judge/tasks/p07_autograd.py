"""Prerequisite 7: Autograd essentials (multi-exercise task)."""

TASK = {
    "title": "Prereq 7: Autograd Essentials",
    "difficulty": "Prereq",
    "order": 7,
    "function_name": "grad_of",
    "function_names": [
        "grad_of",
        "sgd_step_",
        "stop_gradient_mse",
        "accumulate_grads",
        "straight_through_round",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `grad_of`: make a fresh leaf with `x.detach().requires_grad_(True)`, then `torch.autograd.grad(f(x_), x_)[0]`.\n"
        "- `sgd_step_`: inside `with torch.no_grad():` do `p -= lr * p.grad` (in place), then `p.grad = None`. `p = p - ...` only rebinds a local name.\n"
        "- `stop_gradient_mse`: use `target.detach()` inside the loss.\n"
        "- `accumulate_grads`: set `w.grad = None` first, then for each micro-batch call `(loss / len(batches)).backward()`.\n"
        "- `straight_through_round`: `x + (torch.round(x) - x).detach()` has the value of `round(x)` and the gradient of `x`."
    ),
    "tests": [
        {
            "name": "grad_of: values, x untouched",
            "code": """
import torch
x = torch.tensor([1.0, -2.0, 3.0])
g = grad_of(lambda t: (t ** 2).sum(), x)
assert isinstance(g, torch.Tensor) and g.shape == x.shape, f'Expected a tensor of shape {tuple(x.shape)}'
assert torch.allclose(g, 2 * x), f'd/dx sum(x^2) = 2x, got {g}'
assert not x.requires_grad, 'Do not change x (it should not start requiring grad)'
y = torch.tensor([0.5, 1.0], requires_grad=True)
y.grad = torch.tensor([7.0, 7.0])
g = grad_of(lambda t: torch.sin(t).sum(), y)
assert torch.allclose(g, torch.cos(y.detach())), 'Wrong gradient for sin'
assert torch.equal(y.grad, torch.tensor([7.0, 7.0])), 'x.grad must not change: use torch.autograd.grad, not .backward()'
""",
        },
        {
            "name": "sgd_step_: in-place update, still a leaf",
            "code": """
import torch
w = torch.tensor([1.0, 2.0], requires_grad=True)
b = torch.tensor([5.0], requires_grad=True)
(w ** 2).sum().backward()
sgd_step_([w, b], lr=0.1)
assert torch.allclose(w.detach(), torch.tensor([0.8, 1.6])), f'Expected w - lr * grad = [0.8, 1.6], got {w.detach()}. p = p - ... does not update the caller'
assert w.is_leaf and w.requires_grad, 'w must stay a trainable leaf: do the update under torch.no_grad()'
assert w.grad is None or torch.all(w.grad == 0), 'Clear the gradient after the step'
assert torch.equal(b.detach(), torch.tensor([5.0])), 'Parameters with grad None must be skipped'
""",
        },
        {
            "name": "sgd_step_: repeated steps",
            "code": """
import torch
w = torch.tensor([4.0], requires_grad=True)
for _ in range(3):
    (w ** 2).sum().backward()
    sgd_step_([w], lr=0.25)
assert torch.allclose(w.detach(), torch.tensor([0.5])), f'Three steps of w -= 0.25 * 2w from 4 give 0.5, got {w.item():.4f}. Are old gradients accumulating?'
""",
        },
        {
            "name": "stop_gradient_mse",
            "code": """
import torch
pred = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
target = torch.tensor([0.0, 2.0, 1.0], requires_grad=True)
loss = stop_gradient_mse(pred, target)
assert isinstance(loss, torch.Tensor) and loss.ndim == 0, 'Return a scalar tensor'
assert torch.allclose(loss, torch.tensor(10.0 / 3)), f'Expected mean squared error 3.3333, got {loss.item():.4f}'
loss.backward()
assert target.grad is None or torch.all(target.grad == 0), 'No gradient may flow into target: detach it'
assert torch.allclose(pred.grad, 2 * (pred - target).detach() / 3), 'Wrong gradient for pred'
""",
        },
        {
            "name": "accumulate_grads: equals full-batch gradient",
            "code": """
import torch
torch.manual_seed(0)
w = torch.randn(3, requires_grad=True)
X, Y = torch.randn(12, 3), torch.randn(12)
batches = [(X[i:i+4], Y[i:i+4]) for i in range(0, 12, 4)]
w.grad = torch.full((3,), 100.0)              # stale gradient from an earlier step
g = accumulate_grads(w, batches)
w_ = w.detach().requires_grad_(True)
(full,) = torch.autograd.grad(((X @ w_ - Y) ** 2).mean(), w_)
assert isinstance(g, torch.Tensor), 'Return w.grad'
assert torch.allclose(g, full, atol=1e-5), f'Expected the full-batch gradient {full}, got {g}. Cleared the stale grad? Divided by the number of micro-batches?'
assert torch.allclose(accumulate_grads(w, batches), full, atol=1e-5), 'Calling twice should give the same gradient'
""",
        },
        {
            "name": "accumulate_grads: one backward per micro-batch",
            "code": """
import torch
w = torch.randn(2, requires_grad=True)
calls = []
w.register_hook(lambda g: calls.append(1))
batches = [(torch.randn(3, 2), torch.randn(3)) for _ in range(4)]
accumulate_grads(w, batches)
assert len(calls) == 4, f'Expected one backward pass per micro-batch (4), saw {len(calls)}'
""",
        },
        {
            "name": "straight_through_round",
            "code": """
import torch
x = torch.tensor([0.2, 1.7, -2.6, 3.0], requires_grad=True)
out = straight_through_round(x)
assert isinstance(out, torch.Tensor) and torch.equal(out.detach(), torch.round(x.detach())), f'Forward values must be round(x), got {out}'
(out * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum().backward()
assert x.grad is not None, 'The gradient must reach x'
assert torch.equal(x.grad, torch.tensor([1.0, 2.0, 3.0, 4.0])), f'Gradient should pass straight through, got {x.grad}'
""",
        },
    ],
}
