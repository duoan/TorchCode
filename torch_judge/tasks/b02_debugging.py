"""Bonus 2: Debugging & testing tensors (multi-exercise task)."""

TASK = {
    "title": "Bonus 2: Debugging & Testing Tensors",
    "difficulty": "Bonus",
    "order": 2,
    "function_name": "first_nan_module",
    "function_names": [
        "first_nan_module",
        "params_without_grad",
        "diff_report",
        "grad_norms_by_name",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `first_nan_module`: hook every module with no children; in the hook record the name if `torch.isfinite(out).all()` is False and nothing was recorded yet; remove hooks in a `finally`.\n"
        "- `params_without_grad`: `[n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]`.\n"
        "- `diff_report`: `diff = (a - b).abs()` with NaN replaced by `inf`; `argmax` + `torch.unravel_index` for the index; convert everything with `.item()` / `int`.\n"
        "- `grad_norms_by_name`: `{n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}`."
    ),
    "tests": [
        {
            "name": "first_nan_module: finds the first bad layer, nested names",
            "code": """
import torch
from torch import nn
class Log(nn.Module):
    def forward(self, x): return torch.log(x)
class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(4, 4); self.act = nn.ReLU(); self.log = Log()
    def forward(self, x): return self.log(self.act(self.fc(x)))
model = nn.Sequential(nn.Linear(4, 4), Block(), Log(), nn.Linear(4, 2))
torch.manual_seed(0)
name = first_nan_module(model, torch.randn(16, 4))
assert name == '1.log', f'Expected the first non-finite module to be "1.log" (nested names from named_modules), got {name!r}'
assert first_nan_module(nn.Sequential(nn.Linear(4, 3), nn.Tanh()), torch.randn(5, 4)) is None, 'All-finite model should return None'
""",
        },
        {
            "name": "first_nan_module: removes its hooks, even on errors",
            "code": """
import torch
from torch import nn
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
calls = []
model[2].register_forward_hook(lambda *a: calls.append(1))
first_nan_module(model, torch.randn(3, 4))
assert calls, 'first_nan_module must run the model forward'
model[2]._forward_hooks.clear()
assert all(len(m._forward_hooks) == 0 for m in model.modules()), 'Remove every hook before returning'
class Boom(nn.Module):
    def forward(self, x): raise ValueError('boom')
bad = nn.Sequential(nn.Linear(4, 4), Boom())
raised = False
try:
    first_nan_module(bad, torch.randn(3, 4))
except ValueError:
    raised = True
assert raised, 'Let errors from the forward pass propagate (do not swallow them)'
assert all(len(m._forward_hooks) == 0 for m in bad.modules()), 'Hooks must be removed even if the forward pass raises (use try/finally)'
""",
        },
        {
            "name": "params_without_grad",
            "code": """
import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4); self.frozen.requires_grad_(False)
        self.used = nn.Linear(4, 1); self.unused = nn.Linear(4, 1); self.extra = nn.Parameter(torch.ones(2))
    def forward(self, x): return self.used(self.frozen(x))
m = Net()
m(torch.randn(3, 4)).sum().backward()
got = params_without_grad(m)
assert isinstance(got, list), f'Return a list, got {type(got).__name__}'
assert got == ['extra', 'unused.weight', 'unused.bias'], f"Expected ['extra', 'unused.weight', 'unused.bias'] (named_parameters order: own parameters first; frozen params excluded), got {got}"
""",
        },
        {
            "name": "diff_report: values, index, mismatches",
            "code": """
import torch
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = a.clone(); b[1, 2] = 6.5; b[0, 0] = 1.0 + 1e-9
r = diff_report(a, b)
assert isinstance(r, dict), 'Return a dict'
assert abs(r['max_abs_diff'] - 0.5) < 1e-6 and type(r['max_abs_diff']) is float, f'max_abs_diff should be the float 0.5, got {r.get("max_abs_diff")!r}'
assert r['index'] == (1, 2) and all(type(i) is int for i in r['index']), f'index should be the tuple of ints (1, 2), got {r.get("index")!r}'
assert r['a'] == 6.0 and r['b'] == 6.5, f'Values at the index should be 6.0 and 6.5, got {r.get("a")}, {r.get("b")}'
assert r['num_mismatched'] == 1 and type(r['num_mismatched']) is int, f'Exactly 1 element is outside the tolerance, got {r.get("num_mismatched")!r}'
""",
        },
        {
            "name": "diff_report: NaN counts as an infinite difference",
            "code": """
import torch
a = torch.tensor([1.0, float('nan'), 3.0]); b = torch.tensor([1.5, 2.0, 3.0])
r = diff_report(a, b)
assert r['max_abs_diff'] == float('inf') and r['index'] == (1,), f'A NaN should be reported as the largest (infinite) difference, at index (1,); got {r}'
assert r['num_mismatched'] == 2, f'Both the NaN and the 0.5 difference are mismatches, got {r["num_mismatched"]}'
""",
        },
        {
            "name": "grad_norms_by_name",
            "code": """
import torch
from torch import nn
torch.manual_seed(0)
m = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
m[0].bias.requires_grad_(False)
m(torch.randn(6, 4)).pow(2).sum().backward()
got = grad_norms_by_name(m)
assert isinstance(got, dict) and list(got) == ['0.weight', '2.weight', '2.bias'], f'Expected keys 0.weight, 2.weight, 2.bias (skip params with no grad), got {list(got) if isinstance(got, dict) else got}'
for n, p in m.named_parameters():
    if p.grad is not None:
        assert type(got[n]) is float, f'Norms must be Python floats (use .item()), got {type(got[n]).__name__} for {n}'
        assert abs(got[n] - p.grad.norm().item()) < 1e-6, f'Wrong norm for {n}'
""",
        },
    ],
}
