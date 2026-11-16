"""Prerequisite 8: nn.Module anatomy (multi-exercise task)."""

TASK = {
    "title": "Prereq 8: nn.Module Anatomy",
    "difficulty": "Prereq",
    "order": 8,
    "function_name": "ScaleShift",
    "function_names": [
        "ScaleShift",
        "RunningMean",
        "MLPStack",
        "freeze_",
        "num_trainable",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `ScaleShift`: `self.gamma = nn.Parameter(torch.ones(dim))` and `self.beta = nn.Parameter(torch.zeros(dim))`.\n"
        "- `RunningMean`: `self.register_buffer('running_mean', torch.zeros(dim))`; branch on `self.training`; update with `mul_` / `add_` inside `torch.no_grad()`.\n"
        "- `MLPStack`: collect `Linear` and `ReLU` layers in a list, then wrap it in `nn.Sequential(*layers)`. A plain Python list is not registered.\n"
        "- `freeze_`: `p.requires_grad_(False)` for `p in module.parameters()` (this includes submodules), then return the module.\n"
        "- `num_trainable`: `sum(p.numel() for p in module.parameters() if p.requires_grad)`."
    ),
    "tests": [
        {
            "name": "ScaleShift: registered parameters",
            "code": """
import torch
from torch import nn
m = ScaleShift(4)
assert isinstance(m, nn.Module), 'ScaleShift must subclass nn.Module'
names = sorted(n for n, _ in m.named_parameters())
assert names == ['beta', 'gamma'], f'Expected parameters named gamma and beta, found {names}'
params = dict(m.named_parameters())
assert torch.equal(params['gamma'].detach(), torch.ones(4)) and torch.equal(params['beta'].detach(), torch.zeros(4)), 'gamma starts at ones, beta at zeros'
""",
        },
        {
            "name": "ScaleShift: forward and gradients",
            "code": """
import torch
m = ScaleShift(3)
with torch.no_grad():
    m.gamma.copy_(torch.tensor([1.0, 2.0, 3.0])); m.beta.copy_(torch.tensor([0.5, 0.0, -1.0]))
x = torch.ones(2, 5, 3)
out = m(x)
assert isinstance(out, torch.Tensor) and out.shape == (2, 5, 3), f'Expected shape (2, 5, 3), got {getattr(out, "shape", out)}'
assert torch.allclose(out[0, 0], torch.tensor([1.5, 2.0, 2.0])), f'Expected x * gamma + beta, got {out[0, 0]}'
out.sum().backward()
assert m.gamma.grad is not None and m.beta.grad is not None, 'Gradients must reach gamma and beta'
""",
        },
        {
            "name": "RunningMean: buffer, not parameter",
            "code": """
import torch
from torch import nn
m = RunningMean(3)
assert isinstance(m, nn.Module), 'RunningMean must subclass nn.Module'
assert [n for n, _ in m.named_buffers()] == ['running_mean'], f'Expected one buffer named running_mean, found {[n for n, _ in m.named_buffers()]}'
assert len(list(m.parameters())) == 0, 'running_mean must be a buffer, not a parameter'
assert 'running_mean' in m.state_dict(), 'Buffers belong in state_dict()'
m = m.to(torch.float64)
assert m.running_mean.dtype == torch.float64, '.to() should convert registered buffers'
""",
        },
        {
            "name": "RunningMean: train vs eval",
            "code": """
import torch
m = RunningMean(2, momentum=0.5)
x = torch.tensor([[1.0, 10.0], [3.0, 30.0]], requires_grad=True)
out = m(x)
assert isinstance(out, torch.Tensor), 'forward must return a tensor'
assert torch.allclose(out, torch.tensor([[-1.0, -10.0], [1.0, 10.0]])), f'Training: return x - batch_mean, got {out}'
assert torch.allclose(m.running_mean, torch.tensor([1.0, 10.0])), f'Expected running_mean [1, 10], got {m.running_mean}'
assert m.running_mean.grad_fn is None and not m.running_mean.requires_grad, 'running_mean must not be part of the graph'
m.eval()
out = m(x)
assert torch.allclose(out, x.detach() - torch.tensor([1.0, 10.0])), 'Eval: return x - running_mean'
assert torch.allclose(m.running_mean, torch.tensor([1.0, 10.0])), 'Eval must not update running_mean'
m2 = RunningMean(2); m2(torch.tensor([[10.0, 20.0]]))
assert torch.allclose(m2.running_mean, torch.tensor([1.0, 2.0])), 'Default momentum should be 0.1'
""",
        },
        {
            "name": "MLPStack: registered layers",
            "code": """
import torch
from torch import nn
m = MLPStack([16, 32, 32, 4])
assert isinstance(m, nn.Module), 'MLPStack must subclass nn.Module'
n = sum(p.numel() for p in m.parameters())
expected = 16*32 + 32 + 32*32 + 32 + 32*4 + 4
assert n == expected, f'Expected {expected} registered parameter elements, found {n}. Layers in a plain Python list are not registered'
assert len(m.state_dict()) == 6, f'Expected 6 tensors in state_dict (3 weights + 3 biases), found {len(m.state_dict())}'
""",
        },
        {
            "name": "MLPStack: ReLU between layers only",
            "code": """
import torch
from torch import nn
torch.manual_seed(0)
m = MLPStack([8, 16, 16, 3])
linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
assert len(linears) == 3, f'Expected 3 nn.Linear layers, found {len(linears)}'
x = torch.randn(64, 8)
h = x
for k, lin in enumerate(linears):
    h = lin(h)
    if k < len(linears) - 1:
        h = torch.relu(h)
out = m(x)
assert isinstance(out, torch.Tensor) and out.shape == (64, 3), f'Expected shape (64, 3), got {getattr(out, "shape", out)}'
assert torch.allclose(out, h, atol=1e-6), 'Expected Linear -> ReLU -> Linear -> ReLU -> Linear (no ReLU after the last layer)'
""",
        },
        {
            "name": "freeze_",
            "code": """
import torch
from torch import nn
m = nn.Sequential(nn.Linear(4, 8), nn.Sequential(nn.ReLU(), nn.Linear(8, 2)))
out = freeze_(m)
assert out is m, 'Return the same module object'
assert all(not p.requires_grad for p in m.parameters()), 'Every parameter, including nested ones, must be frozen'
assert not m(torch.randn(3, 4)).requires_grad, 'A fully frozen model should produce outputs that need no grad'
""",
        },
        {
            "name": "num_trainable",
            "code": """
import torch
from torch import nn
lin = nn.Linear(10, 5)
n = num_trainable(lin)
assert type(n) is int, f'Return a Python int, got {type(n).__name__}'
assert n == 55, f'nn.Linear(10, 5) has 55 trainable elements, got {n}'
lin.weight.requires_grad_(False)
assert num_trainable(lin) == 5, f'After freezing the weight only the 5 bias elements remain, got {num_trainable(lin)}'
assert num_trainable(nn.ReLU()) == 0, 'A module without parameters has 0 trainable elements'
""",
        },
    ],
}
