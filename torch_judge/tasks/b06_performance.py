"""Bonus 6: Memory & performance (multi-exercise task)."""

_DISPATCH = """
import torch
from torch.utils._python_dispatch import TorchDispatchMode
class _Trace(TorchDispatchMode):
    def __init__(self): super().__init__(); self.n = 0; self.shapes = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.n += 1
        if isinstance(out, torch.Tensor): self.shapes.append(tuple(out.shape))
        return out
"""

TASK = {
    "title": "Bonus 6: Memory & Performance",
    "difficulty": "Bonus",
    "order": 6,
    "function_name": "is_view_of",
    "function_names": [
        "is_view_of",
        "moving_average",
        "chunked_attention",
        "checkpointed_mlp",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `is_view_of`: compare `a.untyped_storage().data_ptr()` with `b`'s (plain `data_ptr()` differs for offset slices).\n"
        "- `moving_average`: `c = cat([0, x.cumsum(-1)])`; the window sums are `c[..., k:] - c[..., :-k]`.\n"
        "- `chunked_attention`: loop over query blocks `Q[:, s:s+chunk]`; scores `(B, chunk, S)`, softmax, `@ V`; `torch.cat` the blocks.\n"
        "- `checkpointed_mlp`: `h = checkpoint(block, h, use_reentrant=False)` for each block."
    ),
    "tests": [
        {
            "name": "is_view_of",
            "code": """
import torch
x = torch.arange(24.).view(4, 6)
views = {'slice': x[1:3], 'column': x[:, 2], 'transpose': x.T, 'expand': x[:1].expand(5, 6), 'detach': x.detach(), 'unfold': x.unfold(1, 2, 2)}
copies = {'clone': x.clone(), 'fancy index': x[[0, 2]], 'mask': x[x > 3], 'reshape of transpose': x.T.reshape(24), 'contiguous of transpose': x.T.contiguous(), 'arithmetic': x + 0}
for name, t in views.items():
    r = is_view_of(t, x)
    assert type(r) is bool, f'Return a Python bool, got {type(r).__name__}'
    assert r is True, f'{name} is a view of x: expected True'
for name, t in copies.items():
    assert is_view_of(t, x) is False, f'{name} is a copy: expected False'
""",
        },
        {
            "name": "moving_average: values",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(3, 50, dtype=torch.float64)
for k in (1, 4, 50):
    out = moving_average(x, k)
    assert isinstance(out, torch.Tensor) and out.shape == (3, 51 - k), f'k={k}: expected shape (3, {51 - k}), got {getattr(out, "shape", out)}'
    ref = torch.stack([x[:, i:i + k].mean(-1) for i in range(51 - k)], dim=-1)
    assert torch.allclose(out, ref, atol=1e-9), f'k={k}: wrong values'
""",
        },
        {
            "name": "moving_average: vectorized",
            "code": _DISPATCH + """
out = moving_average(torch.randn(1000), 10)
assert isinstance(out, torch.Tensor) and out.shape == (991,), 'moving_average must return the window means'
def ops(N):
    with _Trace() as t:
        moving_average(torch.randn(N), 5)
    return t.n
n_small, n_big = ops(100), ops(5000)
assert n_big <= n_small + 5, f'{n_small} ops for N=100 but {n_big} for N=5000: looks like a Python loop over positions'
""",
        },
        {
            "name": "chunked_attention: matches attention, any chunk size",
            "code": """
import torch, math
torch.manual_seed(0)
Q, K, V = torch.randn(2, 13, 8), torch.randn(2, 13, 8), torch.randn(2, 13, 8)
ref = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(8), dim=-1) @ V
for chunk in (1, 4, 13, 50):
    out = chunked_attention(Q, K, V, chunk)
    assert isinstance(out, torch.Tensor) and out.shape == (2, 13, 8), f'chunk={chunk}: expected (2, 13, 8), got {getattr(out, "shape", out)}'
    assert torch.allclose(out, ref, atol=1e-5), f'chunk={chunk}: differs from full attention (does the last, smaller chunk work?)'
""",
        },
        {
            "name": "chunked_attention: never builds the S x S scores",
            "code": _DISPATCH + """
S = 64
Q, K, V = torch.randn(1, S, 8), torch.randn(1, S, 8), torch.randn(1, S, 8)
with _Trace() as t:
    out = chunked_attention(Q, K, V, 16)
assert isinstance(out, torch.Tensor) and out.shape == (1, S, 8), 'Return the attention output'
big = [s for s in t.shapes if len(s) >= 2 and s[-2:] == (S, S)]
assert not big, f'Built a ({S}, {S}) score matrix; the largest should be (B, chunk, S). Shapes seen: {sorted(set(t.shapes))[:8]}'
""",
        },
        {
            "name": "checkpointed_mlp: same values and gradients, recomputes in backward",
            "code": """
import torch, copy
from torch import nn
torch.manual_seed(0)
blocks = [nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)) for _ in range(3)]
plain = copy.deepcopy(blocks)
calls = [0] * 3
for i, b in enumerate(blocks):
    b[0].register_forward_hook(lambda *a, i=i: calls.__setitem__(i, calls[i] + 1))
x = torch.randn(4, 8, requires_grad=True)
x2 = x.detach().clone().requires_grad_(True)
out = checkpointed_mlp(x, blocks)
ref = x2
for b in plain:
    ref = b(ref)
assert isinstance(out, torch.Tensor) and torch.allclose(out, ref, atol=1e-6), 'Output must equal running the blocks normally'
out.pow(2).sum().backward(); ref.pow(2).sum().backward()
assert torch.allclose(x.grad, x2.grad, atol=1e-5), 'Input gradients differ'
assert torch.allclose(blocks[0][0].weight.grad, plain[0][0].weight.grad, atol=1e-5), 'Parameter gradients differ'
assert calls == [2, 2, 2], f'Each block should run twice (forward, then recomputed in backward); counts: {calls}. Wrap every block in checkpoint(...)'
""",
        },
    ],
}
