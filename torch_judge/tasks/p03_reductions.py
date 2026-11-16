"""Prerequisite 3: Reductions along dims (multi-exercise task)."""

TASK = {
    "title": "Prereq 3: Reductions Along Dims",
    "difficulty": "Prereq",
    "order": 3,
    "function_name": "stable_shift",
    "function_names": [
        "stable_shift",
        "normalize_rows",
        "channel_stats",
        "masked_mean",
        "global_l2_norm",
        "exclusive_cumsum",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `stable_shift`: `x.max(dim=-1, keepdim=True)` returns `(values, indices)`; subtract `.values`.\n"
        "- `normalize_rows`: divide by `x.sum(dim=-1, keepdim=True)`. Without `keepdim` a square input divides by the wrong axis.\n"
        "- `channel_stats`: reduce over `dim=(0, 2, 3)`, and pass `unbiased=False` to `var`.\n"
        "- `masked_mean`: turn the mask into `x.dtype` with shape `(B, S, 1)`, sum `x * m` over dim 1, and divide by `m.sum(dim=1).clamp_min(1)`.\n"
        "- `global_l2_norm`: reduce each tensor to `t.pow(2).sum()`, stack the scalars, sum, then `sqrt`.\n"
        "- `exclusive_cumsum`: `x.cumsum(dim) - x` removes each element from its own running total."
    ),
    "tests": [
        {
            "name": "stable_shift: row max becomes 0",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 4) * 10
out = stable_shift(x)
assert isinstance(out, torch.Tensor) and out.shape == x.shape, f'Expected a tensor of shape {tuple(x.shape)}'
assert torch.allclose(out.amax(dim=-1), torch.zeros(4)), 'Each ROW max should be 0. Reducing over the wrong dim?'
assert torch.allclose(out, x - x.amax(dim=-1, keepdim=True)), 'Wrong values'
""",
        },
        {
            "name": "stable_shift: 3-D input, huge values",
            "code": """
import torch
x = torch.tensor([[[1000., 1001., 999.]], [[-5., -7., -6.]]])
out = stable_shift(x)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
assert torch.allclose(out, torch.tensor([[[-1., 0., -2.]], [[0., -2., -1.]]])), f'Got {out}'
assert torch.isfinite(torch.exp(out)).all(), 'exp of the shifted values should not overflow'
""",
        },
        {
            "name": "normalize_rows: square input",
            "code": """
import torch
x = torch.tensor([[1., 1., 2.], [3., 1., 0.], [1., 1., 1.]])
out = normalize_rows(x)
assert isinstance(out, torch.Tensor) and out.shape == x.shape, f'Expected a tensor of shape {tuple(x.shape)}'
assert torch.allclose(out.sum(dim=-1), torch.ones(3)), f'Rows must sum to 1, got {out.sum(dim=-1)}'
assert torch.allclose(out[0], torch.tensor([0.25, 0.25, 0.5])), f'Row 0 wrong: {out[0]}'
""",
        },
        {
            "name": "normalize_rows: batched input and gradient",
            "code": """
import torch
x = (torch.rand(2, 3, 5) + 0.1).requires_grad_()
out = normalize_rows(x)
assert out.shape == (2, 3, 5), f'Shape mismatch: {out.shape}'
assert torch.allclose(out.sum(dim=-1), torch.ones(2, 3), atol=1e-6), 'Every row (last dim) must sum to 1'
out[..., 0].sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all(), 'Gradient should flow'
""",
        },
        {
            "name": "channel_stats: per-channel, biased",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 3, 2, 2)
res = channel_stats(x)
assert isinstance(res, tuple) and len(res) == 2, 'Return a tuple (mean, var)'
mean, var = res
assert mean.shape == (3,) and var.shape == (3,), f'Expected shapes (C,) = (3,), got {mean.shape} and {var.shape}'
assert torch.allclose(mean, x.mean(dim=(0, 2, 3)), atol=1e-6), 'Wrong mean: reduce over batch AND spatial dims'
if torch.allclose(var, x.var(dim=(0, 2, 3), unbiased=True), atol=1e-6):
    raise AssertionError('That is the unbiased variance (divides by N-1). Use unbiased=False')
assert torch.allclose(var, x.var(dim=(0, 2, 3), unbiased=False), atol=1e-6), 'Wrong variance'
""",
        },
        {
            "name": "masked_mean: bool and float masks",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(3, 5, 4)
mask_b = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0]], dtype=torch.bool)
expected = torch.stack([x[b][mask_b[b]].mean(dim=0) for b in range(3)])
for mask in (mask_b, mask_b.float()):
    out = masked_mean(x, mask)
    assert isinstance(out, torch.Tensor) and out.shape == (3, 4), f'Expected shape (B, D) = (3, 4), got {getattr(out, "shape", out)}'
    assert torch.allclose(out, expected, atol=1e-6), f'Wrong values for a {mask.dtype} mask'
""",
        },
        {
            "name": "masked_mean: fully masked sequence gives zeros",
            "code": """
import torch
x = torch.randn(2, 3, 4, requires_grad=True)
mask = torch.tensor([[True, False, True], [False, False, False]])
out = masked_mean(x, mask)
assert torch.isfinite(out).all(), f'No NaN/inf allowed, got {out}'
assert torch.equal(out[1], torch.zeros(4)), f'A sequence with no valid tokens should give zeros, got {out[1]}'
out.sum().backward()
assert torch.isfinite(x.grad).all() and torch.equal(x.grad[0, 1], torch.zeros(4)), 'Masked tokens must get zero gradient'
""",
        },
        {
            "name": "global_l2_norm",
            "code": """
import torch
torch.manual_seed(0)
ts = [torch.randn(3, 4), torch.randn(7), torch.randn(2, 2, 2)]
out = global_l2_norm(ts)
assert isinstance(out, torch.Tensor), f'Return a tensor, got {type(out)}'
assert out.ndim == 0, f'Return a 0-d tensor, got shape {tuple(out.shape)}'
expected = torch.cat([t.flatten() for t in ts]).norm()
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected.item():.5f}, got {out.item():.5f}'
assert torch.allclose(global_l2_norm([torch.tensor([3., 0.]), torch.tensor([[0., 4.]])]), torch.tensor(5.)), '3-4-5 check failed'
""",
        },
        {
            "name": "exclusive_cumsum: values and dims",
            "code": """
import torch
x = torch.tensor([[1., 2., 3., 4.], [0.5, 0.25, 0.25, 0.]])
out = exclusive_cumsum(x)
assert isinstance(out, torch.Tensor) and out.shape == x.shape, f'Expected shape {tuple(x.shape)}'
expected = torch.tensor([[0., 1., 3., 6.], [0., 0.5, 0.75, 1.]])
if torch.allclose(out, x.cumsum(-1)):
    raise AssertionError('That is the inclusive cumsum: exclude the element itself')
assert torch.allclose(out, expected), f'Got {out}'
y = torch.randn(4, 3, 5)
assert torch.allclose(exclusive_cumsum(y, dim=1), y.cumsum(1) - y, atol=1e-6), 'dim argument not respected'
assert torch.equal(exclusive_cumsum(y, dim=1)[:, 0], torch.zeros(4, 5)), 'The first position must be 0'
""",
        },
        {
            "name": "exclusive_cumsum: top-p keep mask and gradient",
            "code": """
import torch
p = torch.tensor([0.5, 0.3, 0.15, 0.05], requires_grad=True)
keep = exclusive_cumsum(p, dim=0) < 0.7
assert keep.tolist() == [True, True, False, False], f'Top-p (p=0.7) should keep the first two tokens, got {keep.tolist()}'
exclusive_cumsum(p, dim=0).sum().backward()
assert torch.allclose(p.grad, torch.tensor([3., 2., 1., 0.])), f'Gradient should count later positions, got {p.grad}'
""",
        },
        {
            "name": "Inputs are not modified",
            "code": """
import torch
def _changed(fn, *args):
    before = [a.clone() if isinstance(a, torch.Tensor) else [t.clone() for t in a] if isinstance(a, list) else a for a in args]
    out = fn(*args)
    assert out is not None, f"{getattr(fn, '__name__', 'function')} returned None: implement it first"
    for i, (a, b) in enumerate(zip(args, before)):
        same = torch.equal(a, b) if isinstance(a, torch.Tensor) else all(torch.equal(x, y) for x, y in zip(a, b)) if isinstance(a, list) else True
        if not same:
            return i
    return None
def _check(name, fn, *args):
    i = _changed(fn, *args)
    assert i is None, f'{name} modified its input #{i + 1}: build and return a new tensor instead of writing into the input'

_check('stable_shift', stable_shift, torch.randn(3, 5))
_check('normalize_rows', normalize_rows, torch.rand(3, 5) + 0.1)
_check('channel_stats', channel_stats, torch.randn(2, 3, 4, 4))
_check('masked_mean', masked_mean, torch.randn(2, 4, 3), torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=torch.bool))
_check('global_l2_norm', global_l2_norm, [torch.randn(3), torch.randn(2, 2)])
_check('exclusive_cumsum', exclusive_cumsum, torch.randn(3, 6))
""",
        },
    ],
}
