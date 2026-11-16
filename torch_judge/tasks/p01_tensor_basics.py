"""Prerequisite 1: Tensor basics (multi-exercise task)."""

TASK = {
    "title": "Prereq 1: Tensor Basics",
    "difficulty": "Prereq",
    "order": 1,
    "function_name": "positive_mask",
    "function_names": [
        "positive_mask",
        "running_max_init",
        "scale_inplace_",
        "quantize_int8",
        "fan_in_out",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `positive_mask`: `x > 0` is a bool tensor; convert it with `.to(x.dtype)` (not `.float()`, which always gives float32).\n"
        "- `running_max_init`: `scores.new_full((B, S, 1), float('-inf'))` inherits dtype and device.\n"
        "- `scale_inplace_`: use an in-place method (`mul_`) and return `x`. `x = x * s` makes a new tensor.\n"
        "- `quantize_int8`: `torch.round(w / scale)`, then `.clamp(-128, 127)`, then `.to(torch.int8)`. Cast last.\n"
        "- `fan_in_out`: the receptive field is `weight[0][0].numel()` (1 for Linear); multiply it by `shape[1]` and `shape[0]`."
    ),
    "tests": [
        {
            "name": "positive_mask: values",
            "code": """
import torch
x = torch.tensor([-2.0, -0.0, 0.0, 1e-6, 3.0])
out = positive_mask(x)
assert isinstance(out, torch.Tensor), f'Expected a tensor, got {type(out)}'
assert torch.equal(out, torch.tensor([0., 0., 0., 1., 1.])), f'Got {out}'
x = torch.randn(4, 5)
assert torch.equal(positive_mask(x), (x > 0).float()), 'Wrong values on a 2-D input'
""",
        },
        {
            "name": "positive_mask: keeps the input dtype",
            "code": """
import torch
for dtype in (torch.float64, torch.float16):
    out = positive_mask(torch.tensor([-1.0, 2.0], dtype=dtype))
    assert out.dtype == dtype, f'For a {dtype} input the mask should be {dtype}, got {out.dtype}. Use .to(x.dtype)'
""",
        },
        {
            "name": "running_max_init: shape, value, dtype",
            "code": """
import torch
scores = torch.randn(2, 5, 7, dtype=torch.float64)
out = running_max_init(scores)
assert isinstance(out, torch.Tensor), f'Expected a tensor, got {type(out)}'
assert out.shape == (2, 5, 1), f'Shape mismatch: {out.shape}, expected (B, S, 1) = (2, 5, 1)'
assert out.dtype == torch.float64, f'dtype should match scores (float64), got {out.dtype}'
assert out.device == scores.device, 'device should match scores'
assert torch.isinf(out).all() and (out < 0).all(), 'Every entry must be -inf'
""",
        },
        {
            "name": "scale_inplace_: modifies and returns the same tensor",
            "code": """
import torch
t = torch.arange(6.).view(2, 3)
ptr = t.data_ptr()
out = scale_inplace_(t, 0.5)
assert out is t, 'Return x itself, not a new tensor'
assert t.data_ptr() == ptr, 'x must be modified in place'
assert torch.equal(t, torch.arange(6.).view(2, 3) * 0.5), f'Caller tensor not scaled: {t}'
""",
        },
        {
            "name": "scale_inplace_: works on a view",
            "code": """
import torch
m = torch.ones(3, 4)
scale_inplace_(m[1], 3.0)
expected = torch.ones(3, 4); expected[1] = 3.0
assert torch.equal(m, expected), 'Scaling the view m[1] in place should change row 1 of m'
""",
        },
        {
            "name": "quantize_int8: round, clamp, cast",
            "code": """
import torch
w = torch.tensor([0.04, -0.16, 1.0, 50.0, -50.0, 12.8, -12.9])
out = quantize_int8(w, 0.1)
assert isinstance(out, torch.Tensor), f'Expected a tensor, got {type(out)}'
assert out.dtype == torch.int8, f'Expected torch.int8, got {out.dtype}'
expected = torch.tensor([0, -2, 10, 127, -128, 127, -128], dtype=torch.int8)
assert torch.equal(out, expected), f'Got {out.tolist()}, expected {expected.tolist()}. Clamp before casting?'
""",
        },
        {
            "name": "quantize_int8: per-row tensor scale",
            "code": """
import torch
w = torch.tensor([[1.0, -2.0], [30.0, 60.0]])
scale = torch.tensor([[0.5], [0.5]])
out = quantize_int8(w, scale)
assert out.shape == (2, 2) and out.dtype == torch.int8, f'Got shape {out.shape}, dtype {out.dtype}'
assert torch.equal(out, torch.tensor([[2, -4], [60, 120]], dtype=torch.int8)), f'Got {out.tolist()}'
""",
        },
        {
            "name": "fan_in_out: Linear and conv weights",
            "code": """
import torch
res = fan_in_out(torch.empty(64, 32))
assert res == (32, 64), f'Linear (64, 32): expected (32, 64), got {res}'
res = fan_in_out(torch.empty(16, 3, 5, 5))
assert res == (75, 400), f'Conv (16, 3, 5, 5): expected (fan_in=3*5*5, fan_out=16*5*5) = (75, 400), got {res}'
assert all(type(v) is int for v in res), f'Return Python ints, got {[type(v).__name__ for v in res]}'
""",
        },
    ],
}
