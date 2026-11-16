"""Bonus 3: Mixed precision & devices (multi-exercise task)."""

TASK = {
    "title": "Bonus 3: Mixed Precision & Devices",
    "difficulty": "Bonus",
    "order": 3,
    "function_name": "count_overflow",
    "function_names": [
        "count_overflow",
        "softmax_lowp",
        "unscale_and_check",
        "to_device_like",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `count_overflow`: `torch.isfinite(x) & torch.isinf(x.to(dtype))`, summed, as an `int`.\n"
        "- `softmax_lowp`: `xf = x.float()`, a stable softmax on `xf`, then `.to(x.dtype)`.\n"
        "- `unscale_and_check`: `p.grad.div_(scale)` for each grad that exists; return `True` if any is not finite.\n"
        "- `to_device_like`: recurse into dicts / lists / tuples (rebuild with `type(obj)(...)`); cast only `is_floating_point()` tensors."
    ),
    "tests": [
        {
            "name": "count_overflow",
            "code": """
import torch
x = torch.tensor([1.0, 65504.0, 65520.0, -1e5, float('inf'), float('nan'), 3e38])
n = count_overflow(x, torch.float16)
assert type(n) is int, f'Return a Python int, got {type(n).__name__}'
assert n == 3, f'Expected 3 (65520 rounds up, -1e5, 3e38); values already inf/NaN do not count. Got {n}'
assert count_overflow(torch.tensor([1e30, -1e30]), torch.bfloat16) == 0, 'bfloat16 has float32\\'s range: 1e30 does not overflow'
assert count_overflow(torch.tensor([1e30, -1e30]), torch.float16) == 2
""",
        },
        {
            "name": "softmax_lowp: dtype, accuracy, large vocabulary",
            "code": """
import torch
torch.manual_seed(0)
for dt in (torch.float16, torch.bfloat16):
    x = (torch.randn(3, 50) * 8).to(dt)
    p = softmax_lowp(x)
    assert isinstance(p, torch.Tensor) and p.dtype == dt and p.shape == x.shape, f'Return the input dtype {dt} and shape'
    ref = torch.softmax(x.double(), -1)
    assert torch.allclose(p.double(), ref, atol=2e-3), f'{dt}: not accurate enough; compute in float32'
x = torch.zeros(1, 100_000, dtype=torch.float16)
p = softmax_lowp(x)
assert torch.isfinite(p).all() and abs(p[0, 0].item() - 1e-5) < 1e-6, f'Uniform over 100k tokens should give ~1e-5 each, got {p[0, 0].item()}'
assert abs(p.float().sum().item() - 1.0) < 1e-2, 'Probabilities should sum to ~1'
q = softmax_lowp(torch.tensor([[60000.0], [-60000.0], [0.0]], dtype=torch.float16), dim=0)
assert torch.allclose(q.float().flatten(), torch.tensor([1.0, 0.0, 0.0])), f'dim argument / stability: got {q.flatten().tolist()}'
""",
        },
        {
            "name": "unscale_and_check",
            "code": """
import torch
a = torch.zeros(3, requires_grad=True); b = torch.zeros(2, requires_grad=True); c = torch.zeros(1, requires_grad=True)
(a * torch.tensor([1.0, 2.0, 3.0]) * 512 + 0).sum().backward()
b.grad = torch.tensor([512.0, -1024.0])
ptr = a.grad.data_ptr()
found = unscale_and_check([a, b, c], 512.0)
assert type(found) is bool and found is False, f'All grads finite: expected False (a bool), got {found!r}'
assert a.grad.data_ptr() == ptr, 'Unscale the gradients in place'
assert torch.allclose(a.grad, torch.tensor([1.0, 2.0, 3.0])) and torch.allclose(b.grad, torch.tensor([1.0, -2.0])), 'Gradients should be divided by the scale'
b.grad[1] = float('nan')
assert unscale_and_check([a, b, c], 2.0) is True, 'A NaN gradient means the step must be skipped'
a.grad[0] = float('inf')
assert unscale_and_check([a], 2.0) is True, 'An inf gradient means the step must be skipped'
""",
        },
        {
            "name": "to_device_like: structure and dtypes",
            "code": """
import torch
ids = torch.tensor([[1, 2, 3]]); feats = torch.randn(2, dtype=torch.float32)
batch = {'ids': ids, 'mask': torch.ones(1, 3, dtype=torch.bool), 'feats': [feats, (torch.randn(1), 'meta', None)], 'n': 3}
out = to_device_like(batch, torch.empty(0), float_dtype=torch.bfloat16)
assert isinstance(out, dict) and out is not batch and set(out) == set(batch), 'Return a new dict with the same keys'
assert out['ids'].dtype == torch.long and out['mask'].dtype == torch.bool, 'Integer and bool tensors must keep their dtype'
assert out['feats'][0].dtype == torch.bfloat16 and out['feats'][1][0].dtype == torch.bfloat16, 'Floating tensors should be cast to float_dtype'
assert isinstance(out['feats'], list) and isinstance(out['feats'][1], tuple), 'Lists stay lists and tuples stay tuples'
assert out['feats'][1][1] == 'meta' and out['feats'][1][2] is None and out['n'] == 3, 'Non-tensors are returned unchanged'
assert feats.dtype == torch.float32 and batch['feats'][0] is feats, 'Do not modify the input structure'
same = to_device_like([feats, ids], torch.empty(0))
assert same[0].dtype == torch.float32 and same[1].dtype == torch.long, 'Without float_dtype, dtypes are unchanged'
""",
        },
    ],
}
