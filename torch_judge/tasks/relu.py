"""ReLU implementation task."""

TASK = {
    "title": "Implement ReLU",
    "difficulty": "Easy",
    "function_name": "relu",
    "hint": "ReLU(x) = max(0, x). Think about element-wise comparison with zero.",
    "tests": [
        {
            "name": "Basic values",
            "code": """
import torch
x = torch.tensor([-2., -1., 0., 1., 2.])
out = {fn}(x)
expected = torch.tensor([0., 0., 0., 1., 2.])
assert out.shape == expected.shape, f'Shape mismatch: {out.shape} vs {expected.shape}'
assert torch.allclose(out, expected), f'Wrong Answer: {out} vs {expected}'
""",
        },
        {
            "name": "2-D tensor",
            "code": """
import torch
x = torch.randn(4, 8)
out = {fn}(x)
assert out.shape == x.shape, f'Shape mismatch on 2-D input'
assert (out >= 0).all(), 'ReLU output must be non-negative'
assert torch.allclose(out, x.clamp(min=0)), 'Value mismatch on random input'
""",
        },
        {
            "name": "Gradient check",
            "code": """
import torch
x = torch.tensor([-1., 0., 1., 2.], requires_grad=True)
out = {fn}(x)
out.sum().backward()
assert x.grad is not None, 'Gradient not computed'
assert x.grad[0] == 0., f'grad at x=-1 should be 0, got {x.grad[0]}'
assert x.grad[2] == 1., f'grad at x=1 should be 1, got {x.grad[2]}'
assert x.grad[3] == 1., f'grad at x=2 should be 1, got {x.grad[3]}'
assert x.grad[1] in (0., 1.), f'grad at x=0 should be 0 or 1, got {x.grad[1]}'
""",
        },
        {
            "name": "Performance",
            "code": """
import torch, time
big = torch.randn(1024, 1024)
t0 = time.perf_counter()
for _ in range(100):
    out = {fn}(big)
elapsed = time.perf_counter() - t0
assert isinstance(out, torch.Tensor) and out.shape == big.shape, 'Implement relu first: it must return a tensor of the same shape'
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s for 100 iterations'
""",
        },
    ],
}
