"""Prerequisite 2: Broadcasting (multi-exercise task)."""

TASK = {
    "title": "Prereq 2: Broadcasting",
    "difficulty": "Prereq",
    "order": 2,
    "function_name": "add_channel_bias",
    "function_names": [
        "add_channel_bias",
        "scale_rows",
        "abs_distance_matrix",
        "outer_angles",
        "pairwise_sq_dists",
    ],
    "hint": (
        "Line up the dims you mean by inserting size-1 axes: b.view(1, -1, 1, 1), s[:, None], "
        "idx[:, None] - idx[None, :]. For pairwise distances build an (N, M, D) difference with "
        "a[:, None, :] - b[None, :, :] and reduce over the last dim."
    ),
    "tests": [
        {
            "name": "add_channel_bias: values",
            "code": """
import torch
x = torch.randn(2, 3, 4, 5)
b = torch.tensor([1., -2., 3.])
out = add_channel_bias(x, b)
assert out.shape == (2, 3, 4, 5), f'Shape mismatch: {out.shape}'
for c in range(3):
    assert torch.allclose(out[:, c], x[:, c] + b[c]), f'Channel {c} has the wrong bias'
""",
        },
        {
            "name": "add_channel_bias: square images (C == H == W)",
            "code": """
import torch
x = torch.zeros(1, 4, 4, 4)
b = torch.arange(4).float()
out = add_channel_bias(x, b)
assert out.shape == (1, 4, 4, 4), f'Shape mismatch: {out.shape}'
assert torch.equal(out[0, :, 0, 0], b), 'Bias must vary along the channel dim (dim 1), not along W'
assert torch.equal(out[0, 2], torch.full((4, 4), 2.)), 'Bias aligned with the wrong dim'
""",
        },
        {
            "name": "scale_rows: square matrix (N == D)",
            "code": """
import torch
x = torch.ones(3, 3)
s = torch.tensor([1., 2., 3.])
out = scale_rows(x, s)
expected = torch.tensor([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
assert torch.equal(out, expected), f'Rows must be scaled (not columns). Got:\\n{out}'
""",
        },
        {
            "name": "scale_rows: gradient flows",
            "code": """
import torch
x = torch.randn(5, 7, requires_grad=True)
s = torch.randn(5, requires_grad=True)
out = scale_rows(x, s)
assert out.shape == (5, 7), f'Shape mismatch: {out.shape}'
out.sum().backward()
assert torch.allclose(s.grad, x.detach().sum(dim=1), atol=1e-5), 'Wrong gradient w.r.t. s'
""",
        },
        {
            "name": "abs_distance_matrix",
            "code": """
import torch
D = abs_distance_matrix(6)
assert D.shape == (6, 6), f'Shape mismatch: {D.shape}'
assert not D.is_floating_point(), f'Expected an integer tensor, got {D.dtype}'
for i in range(6):
    for j in range(6):
        assert D[i, j].item() == abs(i - j), f'D[{i},{j}] = {D[i, j].item()}, expected {abs(i - j)}'
assert abs_distance_matrix(1).shape == (1, 1), 'n=1 should give a (1, 1) matrix'
""",
        },
        {
            "name": "outer_angles",
            "code": """
import torch
pos = torch.arange(7).float()
freqs = torch.tensor([1.0, 0.5, 0.1])
out = outer_angles(pos, freqs)
assert out.shape == (7, 3), f'Shape mismatch: {out.shape} (expected (S, F) = (7, 3))'
assert torch.allclose(out, torch.outer(pos, freqs)), 'Wrong values'
sq = outer_angles(torch.arange(3).float(), torch.tensor([1., 10., 100.]))
assert torch.allclose(sq[2], torch.tensor([2., 20., 200.])), 'Rows must index positions, columns index freqs'
""",
        },
        {
            "name": "pairwise_sq_dists",
            "code": """
import torch
torch.manual_seed(0)
a = torch.randn(6, 4)
b = torch.randn(9, 4)
out = pairwise_sq_dists(a, b)
assert out.shape == (6, 9), f'Shape mismatch: {out.shape}'
expected = torch.cdist(a, b) ** 2
assert torch.allclose(out, expected, atol=1e-4), 'Wrong values'
""",
        },
        {
            "name": "Performance (no Python loops)",
            "code": """
import torch, time
x = torch.randn(32, 64, 32, 32); bias = torch.randn(64)
a = torch.randn(512, 32); b = torch.randn(512, 32)
t0 = time.perf_counter()
for _ in range(5):
    outs = (add_channel_bias(x, bias), abs_distance_matrix(512), pairwise_sq_dists(a, b))
elapsed = time.perf_counter() - t0
assert all(isinstance(o, torch.Tensor) for o in outs), 'All exercises must return tensors before timing counts'
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s. Are you looping over elements?'
""",
        },
    ],
}
