"""Linear Self-Attention task."""

TASK = {
    "title": "Linear Self-Attention",
    "difficulty": "Hard",
    "function_name": "linear_attention",
    "hint": "Feature map: phi(x) = elu(x) + 1. Compute phi(Q) @ (phi(K)^T @ V) instead of softmax(Q @ K^T) @ V. Normalize by phi(Q) @ sum(phi(K)).",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
out = {fn}(torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 32))
assert out.shape == (2, 8, 32), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "No NaN or Inf",
            "code": """
import torch
torch.manual_seed(0)
out = {fn}(torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 16, 8))
assert not torch.isnan(out).any(), 'NaN in output'
assert not torch.isinf(out).any(), 'Inf in output'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
Q = torch.randn(1, 4, 8, requires_grad=True)
K = torch.randn(1, 4, 8, requires_grad=True)
V = torch.randn(1, 4, 8, requires_grad=True)
{fn}(Q, K, V).sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'
""",
        },
        {
            "name": "Runs fast on long sequences (linear complexity)",
            "code": """
import torch, time
torch.manual_seed(0)
Q = torch.randn(1, 2048, 64)
K = torch.randn(1, 2048, 64)
V = torch.randn(1, 2048, 64)
t0 = time.perf_counter()
for _ in range(10):
    out = {fn}(Q, K, V)
elapsed = time.perf_counter() - t0
assert isinstance(out, torch.Tensor) and out.shape == V.shape, 'Implement linear_attention first: it must return a (B, S, D_v) tensor'
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s — should be O(S*D^2) not O(S^2*D)'
""",
        },
    ],
}
