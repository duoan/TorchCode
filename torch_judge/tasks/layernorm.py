"""LayerNorm implementation task."""

TASK = {
    "title": "Implement LayerNorm",
    "difficulty": "Medium",
    "function_name": "my_layer_norm",
    "hint": "Normalize over the last dim: $(x - \\mu) / \\sqrt{\\sigma^2 + \\epsilon}$, then scale by $\\gamma$ and shift by $\\beta$.",
    "tests": [
        {
            "name": "Shape and basic behavior",
            "code": """
import torch
x = torch.randn(2, 3, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = {fn}(x, gamma, beta)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch vs F.layer_norm'
""",
        },
        {
            "name": "With learned parameters",
            "code": """
import torch
x = torch.randn(4, 16)
gamma = torch.randn(16)
beta = torch.randn(16)
out = {fn}(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [16], gamma, beta)
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch with non-trivial gamma/beta'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
x = torch.randn(2, 8, requires_grad=True)
gamma = torch.ones(8, requires_grad=True)
beta = torch.zeros(8, requires_grad=True)
out = {fn}(x, gamma, beta)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gamma.grad is not None, 'gamma.grad is None'
""",
        },
    ],
}
