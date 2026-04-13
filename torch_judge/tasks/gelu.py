"""GELU Activation task."""

TASK = {
    "title": "GELU Activation",
    "difficulty": "Easy",
    "function_name": "my_gelu",
    "hint": "Exact: $x \\cdot 0.5 \\cdot (1 + \\text{erf}(x / \\sqrt{2}))$. Or approximate: $0.5x(1+\\tanh(\\sqrt{2/\\pi}(x+0.044715x^3)))$.",
    "tests": [
        {
            "name": "Matches F.gelu",
            "code": "\nimport torch\ntorch.manual_seed(0)\nx = torch.randn(4, 8)\nout = {fn}(x)\nref = torch.nn.functional.gelu(x)\nassert torch.allclose(out, ref, atol=1e-4), 'Does not match F.gelu'\n"
        },
        {
            "name": "gelu(0) = 0",
            "code": "\nimport torch\nout = {fn}(torch.tensor([0.0]))\nassert torch.allclose(out, torch.tensor([0.0]), atol=1e-7), f'gelu(0) = {out.item()}'\n"
        },
        {
            "name": "Shape preservation",
            "code": "\nimport torch\nx = torch.randn(2, 3, 4)\nassert {fn}(x).shape == x.shape, 'Shape mismatch'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\nx = torch.randn(4, 8, requires_grad=True)\n{fn}(x).sum().backward()\nassert x.grad is not None and x.grad.shape == x.shape, 'Gradient issue'\n"
        }
    ]
}
