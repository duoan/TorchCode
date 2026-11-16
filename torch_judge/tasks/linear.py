"""Simple Linear Layer task."""

TASK = {
    "title": "Simple Linear Layer",
    "difficulty": "Medium",
    "function_name": "SimpleLinear",
    "hint": "Subclass nn.Module, call super().__init__(), and wrap weight and bias in nn.Parameter. y = x @ W^T + b; initialize the weight with randn * (1/sqrt(in_features)).",
    "tests": [
        {
            "name": "Is an nn.Module with registered parameters",
            "code": """
import torch
from torch import nn
layer = {fn}(8, 4)
assert isinstance(layer, nn.Module), 'SimpleLinear must subclass nn.Module'
names = sorted(n for n, _ in layer.named_parameters())
assert names == ['bias', 'weight'], f'weight and bias must be registered nn.Parameter attributes; found parameters {names}'
x = torch.randn(3, 8)
assert torch.allclose(layer(x), layer.forward(x)), 'Calling layer(x) should run forward'
""",
        },
        {
            "name": "Weight & bias shape",
            "code": """
import torch
layer = {fn}(8, 4)
assert layer.weight.shape == (4, 8), f'Weight shape: {layer.weight.shape}'
assert layer.bias.shape == (4,), f'Bias shape: {layer.bias.shape}'
assert layer.weight.requires_grad, 'weight must require grad'
assert layer.bias.requires_grad, 'bias must require grad'
""",
        },
        {
            "name": "Forward pass",
            "code": """
import torch
layer = {fn}(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
assert y.shape == (2, 4), f'Output shape: {y.shape}'
expected = x @ layer.weight.T + layer.bias
assert torch.allclose(y, expected, atol=1e-5), 'Forward != x @ W^T + b'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
layer = {fn}(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
y.sum().backward()
assert layer.weight.grad is not None, 'weight.grad is None'
assert layer.bias.grad is not None, 'bias.grad is None'
""",
        },
    ],
}
