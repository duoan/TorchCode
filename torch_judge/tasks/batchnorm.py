"""BatchNorm implementation task."""

TASK = {
    "title": "Implement BatchNorm",
    "difficulty": "Medium",
    "function_name": "my_batch_norm",
    "hint": (
        "Implement train/eval BatchNorm: in training, normalize with batch stats over dim=0 "
        "(biased variance) and update running_mean/running_var with momentum, using the "
        "UNBIASED batch variance (x.var(dim=0, unbiased=True)) for running_var, like "
        "nn.BatchNorm1d; in inference, normalize using the running statistics only."
    ),
    "tests": [
        {
            "name": "Training mode — zero mean per feature",
            "code": """
import torch
x = torch.randn(8, 4)
gamma = torch.ones(4)
beta = torch.zeros(4)
running_mean = torch.zeros(4)
running_var = torch.ones(4)
out = {fn}(x, gamma, beta, running_mean, running_var, training=True)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
col_means = out.mean(dim=0)
assert torch.allclose(col_means, torch.zeros(4), atol=1e-5), f'Column means not zero: {col_means}'
""",
        },
        {
            "name": "Training mode — numerical correctness and running stats update",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(16, 8)
gamma = torch.randn(8)
beta = torch.randn(8)
running_mean = torch.zeros(8)
running_var = torch.ones(8)
momentum = 0.1
out = {fn}(x, gamma, beta, running_mean, running_var, momentum=momentum, training=True)

# Reference using batch stats
mean = x.mean(dim=0)
var = x.var(dim=0, unbiased=False)
ref = gamma * (x - mean) / torch.sqrt(var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch'

# Running stats should have moved toward batch stats (unbiased variance for running_var, like PyTorch)
expected_mean = (1 - momentum) * torch.zeros_like(mean) + momentum * mean
expected_var = (1 - momentum) * torch.ones_like(var) + momentum * x.var(dim=0, unbiased=True)
biased_var = (1 - momentum) * torch.ones_like(var) + momentum * var
assert torch.allclose(running_mean, expected_mean, atol=1e-6), 'running_mean not updated correctly'
if torch.allclose(running_var, biased_var, atol=1e-6):
    raise AssertionError('running_var used the biased variance. Like nn.BatchNorm1d, update it with x.var(dim=0, unbiased=True)')
assert torch.allclose(running_var, expected_var, atol=1e-6), 'running_var not updated correctly'
""",
        },
        {
            "name": "Matches nn.BatchNorm1d over several steps",
            "code": """
import torch
from torch import nn
torch.manual_seed(0)
bn = nn.BatchNorm1d(6, momentum=0.2)
gamma = torch.randn(6); beta = torch.randn(6)
with torch.no_grad():
    bn.weight.copy_(gamma); bn.bias.copy_(beta)
running_mean = torch.zeros(6); running_var = torch.ones(6)
bn.train()
for step in range(3):
    x = torch.randn(10, 6) * 3 + 1
    out = {fn}(x, gamma, beta, running_mean, running_var, momentum=0.2, training=True)
    assert torch.allclose(out, bn(x).detach(), atol=1e-4), f'Training output differs from nn.BatchNorm1d at step {step}'
assert torch.allclose(running_mean, bn.running_mean, atol=1e-5), 'running_mean differs from nn.BatchNorm1d'
assert torch.allclose(running_var, bn.running_var, atol=1e-5), f'running_var differs from nn.BatchNorm1d: {running_var} vs {bn.running_var}'
bn.eval()
x = torch.randn(5, 6)
out = {fn}(x, gamma, beta, running_mean, running_var, training=False)
assert torch.allclose(out, bn(x).detach(), atol=1e-4), 'Eval output differs from nn.BatchNorm1d'
""",
        },
        {
            "name": "Inference mode — uses running statistics",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 8)
gamma = torch.randn(8)
beta = torch.randn(8)

# Pretend these came from previous training
running_mean = torch.randn(8)
running_var = torch.rand(8) + 0.5  # positive

out = {fn}(x, gamma, beta, running_mean.clone(), running_var.clone(), training=False)
ref = gamma * (x - running_mean) / torch.sqrt(running_var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-4), 'Inference should use running stats'
""",
        },
        {
            "name": "Gradient flow w.r.t inputs and affine params",
            "code": """
import torch
x = torch.randn(4, 8, requires_grad=True)
gamma = torch.ones(8, requires_grad=True)
beta = torch.zeros(8, requires_grad=True)
running_mean = torch.zeros(8)
running_var = torch.ones(8)
out = {fn}(x, gamma, beta, running_mean, running_var, training=True)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gamma.grad is not None, 'gamma.grad is None'
assert beta.grad is not None, 'beta.grad is None'
""",
        },
    ],
}
