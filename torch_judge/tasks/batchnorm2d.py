"""BatchNorm2d implementation task."""

TASK = {
    "title": "Implement BatchNorm2d",
    "difficulty": "Medium",
    "function_name": "my_batch_norm2d",
    "hint": (
        "Reduce over dims (0, 2, 3) to get per-channel stats of shape (C,), and reshape every "
        "(C,) tensor to (1, C, 1, 1) before combining it with x. Normalize with the biased "
        "variance; update running_var with the unbiased one, x.var(dim=(0, 2, 3), unbiased=True), "
        "whose count is N * H * W."
    ),
    "tests": [
        {
            "name": "Training mode: per-channel normalization",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 3, 5, 5) * torch.tensor([1.0, 5.0, 0.5]).view(1, 3, 1, 1) + 2
out = {fn}(x, torch.ones(3), torch.zeros(3), torch.zeros(3), torch.ones(3), training=True)
assert isinstance(out, torch.Tensor) and out.shape == x.shape, f'Expected shape {tuple(x.shape)}, got {getattr(out, "shape", out)}'
m = out.mean(dim=(0, 2, 3)); v = out.var(dim=(0, 2, 3), unbiased=False)
assert torch.allclose(m, torch.zeros(3), atol=1e-5), f'Each CHANNEL should have mean 0 over (N, H, W), got {m}'
assert torch.allclose(v, torch.ones(3), atol=1e-3), f'Each CHANNEL should have variance 1 over (N, H, W), got {v}'
""",
        },
        {
            "name": "Running stats use the unbiased variance over N*H*W",
            "code": """
import torch
torch.manual_seed(1)
N, C, H, W = 2, 4, 3, 3
x = torch.randn(N, C, H, W)
rm, rv = torch.zeros(C), torch.ones(C)
{fn}(x, torch.ones(C), torch.zeros(C), rm, rv, momentum=0.5, training=True)
dims = (0, 2, 3)
assert torch.allclose(rm, 0.5 * x.mean(dim=dims), atol=1e-6), 'running_mean not updated correctly (in place, with momentum)'
expected = 0.5 + 0.5 * x.var(dim=dims, unbiased=True)
biased = 0.5 + 0.5 * x.var(dim=dims, unbiased=False)
batch_only = 0.5 + 0.5 * x.var(dim=dims, unbiased=False) * N / (N - 1)
if torch.allclose(rv, biased, atol=1e-6):
    raise AssertionError('running_var used the biased variance; nn.BatchNorm2d uses the unbiased one')
if torch.allclose(rv, batch_only, atol=1e-6):
    raise AssertionError('Bessel correction used N / (N - 1); the sample count per channel is N * H * W')
assert torch.allclose(rv, expected, atol=1e-6), f'running_var not updated correctly: {rv} vs {expected}'
""",
        },
        {
            "name": "Matches nn.BatchNorm2d over several steps",
            "code": """
import torch
from torch import nn
torch.manual_seed(2)
C = 5
bn = nn.BatchNorm2d(C, momentum=0.2)
gamma, beta = torch.randn(C), torch.randn(C)
with torch.no_grad():
    bn.weight.copy_(gamma); bn.bias.copy_(beta)
rm, rv = torch.zeros(C), torch.ones(C)
bn.train()
for step in range(3):
    x = torch.randn(3, C, 4, 6) * 2 - 1
    out = {fn}(x, gamma, beta, rm, rv, momentum=0.2, training=True)
    assert torch.allclose(out, bn(x).detach(), atol=1e-4), f'Training output differs from nn.BatchNorm2d at step {step}'
assert torch.allclose(rm, bn.running_mean, atol=1e-5) and torch.allclose(rv, bn.running_var, atol=1e-5), 'Running stats differ from nn.BatchNorm2d'
bn.eval()
x = torch.randn(2, C, 4, 6)
before = rm.clone(), rv.clone()
out = {fn}(x, gamma, beta, rm, rv, training=False)
assert torch.allclose(out, bn(x).detach(), atol=1e-4), 'Eval output differs from nn.BatchNorm2d: use the running stats'
assert torch.equal(rm, before[0]) and torch.equal(rv, before[1]), 'Eval mode must not change the running stats'
""",
        },
        {
            "name": "Gradients flow to x, gamma, beta; not into running stats",
            "code": """
import torch
x = torch.randn(2, 3, 4, 4, requires_grad=True)
gamma = torch.ones(3, requires_grad=True); beta = torch.zeros(3, requires_grad=True)
rm, rv = torch.zeros(3), torch.ones(3)
out = {fn}(x, gamma, beta, rm, rv, training=True)
(out * torch.randn_like(out)).sum().backward()
assert x.grad is not None and gamma.grad is not None and beta.grad is not None, 'Gradients must reach x, gamma and beta'
assert not rm.requires_grad and not rv.requires_grad and rm.grad_fn is None and rv.grad_fn is None, 'Running stats must not be part of the graph: detach the batch stats before updating them'
""",
        },
        {
            "name": "Works on non-square images and a single image",
            "code": """
import torch
from torch import nn
torch.manual_seed(3)
x = torch.randn(1, 2, 3, 7)
bn = nn.BatchNorm2d(2).train()
rm, rv = torch.zeros(2), torch.ones(2)
out = {fn}(x, torch.ones(2), torch.zeros(2), rm, rv, training=True)
assert torch.allclose(out, bn(x).detach(), atol=1e-4), 'N = 1 is fine for BatchNorm2d as long as H * W > 1'
assert torch.allclose(rv, bn.running_var, atol=1e-5), 'running_var differs from nn.BatchNorm2d for N = 1'
""",
        },
    ],
}
