"""Prerequisite 10: Training loop & randomness (multi-exercise task)."""

TASK = {
    "title": "Prereq 10: Training Loop & Randomness",
    "difficulty": "Prereq",
    "order": 10,
    "function_name": "train_steps",
    "function_names": [
        "train_steps",
        "set_lr_",
        "keep_mask",
        "sample_tokens",
        "shuffled_batches",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `train_steps`: each step `optimizer.zero_grad()`, `loss = loss_fn(model(x), y)`, `loss.backward()`, `optimizer.step()`; append `loss.item()`.\n"
        "- `set_lr_`: `for g in optimizer.param_groups: g['lr'] = lr`.\n"
        "- `keep_mask`: `torch.rand(shape, generator=generator) >= p_drop`.\n"
        "- `sample_tokens`: `torch.multinomial(probs, n, replacement=True, generator=generator)`.\n"
        "- `shuffled_batches`: `perm = torch.randperm(N, generator=generator)`, then `torch.split(X[perm], batch_size)` and the same for `y`."
    ),
    "tests": [
        {
            "name": "train_steps: matches the canonical loop",
            "code": """
import torch, copy
from torch import nn
torch.manual_seed(0)
model = nn.Linear(3, 1)
ref = copy.deepcopy(model)
x = torch.randn(16, 3); y = x @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.3
losses = train_steps(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.MSELoss(), x, y, 20)
assert isinstance(losses, list) and len(losses) == 20, f'Return a list of 20 losses, got {type(losses).__name__} of length {len(losses) if hasattr(losses, "__len__") else "?"}'
assert all(type(l) is float for l in losses), 'Losses must be Python floats: use loss.item()'
opt = torch.optim.SGD(ref.parameters(), lr=0.1)
expected = []
for _ in range(20):
    opt.zero_grad(); loss = nn.MSELoss()(ref(x), y); loss.backward(); opt.step(); expected.append(loss.item())
assert all(abs(a - b) < 1e-5 for a, b in zip(losses, expected)), 'Losses differ from the canonical loop. Zeroing gradients every step? Recording the loss before the update?'
assert torch.allclose(model.weight, ref.weight, atol=1e-5), 'Final weights differ from the canonical loop'
assert losses[-1] < losses[0], 'Loss should decrease'
""",
        },
        {
            "name": "set_lr_",
            "code": """
import torch
opt = torch.optim.SGD([{'params': [torch.zeros(1, requires_grad=True)], 'lr': 0.1},
                       {'params': [torch.zeros(2, requires_grad=True)], 'lr': 0.2}])
set_lr_(opt, 0.03)
assert [g['lr'] for g in opt.param_groups] == [0.03, 0.03], f'Every param group must get the new lr, got {[g["lr"] for g in opt.param_groups]}'
""",
        },
        {
            "name": "keep_mask: dtype, rate, edge cases",
            "code": """
import torch
g = torch.Generator().manual_seed(0)
m = keep_mask((400, 250), 0.3, g)
assert isinstance(m, torch.Tensor) and m.dtype == torch.bool and m.shape == (400, 250), f'Expected a bool tensor of shape (400, 250), got {getattr(m, "dtype", type(m))} {getattr(m, "shape", "")}'
rate = m.float().mean().item()
assert abs(rate - 0.7) < 0.01, f'Keep rate should be about 1 - p_drop = 0.7, got {rate:.3f}'
assert keep_mask((50,), 0.0, g).all(), 'p_drop=0 must keep everything'
assert not keep_mask((50,), 1.0, g).any(), 'p_drop=1 must keep nothing'
""",
        },
        {
            "name": "keep_mask: uses the generator",
            "code": """
import torch
torch.manual_seed(0); a = keep_mask((1000,), 0.5, torch.Generator().manual_seed(7))
torch.manual_seed(123); b = keep_mask((1000,), 0.5, torch.Generator().manual_seed(7))
assert torch.equal(a, b), 'Same generator seed must give the same mask regardless of the global seed: pass generator=generator'
c = keep_mask((1000,), 0.5, torch.Generator().manual_seed(8))
assert not torch.equal(a, c), 'Different generator seeds should give different masks'
""",
        },
        {
            "name": "sample_tokens: distribution and generator",
            "code": """
import torch
probs = torch.tensor([0.5, 0.0, 0.3, 0.2])
torch.manual_seed(0); a = sample_tokens(probs, 40000, torch.Generator().manual_seed(3))
assert isinstance(a, torch.Tensor) and a.shape == (40000,) and a.dtype == torch.long, f'Expected a (n,) long tensor, got {getattr(a, "dtype", type(a))} {getattr(a, "shape", "")}'
freq = torch.bincount(a, minlength=4).float() / 40000
assert freq[1] == 0, 'A token with probability 0 must never be sampled'
assert torch.allclose(freq, probs, atol=0.01), f'Frequencies {freq.tolist()} should be close to {probs.tolist()}. Sampling with replacement?'
torch.manual_seed(99); b = sample_tokens(probs, 40000, torch.Generator().manual_seed(3))
assert torch.equal(a, b), 'Same generator seed must give the same samples: pass generator=generator'
""",
        },
        {
            "name": "shuffled_batches: coverage, pairing, sizes",
            "code": """
import torch
X = torch.arange(50).float()[:, None]
y = X.squeeze(1) * 10
batches = shuffled_batches(X, y, 16, torch.Generator().manual_seed(0))
assert isinstance(batches, list) and all(isinstance(b, tuple) and len(b) == 2 for b in batches), 'Return a list of (X_batch, y_batch) tuples'
assert [len(xb) for xb, _ in batches] == [16, 16, 16, 2], f'Expected batch sizes [16, 16, 16, 2], got {[len(xb) for xb, _ in batches]}'
xs = torch.cat([xb for xb, _ in batches]).flatten(); ys = torch.cat([yb for _, yb in batches])
assert torch.equal(xs.sort().values, X.flatten()), 'Every row must appear exactly once'
assert torch.equal(ys, xs * 10), 'X and y rows must stay paired (shuffle both with the same permutation)'
assert not torch.equal(xs, X.flatten()), 'Rows should be shuffled'
""",
        },
        {
            "name": "shuffled_batches: uses the generator",
            "code": """
import torch
X = torch.arange(30).float()[:, None]; y = X.squeeze(1)
torch.manual_seed(0); a = shuffled_batches(X, y, 8, torch.Generator().manual_seed(5))
torch.manual_seed(42); b = shuffled_batches(X, y, 8, torch.Generator().manual_seed(5))
assert all(torch.equal(p[0], q[0]) for p, q in zip(a, b)), 'Same generator seed must give the same order regardless of the global seed'
""",
        },
    ],
}
