"""Prerequisite 9: Numerical stability & log-space (multi-exercise task)."""

TASK = {
    "title": "Prereq 9: Numerical Stability & Log-Space",
    "difficulty": "Prereq",
    "order": 9,
    "function_name": "stable_logsumexp",
    "function_names": [
        "stable_logsumexp",
        "log_softmax",
        "log_sigmoid",
        "kl_from_logits",
        "safe_masked_softmax",
        "merge_softmax_stats",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `stable_logsumexp`: `m = x.amax(dim, keepdim=True)`; replace infinite `m` with 0; return `m + log(sum(exp(x - m)))`, squeezed.\n"
        "- `log_softmax`: `x - stable_logsumexp(x, dim).unsqueeze(dim)`.\n"
        "- `log_sigmoid`: `torch.minimum(x, 0) - torch.log1p(torch.exp(-x.abs()))`.\n"
        "- `kl_from_logits`: `log_p, log_q = log_softmax(p), log_softmax(q)`; `(log_p.exp() * (log_p - log_q)).sum(-1)`.\n"
        "- `safe_masked_softmax`: `masked_fill(-inf)`, shift by a finite max, `exp`, then divide by `torch.where(denom > 0, denom, 1)`. Not `where(..., e / denom, 0)`.\n"
        "- `merge_softmax_stats`: `m = torch.maximum(m1, m2)`; rescale each sum by `exp(m_i - m)`; if `m` is `-inf`, shift by 0 instead to avoid `-inf - (-inf)`."
    ),
    "tests": [
        {
            "name": "stable_logsumexp: values, dims, huge inputs",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 6) * 3
out = stable_logsumexp(x)
assert isinstance(out, torch.Tensor) and out.shape == (4,), f'Expected shape (4,), got {getattr(out, "shape", out)}'
assert torch.allclose(out, torch.logsumexp(x, dim=-1), atol=1e-5), 'Wrong values'
assert torch.allclose(stable_logsumexp(x, dim=0), torch.logsumexp(x, dim=0), atol=1e-5), 'dim argument not respected'
big = torch.tensor([1000.0, 1001.0, 1002.0])
assert torch.allclose(stable_logsumexp(big, dim=0), torch.logsumexp(big, dim=0)), f'Overflow on huge inputs: got {stable_logsumexp(big, dim=0)}'
""",
        },
        {
            "name": "stable_logsumexp: -inf entries and gradient",
            "code": """
import torch
ninf = float('-inf')
x = torch.tensor([[0.0, ninf, 1.0], [ninf, ninf, ninf]])
out = stable_logsumexp(x)
assert not torch.isnan(out).any(), f'No NaN allowed, got {out}'
assert torch.allclose(out[0], torch.logsumexp(x[0], dim=0)) and torch.isneginf(out[1]), f'Expected [{torch.logsumexp(x[0], 0).item():.4f}, -inf], got {out}'
z = torch.randn(3, 5, requires_grad=True)
stable_logsumexp(z).sum().backward()
assert torch.allclose(z.grad, torch.softmax(z.detach(), dim=-1), atol=1e-5), 'The gradient of logsumexp should be softmax'
""",
        },
        {
            "name": "log_softmax",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(3, 7) * 5
out = log_softmax(x)
assert isinstance(out, torch.Tensor) and out.shape == x.shape, 'Shape must match'
assert torch.allclose(out, torch.log_softmax(x, dim=-1), atol=1e-5), 'Wrong values'
assert torch.allclose(log_softmax(x, dim=0), torch.log_softmax(x, dim=0), atol=1e-5), 'dim argument not respected'
wide = torch.tensor([0.0, -1000.0, 1000.0])
ref = torch.log_softmax(wide, dim=0)
got = log_softmax(wide, dim=0)
assert torch.isfinite(got).all() and torch.allclose(got, ref), f'Expected finite {ref}, got {got}. log(softmax) underflows to -inf'
""",
        },
        {
            "name": "log_sigmoid: whole range and gradient",
            "code": """
import torch
import torch.nn.functional as F
x = torch.linspace(-300, 300, 601, requires_grad=True)
out = log_sigmoid(x)
assert isinstance(out, torch.Tensor) and out.shape == x.shape, 'Shape must match'
assert torch.isfinite(out).all(), f'Non-finite values, e.g. at x = {x[~torch.isfinite(out)][:3].tolist()}'
assert torch.allclose(out, F.logsigmoid(x.detach()), atol=1e-5), 'Wrong values'
out.sum().backward()
assert torch.isfinite(x.grad).all() and torch.allclose(x.grad, torch.sigmoid(-x.detach()), atol=1e-5), 'Gradient should be sigmoid(-x), and finite'
""",
        },
        {
            "name": "kl_from_logits",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(0)
p, q = torch.randn(2, 4, 10) * 3, torch.randn(2, 4, 10) * 3
out = kl_from_logits(p, q)
assert isinstance(out, torch.Tensor) and out.shape == (2, 4), f'Expected shape (2, 4), got {getattr(out, "shape", out)}'
ref = (F.softmax(p, -1) * (F.log_softmax(p, -1) - F.log_softmax(q, -1))).sum(-1)
assert torch.allclose(out, ref, atol=1e-4), 'Wrong values. KL(P || Q) = sum P * (log P - log Q)'
assert (out >= -1e-6).all(), 'KL must be non-negative'
assert torch.allclose(kl_from_logits(p, p), torch.zeros(2, 4), atol=1e-6), 'KL(P || P) should be 0'
ext = kl_from_logits(torch.tensor([0.0, 0.0]), torch.tensor([500.0, -500.0]))
assert torch.isfinite(ext), f'Extreme logits should give a finite KL, got {ext}'
""",
        },
        {
            "name": "safe_masked_softmax: values",
            "code": """
import torch
torch.manual_seed(0)
scores = torch.randn(2, 4, 4)
mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
out = safe_masked_softmax(scores, mask)
assert isinstance(out, torch.Tensor) and out.shape == scores.shape, 'Shape must match scores'
ref = torch.softmax(scores.masked_fill(mask, float('-inf')), dim=-1)
assert torch.allclose(out, ref, atol=1e-6), 'Rows with visible keys should be a normal masked softmax'
big = safe_masked_softmax(torch.tensor([[1000.0, 0.0]]), torch.tensor([[False, False]]))
assert torch.allclose(big, torch.tensor([[1.0, 0.0]])), f'Overflow with large scores: {big}'
""",
        },
        {
            "name": "safe_masked_softmax: fully masked rows, finite gradients",
            "code": """
import torch
scores = torch.randn(3, 3, requires_grad=True)
mask = torch.tensor([[False, True, True], [False, False, True], [True, True, True]])
out = safe_masked_softmax(scores, mask)
assert not torch.isnan(out).any(), f'No NaN allowed, got\\n{out}'
assert torch.equal(out[2], torch.zeros(3)), f'A fully masked row should be all zeros, got {out[2]}'
assert torch.allclose(out[:2].sum(-1), torch.ones(2)), 'Other rows must still sum to 1'
(out * torch.arange(9.).view(3, 3)).sum().backward()
assert torch.isfinite(scores.grad).all(), f'Gradients must be finite (see the torch.where trap), got\\n{scores.grad}'
""",
        },
        {
            "name": "merge_softmax_stats: streaming equals logsumexp",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 23) * 10
m, s = torch.full((4,), float('-inf')), torch.zeros(4)
for block in x.split(5, dim=1):
    mb = block.amax(dim=1)
    res = merge_softmax_stats(m, s, mb, torch.exp(block - mb[:, None]).sum(dim=1))
    assert isinstance(res, tuple) and len(res) == 2, 'Return a tuple (m, s)'
    m, s = res
assert torch.allclose(m, x.amax(dim=1)), 'Merged max should be the row max'
assert torch.allclose(m + torch.log(s), torch.logsumexp(x, dim=1), atol=1e-4), 'm + log(s) should equal logsumexp over the whole row'
""",
        },
        {
            "name": "merge_softmax_stats: huge values, empty blocks, symmetry",
            "code": """
import torch
ninf = torch.tensor(float('-inf'))
a = torch.tensor([1000.0, 1001.0]); b = torch.tensor([1002.0, 3.0])
sa = (a.max(), torch.exp(a - a.max()).sum()); sb = (b.max(), torch.exp(b - b.max()).sum())
m, s = merge_softmax_stats(*sa, *sb)
assert torch.isfinite(s) and torch.allclose(m + torch.log(s), torch.logsumexp(torch.cat([a, b]), 0)), f'Overflow or wrong value with huge inputs: m={m}, s={s}. Rescale each sum by exp(m_i - m)'
m2, s2 = merge_softmax_stats(*sb, *sa)
assert torch.allclose(m, m2) and torch.allclose(s, s2), 'Merging should not depend on the order of the blocks'
m3, s3 = merge_softmax_stats(ninf, torch.tensor(0.0), *sa)
assert torch.allclose(m3, sa[0]) and torch.allclose(s3, sa[1]), f'Merging with an empty block should return the other block, got ({m3}, {s3})'
m4, s4 = merge_softmax_stats(ninf, torch.tensor(0.0), ninf, torch.tensor(0.0))
assert not torch.isnan(s4) and torch.isneginf(m4) and s4 == 0, f'Two empty blocks should give (-inf, 0), got ({m4}, {s4})'
""",
        },
    ],
}
