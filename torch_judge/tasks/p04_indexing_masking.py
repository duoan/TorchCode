"""Prerequisite 4: Indexing & masking (multi-exercise task)."""

TASK = {
    "title": "Prereq 4: Indexing & Masking",
    "difficulty": "Prereq",
    "order": 4,
    "function_name": "pick_per_row",
    "function_names": [
        "pick_per_row",
        "attention_mask",
        "apply_mask",
        "keep_topk",
        "unsort",
        "group_mean",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `pick_per_row`: index with two tensors, `logits[torch.arange(B), targets]`. `logits[:, targets]` gives `(B, B)`.\n"
        "- `attention_mask`: build `R = i[:, None] - i[None, :]`; causal is `R < 0`, sliding is `R.abs() > w`, causal_sliding is `(R < 0) | (R > w)`.\n"
        "- `apply_mask`: `scores.masked_fill(mask, float('-inf'))` returns a new tensor and broadcasts the mask.\n"
        "- `keep_topk`: `vals, idx = logits.topk(k, dim=-1)`, then scatter `vals` into `torch.full_like(logits, float('-inf'))`.\n"
        "- `unsort`: `torch.empty_like(sorted_values).scatter(-1, sort_idx, sorted_values)`.\n"
        "- `group_mean`: `uniq, inv = torch.unique(ids, return_inverse=True)`; sum with `index_add(0, inv, values)`, count with `bincount(inv)`, then index the means with `inv`."
    ),
    "tests": [
        {
            "name": "pick_per_row",
            "code": """
import torch
torch.manual_seed(0)
logits = torch.randn(5, 7, requires_grad=True)
targets = torch.tensor([0, 6, 3, 3, 1])
out = pick_per_row(logits, targets)
assert isinstance(out, torch.Tensor) and out.shape == (5,), f'Expected shape (B,) = (5,), got {getattr(out, "shape", out)}. logits[:, targets] gives (B, B)'
assert torch.equal(out, logits.gather(1, targets[:, None]).squeeze(1)), 'Wrong values'
out.sum().backward()
assert logits.grad[1, 6] == 1 and logits.grad[1].sum() == 1, 'Gradient should flow only to the picked entries'
""",
        },
        {
            "name": "attention_mask: causal",
            "code": """
import torch
m = attention_mask(5, "causal")
assert isinstance(m, torch.Tensor) and m.dtype == torch.bool, f'Expected a bool tensor, got {getattr(m, "dtype", type(m))}'
assert m.shape == (5, 5), f'Shape mismatch: {m.shape}'
assert torch.equal(m, torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)), f'True must mean BLOCKED: future keys j > i.\\n{m.int()}'
""",
        },
        {
            "name": "attention_mask: sliding and causal_sliding",
            "code": """
import torch
n, w = 7, 2
i = torch.arange(n)[:, None]; j = torch.arange(n)[None, :]
s = attention_mask(n, "sliding", w)
assert torch.equal(s, (i - j).abs() > w), f'sliding: block abs(i - j) > w (two-sided).\\n{s.int()}'
cs = attention_mask(n, "causal_sliding", w)
assert torch.equal(cs, (j > i) | (i - j > w)), f'causal_sliding: block the future AND keys more than w behind.\\n{cs.int()}'
assert torch.equal(attention_mask(n, "causal_sliding", n), attention_mask(n, "causal")), 'A window wider than n should equal causal'
assert (~attention_mask(n, "sliding", 0)).int().sum() == n, 'w=0 sliding: each query sees only itself'
try:
    attention_mask(3, "bogus")
except ValueError:
    pass
else:
    raise AssertionError('Unknown kind should raise ValueError')
""",
        },
        {
            "name": "apply_mask: broadcast, -inf, no in-place",
            "code": """
import torch
torch.manual_seed(0)
scores = torch.randn(2, 3, 4, 4, dtype=torch.float64)
before = scores.clone()
mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
out = apply_mask(scores, mask)
assert isinstance(out, torch.Tensor) and out.shape == scores.shape, 'Shape must match scores'
assert out.dtype == torch.float64, f'dtype must match scores, got {out.dtype}'
assert torch.equal(scores, before), 'Do not modify scores in place (use masked_fill, not masked_fill_)'
assert torch.isneginf(out[..., mask]).all(), 'Blocked positions must be -inf'
assert torch.equal(out[..., ~mask], scores[..., ~mask]), 'Unblocked positions must be unchanged'
""",
        },
        {
            "name": "keep_topk",
            "code": """
import torch
logits = torch.tensor([[0.1, 2.0, -1.0, 3.0, 0.5], [9.0, 8.0, 7.0, 6.0, 5.0]])
before = logits.clone()
out = keep_topk(logits, 2)
inf = float('-inf')
expected = torch.tensor([[inf, 2.0, inf, 3.0, inf], [9.0, 8.0, inf, inf, inf]])
assert isinstance(out, torch.Tensor) and torch.equal(out, expected), f'Got\\n{out}'
assert torch.equal(logits, before), 'Do not modify logits in place'
assert torch.equal(keep_topk(logits, 10), logits), 'k >= V should keep everything'
""",
        },
        {
            "name": "keep_topk: batched rows",
            "code": """
import torch
torch.manual_seed(0)
logits = torch.randn(3, 4, 50)
out = keep_topk(logits, 5)
assert out.shape == logits.shape, f'Shape mismatch: {out.shape}'
assert (torch.isfinite(out).sum(dim=-1) == 5).all(), 'Exactly k entries per row should stay finite'
kth = logits.topk(5, dim=-1).values[..., -1:]
assert torch.equal(torch.isfinite(out), logits >= kth), 'The k LARGEST entries of each row (last dim) should be kept'
""",
        },
        {
            "name": "unsort: round trip",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(3, 4, 9)
for desc in (False, True):
    s, idx = x.sort(dim=-1, descending=desc)
    out = unsort(s, idx)
    assert isinstance(out, torch.Tensor) and out.shape == x.shape, 'Shape must match'
    assert torch.equal(out, x), f'unsort(sort(x)) should give back x (descending={desc})'
""",
        },
        {
            "name": "unsort: after editing in sorted order",
            "code": """
import torch
x = torch.tensor([0.3, -1.0, 2.0, 0.0])
s, idx = x.sort(descending=True)
s = s.clone(); s[2:] = float('-inf')
out = unsort(s, idx)
expected = torch.tensor([0.3, float('-inf'), 2.0, float('-inf')])
assert torch.equal(out, expected), f'Got {out}, expected {expected}. Is it a gather instead of a scatter?'
""",
        },
        {
            "name": "group_mean: arbitrary ids, dtype, gradient",
            "code": """
import torch
ids = torch.tensor([7, 3, 7, 10, 3, 7])
vals = torch.tensor([1., 2., 3., 4., 5., 8.], dtype=torch.float64, requires_grad=True)
out = group_mean(vals, ids)
assert isinstance(out, torch.Tensor) and out.shape == (6,), f'Expected shape (6,), got {getattr(out, "shape", out)}'
assert out.dtype == torch.float64, f'dtype must match values, got {out.dtype}'
expected = torch.tensor([4., 3.5, 4., 4., 3.5, 4.], dtype=torch.float64)
assert torch.allclose(out, expected), f'Expected {expected.tolist()}, got {out.tolist()}'
out.sum().backward()
assert torch.allclose(vals.grad, torch.ones(6, dtype=torch.float64)), 'Gradient should flow back to every value'
""",
        },
        {
            "name": "group_mean: many groups, no loop over groups",
            "code": """
import torch
from torch.overrides import TorchFunctionMode
class _Count(TorchFunctionMode):
    def __init__(self): super().__init__(); self.n = 0
    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))
torch.manual_seed(0)
ids = torch.randint(0, 100_000, (5000,))
vals = torch.randn(5000)
with _Count() as c:
    out = group_mean(vals, ids)
uniq = ids.unique()
assert c.n < 200, f'{c.n} torch calls for {len(uniq)} groups: looks like a Python loop over groups. Use unique / bincount / index_add'
ref = torch.empty(5000)
for g in uniq[:50]:
    m = ids == g
    ref[m] = vals[m].mean()
sel = torch.isin(ids, uniq[:50])
assert torch.allclose(out[sel], ref[sel], atol=1e-5), 'Wrong values with many groups'
""",
        },
    ],
}
