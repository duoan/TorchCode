"""Prerequisite 4: Indexing & masking.

Content for templates/00_p04_indexing_masking.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P4",
    "file": "00_p04_indexing_masking",
    "task_id": "p04_indexing_masking",
    "title": "Indexing & Masking",
    "intro": r'''
# 🔵 Prerequisite 4: Indexing & Masking

Picking the logit of the correct class, looking up embeddings, hiding future tokens from attention, keeping only the top-k logits when sampling, routing tokens to experts: all of these **select** or **overwrite** parts of a tensor by index or by mask.

PyTorch has several ways to do this (slicing, integer "fancy" indexing, boolean masks, `masked_fill`, `topk`, `gather`, `scatter`). Picking the right one keeps the code loop-free and differentiable.

### What you'll learn
1. Slicing, including strides like `x[..., 0::2]`
2. Integer (fancy) indexing: `weight[indices]` and `x[rows, cols]`
3. Boolean masks: reading `x[mask]` and writing `x[mask] = ...`
4. `masked_fill` with `-inf`, and building masks with `triu` / comparisons
5. `topk` and `sort`: values *and* indices
6. `gather` and `scatter`: index-based reads and writes along a dim
7. Grouping by id: `unique`, `bincount`, `index_add`

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `pick_per_row(logits, targets)` | #16 Cross-Entropy |
| 2 | `attention_mask(n, kind, w)` | #09 Causal, #11 Sliding Window, #13 GPT-2 Block, #14 KV Cache |
| 3 | `apply_mask(scores, mask)` | every masked attention problem |
| 4 | `keep_topk(logits, k)` | #32 Top-k Sampling, #28 MoE |
| 5 | `unsort(sorted_values, sort_idx)` | #32 Top-p Sampling |
| 6 | `group_mean(values, group_ids)` | #38 GRPO |

### Rules
- **No Python loops** over tensor elements, or over groups (Exercise 6).
- Do **not** use `F.cross_entropy` / `F.nll_loss` (Exercise 1) or `torch.argsort` on the indices to undo the sort (Exercise 5). Use `scatter`.
''',
    "lesson": [
        ("md", r'''
## 1. Slicing

Slices return **views** (no copy). A step selects every k-th element: `x[..., 0::2]` takes even positions and `x[..., 1::2]` odd ones. RoPE uses exactly this to split vectors into pairs.
`...` (Ellipsis) means "all leading dims", so the same code works for `(S, D)` and `(B, H, S, D)`.
'''),
        ("code", r'''
x = torch.arange(12).view(2, 6)
print(x[:, 1:4])        # columns 1..3
print(x[..., 0::2])     # even columns
print(x[..., 1::2])     # odd columns
print(x[-1])            # last row
'''),
        ("md", r'''
## 2. Integer (fancy) indexing

Indexing with an integer **tensor** selects rows: `weight[indices]` is an embedding lookup, and works for any shape of `indices`.
Indexing with **two** index tensors pairs them up element-wise: `x[rows, cols]` returns `x[rows[0], cols[0]], x[rows[1], cols[1]], …`. This is **not** the same as `x[rows][:, cols]`.
'''),
        ("code", r'''
weight = torch.arange(10.).view(5, 2)          # 5 embeddings of size 2
idx = torch.tensor([[0, 4], [2, 2]])
print(weight[idx].shape)                       # (2, 2, 2): idx shape + embedding dim

x = torch.tensor([[10., 11., 12.],
                  [20., 21., 22.]])
print(x[torch.tensor([0, 1]), torch.tensor([2, 0])])   # tensor([12., 20.])
'''),
        ("md", r'''
## 3. Boolean masks

- **Reading** `x[mask]` returns a **1-D** tensor of the selected elements. The shape is lost.
- **Writing** `x[mask] = value` (or `x[mask] += ...`) updates only the selected positions.
- `torch.where(mask, a, b)` picks element-wise and **keeps the shape**.
'''),
        ("code", r'''
x = torch.tensor([[1., -2.], [-3., 4.]])
mask = x < 0
print(x[mask])                        # tensor([-2., -3.]): flattened
print(torch.where(mask, 0.0, x))      # same shape, negatives replaced
y = x.clone(); y[mask] = 0.0; print(y)
'''),
        ("md", r'''
## 4. `masked_fill` and building masks

`scores.masked_fill(mask, float("-inf"))` returns a copy with `-inf` wherever `mask` is `True`. The mask **broadcasts**, so one `(S, S)` mask can cover a whole `(B, H, S, S)` score tensor. After softmax, `-inf` becomes probability `0`.
Masks are built from comparisons, or with `torch.triu` / `torch.tril`. `diagonal=k` shifts which diagonal is kept.

> ⚠️ If an entire row is `-inf`, softmax returns `NaN` for that row. Make sure every query can see at least one key (P9 covers this).
'''),
        ("code", r'''
S = 4
future = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)   # True above the diagonal
print(future)
i = torch.arange(S)
print(torch.equal(future, i[None, :] > i[:, None]))   # same mask, from a comparison (P2)

scores = torch.zeros(2, S, S)
print(torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)[0])
'''),
        ("md", r'''
## 5. `topk` and `sort`

Both return `(values, indices)` along a dim (default: last). `topk(k)` gives the `k` largest, in descending order. `sort` orders everything.
The **indices** tell you where each value came from, and you need them to put values back (§6).
'''),
        ("code", r'''
logits = torch.tensor([0.1, 2.0, -1.0, 3.0, 0.5])
vals, idx = logits.topk(2)
print(vals, idx)                    # tensor([3., 2.]) tensor([3, 1])
s_vals, s_idx = logits.sort(descending=True)
print(s_vals, s_idx)
'''),
        ("md", r'''
## 6. `gather` and `scatter`

- `x.gather(dim, idx)` **reads**: `out[i][j] = x[i][idx[i][j]]` (for `dim=1`). `out` has the shape of `idx`.
- `x.scatter(dim, idx, src)` **writes**: `out[i][idx[i][j]] = src[i][j]`. It returns a copy of `x` with those positions overwritten (`scatter_` is the in-place version).

They are inverses: `gather` pulls values *from* positions, `scatter` pushes values *to* positions.
'''),
        ("code", r'''
x = torch.tensor([[10., 11., 12.],
                  [20., 21., 22.]])
idx = torch.tensor([[2, 0],
                    [1, 1]])
print(x.gather(1, idx))                          # [[12., 10.], [21., 21.]]

base = torch.zeros(2, 3)
print(base.scatter(1, torch.tensor([[2], [0]]), torch.tensor([[5.], [7.]])))
# [[0., 0., 5.], [7., 0., 0.]]
'''),
        ("md", r'''
## 7. Grouping by id

GRPO (#38) normalizes each reward against the other rewards **from the same prompt**. Group ids can be arbitrary integers, like `[7, 3, 7, 10, 3]`.

- `torch.unique(ids, return_inverse=True)` returns `(unique_ids, inverse)`, where `inverse[i]` is the position of `ids[i]` in `unique_ids`. That relabels the groups as `0 … G-1`.
- `torch.bincount(inverse)` counts the members of each group.
- `torch.zeros(G).index_add(0, inverse, values)` adds every value into its group's slot (`bincount(inverse, weights=values)` also works).
- `group_stat[inverse]` broadcasts a per-group result back to every element.

No loop over groups needed, no matter how many there are.
'''),
        ("code", r'''
ids = torch.tensor([7, 3, 7, 10, 3])
vals = torch.tensor([1., 2., 3., 4., 5.])
uniq, inv = ids.unique(return_inverse=True)
print(uniq, inv)                                  # tensor([ 3,  7, 10]) tensor([1, 0, 1, 2, 0])
counts = torch.bincount(inv)
sums = torch.zeros(len(uniq)).index_add(0, inv, vals)
print(counts, sums)                               # [2, 2, 1]  [7., 4., 4.]
print((sums / counts)[inv])                       # each element's group mean: [2., 3.5, 2., 4., 3.5]
'''),
    ],
    "exercises_intro": r'''
---
# ✏️ Exercises

Implement the six functions below. Each one has a debug cell you can use before you submit.
''',
    "exercises": [
        {
            "spec": r'''
### Exercise 1: `pick_per_row(logits, targets)`

For each row, pick the value at that row's target column.

- `logits`: `(B, C)`
- `targets`: `(B,)` integer class ids
- returns: `(B,)` with `out[b] = logits[b, targets[b]]`

*Cross-entropy (#16) is `-pick_per_row(log_probs, targets).mean()`.*
''',
            "stub": r'''
def pick_per_row(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def pick_per_row(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, targets]
''',
            "debug": r'''
logits = torch.tensor([[0.1, 0.7, 0.2],
                       [0.5, 0.3, 0.2]])
print(pick_per_row(logits, torch.tensor([1, 0])))   # expect tensor([0.7000, 0.5000])
''',
        },
        {
            "spec": r'''
### Exercise 2: `attention_mask(n, kind, w=0)`

Build an `(n, n)` **boolean** mask for attention, where `True` means **blocked** (query `i` may *not* attend to key `j`).

- `kind="causal"`: block the future, `j > i`
- `kind="sliding"`: two-sided window (Longformer-style, #11), block `abs(i - j) > w`
- `kind="causal_sliding"`: causal window (Mistral-style), block `j > i` **or** `i - j > w`
- any other `kind`: raise `ValueError`

*Hint: P2's `relative_position(n)` (`R[i, j] = i - j`) makes each kind a single comparison.*
''',
            "stub": r'''
def attention_mask(n: int, kind: str, w: int = 0) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def attention_mask(n: int, kind: str, w: int = 0) -> torch.Tensor:
    idx = torch.arange(n)
    R = idx[:, None] - idx[None, :]          # R[i, j] = i - j
    if kind == "causal":
        return R < 0
    if kind == "sliding":
        return R.abs() > w
    if kind == "causal_sliding":
        return (R < 0) | (R > w)
    raise ValueError(f"unknown mask kind: {kind!r}")
''',
            "debug": r'''
for kind in ("causal", "sliding", "causal_sliding"):
    print(kind)
    print(attention_mask(5, kind, w=1).int())
# causal_sliding with w=1 should leave exactly 2 visible keys per row (itself and the previous one),
# except row 0, which only sees itself.
''',
        },
        {
            "spec": r'''
### Exercise 3: `apply_mask(scores, mask)`

Set blocked attention scores to `-inf` so softmax gives them zero probability.

- `scores`: `(..., S_q, S_k)` float, any number of leading dims (batch, heads)
- `mask`: `(S_q, S_k)` bool, `True` = blocked
- returns: same shape and dtype as `scores`
- do **not** modify `scores` in place
''',
            "stub": r'''
def apply_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def apply_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(mask, float("-inf"))
''',
            "debug": r'''
scores = torch.zeros(2, 3, 3)
masked = apply_mask(scores, attention_mask(3, "causal"))
print(masked[0])
print(torch.softmax(masked, dim=-1)[0])   # row i spreads evenly over keys 0..i
''',
        },
        {
            "spec": r'''
### Exercise 4: `keep_topk(logits, k)`

Keep the `k` largest logits in each row (last dim) and set every other entry to `-inf`.

- `logits`: `(..., V)` float
- `k`: int, `1 <= k`; if `k >= V`, keep everything
- returns: same shape; kept entries **unchanged**, others `-inf`
- do **not** modify `logits` in place

*#32 samples only from the top-k tokens; #28 MoE routes each token to its top-k experts.*
''',
            "stub": r'''
def keep_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def keep_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.shape[-1])
    vals, idx = logits.topk(k, dim=-1)
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, vals)
''',
            "debug": r'''
logits = torch.tensor([[0.1, 2.0, -1.0, 3.0, 0.5],
                       [9.0, 8.0, 7.0, 6.0, 5.0]])
print(keep_topk(logits, 2))
# expect [[-inf, 2., -inf, 3., -inf],
#         [ 9.,  8., -inf, -inf, -inf]]
''',
        },
        {
            "spec": r'''
### Exercise 5: `unsort(sorted_values, sort_idx)`

Undo a sort: given `sorted_values, sort_idx = x.sort(dim=-1)` (possibly after editing `sorted_values`), put each value back at its original position.

- `sorted_values`: `(..., V)`
- `sort_idx`: `(..., V)` long, from `x.sort(dim=-1)`; `sorted_values[..., j]` came from position `sort_idx[..., j]`
- returns: `(..., V)` with `out[..., sort_idx[..., j]] = sorted_values[..., j]`

*Top-p sampling (#32) sorts the logits, masks the tail in sorted order, then has to put them back.*
''',
            "stub": r'''
def unsort(sorted_values: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def unsort(sorted_values: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(sorted_values).scatter(-1, sort_idx, sorted_values)
''',
            "debug": r'''
x = torch.tensor([0.3, -1.0, 2.0, 0.0])
s, i = x.sort(descending=True)
print(s, i)
print(unsort(s, i))                      # expect the original x
s[2:] = float("-inf")                    # drop the two smallest, in sorted order
print(unsort(s, i))                      # expect tensor([0.3, -inf, 2.0, -inf])
''',
        },
        {
            "spec": r'''
### Exercise 6: `group_mean(values, group_ids)`

Replace every value with the mean of its group.

- `values`: `(B,)` float
- `group_ids`: `(B,)` integer ids; arbitrary values, not necessarily `0 … G-1` or sorted
- returns: `(B,)` with `out[i] = mean of values[j] over all j with group_ids[j] == group_ids[i]`
- same dtype as `values`, gradients must flow back to `values`, and **no Python loop over groups**

*#38 GRPO computes this (and a per-group std) to turn rewards into advantages.*
''',
            "stub": r'''
def group_mean(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def group_mean(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    uniq, inv = torch.unique(group_ids, return_inverse=True)
    sums = torch.zeros(len(uniq), dtype=values.dtype, device=values.device).index_add(0, inv, values)
    counts = torch.bincount(inv, minlength=len(uniq)).to(values.dtype)
    return (sums / counts)[inv]
''',
            "debug": r'''
print(group_mean(torch.tensor([1., 2., 3., 4., 5.]), torch.tensor([7, 3, 7, 10, 3])))
# expect tensor([2.0000, 3.5000, 2.0000, 4.0000, 3.5000])
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (9, "`attention_mask(n, 'causal')` plus `apply_mask`"),
        (11, "`attention_mask(n, 'sliding', w)`; `'causal_sliding'` is the Mistral variant"),
        (13, "The causal mask and `masked_fill`, inside a full transformer block"),
        (14, "A causal mask shifted by the cache length: `torch.triu(..., diagonal=S_past + 1)` (lesson §4)"),
        (16, "`pick_per_row`: the log-probability of the correct class"),
        (18, "`weight[indices]`: the embedding lookup (lesson §2)"),
        (24, "Strided slicing `x[..., 0::2]` and `x[..., 1::2]` for RoPE pairs (lesson §1)"),
        (28, "`topk` routing, then boolean-mask reads and writes like `output[mask] += ...` (lesson §3)"),
        (32, "`keep_topk` for top-k, then `sort`, mask, and `unsort` for top-p"),
        (38, "`group_mean`: per-group statistics with `unique` / `bincount` / `index_add`, no loop over groups"),
    ],
}
