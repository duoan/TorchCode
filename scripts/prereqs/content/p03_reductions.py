"""Prerequisite 3: Reductions along dims.

Content for templates/00_p03_reductions.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P3",
    "file": "00_p03_reductions",
    "task_id": "p03_reductions",
    "title": "Reductions Along Dims",
    "intro": r'''
# 🔵 Prerequisite 3: Reductions Along Dims

Softmax, LayerNorm, BatchNorm, RMSNorm, gradient clipping, quantization scales, and GRPO advantages all **reduce** a tensor along some dimension (sum, mean, max, variance, norm), then usually **broadcast the result back** against the original.

Two choices decide whether that works: **which `dim`** you reduce over, and whether you keep it with **`keepdim=True`**. A third choice, **biased vs unbiased variance**, silently changes your numbers.

### What you'll learn
1. What `dim` means: the dimension that **disappears**
2. `keepdim=True` so the result broadcasts back
3. `max` / `min` return `(values, indices)`; `amax` / `argmax`
4. `var` / `std`: the unbiased default and when you need `unbiased=False`
5. Reducing over several dims at once
6. Norms, and reducing across a *list* of tensors

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `stable_shift(x)` | #02 Softmax, #25 Flash Attention |
| 2 | `normalize_rows(x)` | #02 Softmax, #12 Linear Attention |
| 3 | `channel_stats(x)` | #07 BatchNorm |
| 4 | `masked_mean(x, mask)` | #41 OPD Loss, sequence pooling |
| 5 | `global_l2_norm(tensors)` | #21 Gradient Clipping |

### Rules
- **No Python loops** over tensor elements or channels. (Exercise 5 may loop over the *list* of tensors.)
- Do **not** use `torch.softmax`, `F.normalize`, `torch.nn.utils.clip_grad_norm_`, or `torch.nn.functional` normalization layers.
''',
    "lesson": [
        ("md", r'''
## 1. `dim` is the dimension that disappears

For a `(B, N)` tensor, `x.sum(dim=-1)` adds **along** the last dim, giving one number per row: shape `(B,)`.
`x.sum(dim=0)` adds down the columns, giving one number per column: shape `(N,)`.
'''),
        ("code", r'''
x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
print(x.sum(dim=-1))   # tensor([ 6., 15.])     one per row
print(x.sum(dim=0))    # tensor([5., 7., 9.])   one per column
print(x.sum())         # tensor(21.)            everything -> 0-d tensor
'''),
        ("md", r'''
## 2. `keepdim=True` to broadcast back

Without `keepdim`, a `(B, N)` → `(B,)` result can't be subtracted from `x` row-wise. Broadcasting aligns from the **right**, so `(B,)` lines up with `N`.
With `keepdim=True` you get `(B, 1)`, which broadcasts correctly. *(See P2 Broadcasting.)*
'''),
        ("code", r'''
x = torch.randn(3, 3)
wrong = x - x.mean(dim=-1)                 # (3,3) - (3,) -> subtracts per COLUMN, silently!
right = x - x.mean(dim=-1, keepdim=True)   # (3,3) - (3,1) -> per row
print(right.mean(dim=-1))                  # ~0 for every row
print(wrong.mean(dim=-1))                  # not 0
'''),
        ("md", r'''
## 3. `max` returns values *and* indices

`x.max(dim=...)` returns a named tuple `(values, indices)`. Use `.values`, or unpack it. `x.amax(dim=...)` returns only the values (and accepts several dims); `x.argmax(dim=...)` returns only the indices.
'''),
        ("code", r'''
x = torch.tensor([[3., 9., 1.],
                  [7., 2., 8.]])
vals, idx = x.max(dim=-1)
print(vals, idx)                           # tensor([9., 8.]) tensor([1, 2])
print(x.max(dim=-1, keepdim=True).values)  # (2, 1)
print(x.amax(dim=-1), x.argmax(dim=-1))
'''),
        ("md", r'''
## 4. ⚠️ Variance: unbiased by default

`x.var()` and `x.std()` divide by `N - 1` by default (the *unbiased* sample estimate).
**Normalization layers divide by `N`**: LayerNorm, BatchNorm's normalization, and RMSNorm. Pass `unbiased=False` (or `correction=0`).
With small `N` the difference is large; tests will catch it.
'''),
        ("code", r'''
x = torch.tensor([1., 2., 3., 4.])
print(x.var())                    # 1.6667  divides by N-1 = 3
print(x.var(unbiased=False))      # 1.25    divides by N   = 4
print(((x - x.mean()) ** 2).mean())   # 1.25, the definition
'''),
        ("md", r'''
## 5. Several dims at once

Pass a tuple: `x.mean(dim=(0, 2, 3))` reduces a `(B, C, H, W)` tensor to one value per channel `(C,)`. That is how BatchNorm2d computes its statistics.
'''),
        ("code", r'''
x = torch.randn(8, 3, 4, 4)
print(x.mean(dim=(0, 2, 3)).shape)                  # torch.Size([3])
print(x.amax(dim=(-2, -1), keepdim=True).shape)     # torch.Size([8, 3, 1, 1])
'''),
        ("md", r'''
## 6. Norms and lists of tensors

`x.norm()` is the L2 norm of **all** elements: `sqrt(sum(x**2))`. `x.norm(dim=-1)` gives one norm per row.
To reduce across a **list** of differently-shaped tensors (like every parameter's gradient), reduce each one to a scalar, then combine the scalars. Never concatenate everything into one huge tensor.
'''),
        ("code", r'''
grads = [torch.randn(4, 4), torch.randn(10)]
per_tensor = torch.stack([g.pow(2).sum() for g in grads])  # one scalar each
print(per_tensor.sum().sqrt())
print(torch.cat([g.flatten() for g in grads]).norm())      # same value, but copies everything
'''),
    ],
    "exercises_intro": r'''
---
# ✏️ Exercises

Implement the five functions below. Each one has a debug cell you can use before you submit.
''',
    "exercises": [
        {
            "spec": r'''
### Exercise 1: `stable_shift(x)`

Subtract each row's maximum from that row, so the largest value in every row becomes `0`.

- `x`: any shape; "row" means the **last** dim
- returns: same shape as `x`

*Softmax is invariant to this shift, and it prevents `exp` from overflowing (#02). Flash Attention does the same with a running max (#25).*
''',
            "stub": r'''
def stable_shift(x: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def stable_shift(x: torch.Tensor) -> torch.Tensor:
    return x - x.max(dim=-1, keepdim=True).values
''',
            "debug": r'''
x = torch.tensor([[1., 1000., 3.],
                  [5., 2., 4.]])
print(stable_shift(x))
# expect [[-999.,    0., -997.],
#         [   0.,   -3.,   -1.]]
''',
        },
        {
            "spec": r'''
### Exercise 2: `normalize_rows(x)`

Scale each row so it sums to `1`.

- `x`: positive values, any shape; "row" means the **last** dim
- returns: same shape, with `out.sum(dim=-1) == 1` everywhere

*Softmax is `normalize_rows(exp(stable_shift(x)))`. Linear attention divides by a summed normalizer the same way (#12).*
''',
            "stub": r'''
def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True)
''',
            "debug": r'''
x = torch.tensor([[1., 1., 2.],
                  [3., 1., 0.]])
p = normalize_rows(x)
print(p)               # expect [[0.25, 0.25, 0.50], [0.75, 0.25, 0.00]]
print(p.sum(dim=-1))   # expect tensor([1., 1.])
''',
        },
        {
            "spec": r'''
### Exercise 3: `channel_stats(x)`

Compute per-channel mean and **biased** variance of a batch of images, as BatchNorm2d does in training mode.

- `x`: `(B, C, H, W)`
- returns: a tuple `(mean, var)`, each of shape `(C,)`, reducing over batch **and** spatial dims
- use the **biased** variance (divide by the number of elements, not by that number minus 1)

*#07 BatchNorm does the same for `(N, D)` inputs, reducing over dim 0 only.*
''',
            "stub": r'''
def channel_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass  # Replace this
''',
            "solution": r'''
def channel_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dims = (0, 2, 3)
    return x.mean(dim=dims), x.var(dim=dims, unbiased=False)
''',
            "debug": r'''
x = torch.zeros(2, 3, 2, 2)
x[:, 1] = 5.0                      # channel 1 is constant 5
x[0, 2], x[1, 2] = 1.0, 3.0        # channel 2 is half 1s, half 3s
mean, var = channel_stats(x)
print(mean)   # expect tensor([0., 5., 2.])
print(var)    # expect tensor([0., 0., 1.])   (unbiased would give 1.1429)
''',
        },
        {
            "spec": r'''
### Exercise 4: `masked_mean(x, mask)`

Average token vectors over the sequence, counting only the **valid** positions.

- `x`: `(B, S, D)`
- `mask`: `(B, S)`, either `bool` or `0/1` floats; `True` / `1` = valid
- returns: `(B, D)`, where `out[b] = mean of x[b, s] over valid s`
- if a sequence has **no** valid positions, return zeros for it (not `NaN`)

*#41 OPD averages a per-token loss over unmasked tokens with exactly this pattern.*
''',
            "stub": r'''
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype).unsqueeze(-1)            # (B, S, 1)
    total = (x * m).sum(dim=1)                    # (B, D)
    count = m.sum(dim=1).clamp_min(1.0)           # (B, 1); avoid 0 / 0
    return total / count
''',
            "debug": r'''
x = torch.tensor([[[1., 10.], [3., 30.], [100., 100.]],
                  [[5., 50.], [7., 70.], [9., 90.]]])
mask = torch.tensor([[True, True, False],
                     [False, False, False]])
print(masked_mean(x, mask))
# expect [[2., 20.],    mean of the first two tokens
#         [0.,  0.]]    no valid tokens -> zeros
''',
        },
        {
            "spec": r'''
### Exercise 5: `global_l2_norm(tensors)`

Compute the L2 norm of **all** elements of **all** tensors in a list, as if they were concatenated into one long vector.

- `tensors`: a list of tensors with different shapes
- returns: a 0-d tensor, `sqrt(sum over every tensor of sum(t ** 2))`

*This is the total gradient norm in #21 Gradient Clipping.*
''',
            "stub": r'''
def global_l2_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def global_l2_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([t.pow(2).sum() for t in tensors]).sum().sqrt()
''',
            "debug": r'''
ts = [torch.tensor([3., 0.]), torch.tensor([[0., 4.]])]
print(global_l2_norm(ts))   # expect tensor(5.)
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (2, "`stable_shift` then `normalize_rows`: softmax is exactly these two steps around `exp`"),
        (4, "Mean and **biased** variance over the last dim, with `keepdim=True`"),
        (7, "`channel_stats`: batch statistics with `unbiased=False`"),
        (8, "`x.pow(2).mean(dim=-1, keepdim=True)`: the mean square inside RMSNorm"),
        (12, "Dividing by a summed normalizer, like `normalize_rows`"),
        (21, "`global_l2_norm`: the total norm across all parameter gradients"),
        (25, "`stable_shift` with a running max: the online-softmax trick"),
        (36, "Per-row `abs().amax(dim=1, keepdim=True)` for the quantization scale"),
        (38, "Per-group mean and **biased** std of rewards for the advantages"),
        (41, "`masked_mean`: average a per-token loss over valid tokens only"),
    ],
}
