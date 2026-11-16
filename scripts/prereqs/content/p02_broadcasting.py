"""Prerequisite 2: Broadcasting.

Content for templates/00_p02_broadcasting.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P2",
    "file": "00_p02_broadcasting",
    "task_id": "p02_broadcasting",
    "title": "Broadcasting",
    # Shape symbols used in this notebook, shown in the "📐 Notation" box
    "notation": [
        ('B', 'batch size'),
        ('C', 'channels'),
        ('H', 'image height'),
        ('W', 'image width'),
        ('L', 'length of a 1-D signal, in the `(B, C, L)` example'),
        ('N', 'number of rows'),
        ('M', 'number of rows of the second set (pairwise distances)'),
        ('D', 'feature dimension'),
        ('S', 'sequence length (number of positions)'),
        ('F', 'number of frequencies (RoPE)'),
        ('S_past', 'number of positions already in a KV cache'),
        ('S_new', 'number of new query positions'),
    ],
    "intro": r'''
# 🔵 Prerequisite 2: Broadcasting

Almost every problem in TorchCode quietly relies on **broadcasting**: adding a bias of shape `(C,)` to a feature map of shape `(B, C, H, W)`, scaling normalized activations by `gamma`, building an `|i - j|` mask for sliding-window attention, or multiplying positions by frequencies in RoPE.

When you get broadcasting right, the code is short and fast. When you get it wrong, PyTorch often **does not raise an error**. It silently produces a tensor of the wrong shape.

### What you'll learn
1. The broadcasting rule (align from the right)
2. Adding size-1 dimensions on purpose (`unsqueeze`, `None`, `view`)
3. "Outer" operations: combining every element of `a` with every element of `b`
4. The classic silent bug: `(N,)` vs `(N, 1)`

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `add_channel_bias(x, b)` | #22 Conv2d |
| 2 | `scale_rows(x, s)` | #04 LayerNorm, #07 BatchNorm, #28 MoE |
| 3 | `abs_distance_matrix(n)` | #11 Sliding Window Attention |
| 4 | `relative_position(n)` | #09 Causal Attention, #11 Sliding Window, #13 GPT-2 Block |
| 5 | `outer_angles(positions, freqs)` | #24 RoPE |
| 6 | `pairwise_sq_dists(a, b)` | k-NN, clustering, contrastive losses |

### Rules
- **No Python loops** over tensor elements. Every exercise is a one- or two-liner.
- **Return new tensors**: don't modify the inputs (the tests check).
- Do **not** use `torch.outer`, `torch.cdist`, `torch.einsum`, or `torch.Tensor.expand`. The point is to make broadcasting do the work.
''',
    "lesson": [
        ("md", r'''
## 1. The rule

Two shapes are compatible if, **comparing from the rightmost dimension**, each pair of sizes is either equal, or one of them is `1`, or one of them is missing.
The result takes the larger size in each position.

```
x:      (4, 3, 5)
y:         (3, 1)
result: (4, 3, 5)     # y is treated as (1, 3, 1) and stretched
```

No data is copied. The size-1 dimension is "stretched" virtually.
'''),
        ("code", r'''
x = torch.randn(4, 3, 5)
y = torch.randn(3, 1)
print((x + y).shape)          # torch.Size([4, 3, 5])

# Incompatible: compare right-to-left: 5 vs 3 -> neither is 1 -> error
try:
    torch.randn(4, 5) + torch.randn(3)
except RuntimeError as e:
    print("RuntimeError:", str(e)[:80])
'''),
        ("md", r'''
## 2. Adding size-1 dimensions on purpose

Often the dimensions you want to match are **not** the rightmost ones. Then you insert size-1 dimensions so the shapes line up the way you mean.
All three spellings below are equivalent:

| Spelling | Result for `b` of shape `(C,)` |
|----------|-------------------------------|
| `b.unsqueeze(0).unsqueeze(-1)` | `(1, C, 1)` |
| `b[None, :, None]` | `(1, C, 1)` |
| `b.view(1, -1, 1)` | `(1, C, 1)` |

All three return **views** of `b`: nothing is copied and there's no meaningful speed difference, so pick whichever reads most clearly.
One caution about `view`: `-1` means "infer this size", so on a **multi-dimensional** `x`, `x.view(1, -1, 1)` **flattens** the existing dimensions into one (and fails if `x` is non-contiguous, see P5 §1). When you only want to *add* dimensions, `unsqueeze` or `None` say so directly.

Inside `[...]`, **each `None` adds exactly one dimension** and each `:` keeps one existing dimension. Dimensions you don't mention at the end are kept as they are. So for `a` of shape `(N, D)`, `a[:, None]` is `(N, 1, D)` (the `D` is kept implicitly), while `a[None, :, None]` is `(1, N, 1, D)`, which is 4-D, not 3-D.
'''),
        ("code", r'''
x = torch.randn(2, 3, 4)       # (B, C, L)
b = torch.tensor([10., 20., 30.])  # one value per channel C

# x + b  would try to align b with L=4 -> error. Line it up with C instead:
out = x + b[None, :, None]
print(out.shape)
print(out[0, :, 0] - x[0, :, 0])   # tensor([10., 20., 30.])

a = torch.randn(5, 2)              # (N, D)
print(a[:, None].shape)            # torch.Size([5, 1, 2]): one None, D kept
print(a[None, :, None].shape)      # torch.Size([1, 5, 1, 2]): two Nones -> 4-D
'''),
        ("md", r'''
## 3. Outer operations

Combining a column `(N, 1)` with a row `(1, M)` gives an `(N, M)` grid where entry `[i, j]` uses `a[i]` and `b[j]`.
The **column** supplies the row index `i`; the **row** supplies the column index `j`.
This is how you build attention masks, position-frequency tables, and pairwise distances **without loops**.
'''),
        ("code", r'''
i = torch.arange(3)                    # 3 rows
j = torch.arange(4)                    # 4 columns
grid = i[:, None] * 10 + j[None, :]    # (3, 1) op (1, 4) -> (3, 4)
print(grid)
# [[ 0,  1,  2,  3],
#  [10, 11, 12, 13],
#  [20, 21, 22, 23]]   tens digit = row i, ones digit = column j

# A causal mask is the same outer comparison, of POSITIONS: query at position p may see
# key at position k only if k <= p. Its shape is (num queries, num keys), which is only
# square in plain self-attention. With a KV cache (#14), 2 new queries attend to 5 keys:
S_past, S_new = 3, 2
q = torch.arange(S_past, S_past + S_new)   # query positions: 3, 4
k = torch.arange(S_past + S_new)           # key positions:   0, 1, 2, 3, 4
causal = k[None, :] > q[:, None]           # (2, 5): True where the key is in the query's future
print(causal)
# [[False, False, False, False,  True],
#  [False, False, False, False, False]]
'''),
        ("md", r'''
## 4. ⚠️ The silent bug: `(N,)` vs `(N, 1)`

A model's prediction often has shape `(N, 1)` while the target has shape `(N,)`.
Subtracting them **does not fail**. It broadcasts to `(N, N)`, and your loss is then averaged over the wrong thing.
'''),
        ("code", r'''
pred = torch.randn(5, 1)
target = torch.randn(5)
diff = pred - target
print("pred - target shape:", diff.shape, "<- (5, 5)! almost certainly a bug")
print("fixed:", (pred.squeeze(-1) - target).shape)
'''),
        ("md", r'''
> **Habit to build:** after every broadcasting op you're not 100% sure about, `print(t.shape)`. Assert shapes in your own code: `assert out.shape == (B, C, H, W)`.
'''),
    ],
    "exercises_intro": r'''
---
# ✏️ Exercises

Implement the six functions below. Each one has a debug cell you can use to check shapes before you submit.
''',
    "exercises": [
        {
            "spec": r'''
### Exercise 1: `add_channel_bias(x, b)`

Add a per-channel bias to a batch of images.

- `x`: `(B, C, H, W)`
- `b`: `(C,)`
- returns: `(B, C, H, W)` where `out[n, c, h, w] = x[n, c, h, w] + b[c]`

*This is exactly the last line of #22 Conv2d.*
''',
            "stub": r'''
def add_channel_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def add_channel_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return x + b.view(1, -1, 1, 1)
''',
            "debug": r'''
x = torch.zeros(2, 3, 4, 4)
b = torch.tensor([1., 2., 3.])
out = add_channel_bias(x, b)
print(out.shape)          # expect torch.Size([2, 3, 4, 4])
print(out[0, :, 0, 0])    # expect tensor([1., 2., 3.])
''',
        },
        {
            "spec": r'''
### Exercise 2: `scale_rows(x, s)`

Multiply each row of a matrix by its own scalar.

- `x`: `(N, D)`
- `s`: `(N,)`
- returns: `(N, D)` where `out[i, :] = s[i] * x[i, :]`

*In #28 MoE, each token's expert output is scaled by its routing weight in exactly this way.*
''',
            "stub": r'''
def scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def scale_rows(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return x * s[:, None]
''',
            "debug": r'''
x = torch.ones(3, 4)
s = torch.tensor([1., 2., 3.])
print(scale_rows(x, s))   # rows of 1s, 2s, 3s
# Hint: what does x * s do when N == D? Try torch.ones(3, 3) * s.
''',
        },
        {
            "spec": r'''
### Exercise 3: `abs_distance_matrix(n)`

Build the `(n, n)` matrix of absolute index distances `D[i, j] = |i - j|`.

- `n`: int
- returns: `(n, n)` integer tensor (dtype `torch.long`)

*#11 Sliding Window Attention masks every position where `|i - j| > window_size`.*
''',
            "stub": r'''
def abs_distance_matrix(n: int) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def abs_distance_matrix(n: int) -> torch.Tensor:
    idx = torch.arange(n)
    return (idx[:, None] - idx[None, :]).abs()
''',
            "debug": r'''
print(abs_distance_matrix(5))
# Expected:
# [[0, 1, 2, 3, 4],
#  [1, 0, 1, 2, 3],
#  [2, 1, 0, 1, 2],
#  [3, 2, 1, 0, 1],
#  [4, 3, 2, 1, 0]]
''',
        },
        {
            "spec": r'''
### Exercise 4: `relative_position(n)`

The **signed** version of Exercise 3: `R[i, j] = i - j`, i.e. how far key `j` is *behind* query `i`.

- `n`: int
- returns: `(n, n)` integer tensor (dtype `torch.long`)
- `R[i, j] > 0`: key `j` is in the **past** of query `i`
- `R[i, j] == 0`: the query itself
- `R[i, j] < 0`: key `j` is in the **future**

**Why the sign matters:** `abs_distance_matrix` can't tell past from future. With `R`, every attention mask in TorchCode is a single comparison:

- **Causal** (#09, #13): mask where `R < 0`
- **Two-sided sliding window**, Longformer-style (#11 as written): mask where `R.abs() > w`
- **Causal sliding window**, Mistral-style: mask where `(R < 0) | (R > w)`

*Applying masks with `masked_fill` is covered in P4 Indexing & masking. Here you only build the grid.*
''',
            "stub": r'''
def relative_position(n: int) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def relative_position(n: int) -> torch.Tensor:
    idx = torch.arange(n)
    return idx[:, None] - idx[None, :]
''',
            "debug": r'''
R = relative_position(5)
print(R)
# Expected:
# [[ 0, -1, -2, -3, -4],
#  [ 1,  0, -1, -2, -3],
#  [ 2,  1,  0, -1, -2],
#  [ 3,  2,  1,  0, -1],
#  [ 4,  3,  2,  1,  0]]

# Every mask is one comparison on R (1 = masked out):
w = 1
print("causal:\n", (R < 0).int())
print("two-sided window (Longformer):\n", (R.abs() > w).int())
print("causal window (Mistral):\n", ((R < 0) | (R > w)).int())
''',
        },
        {
            "spec": r'''
### Exercise 5: `outer_angles(positions, freqs)`

Build the table of rotation angles used by RoPE: `angles[p, k] = positions[p] * freqs[k]`.

- `positions`: `(S,)` float
- `freqs`: `(F,)` float
- returns: `(S, F)`

*This is the `angles = pos * freqs` line of #24 RoPE. It looks repetitive, but each column grows at a different speed: the first frequency is 1, the next ones are 10×, 100×, … smaller. Like clock hands moving at different speeds, the columns together encode position at many scales.*
''',
            "stub": r'''
def outer_angles(positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def outer_angles(positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    return positions[:, None] * freqs[None, :]
''',
            "debug": r'''
pos = torch.arange(4).float()
freqs = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
ang = outer_angles(pos, freqs)
print(ang.shape)   # expect torch.Size([4, 4])
torch.set_printoptions(sci_mode=False, precision=4)   # scientific notation hides that the columns differ
print(freqs)   # expect [1.0, 0.1, 0.01, 0.001]: each column 10x slower
print(ang)     # row p = position p times each frequency
torch.set_printoptions(profile="default")            # back to normal printing
''',
        },
        {
            "spec": r'''
### Exercise 6: `pairwise_sq_dists(a, b)`

Compute squared Euclidean distances between **every** row of `a` and **every** row of `b`.

- `a`: `(N, D)`
- `b`: `(M, D)`
- returns: `(N, M)` where `out[i, j] = sum_d (a[i, d] - b[j, d])**2`: **one row per point of `a`, one column per point of `b`**

*Hint: build a 3-D intermediate of shape `(N, M, D)`: axis 0 picks a point of `a`, axis 1 a point of `b`, axis 2 a coordinate. To get there, insert **one** size-1 dimension into each input, in **different** places, so their point dimensions spread out against each other while their coordinates stay aligned.*

*Squaring the intermediate in place (`diff.square_()`) is fine: `diff` is a tensor you just created. The rule against in-place ops is about the caller's inputs `a` and `b` (P1 §6).*
''',
            "stub": r'''
def pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a[:, None, :] - b[None, :, :]   # (N, M, D)
    return (diff ** 2).sum(dim=-1)
''',
            "debug": r'''
a = torch.tensor([[0., 0.], [1., 1.]])
b = torch.tensor([[0., 0.], [3., 4.], [1., 0.]])
print(pairwise_sq_dists(a, b))
# Expected (rows = points of a, columns = points of b):
#            b[0]=[0,0]  b[1]=[3,4]  b[2]=[1,0]
# a[0]=[0,0]     0.         25.          1.
# a[1]=[1,1]     2.         13.          1.
# e.g. 25 = (0 - 3)**2 + (0 - 4)**2
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (4, "`gamma * x_norm + beta` broadcasts per-feature params over the batch; `keepdim=True` stats broadcast back"),
        (8, "Dividing by a `(..., 1)` RMS and scaling by a `(D,)` weight"),
        (7, "Per-feature batch statistics `(D,)` broadcast across the batch dim"),
        (9, "`relative_position`: the causal mask is `R < 0` (keys in the future)"),
        (11, "`abs_distance_matrix`: mask keys more than `window_size` away (`abs(i - j) > window_size`); `relative_position` gives the causal, Mistral-style variant"),
        (13, "`relative_position`: the same causal mask, inside a full transformer block"),
        (22, "`add_channel_bias`: the final `out + bias.view(1, -1, 1, 1)`"),
        (24, "`outer_angles`: the `pos * freqs` table of rotation angles"),
        (28, "`scale_rows`: each token's expert output times its routing weight"),
        (42, "Reshaping per-channel `(C,)` statistics to `(1, C, 1, 1)`, like `add_channel_bias`"),
        (40, "The `(N,)` vs `(N, 1)` trap: `layer(X)` is `(N, 1)`, the target is `(N,)`"),
    ],
}
