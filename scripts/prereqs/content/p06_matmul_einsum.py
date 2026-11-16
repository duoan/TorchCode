"""Prerequisite 6: Matmul & einsum.

Content for templates/00_p06_matmul_einsum.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P6",
    "file": "00_p06_matmul_einsum",
    "task_id": "p06_matmul_einsum",
    "title": "Matmul & Einsum",
    # Shape symbols used in this notebook, shown in the "📐 Notation" box
    "notation": [
        ('B', 'batch size'),
        ('S', 'sequence length'),
        ('S_q', 'number of queries'),
        ('S_k', 'number of keys'),
        ('H', 'number of attention heads (lessons §2–4); **image height** in `patch_conv`'),
        ('d', 'dimension of each query / key vector (per head)'),
        ('d_k', 'query / key dimension'),
        ('d_v', 'value dimension'),
        ('W', 'a weight matrix `(out_features, in_features)`; `W_out` is an output **width**'),
        ('N', 'number of rows / examples'),
        ('r', 'LoRA rank'),
        ('C_in', 'input channels'),
        ('C_out', 'output channels'),
        ('kH', 'kernel height'),
        ('kW', 'kernel width'),
        ('H_out', 'output height'),
        ('W_out', 'output width'),
    ],
    "intro": r'''
# 🔵 Prerequisite 6: Matmul & Einsum

Linear layers, attention scores, attention outputs, LoRA, convolution, and linear regression are all **matrix multiplications** with a few extra dimensions in front.
Getting them right is mostly about **shape bookkeeping**: which dims are batch dims, which dim gets contracted, and where a transpose goes.

### What you'll learn
1. The `x @ W.T + b` convention of `nn.Linear`
2. `@` / `matmul` vs `mm` vs `bmm`, and batch broadcasting
3. Transposing the last two dims: `K.transpose(-2, -1)`
4. `einsum`: name every dim, and the contraction becomes obvious
5. Associativity and cost: `(A @ B) @ C` vs `A @ (B @ C)`
6. Solving linear systems without inverting a matrix

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `linear_forward(x, W, b)` | #03 Linear, #36 INT8, #40 Linear Regression |
| 2 | `attention_scores(Q, K)` | #05, #06, #09, #23 attention |
| 3 | `patch_conv(patches, weight)` | #22 Conv2d |
| 4 | `qkv_reassociated(Q, K, V)` | #12 Linear Attention |
| 5 | `lora_forward(x, W, A, B, scaling)` | #26 LoRA |

### Rules
- **No Python loops** over tensor elements, batches, or heads.
- Do **not** use `F.linear`, `nn.Linear`, `F.conv2d`, or `F.scaled_dot_product_attention`.
''',
    "lesson": [
        ("md", r'''
## 1. `x @ W.T + b`

`nn.Linear(in_features, out_features)` stores its weight as **`(out_features, in_features)`**, so its forward pass is `x @ W.T + b`.
`x` can have any number of leading dims: `(N, in)`, `(B, S, in)`, … The matmul contracts the **last** dim of `x` with the **last** dim of `W`.
'''),
        ("code", r'''
lin = torch.nn.Linear(4, 3)
print(lin.weight.shape)                                  # torch.Size([3, 4]) = (out, in)
x = torch.randn(2, 5, 4)
print(torch.allclose(lin(x), x @ lin.weight.T + lin.bias, atol=1e-6))
'''),
        ("md", r'''
## 2. `@` vs `mm` vs `bmm`

| Op | Inputs | Notes |
|----|--------|-------|
| `torch.mm(a, b)` | exactly 2-D | `(n, k) @ (k, m)` |
| `torch.bmm(a, b)` | exactly 3-D, same batch | `(B, n, k) @ (B, k, m)` |
| `a @ b` / `torch.matmul` | any dims ≥ 1 | leading dims **broadcast**: `(B, H, n, k) @ (B, H, k, m)` or even `(B, H, n, k) @ (k, m)` |

`@` is the safe default. `bmm` is fine when you know you have exactly 3-D tensors.
'''),
        ("code", r'''
a = torch.randn(2, 3, 5, 4)      # (B, H, n, k)
b = torch.randn(2, 3, 4, 6)      # (B, H, k, m)
print((a @ b).shape)             # (2, 3, 5, 6)
print((a @ torch.randn(4, 6)).shape)   # the (4, 6) matrix broadcasts over B and H
try:
    torch.bmm(a, b)
except RuntimeError as e:
    print("bmm needs 3-D:", str(e)[:50])
'''),
        ("md", r'''
## 3. Transpose the last two dims

To compare every query with every key you need `Q @ Kᵀ` over the **last two** dims only.
`K.T` is wrong for batched tensors: it reverses **all** dims. Use `K.transpose(-2, -1)` (or `K.mT`).
'''),
        ("code", r'''
Q = torch.randn(2, 5, 8)         # (B, S_q, d)
K = torch.randn(2, 7, 8)         # (B, S_k, d)
print(K.transpose(-2, -1).shape) # (2, 8, 7)
print((Q @ K.transpose(-2, -1)).shape)   # (2, 5, 7): one score per (query, key)
'''),
        ("md", r'''
## 4. `einsum`

Give every dim a letter. Letters in the inputs that are **missing from the output are summed over**; all others are kept.

| Expression | Meaning |
|------------|---------|
| `"ij,jk->ik"` | matrix multiply |
| `"bij,bjk->bik"` | batched matmul (`bmm`) |
| `"bhqd,bhkd->bhqk"` | attention scores per head, without any transpose |
| `"i,j->ij"` | outer product |
| `"ii->"` | trace |
'''),
        ("code", r'''
Q = torch.randn(2, 3, 5, 8)      # (b, h, q, d)
K = torch.randn(2, 3, 7, 8)      # (b, h, k, d)
s1 = torch.einsum("bhqd,bhkd->bhqk", Q, K)
s2 = Q @ K.transpose(-2, -1)
print(s1.shape, torch.allclose(s1, s2, atol=1e-5))
'''),
        ("md", r'''
## 5. Associativity and cost

Matrix products are associative, so `(A @ B) @ C == A @ (B @ C)`, but the **cost** can differ enormously.
With `Q, K, V` of shape `(S, d)`:
- `(Q @ Kᵀ) @ V` builds an `(S, S)` matrix: about `2·S²·d` multiply-adds, `S²` memory.
- `Q @ (Kᵀ @ V)` builds a `(d, d)` matrix: about `2·S·d²`. For long sequences (`S ≫ d`) this is far cheaper.

That is the core idea of linear attention (#12). LoRA (#26) uses the same trick: `(x @ Aᵀ) @ Bᵀ` never forms the large `B @ A`.
'''),
        ("code", r'''
import time
S, d = 4096, 32
Q, K, V = torch.randn(S, d), torch.randn(S, d), torch.randn(S, d)
t0 = time.perf_counter(); left = (Q @ K.T) @ V; t1 = time.perf_counter()
right = Q @ (K.T @ V); t2 = time.perf_counter()
print(f"(QK^T)V: {1000*(t1-t0):.1f} ms   Q(K^TV): {1000*(t2-t1):.1f} ms")
print(torch.allclose(left, right, rtol=1e-3, atol=1e-2))
'''),
        ("md", r'''
## 6. Solving linear systems

To solve `A w = b`, **don't** compute `torch.linalg.inv(A) @ b`. It is slower and less accurate. Instead:

| Problem | Use |
|---------|-----|
| Square system `A w = b` | `torch.linalg.solve(A, b)` |
| Least squares: minimize `‖X w − y‖²` with more rows than columns | `torch.linalg.lstsq(X, y).solution` |

The textbook normal equation `(XᵀX) w = Xᵀy` gives the same answer, but forming `XᵀX` **squares the condition number** of the problem, so it loses precision quickly when columns are nearly redundant. `lstsq` works on `X` directly. #40's closed-form method uses it, with a column of ones appended for the bias.
'''),
        ("code", r'''
torch.manual_seed(0)
X = torch.randn(200, 3, dtype=torch.float64)
w_true = torch.tensor([[1.5], [-2.0], [0.5]], dtype=torch.float64)
y = X @ w_true + 0.01 * torch.randn(200, 1, dtype=torch.float64)

w_lstsq = torch.linalg.lstsq(X, y).solution                  # (3, 1)
w_normal = torch.linalg.solve(X.T @ X, X.T @ y)             # same answer here
print(w_lstsq.flatten(), w_normal.flatten())
print(f"cond(X) = {torch.linalg.cond(X):.3g}, cond(X^T X) = {torch.linalg.cond(X.T @ X):.3g}")   # squared

# Nearly redundant columns: the squaring is what hurts
Xc = X.clone()
Xc[:, 2] = Xc[:, 0] + 1e-4 * torch.randn(200, dtype=torch.float64)
print(f"cond(Xc) = {torch.linalg.cond(Xc):.3g}, cond(Xc^T Xc) = {torch.linalg.cond(Xc.T @ Xc):.3g}")
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
### Exercise 1: `linear_forward(x, W, b)`

The forward pass of a linear layer.

- `x`: `(..., in_features)`, any number of leading dims
- `W`: `(out_features, in_features)`, the `nn.Linear` layout
- `b`: `(out_features,)`
- returns: `(..., out_features)`
''',
            "stub": r'''
def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return x @ W.T + b
''',
            "debug": r'''
x = torch.randn(2, 5, 4)
W, b = torch.randn(3, 4), torch.randn(3)
print(linear_forward(x, W, b).shape)   # expect torch.Size([2, 5, 3])
''',
        },
        {
            "spec": r'''
### Exercise 2: `attention_scores(Q, K)`

Scaled dot-product attention scores: `Q @ Kᵀ / sqrt(d)`.

- `Q`: `(..., S_q, d)`
- `K`: `(..., S_k, d)`, same leading dims as `Q` (e.g. `(B, S, d)` or `(B, H, S, d)`)
- returns: `(..., S_q, S_k)`

*`S_q` and `S_k` may differ (cross-attention, #23).*
''',
            "stub": r'''
import math

def attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
import math

def attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
''',
            "debug": r'''
Q = torch.randn(2, 3, 5, 16)     # (B, H, S_q, d)
K = torch.randn(2, 3, 7, 16)     # (B, H, S_k, d)
print(attention_scores(Q, K).shape)   # expect torch.Size([2, 3, 5, 7])
''',
        },
        {
            "spec": r'''
### Exercise 3: `patch_conv(patches, weight)`

#22 Conv2d extracts every `kH × kW` patch with `unfold`. The convolution is then a single contraction. Write it with `torch.einsum`.

- `patches`: `(B, C_in, H_out, W_out, kH, kW)`, where `patches[b, c, h, w]` is the input patch at output position `(h, w)`
- `weight`: `(C_out, C_in, kH, kW)`
- returns: `(B, C_out, H_out, W_out)` with `out[b, o, h, w] = sum over c, i, j of patches[b, c, h, w, i, j] * weight[o, c, i, j]`
''',
            "stub": r'''
def patch_conv(patches: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def patch_conv(patches: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bchwij,ocij->bohw", patches, weight)
''',
            "debug": r'''
x = torch.randn(1, 2, 5, 5)
patches = x.unfold(2, 3, 1).unfold(3, 3, 1)          # (1, 2, 3, 3, 3, 3)
weight = torch.randn(4, 2, 3, 3)
out = patch_conv(patches, weight)
print(out.shape)                                     # expect torch.Size([1, 4, 3, 3])
print(torch.allclose(out, torch.nn.functional.conv2d(x, weight), atol=1e-5))   # expect True
''',
        },
        {
            "spec": r'''
### Exercise 4: `qkv_reassociated(Q, K, V)`

Compute `(Q @ Kᵀ) @ V` **without** building the `(S, S)` matrix: multiply `Kᵀ @ V` first.

- `Q`, `K`: `(B, S, d_k)`
- `V`: `(B, S, d_v)`
- returns: `(B, S, d_v)`, equal to `(Q @ Kᵀ) @ V` up to float rounding

*Without the softmax in between, attention can be re-associated. That is how linear attention (#12) becomes `O(S·d²)`.*
''',
            "stub": r'''
def qkv_reassociated(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def qkv_reassociated(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    KV = K.transpose(-2, -1) @ V        # (B, d_k, d_v): small
    return Q @ KV
''',
            "debug": r'''
Q, K, V = torch.randn(2, 6, 4), torch.randn(2, 6, 4), torch.randn(2, 6, 3)
out = qkv_reassociated(Q, K, V)
print(out.shape, torch.allclose(out, (Q @ K.transpose(-2, -1)) @ V, atol=1e-4))   # expect (2, 6, 3) True
''',
        },
        {
            "spec": r'''
### Exercise 5: `lora_forward(x, W, A, B, scaling)`

A LoRA layer adds a low-rank update to a frozen weight: `y = x Wᵀ + scaling · x (BA)ᵀ`.
Compute it **without forming `B @ A`**, which would be a full `(out, in)` matrix. Go through the rank-`r` bottleneck instead: `(x @ Aᵀ) @ Bᵀ`.

- `x`: `(..., in)`
- `W`: `(out, in)`, the frozen base weight
- `A`: `(r, in)`, `B`: `(out, r)`
- `scaling`: float
- returns: `(..., out)`
''',
            "stub": r'''
def lora_forward(x: torch.Tensor, W: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                 scaling: float) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def lora_forward(x: torch.Tensor, W: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                 scaling: float) -> torch.Tensor:
    return x @ W.T + (x @ A.T) @ B.T * scaling
''',
            "debug": r'''
x = torch.randn(2, 5, 16)
W, A, B = torch.randn(8, 16), torch.randn(2, 16), torch.randn(8, 2)
out = lora_forward(x, W, A, B, scaling=0.5)
print(out.shape)                                                     # expect torch.Size([2, 5, 8])
print(torch.allclose(out, x @ (W + 0.5 * B @ A).T, atol=1e-4))       # expect True
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (3, "`linear_forward`: `x @ W.T + b` with the `(out, in)` weight layout"),
        (5, "`attention_scores`, then `weights @ V`"),
        (6, "`attention_scores` on 4-D `(B, H, S, d)` tensors, one score matrix per head"),
        (9, "`attention_scores` before the causal mask"),
        (12, "`qkv_reassociated`: compute `Kᵀ @ V` first for linear complexity"),
        (22, "`patch_conv`: the convolution as one `einsum` over unfolded patches"),
        (23, "`attention_scores` with `S_q != S_k`"),
        (26, "`lora_forward`: the low-rank update without forming `B @ A`"),
        (36, "`x @ w.T` with the dequantized weight"),
        (40, "`X @ w + b`, `X.T @ error` for the gradient, and `torch.linalg.lstsq` for the closed form (lesson §6)"),
    ],
}
