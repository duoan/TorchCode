"""Prerequisite 5: Shape manipulation.

Content for templates/00_p05_shape_manipulation.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P5",
    "file": "00_p05_shape_manipulation",
    "task_id": "p05_shape_manipulation",
    "title": "Shape Manipulation",
    "intro": r'''
# 🔵 Prerequisite 5: Shape Manipulation

Multi-head attention, GQA, KV cache, cross-attention, the GPT-2 block, RoPE, and ViT patch embedding all rely on the same few moves: **reshape → transpose → (compute) → transpose → reshape**.
In practice, most attention bugs are shape bugs.

### What you'll learn
1. `view` vs `reshape`, and why `.contiguous()` exists
2. `transpose` vs `permute`
3. `squeeze` / `unsqueeze` / `flatten`
4. `cat` vs `stack`
5. `repeat_interleave` vs `repeat`

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `split_heads(x, num_heads)` | #06 MHA, #10 GQA, #13 GPT-2, #14 KV Cache, #23 Cross-Attn |
| 2 | `merge_heads(x)` | same as above (the inverse) |
| 3 | `repeat_kv(x, n_rep)` | #10 GQA |
| 4 | `interleave(a, b)` | #24 RoPE |
| 5 | `patchify(img, patch_size)` | #27 ViT Patch Embedding |

### Rules
- **No Python loops.** Use only view/reshape/transpose/permute/cat/stack/repeat-style ops.
- Do **not** use `einops`.
''',
    "lesson": [
        ("md", r'''
## 1. `view` vs `reshape` and contiguity

A tensor is a flat block of memory plus **strides** that say how far to jump for each dimension.
- `view` only re-interprets the strides. It never copies data, so it **fails** if the memory layout doesn't allow the new shape.
- `transpose` / `permute` just swap strides. The result is usually **non-contiguous**.
- `reshape` = `view` if possible, otherwise copy. `.contiguous()` forces a copy into the standard layout.
'''),
        ("code", r'''
x = torch.arange(6).view(2, 3)
print(x, x.stride(), x.is_contiguous())

xt = x.transpose(0, 1)                       # (3, 2)
print(xt, xt.stride(), xt.is_contiguous())   # strides swapped, not contiguous

try:
    xt.view(6)
except RuntimeError as e:
    print("view failed:", str(e)[:60], "...")
print("reshape works:", xt.reshape(6))
print("contiguous().view works:", xt.contiguous().view(6))
'''),
        ("md", r'''
> **Reading order matters.** `x.view(2, 3)` fills the new shape in **row-major** order (last dim varies fastest). Splitting a dimension `D` into `(H, d)` with `view(..., H, d)` therefore puts the first `d` features into head 0, the next `d` into head 1, and so on.

## 2. `transpose` vs `permute`

- `transpose(i, j)` swaps exactly two dims.
- `permute(...)` lists the new order of **all** dims.
'''),
        ("code", r'''
x = torch.randn(2, 3, 4, 5)
print(x.transpose(1, 2).shape)          # (2, 4, 3, 5)
print(x.permute(0, 2, 3, 1).shape)      # (2, 4, 5, 3)  e.g. NCHW -> NHWC
print(x.transpose(-2, -1).shape)        # (2, 3, 5, 4)  the K^T in attention
'''),
        ("md", r'''
## 3. `squeeze`, `unsqueeze`, `flatten`
'''),
        ("code", r'''
x = torch.randn(3, 1, 4)
print(x.squeeze(1).shape)       # (3, 4): remove a size-1 dim
print(x.unsqueeze(0).shape)     # (1, 3, 1, 4): add one
print(x.flatten(1).shape)       # (3, 4): merge dims 1..end
print(torch.randn(2, 3, 4).flatten(-2).shape)  # (2, 12): merge the last two
'''),
        ("md", r'''
## 4. `cat` vs `stack`

- `cat` joins along an **existing** dim. Shapes must match everywhere else.
- `stack` creates a **new** dim. All shapes must match exactly.

*#14 KV Cache appends new keys with `cat` along the sequence dim.*
'''),
        ("code", r'''
a, b = torch.zeros(2, 3), torch.ones(2, 3)
print(torch.cat([a, b], dim=0).shape)    # (4, 3)
print(torch.cat([a, b], dim=1).shape)    # (2, 6)
print(torch.stack([a, b], dim=0).shape)  # (2, 2, 3)
print(torch.stack([a, b], dim=-1).shape) # (2, 3, 2)
'''),
        ("md", r'''
## 5. `repeat_interleave` vs `repeat`

Both copy data, but in a different **order**:
'''),
        ("code", r'''
x = torch.tensor([1, 2, 3])
print(x.repeat_interleave(2))   # [1, 1, 2, 2, 3, 3]: each element repeated in place
print(x.repeat(2))              # [1, 2, 3, 1, 2, 3]: the whole tensor tiled
'''),
    ],
    "exercises_intro": r'''
---
# ✏️ Exercises

Use the shape conventions from the attention problems: `B` = batch, `S` = sequence length, `D` = model dim, `H` = number of heads, `d = D // H` = head dim.
''',
    "exercises": [
        {
            "spec": r'''
### Exercise 1: `split_heads(x, num_heads)`

Split the model dimension into heads and move the head axis in front of the sequence axis.

- `x`: `(B, S, D)` with `D % num_heads == 0`
- returns: `(B, H, S, d)` where `out[b, h, s, :] = x[b, s, h*d:(h+1)*d]`
''',
            "stub": r'''
def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    B, S, D = x.shape
    return x.view(B, S, num_heads, D // num_heads).transpose(1, 2)
''',
            "debug": r'''
x = torch.arange(2 * 3 * 8).float().view(2, 3, 8)
h = split_heads(x, num_heads=4)
print(h.shape)        # expect torch.Size([2, 4, 3, 2])
print(h[0, 1, 0])     # expect x[0, 0, 2:4] = tensor([2., 3.])
''',
        },
        {
            "spec": r'''
### Exercise 2: `merge_heads(x)`

The inverse of `split_heads`.

- `x`: `(B, H, S, d)`
- returns: `(B, S, H*d)`, so that `merge_heads(split_heads(x, H))` equals `x`

*Watch out: after a transpose, `view` will fail. Why?*
''',
            "stub": r'''
def merge_heads(x: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, S, d = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, H * d)
''',
            "debug": r'''
x = torch.randn(2, 5, 12)
roundtrip = merge_heads(split_heads(x, num_heads=3))
print(roundtrip.shape, torch.equal(roundtrip, x))   # expect torch.Size([2, 5, 12]) True
''',
        },
        {
            "spec": r'''
### Exercise 3: `repeat_kv(x, n_rep)`

In Grouped-Query Attention, several query heads share one key/value head. Expand the KV heads so the shapes match the queries.

- `x`: `(B, H_kv, S, d)`
- returns: `(B, H_kv * n_rep, S, d)` where heads are ordered `[kv0, kv0, ..., kv1, kv1, ...]`: each KV head is repeated `n_rep` times **consecutively**.

*Query head `h` must use KV head `h // n_rep`. Which of `repeat` / `repeat_interleave` gives that order?*
''',
            "stub": r'''
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    return x.repeat_interleave(n_rep, dim=1)
''',
            "debug": r'''
x = torch.arange(2).float().view(1, 2, 1, 1)   # two KV heads with values 0 and 1
print(repeat_kv(x, 3).flatten())               # expect tensor([0., 0., 0., 1., 1., 1.])
''',
        },
        {
            "spec": r'''
### Exercise 4: `interleave(a, b)`

Merge two tensors along the last dim, alternating elements: `[a0, b0, a1, b1, ...]`.

- `a`, `b`: `(..., N)` (same shape, any number of leading dims)
- returns: `(..., 2N)`

*RoPE rotates pairs `(x[2k], x[2k+1])` and then has to put the rotated halves back in interleaved order.*
''',
            "stub": r'''
def interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack([a, b], dim=-1).flatten(-2)
''',
            "debug": r'''
a = torch.tensor([[1, 2, 3], [7, 8, 9]])
b = torch.tensor([[10, 20, 30], [70, 80, 90]])
print(interleave(a, b))
# expect [[ 1, 10,  2, 20,  3, 30],
#         [ 7, 70,  8, 80,  9, 90]]
''',
        },
        {
            "spec": r'''
### Exercise 5: `patchify(img, patch_size)`

Cut images into non-overlapping `p × p` patches and flatten each patch into a vector, as in ViT.

- `img`: `(B, C, H, W)`, where `H` and `W` are divisible by `p`
- returns: `(B, N, C*p*p)` where `N = (H/p) * (W/p)`
- **Patch order:** row-major over the patch grid (left→right, then top→bottom)
- **Within a patch:** flatten in `(C, p_row, p_col)` order, i.e. `out[b, n] == img[b, :, r*p:(r+1)*p, c*p:(c+1)*p].flatten()` for patch `n = r * (W/p) + c`

*Hint: `reshape` to 6 dims, `permute`, `reshape` back.*
''',
            "stub": r'''
def patchify(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def patchify(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, C, H, W = img.shape
    p = patch_size
    n_h, n_w = H // p, W // p
    x = img.reshape(B, C, n_h, p, n_w, p)        # split H and W into (grid, within-patch)
    x = x.permute(0, 2, 4, 1, 3, 5)              # (B, n_h, n_w, C, p, p)
    return x.reshape(B, n_h * n_w, C * p * p)
''',
            "debug": r'''
img = torch.arange(1 * 1 * 4 * 4).float().view(1, 1, 4, 4)
print(img[0, 0])
patches = patchify(img, 2)
print(patches.shape)   # expect torch.Size([1, 4, 4])
print(patches[0])
# expect [[ 0,  1,  4,  5],   top-left
#         [ 2,  3,  6,  7],   top-right
#         [ 8,  9, 12, 13],   bottom-left
#         [10, 11, 14, 15]]   bottom-right
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (6, "`split_heads` → attention → `merge_heads`, the core MHA pattern"),
        (23, "Same split/merge, with different sequence lengths for queries and keys"),
        (10, "`repeat_kv`: share each KV head across a group of query heads"),
        (13, "Split/merge heads inside a full pre-norm transformer block"),
        (14, "Split/merge heads, plus `torch.cat` of past and new keys along the sequence dim"),
        (24, "`interleave`: put rotated even/odd pairs back in order"),
        (27, "`patchify`: the reshape → permute → reshape at the heart of ViT"),
        (28, "`reshape(-1, D)` to flatten `(B, S, D)` tokens for routing, then reshape back"),
    ],
}
