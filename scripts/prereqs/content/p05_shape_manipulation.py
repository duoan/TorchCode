"""Prerequisite 5: Shape manipulation.

Content for templates/00_p05_shape_manipulation.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P5",
    "file": "00_p05_shape_manipulation",
    "task_id": "p05_shape_manipulation",
    "title": "Shape Manipulation",
    # Shape symbols used in this notebook, shown in the "📐 Notation" box
    "notation": [
        ('B', 'batch size'),
        ('S', 'sequence length'),
        ('D', 'model dimension'),
        ('H', 'number of attention heads (Exercises 1–3); **image height** in Exercises 5–6 and lesson §6'),
        ('d', 'dimension per head, `D // H`'),
        ('H_kv', 'number of key/value heads (GQA), also written `Hkv`'),
        ('n_rep', 'how many query heads share each key/value head'),
        ('C', 'image channels'),
        ('W', 'image width'),
        ('p', 'patch size (`patchify`)'),
        ('N', 'number of patches'),
        ('n_h', 'patches per column, `H // p`'),
        ('n_w', 'patches per row, `W // p`'),
        ('p_row', 'row inside a patch'),
        ('p_col', 'column inside a patch'),
        ('k', 'window (kernel) size in `sliding_windows`'),
        ('H_out', 'number of windows along the height'),
        ('W_out', 'number of windows along the width'),
    ],
    "imports": "import torch\nimport torch.nn.functional as F",
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
6. Sliding windows with `F.pad` and `unfold`
7. `expand` vs `repeat`, `unflatten`, `movedim`, `roll`, and the `*_as` helpers
8. Many ways to do it: equivalent reshapes and their trade-offs

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `split_heads(x, num_heads)` | #06 MHA, #10 GQA, #13 GPT-2, #14 KV Cache, #23 Cross-Attn |
| 2 | `merge_heads(x)` | same as above (the inverse) |
| 3 | `repeat_kv(x, n_rep)` | #10 GQA |
| 4 | `interleave(a, b)` | #24 RoPE |
| 5 | `patchify(img, patch_size)` | #27 ViT Patch Embedding |
| 6 | `sliding_windows(x, k, stride, padding)` | #22 Conv2d |

### Rules
- **No Python loops.** Use only view/reshape/transpose/permute/cat/stack/repeat-style ops, plus `unfold` and `F.pad` (Exercise 6).
- Do **not** use `F.unfold` (the im2col function) or `F.conv2d`.
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
        ("md", r'''
## 6. Sliding windows: `F.pad` and `unfold`

`x.unfold(dim, size, step)` returns a **view** containing every window of length `size` along `dim`, taken every `step` elements. The windows become a **new last dim**.
Unfold dim 2 and then dim 3 of an image `(B, C, H, W)` to get every `k × k` patch: `(B, C, H_out, W_out, k, k)`.

The number of windows along a dim of size `H`, with padding `p` and stride `s`:

$$H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1$$

`F.pad(x, (left, right, top, bottom))` pads the **last** dims first: the first pair pads the last dim, the second pair the one before it.
'''),
        ("code", r'''
x = torch.arange(16.).view(1, 1, 4, 4)
w = x.unfold(2, 2, 2).unfold(3, 2, 2)      # 2x2 windows, stride 2
print(w.shape)                             # (1, 1, 2, 2, 2, 2)
print(w[0, 0, 0, 1])                       # top-right window: [[2, 3], [6, 7]]
print(F.pad(x, (1, 1, 1, 1)).shape)        # (1, 1, 6, 6): one zero on every side
print(x.unfold(3, 3, 1).shape)             # 1-D windows along W: (1, 1, 4, 2, 3)
'''),
        ("md", r'''
## 7. `expand` vs `repeat`, and other helpers

- **`x.expand(*sizes)`** broadcasts size-1 dims **without copying**: the new dims get stride `0`, so every "copy" reads the same memory. **`x.repeat(*reps)`** really copies. Use `expand` when you only read the result; never write into an expanded tensor.
- **`x.unflatten(dim, sizes)`** is the readable inverse of `flatten`: `x.unflatten(-1, (H, d))` splits the model dim into heads.
- **`x.movedim(src, dst)`** moves one dim to a new position; `swapaxes` / `transpose` swap two.
- **`torch.roll(x, shifts, dims)`** shifts elements cyclically (wrapping around the end).
- **`*_as` helpers** take the shape from another tensor: `view_as(y)`, `reshape_as(y)`, `expand_as(y)`.
'''),
        ("code", r'''
x = torch.arange(3.).view(3, 1)
e = x.expand(3, 4)
r = x.repeat(1, 4)
print(e.stride(), e.data_ptr() == x.data_ptr())   # (1, 0) True: no copy
print(r.stride(), r.data_ptr() == x.data_ptr())   # (4, 1) False: a real copy

h = torch.randn(2, 5, 12)
print(h.unflatten(-1, (3, 4)).shape)              # (2, 5, 3, 4): heads split off
print(torch.randn(2, 3, 4, 5).movedim(1, -1).shape)   # (2, 4, 5, 3): channels last
print(torch.roll(torch.arange(5), shifts=1))      # [4, 0, 1, 2, 3]
print(torch.zeros(2, 3).expand_as(torch.empty(4, 2, 3)).shape)
'''),
        ("md", r'''
## 8. Many ways to do it

| Goal | Equivalent forms | Trade-off |
|------|------------------|-----------|
| Split heads | `x.view(B, S, H, d).transpose(1, 2)`, `x.unflatten(-1, (H, d)).transpose(1, 2)`, `x.reshape(B, S, H, d).permute(0, 2, 1, 3)` | `view` fails loudly on non-contiguous input; `reshape` silently copies; `unflatten` doesn't need `B` and `S` |
| Repeat KV heads (GQA) | `k.repeat_interleave(n, dim=1)`, `k[:, :, None].expand(B, Hkv, n, S, d).reshape(B, Hkv * n, S, d)` | the `expand` form is how many production implementations do it: no copy until `reshape` |
| Add a dim | `x.unsqueeze(1)`, `x[:, None]`, `x.view(B, 1, D)` | all views; `[:, None]` is the most compact inside expressions |
| Move channels last | `x.permute(0, 2, 3, 1)`, `x.movedim(1, -1)` | `movedim` states the intent and doesn't need every dim listed |
'''),
        ("code", r'''
B, Hkv, S, d, n = 2, 2, 5, 4, 3
k = torch.randn(B, Hkv, S, d)
a = k.repeat_interleave(n, dim=1)
b = k[:, :, None].expand(B, Hkv, n, S, d).reshape(B, Hkv * n, S, d)
print(torch.equal(a, b))

x = torch.randn(B, S, 12)
print(torch.equal(x.view(B, S, 3, 4).transpose(1, 2), x.unflatten(-1, (3, 4)).transpose(1, 2)))
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
        {
            "spec": r'''
### Exercise 6: `sliding_windows(x, k, stride=1, padding=0)`

Extract every `k × k` window of an image batch, as a convolution sees it.

- `x`: `(B, C, H, W)`
- zero-pad `padding` pixels on **every** side first, then take windows of size `k` every `stride` pixels
- returns: `(B, C, H_out, W_out, k, k)` with `H_out = (H + 2·padding − k) // stride + 1` (same for `W_out`)

*#22 Conv2d is exactly this followed by P6's `patch_conv`.*
''',
            "stub": r'''
def sliding_windows(x: torch.Tensor, k: int, stride: int = 1, padding: int = 0) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def sliding_windows(x: torch.Tensor, k: int, stride: int = 1, padding: int = 0) -> torch.Tensor:
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))
    return x.unfold(2, k, stride).unfold(3, k, stride)
''',
            "debug": r'''
x = torch.arange(16.).view(1, 1, 4, 4)
win = sliding_windows(x, k=3, stride=1, padding=1)
print(win.shape)          # expect torch.Size([1, 1, 4, 4, 3, 3])
print(win[0, 0, 0, 0])    # expect [[0, 0, 0], [0, 0, 1], [0, 4, 5]]: top-left window, padded
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
        (22, "`sliding_windows`: `F.pad` + `unfold`, the patch extraction inside Conv2d"),
        (27, "`patchify`: the reshape → permute → reshape at the heart of ViT"),
        (28, "`reshape(-1, D)` to flatten `(B, S, D)` tokens for routing, then reshape back"),
    ],
}
