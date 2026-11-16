"""Bonus 6: Memory & performance.

Content for templates/00_b06_performance.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "B6",
    "file": "00_b06_performance",
    "task_id": "b06_performance",
    "title": "Memory & Performance",
    "covers": "Views vs copies (shared storage), in-place ops and autograd, measuring speed with `torch.utils.benchmark`, hidden syncs (`.item()`), peak memory: chunking, `no_grad`, activation checkpointing",
    "imports": "import torch\nfrom torch import nn",
    "intro": r'''
# ⭐ Bonus 6: Memory & Performance

Correct code can still be too slow or run out of memory. Most PyTorch performance problems come from a few patterns: copying when a view would do (or the reverse), Python loops over tensor elements, hidden GPU synchronizations, and huge intermediate tensors like an `S × S` attention matrix. This notebook shows how to spot each one and what to do about it.

### What you'll learn
1. Views vs copies: which ops share memory, and why it matters
2. In-place operations: memory savings vs autograd errors
3. Measuring speed properly, and why vectorized code wins
4. Hidden synchronizations: `.item()`, printing, data-dependent shapes
5. Peak memory: chunking, `no_grad`, and activation checkpointing

### Exercises
| # | Function | Use it on |
|---|----------|-----------|
| 1 | `is_view_of(a, b)` | #14 KV Cache, #22 Conv2d |
| 2 | `moving_average(x, k)` | any sliding-window statistic |
| 3 | `chunked_attention(Q, K, V, chunk)` | #05 / #09 attention, #25 Flash Attention |
| 4 | `checkpointed_mlp(x, blocks)` | #13 GPT-2 Block stacks |

### Rules
- No Python loops over tensor elements. Looping over chunks (Exercise 3) or blocks (Exercise 4) is fine.
''',
    "lesson": [
        ("md", r'''
## 1. Views vs copies

A **view** shares memory with the tensor it came from; a **copy** has its own. Compare `x.untyped_storage().data_ptr()` to tell (`data_ptr()` alone also changes for an offset slice).

| Views (no copy) | Copies |
|-----------------|--------|
| basic slicing `x[1:3]`, `x[:, 0]` | advanced indexing `x[[0, 2]]`, boolean masks `x[mask]` |
| `view`, `transpose`, `permute`, `expand`, `unsqueeze`, `unfold`, `detach` | `clone`, arithmetic results, `contiguous()` of a non-contiguous tensor |
| `reshape` **when possible** | `reshape` when the layout requires it |

Two consequences: writing to a view changes the original, and **a small view keeps its whole base tensor alive**. Slicing one row out of a 1 GB tensor still holds the 1 GB until you `.clone()` the row.
'''),
        ("code", r'''
def same_storage(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

x = torch.arange(12.).view(3, 4)
print(same_storage(x, x[1:]), same_storage(x, x.T), same_storage(x, x[:, None].expand(3, 2, 4)))   # True True True
print(same_storage(x, x[[0, 2]]), same_storage(x, x.clone()), same_storage(x, x.T.reshape(12)))    # False False False
row = x[1]
row[0] = 100.0
print(x[1, 0])                    # 100.: writing to a view changes the original
'''),
        ("md", r'''
## 2. In-place operations

In-place ops (`add_`, `mul_`, `x[...] = ...`) avoid allocating a new tensor. But autograd saves some tensors for backward and tracks a **version counter** on each. Modifying a saved tensor in place makes backward fail ("modified by an inplace operation", see B2).
**Rule:** use in-place updates for buffers, optimizer states and preallocated caches, under `torch.no_grad()` (P7); prefer out-of-place ops inside the differentiable forward pass.
'''),
        ("code", r'''
x = torch.randn(3, requires_grad=True)
y = x.exp()                       # exp saves its OUTPUT for backward
print(y._version)                 # 0
y.add_(1)                         # in place: version 1, backward would now fail
print(y._version)

cache = torch.zeros(4, 8)         # preallocated, e.g. a KV cache
with torch.no_grad():
    cache[2] = torch.randn(8)     # write into it: no new allocation
'''),
        ("md", r'''
## 3. Measuring speed

- Use **`torch.utils.benchmark.Timer`**: it warms up, repeats, and (on GPU) synchronizes for you. A single `time.perf_counter()` around one call is noisy.
- **Python loops over elements are the #1 slowdown.** Each tensor op has a fixed overhead of microseconds, so a loop of 10,000 tiny ops costs far more than one op over 10,000 elements.
'''),
        ("code", r'''
import torch.utils.benchmark as benchmark
x = torch.randn(10_000)

def loop_sum(x):
    total = torch.tensor(0.0)
    for v in x:
        total += v
    return total

t_loop = benchmark.Timer("loop_sum(x)", globals={"loop_sum": loop_sum, "x": x}).timeit(3)
t_vec = benchmark.Timer("x.sum()", globals={"x": x}).timeit(1000)
print(f"loop: {t_loop.mean * 1e3:.1f} ms   vectorized: {t_vec.mean * 1e6:.1f} µs   speedup ≈ {t_loop.mean / t_vec.mean:,.0f}x")
'''),
        ("md", r'''
## 4. Hidden synchronizations

On a GPU, operations are queued and run **asynchronously**. Anything that needs a value **on the CPU** forces Python to wait for the whole queue: `.item()`, `.tolist()`, `print(tensor)`, `if tensor > 0:`, and ops whose output **shape depends on the data** (`nonzero`, boolean-mask indexing, `unique`).
In a training loop, keep the loss as a tensor and call `.item()` only when you log, not every step.

```python
running = torch.zeros((), device=device)
for step, batch in enumerate(loader):
    loss = train_step(batch)
    running += loss.detach()               # stays on the GPU: no sync
    if step % 100 == 0:
        print(running.item() / 100)        # one sync per 100 steps
        running.zero_()
```
'''),
        ("md", r'''
## 5. Peak memory

Memory runs out at the **peak**: the moment the most tensors are alive at once.
- **Big intermediates:** attention's scores are `B × H × S × S`. At `S = 32k` that's a billion numbers per head. **Chunking** (process queries in blocks, as in Exercise 3) or Flash Attention (#25) never materializes it.
- **Autograd saves activations** for backward. For evaluation, run under `torch.no_grad()` / `inference_mode()` so nothing is saved.
- **Activation checkpointing** (`torch.utils.checkpoint.checkpoint`) doesn't save a block's internal activations at all, and recomputes them during backward: roughly 30% more compute for much less memory. Large transformers checkpoint every block.
'''),
        ("code", r'''
from torch.utils.checkpoint import checkpoint
block = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 16))
calls = []
block[0].register_forward_hook(lambda *a: calls.append(1))
x = torch.randn(4, 16, requires_grad=True)
checkpoint(block, x, use_reentrant=False).sum().backward()
print("forward runs of the block's first layer:", len(calls))   # 2: once forward, once recomputed in backward
'''),
    ],
    "exercises_intro": r'''
---
# ✏️ Exercises

Implement the four functions below. Each one has a debug cell you can use before you submit.
''',
    "exercises": [
        {
            "spec": r'''
### Exercise 1: `is_view_of(a, b)`

Do `a` and `b` share the same underlying memory?

- returns: a Python `bool`, `True` if both tensors use the same storage (even if they start at different offsets or have different shapes)
''',
            "stub": r'''
def is_view_of(a: torch.Tensor, b: torch.Tensor) -> bool:
    pass  # Replace this
''',
            "solution": r'''
def is_view_of(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
''',
            "debug": r'''
x = torch.arange(12.).view(3, 4)
print(is_view_of(x[1:], x), is_view_of(x.T, x), is_view_of(x[[0]], x), is_view_of(x.clone(), x))
# expect True True False False
''',
        },
        {
            "spec": r'''
### Exercise 2: `moving_average(x, k)`

The mean of every window of `k` consecutive values along the last dim, **without a Python loop**.

- `x`: `(..., N)` float; `k`: window size, `1 <= k <= N`
- returns: `(..., N - k + 1)` with `out[..., i] = x[..., i : i + k].mean()`

*Hint: with `c = x.cumsum(-1)`, the sum of `x[i : i + k]` is `c[i + k - 1] - c[i - 1]`. Prepend a zero to make the first window work. (`x.unfold(-1, k, 1).mean(-1)` also works, at `O(N·k)` cost.)*
''',
            "stub": r'''
def moving_average(x: torch.Tensor, k: int) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def moving_average(x: torch.Tensor, k: int) -> torch.Tensor:
    c = torch.cat([torch.zeros_like(x[..., :1]), x.cumsum(-1)], dim=-1)   # c[i] = sum of x[:i]
    return (c[..., k:] - c[..., :-k]) / k
''',
            "debug": r'''
print(moving_average(torch.tensor([1., 2., 3., 4., 5.]), 2))   # expect tensor([1.5, 2.5, 3.5, 4.5])
print(moving_average(torch.tensor([[1., 2., 3.], [4., 4., 7.]]), 3))   # expect [[2.], [5.]]
''',
        },
        {
            "spec": r'''
### Exercise 3: `chunked_attention(Q, K, V, chunk)`

Scaled dot-product attention, `softmax(Q Kᵀ / √d) V`, computed **one block of queries at a time** so the full `S × S` score matrix never exists.

- `Q`, `K`, `V`: `(B, S, d)`; `chunk`: number of queries per block
- returns: `(B, S, d)`, equal to ordinary attention
- the largest score tensor alive at any time must be `(B, chunk, S)`

*Rows of attention are independent, so query blocks can be processed separately. Flash Attention (#25) goes further and chunks the keys too, merging blocks with the online softmax from P9 §7.*
''',
            "stub": r'''
import math

def chunked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, chunk: int) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
import math

def chunked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, chunk: int) -> torch.Tensor:
    scale = 1.0 / math.sqrt(Q.shape[-1])
    outs = []
    for start in range(0, Q.shape[1], chunk):
        q = Q[:, start:start + chunk]                           # (B, c, d)
        scores = (q @ K.transpose(-2, -1)) * scale              # (B, c, S): never (B, S, S)
        outs.append(torch.softmax(scores, dim=-1) @ V)
    return torch.cat(outs, dim=1)
''',
            "debug": r'''
torch.manual_seed(0)
Q, K, V = torch.randn(2, 10, 8), torch.randn(2, 10, 8), torch.randn(2, 10, 8)
out = chunked_attention(Q, K, V, chunk=4)
ref = torch.softmax(Q @ K.transpose(-2, -1) / 8 ** 0.5, dim=-1) @ V
print(out.shape, torch.allclose(out, ref, atol=1e-5))     # expect (2, 10, 8) True
''',
        },
        {
            "spec": r'''
### Exercise 4: `checkpointed_mlp(x, blocks)`

Run a stack of blocks with **activation checkpointing**, so their internal activations are recomputed during backward instead of stored.

- `x`: the input; `blocks`: a list of `nn.Module`s applied in order
- wrap every block in `torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)`
- returns: the final output, with the same values and gradients as running the blocks normally
''',
            "stub": r'''
from torch.utils.checkpoint import checkpoint

def checkpointed_mlp(x: torch.Tensor, blocks: list[nn.Module]) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
from torch.utils.checkpoint import checkpoint

def checkpointed_mlp(x: torch.Tensor, blocks: list[nn.Module]) -> torch.Tensor:
    h = x
    for block in blocks:
        h = checkpoint(block, h, use_reentrant=False)
    return h
''',
            "debug": r'''
torch.manual_seed(0)
blocks = [nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 8)) for _ in range(3)]
x = torch.randn(4, 8, requires_grad=True)
out = checkpointed_mlp(x, blocks)
ref = x
for b in blocks:
    ref = b(ref)
print(torch.allclose(out, ref))      # expect True
''',
        },
    ],
    # Problem number -> how to use this notebook on it ("Where to use it" section)
    "unlocks": [
        (5, "Rewrite attention with `chunked_attention` and compare peak tensor sizes"),
        (13, "Checkpoint each GPT-2 block when stacking several with `checkpointed_mlp`"),
        (14, "Preallocate the KV cache and write into it in place, instead of `torch.cat` copying every step"),
        (22, "`unfold` returns a view: check it with `is_view_of`, then see which step copies"),
        (25, "Chunked attention is the first half of Flash Attention: add key blocks with online softmax"),
    ],
}
