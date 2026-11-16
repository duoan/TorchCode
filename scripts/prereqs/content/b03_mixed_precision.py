"""Bonus 3: Mixed precision & devices.

Content for templates/00_b03_mixed_precision.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "B3",
    "file": "00_b03_mixed_precision",
    "task_id": "b03_mixed_precision",
    "title": "Mixed Precision & Devices",
    "covers": "float32 vs float16 vs bfloat16 (`torch.finfo`), overflow / underflow / accumulation error, what to keep in float32, `torch.autocast`, loss scaling (what `GradScaler` does), device-agnostic code and GPU timing",
    "imports": "import torch\nfrom torch import nn",
    "intro": r'''
# ⭐ Bonus 3: Mixed Precision & Devices

Large models are trained in **16-bit** floats: half the memory, much faster matmuls on modern GPUs. But 16-bit numbers have a much smaller range or much less precision than float32, and code that is correct in float32 can silently return `inf`, `NaN`, or zero gradients in float16. Knowing **which** operations are safe in low precision, and what `torch.autocast` and `GradScaler` do for you, is standard interview material for LLM roles.

*Everything here runs on CPU (with bfloat16 where float16 isn't supported); on a GPU the same code just runs faster.*

### What you'll learn
1. float32 vs float16 vs bfloat16: range, precision, and `torch.finfo`
2. What breaks in low precision: overflow, underflow, and accumulation error
3. `torch.autocast`: low-precision matmuls, float32 where it matters
4. Loss scaling: why float16 training needs `GradScaler`, and what it does
5. Device-agnostic code, and timing GPU code correctly

### Exercises
| # | Function | Use it on |
|---|----------|-----------|
| 1 | `count_overflow(x, dtype)` | checking activations before casting |
| 2 | `softmax_lowp(x, dim)` | #02 Softmax, #05 attention scores |
| 3 | `unscale_and_check(params, scale)` | #29 Adam, #31 Gradient Accumulation |
| 4 | `to_device_like(obj, ref, float_dtype)` | moving batches in any training loop |

### Rules
- No Python loops over tensor elements. Looping over parameters or over a container's items is fine.
- Do **not** use `torch.amp.GradScaler` in Exercise 3: rebuild its core.
''',
    "lesson": [
        ("md", r'''
## 1. Three float formats

| dtype | bits (sign / exponent / mantissa) | largest value | precision (`eps`) | typical use |
|-------|------|---------------|-------------------|-------------|
| `float32` | 1 / 8 / 23 | ≈ 3.4 × 10³⁸ | ≈ 1.2 × 10⁻⁷ | master weights, reductions, losses |
| `float16` | 1 / 5 / 10 | **65504** | ≈ 9.8 × 10⁻⁴ | GPU training with loss scaling |
| `bfloat16` | 1 / 8 / 7 | ≈ 3.4 × 10³⁸ | **≈ 7.8 × 10⁻³** | GPU/TPU training; no loss scaling needed |

**float16** keeps decent precision but has a tiny range: anything above 65504 becomes `inf`, and anything below about 6 × 10⁻⁸ becomes `0`.
**bfloat16** keeps float32's range (same exponent bits) but has only about 2–3 significant decimal digits.
`torch.finfo(dtype)` reports these limits; `torch.iinfo` does the same for integer types.
'''),
        ("code", r'''
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"{str(dt):<15} max={fi.max:.3g}  eps={fi.eps:.3g}  smallest normal={fi.tiny:.3g}")
print(torch.tensor(70000.0).to(torch.float16), torch.tensor(70000.0).to(torch.bfloat16))   # inf vs 70144
print(torch.tensor(1.0, dtype=torch.bfloat16) + 1e-3 == 1.0)                                # True: lost entirely
print(torch.iinfo(torch.int8))
'''),
        ("md", r'''
## 2. What breaks in low precision

- **Overflow (float16):** `exp(12)` is already 162,755 > 65504. So is `300²` in a variance computation.
- **Underflow (float16):** small gradients like `1e-8` become `0`, and Adam's default `eps = 1e-8` is **zero** in float16.
- **Accumulation error (both):** adding many small numbers to a large running total loses them once the total's spacing exceeds the increment. In bfloat16 a running count stops at **256**: 257 isn't representable, so `256 + 1` rounds back to 256.

**Rule of thumb:** do matmuls in 16-bit, but keep **reductions, softmax, normalization statistics, and losses in float32**. That's why LLaMA's RMSNorm calls `x.float()` first, and why optimizers keep a float32 copy of the weights.
'''),
        ("code", r'''
print(torch.exp(torch.tensor(12.0, dtype=torch.float16)))          # inf
print(torch.tensor(300.0, dtype=torch.float16) ** 2)               # inf
print(torch.tensor(1e-8, dtype=torch.float16))                     # 0.
s = torch.tensor(0.0, dtype=torch.bfloat16)
for _ in range(1000):
    s = s + 1
print("bfloat16 count to 1000:", s.item())                         # 256.0
print("float32 count to 1000: ", sum(torch.ones(1000)).item())     # 1000.0
'''),
        ("md", r'''
## 3. `torch.autocast`

`torch.autocast(device_type, dtype=...)` runs eligible ops (matmuls, linear layers, convolutions) in the low-precision dtype **while the weights stay float32**. Ops that are numerically risky (softmax, log, losses, most reductions) run in float32. You write ordinary code; autocast picks the dtype per op.
Gradients flow back to the float32 weights, so the optimizer updates full-precision "master" weights.
'''),
        ("code", r'''
model = nn.Linear(8, 4)
x = torch.randn(2, 8)
with torch.autocast("cpu", dtype=torch.bfloat16):
    y = model(x)
    loss = y.float().pow(2).mean()          # be explicit: compute the loss in float32
print(model.weight.dtype, y.dtype, loss.dtype)    # float32 (weights stay), bfloat16 (matmul), float32
loss.backward()
print(model.weight.grad.dtype)                   # float32: gradients land on the float32 weights
'''),
        ("md", r'''
## 4. Loss scaling (what `GradScaler` does)

In float16, many gradients are smaller than the smallest representable number and **underflow to zero**. The fix:
1. multiply the loss by a large **scale** (e.g. 65536) before `backward()`, so every gradient is scaled up out of the underflow zone;
2. **unscale**: divide the gradients by the same factor before the optimizer step;
3. if any gradient is now `inf` / `NaN`, the scale was too big: **skip** this step and reduce the scale. Otherwise step, and occasionally try a larger scale.

`torch.amp.GradScaler` automates exactly this. bfloat16 has float32's range, so it usually doesn't need it.
'''),
        ("code", r'''
g = torch.tensor([1e-8, 3e-6, 0.5])                 # true float32 gradients
print("float16 directly:", g.to(torch.float16))     # the first one underflows to 0
scale = 65536.0
scaled = (g * scale).to(torch.float16)              # what float16 backward would hold
print("scaled, then unscaled:", scaled.float() / scale)   # 1e-8 survives (approximately)
'''),
        ("md", r'''
## 5. Devices

Write code that runs anywhere: pick the device once, and create every new tensor **next to its inputs** (P1).
- `.to(device, non_blocking=True)` from **pinned** CPU memory (`DataLoader(pin_memory=True)`) lets the copy overlap with GPU compute.
- GPU calls are **asynchronous**: the Python line returns before the GPU finishes. To time GPU code, call `torch.cuda.synchronize()` before reading the clock, or use `torch.utils.benchmark`, which does it for you.
'''),
        ("code", r'''
import time
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print("using", device)

def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()            # wait for earlier GPU work
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()            # wait for this work to finish
    return out, time.perf_counter() - t0

a = torch.randn(512, 512, device=device)
_, seconds = timed(lambda: a @ a, device)
print(f"512x512 matmul: {seconds * 1000:.2f} ms")
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
### Exercise 1: `count_overflow(x, dtype)`

How many **finite** values of `x` would become `±inf` if cast to `dtype`?

- `x`: a float tensor (usually float32); `dtype`: a floating dtype such as `torch.float16`
- returns: a Python `int`
- values that are already `inf` / `NaN` in `x` don't count
''',
            "stub": r'''
def count_overflow(x: torch.Tensor, dtype: torch.dtype) -> int:
    pass  # Replace this
''',
            "solution": r'''
def count_overflow(x: torch.Tensor, dtype: torch.dtype) -> int:
    return int((torch.isfinite(x) & torch.isinf(x.to(dtype))).sum())
''',
            "debug": r'''
x = torch.tensor([1.0, 65504.0, 65520.0, -1e5, float("inf"), 3e38])
print(count_overflow(x, torch.float16))    # expect 3: 65520 (rounds up to inf), -1e5, 3e38
print(count_overflow(x, torch.bfloat16))   # expect 0: bfloat16 shares float32's range
''',
        },
        {
            "spec": r'''
### Exercise 2: `softmax_lowp(x, dim=-1)`

Softmax for a low-precision input, computed safely.

- `x`: float16 or bfloat16 (or float32)
- compute the softmax in **float32** (stable, P9), then return it in `x`'s dtype
- returns: same shape and dtype as `x`, with each slice along `dim` summing to ~1

*Upcasting before the reduction is what autocast does for softmax, and what you should do in a hand-written attention.*
''',
            "stub": r'''
def softmax_lowp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def softmax_lowp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    xf = x.float()
    e = torch.exp(xf - xf.amax(dim=dim, keepdim=True))
    return (e / e.sum(dim=dim, keepdim=True)).to(x.dtype)
''',
            "debug": r'''
x = torch.zeros(1, 100_000, dtype=torch.float16)         # a 100k-token vocabulary, all logits equal
p = softmax_lowp(x)
print(p.dtype, p[0, 0].item(), p.float().sum().item())   # expect float16, ~1e-05, ~1.0
naive = torch.exp(x) / torch.exp(x).sum(-1, keepdim=True)
print("naive float16 sum of exps:", torch.exp(x).sum().item(), "->", naive[0, 0].item())
''',
        },
        {
            "spec": r'''
### Exercise 3: `unscale_and_check(params, scale)`

The core of `GradScaler.unscale_()`: undo the loss scaling on every gradient and report whether the step must be skipped.

- `params`: a list of parameters; some may have `.grad is None` (skip those)
- divide every existing `.grad` **in place** by `scale`
- returns: `True` if any gradient contains `inf` or `NaN` (the optimizer step should be skipped), else `False`, as a Python `bool`
''',
            "stub": r'''
def unscale_and_check(params: list[torch.Tensor], scale: float) -> bool:
    pass  # Replace this
''',
            "solution": r'''
def unscale_and_check(params: list[torch.Tensor], scale: float) -> bool:
    found_inf = False
    for p in params:
        if p.grad is None:
            continue
        p.grad.div_(scale)
        if not torch.isfinite(p.grad).all():
            found_inf = True
    return found_inf
''',
            "debug": r'''
w = torch.zeros(3, requires_grad=True)
(w * torch.tensor([1.0, 2.0, 3.0]) * 1024).sum().backward()   # a loss scaled by 1024
print(unscale_and_check([w], 1024.0), w.grad)                 # expect False tensor([1., 2., 3.])
w.grad[0] = float("inf")
print(unscale_and_check([w], 1024.0))                         # expect True: skip this step
''',
        },
        {
            "spec": r'''
### Exercise 4: `to_device_like(obj, ref, float_dtype=None)`

Move a whole batch (nested dicts, lists and tuples of tensors) to `ref`'s device.

- tensors go to `ref.device`; if `float_dtype` is given, **floating-point** tensors are also cast to it (integer and bool tensors keep their dtype: token ids must stay `long`)
- dicts, lists and tuples are rebuilt with the same type and keys; anything else (strings, numbers, `None`) is returned unchanged
- returns: the new structure; the input is not modified
''',
            "stub": r'''
def to_device_like(obj, ref: torch.Tensor, float_dtype: torch.dtype | None = None):
    pass  # Replace this
''',
            "solution": r'''
def to_device_like(obj, ref: torch.Tensor, float_dtype: torch.dtype | None = None):
    if isinstance(obj, torch.Tensor):
        if float_dtype is not None and obj.is_floating_point():
            return obj.to(device=ref.device, dtype=float_dtype)
        return obj.to(device=ref.device)
    if isinstance(obj, dict):
        return {k: to_device_like(v, ref, float_dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device_like(v, ref, float_dtype) for v in obj)
    return obj
''',
            "debug": r'''
batch = {"ids": torch.tensor([[1, 2, 3]]), "mask": torch.ones(1, 3, dtype=torch.bool),
         "feats": [torch.randn(2), (torch.randn(1), "meta")], "n": 3}
out = to_device_like(batch, torch.empty(0), float_dtype=torch.bfloat16)
print(out["ids"].dtype, out["mask"].dtype, out["feats"][0].dtype, type(out["feats"][1]).__name__, out["feats"][1][1], out["n"])
# expect torch.int64 torch.bool torch.bfloat16 tuple meta 3
''',
        },
    ],
    # Problem number -> how to use this notebook on it ("Where to use it" section)
    "unlocks": [
        (2, "Feed your softmax float16 logits; make it upcast like `softmax_lowp`"),
        (8, "Compute the RMS in float32 and cast back, as LLaMA does"),
        (25, "Keep Flash Attention's running max, sum and output accumulator in float32 while Q, K, V are 16-bit"),
        (29, "Why Adam's `eps = 1e-8` is 0 in float16, and why the optimizer state stays float32"),
        (31, "Combine gradient accumulation with loss scaling: unscale once, after the last micro-batch"),
        (36, "Use `torch.iinfo(torch.int8)` for the clamp range instead of hard-coding -128 and 127"),
    ],
}
