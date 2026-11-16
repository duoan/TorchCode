"""Prerequisite 1: Tensor basics.

Content for templates/00_p01_tensor_basics.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P1",
    "file": "00_p01_tensor_basics",
    "task_id": "p01_tensor_basics",
    "title": "Tensor Basics",
    "intro": r'''
# 🔵 Prerequisite 1: Tensor Basics

Before broadcasting, reductions, or attention, you need to be fluent with the tensor itself: how to **create** one with the right shape, **dtype**, and **device**, how dtypes change under comparisons and casts, and what **in-place** operations really do.

These details rarely appear in a problem statement, but they decide whether your code passes a hidden `float64` test, runs on a GPU, or silently corrupts a value when casting to `int8`.

### What you'll learn
1. Creating tensors (`tensor`, `zeros`, `full`, `arange`, `randn`, `*_like`, `new_*`)
2. Dtypes: defaults, comparisons → `bool`, casting, and integer overflow
3. Shape basics: `.shape`, `.ndim`, `.numel()`
4. Devices: creating tensors next to an existing one
5. Python numbers vs tensors: `.item()` and 0-d tensors
6. In-place operations: the trailing underscore

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `positive_mask(x)` | #01 ReLU |
| 2 | `running_max_init(scores)` | #25 Flash Attention |
| 3 | `scale_inplace_(x, s)` | #07 BatchNorm, #20 Kaiming Init, #21 Gradient Clipping |
| 4 | `quantize_int8(w, scale)` | #36 INT8 Quantization |
| 5 | `fan_in_out(weight)` | #20 Kaiming Init |

### Rules
- **No Python loops** over tensor elements.
- Do **not** use `torch.relu` / `F.relu` / `torch.heaviside` (Exercise 1) or `torch.nn.init` helpers (Exercise 5).
''',
    "lesson": [
        ("md", r'''
## 1. Creating tensors

| You want | Use |
|----------|-----|
| From Python data | `torch.tensor([[1., 2.], [3., 4.]])` |
| A constant | `torch.zeros(2, 3)`, `torch.ones(2, 3)`, `torch.full((2, 3), 7.0)` |
| A range | `torch.arange(5)`, `torch.arange(0, 10, 2)`, `torch.linspace(0, 1, 5)` |
| Random | `torch.randn(2, 3)` (normal), `torch.rand(2, 3)` (uniform) |
| Same shape/dtype/device as `x` | `torch.zeros_like(x)`, `torch.full_like(x, 7.0)`, `torch.rand_like(x)` |
| A **different** shape, same dtype/device as `x` | `x.new_zeros(4, 1)`, `x.new_full((4, 1), 7.0)` |
'''),
        ("code", r'''
x = torch.randn(2, 3, dtype=torch.float64)
print(torch.zeros_like(x).dtype)          # float64: *_like copies dtype (and device)
print(x.new_full((4, 1), -1.0))           # different shape, still float64
print(torch.arange(5), torch.linspace(0, 1, 5))
'''),
        ("md", r'''
## 2. Dtypes

- Python floats become `float32`; Python ints become `int64`.
- **Comparisons return `bool`**: `x > 0` is a mask, not numbers. Convert with `.float()` or `.to(x.dtype)`.
- `.float()` always gives `float32`. `.to(x.dtype)` **keeps** whatever precision `x` had. Prefer it when matching another tensor.
- Casting an out-of-range value to a small integer type gives **garbage** silently (it typically wraps around; the exact result is platform-dependent). Always clamp to the valid range *before* casting.
'''),
        ("code", r'''
print(torch.tensor([1, 2]).dtype, torch.tensor([1.0, 2.0]).dtype)   # int64 float32

x = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
mask = x > 0
print(mask, mask.dtype)                  # bool
print(mask.float().dtype, mask.to(x.dtype).dtype)   # float32 vs float64

big = torch.tensor([200.0, -300.0])
print("cast directly:", big.to(torch.int8))                    # garbage, e.g. [-56, -44]; varies by platform
print("clamp first:  ", big.clamp(-128, 127).to(torch.int8))   # [127, -128]
'''),
        ("md", r'''
## 3. Shape basics

`.shape` is a tuple of plain Python ints, so you can unpack it and do arithmetic with it.
'''),
        ("code", r'''
w = torch.randn(64, 32, 3, 3)            # a conv weight: (C_out, C_in, kH, kW)
C_out, C_in, kH, kW = w.shape
print(w.ndim, w.numel(), C_in * kH * kW, w[0].numel())
print(type(w.shape[1]))                  # <class 'int'>
'''),
        ("md", r'''
## 4. Devices

Every tensor lives on a device (`cpu`, `cuda:0`, `mps`, …). Operations between tensors on **different** devices raise an error.
When your function creates a new tensor, put it on the **same device as its inputs**. Pass `device=x.device`, or use `*_like` / `x.new_*`, which do this automatically.
TorchCode's judge runs on CPU, but code that hard-codes `torch.zeros(...)` breaks the moment it runs on a GPU, and interviewers notice.
'''),
        ("code", r'''
x = torch.randn(3, 4)
good = torch.zeros(3, 1, device=x.device, dtype=x.dtype)   # explicit
also_good = x.new_zeros(3, 1)                              # inherits device + dtype
print(x.device, good.device, also_good.device)
'''),
        ("md", r'''
## 5. Python numbers vs tensors

- Reductions return **0-d tensors**: `x.sum()` is `tensor(3.)`, not `3.0`.
- `.item()` turns a one-element tensor into a Python number (e.g. for logging or returning a float).
- Python's `math` functions work on Python numbers; `torch` functions work on tensors. `math.sqrt(2.0)` as a constant inside tensor math is fine.
'''),
        ("code", r'''
import math
s = torch.tensor([1.0, 2.0]).sum()
print(s, s.ndim, s.item(), type(s.item()))
print(torch.tensor([1.0, 4.0]) / math.sqrt(2.0))
'''),
        ("md", r'''
## 6. In-place operations

Methods ending in `_` (`add_`, `mul_`, `zero_`, `normal_`, `clamp_`) modify the tensor **in place** and return it.
Plain assignment such as `x = x * 2` creates a **new** tensor and rebinds the local name. The caller's tensor is unchanged.
In-place ops on a **view** (like a slice) modify the original tensor too.
'''),
        ("code", r'''
def double_rebind(x):
    x = x * 2          # new tensor; caller unaffected
    return x

def double_inplace(x):
    return x.mul_(2)   # modifies caller's tensor

t = torch.ones(3)
double_rebind(t);  print("after rebind: ", t)
double_inplace(t); print("after in-place:", t)

m = torch.zeros(2, 3)
m[0].add_(1)                             # m[0] is a view into m
print(m)
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
### Exercise 1: `positive_mask(x)`

Return a tensor that is `1` where `x > 0` and `0` elsewhere.

- `x`: any shape, any floating dtype
- returns: same shape **and same dtype** as `x`

*ReLU is just `x * positive_mask(x)` (#01).*
''',
            "stub": r'''
def positive_mask(x: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def positive_mask(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).to(x.dtype)
''',
            "debug": r'''
x = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
m = positive_mask(x)
print(m, m.dtype)     # expect tensor([0., 0., 1.], dtype=torch.float64)
''',
        },
        {
            "spec": r'''
### Exercise 2: `running_max_init(scores)`

Flash Attention keeps a running row-maximum that starts at `-inf`. Create it.

- `scores`: `(B, S, D)`
- returns: `(B, S, 1)` filled with `-inf`, with the **same dtype and device** as `scores`
''',
            "stub": r'''
def running_max_init(scores: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def running_max_init(scores: torch.Tensor) -> torch.Tensor:
    B, S, _ = scores.shape
    return scores.new_full((B, S, 1), float("-inf"))
''',
            "debug": r'''
scores = torch.randn(2, 4, 8, dtype=torch.float64)
m = running_max_init(scores)
print(m.shape, m.dtype, m.device)   # expect torch.Size([2, 4, 1]) torch.float64 cpu
print(m[0, :, 0])                   # expect all -inf
''',
        },
        {
            "spec": r'''
### Exercise 3: `scale_inplace_(x, s)`

Multiply `x` by the number `s` **in place** and return `x` itself (the same tensor object, not a copy).

- `x`: any shape
- `s`: Python float
- returns: `x`

*This is how BatchNorm updates running stats (#07), Kaiming init fills weights (#20), and gradient clipping rescales `.grad` (#21).*
''',
            "stub": r'''
def scale_inplace_(x: torch.Tensor, s: float) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def scale_inplace_(x: torch.Tensor, s: float) -> torch.Tensor:
    return x.mul_(s)
''',
            "debug": r'''
t = torch.ones(2, 3)
out = scale_inplace_(t, 0.5)
print(t)              # expect all 0.5: the caller's tensor changed
print(out is t)       # expect True
''',
        },
        {
            "spec": r'''
### Exercise 4: `quantize_int8(w, scale)`

Quantize float weights to 8-bit integers: divide by `scale`, round to the nearest integer, clamp to `[-128, 127]`, and cast to `torch.int8`.

- `w`: float tensor, any shape
- `scale`: Python float **or** a tensor that broadcasts against `w` (e.g. per-row `(N, 1)`)
- returns: `torch.int8` tensor, same shape as `w`

*The order matters. See lesson §2 for what happens if you cast before clamping. This is the core of #36.*
''',
            "stub": r'''
def quantize_int8(w: torch.Tensor, scale) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def quantize_int8(w: torch.Tensor, scale) -> torch.Tensor:
    return torch.round(w / scale).clamp(-128, 127).to(torch.int8)
''',
            "debug": r'''
w = torch.tensor([0.4, -1.2, 3.0, 500.0, -500.0])
print(quantize_int8(w, 0.1))
# expect tensor([   4,  -12,   30,  127, -128], dtype=torch.int8)
''',
        },
        {
            "spec": r'''
### Exercise 5: `fan_in_out(weight)`

Compute the fan-in and fan-out used by weight initializers.

- `weight`: a `Linear` weight `(out_features, in_features)` **or** a conv weight `(C_out, C_in, kH, kW)`
- returns: a tuple of two **Python ints** `(fan_in, fan_out)`
  - Linear: `fan_in = in_features`, `fan_out = out_features`
  - Conv: `fan_in = C_in * kH * kW`, `fan_out = C_out * kH * kW`

*#20 Kaiming Init uses `std = sqrt(2 / fan_in)`.*
''',
            "stub": r'''
def fan_in_out(weight: torch.Tensor) -> tuple[int, int]:
    pass  # Replace this
''',
            "solution": r'''
def fan_in_out(weight: torch.Tensor) -> tuple[int, int]:
    receptive_field = weight[0][0].numel()   # 1 for Linear, kH * kW for conv
    return weight.shape[1] * receptive_field, weight.shape[0] * receptive_field
''',
            "debug": r'''
print(fan_in_out(torch.empty(64, 32)))          # expect (32, 64)
print(fan_in_out(torch.empty(16, 3, 5, 5)))     # expect (75, 400)
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (1, "`positive_mask`: ReLU is `x` times a bool mask converted to `x.dtype`"),
        (7, "`scale_inplace_`: running stats are updated in place with `mul_` / `add_`"),
        (19, "Mixing Python constants like `math.sqrt(2.0)` with tensor math (lesson §5)"),
        (20, "`fan_in_out` for the std, and in-place `normal_` (`scale_inplace_`)"),
        (21, "`scale_inplace_` on `p.grad`, and returning the norm as a Python float with `.item()`"),
        (25, "`running_max_init`: `-inf` running max and zero accumulators with the right dtype and device"),
        (29, "`zeros_like` optimizer state and in-place parameter updates"),
        (36, "`quantize_int8`: round, clamp, then cast, in that order"),
    ],
}
