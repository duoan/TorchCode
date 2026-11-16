"""Bonus 2: Debugging & testing tensors.

Content for templates/00_b02_debugging.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "B2",
    "file": "00_b02_debugging",
    "task_id": "b02_debugging",
    "title": "Debugging & Testing Tensors",
    "covers": "The common error messages decoded, `torch.testing.assert_close`, finding the first NaN with forward hooks, gradient health (missing and exploding gradients), a debugging workflow",
    "imports": "import torch\nfrom torch import nn",
    "intro": r'''
# ⭐ Bonus 2: Debugging & Testing Tensors

Most time spent on PyTorch code isn't writing it, it's figuring out why it doesn't work: a cryptic shape error, a loss that turns into `NaN` after 300 steps, a layer that silently never trains. This notebook is about finding those problems quickly, and about writing checks that catch them before they waste a training run.

### What you'll learn
1. Reading the most common error messages, and what each one actually means
2. Comparing tensors properly: `torch.testing.assert_close` and reporting differences
3. Finding where `NaN` / `inf` first appears, with forward hooks
4. Gradient health: parameters that never get a gradient, and exploding gradients
5. A debugging workflow that saves time

### Exercises
| # | Function | Use it on |
|---|----------|-----------|
| 1 | `first_nan_module(model, x)` | #13 GPT-2 Block, any model whose loss turns `NaN` |
| 2 | `params_without_grad(model)` | #26 LoRA, #28 MoE |
| 3 | `diff_report(a, b)` | #25 Flash Attention vs a reference |
| 4 | `grad_norms_by_name(model)` | #21 Gradient Clipping |

### Rules
- No Python loops over tensor elements. Looping over a model's modules or parameters is fine.
''',
    "lesson": [
        ("md", r'''
## 1. Read the error message

PyTorch's error messages are precise once you know the pattern. The most common ones:

| Message contains | What it means | Usual fix |
|------------------|---------------|-----------|
| `mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)` | inner dims of a matmul differ | a missing `.T` / `transpose(-2, -1)`, or the wrong weight layout (P6) |
| `The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 0` | broadcasting failed | align dims with `unsqueeze` / `[:, None]` (P2) |
| `expected ... same dtype` / `Found dtype Double but expected Float` | float64 mixed with float32 | `.to(x.dtype)`; `torch.tensor(...)` from NumPy is float64 |
| `Expected all tensors to be on the same device` | CPU tensor meets GPU tensor | create new tensors with `device=x.device` (P1) |
| `element 0 of tensors does not require grad` | `.backward()` on something not connected to a parameter | a `detach()`, `no_grad()`, or `.item()` broke the graph |
| `Trying to backward through the graph a second time` | the graph was freed by the first `.backward()` | recompute the forward pass, or `retain_graph=True` if you really need it twice |
| `modified by an inplace operation` | a tensor saved for backward was changed in place | replace `x.mul_(...)` / `x += ...` with an out-of-place version |
| `view size is not compatible with input tensor's size and stride` | `.view()` on a non-contiguous tensor | `.reshape()` or `.contiguous().view()` (P5) |
'''),
        ("code", r'''
def first_line(fn):
    try:
        fn()
    except Exception as e:
        print(f"{type(e).__name__}: {str(e).splitlines()[0][:95]}")

first_line(lambda: torch.randn(2, 3) @ torch.randn(2, 3))
first_line(lambda: torch.randn(3) + torch.randn(4))
first_line(lambda: torch.randn(2, 2) @ torch.randn(2, 2, dtype=torch.float64))
first_line(lambda: torch.randn(2).sum().backward())

x = torch.randn(3, requires_grad=True)
y = (x * x).sum(); y.backward()
first_line(lambda: y.backward())

def inplace_bug():
    s = torch.randn(3, requires_grad=True).sigmoid()   # sigmoid saves its output for backward
    s.mul_(2)                                         # ... which is then overwritten
    s.sum().backward()
first_line(inplace_bug)
first_line(lambda: torch.randn(3, 4).T.view(-1))
'''),
        ("md", r'''
## 2. Comparing tensors

Never compare floats with `==`. **`torch.testing.assert_close(actual, expected)`** checks values *and* shape, dtype and device, uses tolerances suited to the dtype, and on failure reports **how many** elements differ, the **largest** difference, and **where**. Pass `rtol=` / `atol=` to loosen it, and `check_dtype=False` when comparing across dtypes on purpose.
'''),
        ("code", r'''
a = torch.tensor([1.0, 2.0, 3.0])
torch.testing.assert_close(a + 1e-7, a)                    # passes: within float32 tolerance
try:
    torch.testing.assert_close(torch.tensor([1.0, 2.5, 3.0]), a)
except AssertionError as e:
    print(str(e))
'''),
        ("md", r'''
## 3. Finding where `NaN` first appears

A `NaN` in the loss is a symptom; the cause is usually one specific layer (a `log` of a negative number, a division by zero, an overflowing `exp`). **Forward hooks** (P8) let you check every module's output without editing the model: register a hook on each module, record the **first** one whose output isn't finite, then remove the hooks.
For the backward pass, `torch.autograd.detect_anomaly()` (P7) does the same job.
'''),
        ("code", r'''
class Log(nn.Module):
    def forward(self, x):
        return torch.log(x)

model = nn.Sequential(nn.Linear(4, 4), Log(), nn.Linear(4, 2))
seen = []
handles = [m.register_forward_hook(lambda mod, inp, out, name=name: seen.append((name, torch.isfinite(out).all().item())))
           for name, m in model.named_modules() if name]
model(torch.randn(3, 4))
for h in handles:
    h.remove()
print(seen)      # the Log module (named "1") is the first with a non-finite output
'''),
        ("md", r'''
## 4. Gradient health

After `loss.backward()`, check two things:
- **Parameters with `.grad is None`** were not part of the computation at all: an unused layer, a branch that was skipped, an expert that no token was routed to (#28 MoE), or a parameter accidentally detached. In multi-GPU training (DDP) they even cause a crash.
- **Gradient norms per parameter.** A norm that is huge (exploding) or exactly zero (dead) in one layer points straight at the problem, and #21's clipping hides it rather than fixing it.
'''),
        ("code", r'''
class TwoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 4)
        self.used = nn.Linear(4, 1)
        self.unused = nn.Linear(4, 1)        # never called in forward
    def forward(self, x):
        return self.used(torch.relu(self.body(x)))

m = TwoHeads()
m(torch.randn(8, 4)).sum().backward()
for name, p in m.named_parameters():
    print(f"{name:<14} grad: {'None' if p.grad is None else f'norm {p.grad.norm():.3f}'}")
'''),
        ("md", r'''
## 5. A debugging workflow that saves time

1. **Shrink the problem:** batch of 2, sequence of 4, hidden size 8. Errors appear in seconds, and you can print whole tensors.
2. **Fix the seed** (`torch.manual_seed`) so a bug reproduces every time.
3. **Assert shapes** at every step you're unsure about: `assert q.shape == (B, H, S, d), q.shape`.
4. **Compare against a reference:** a slow, obviously correct loop, or the built-in op (`F.scaled_dot_product_attention`, `nn.LayerNorm`) with `assert_close`.
5. **Overfit one batch:** a model that can't drive the loss near zero on a single batch has a bug, not a tuning problem.
'''),
        ("code", r'''
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
x, y = torch.randn(8, 4), torch.randn(8, 1)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(300):
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
print(f"loss after overfitting one batch: {loss.item():.5f}")   # close to 0: the pipeline can learn
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
### Exercise 1: `first_nan_module(model, x)`

Find the first module whose output contains `NaN` or `inf`.

- `model`: an `nn.Module`; `x`: its input
- run `model(x)` once (no gradients needed) with a forward hook on every **leaf** module (modules with no children)
- returns: the name (as in `model.named_modules()`) of the **first** leaf module, in execution order, whose output has a non-finite value; `None` if all outputs are finite
- **remove every hook** before returning, even if the forward pass raises

*If a module returns a tuple, check every tensor in it.*
''',
            "stub": r'''
def first_nan_module(model: nn.Module, x: torch.Tensor) -> str | None:
    pass  # Replace this
''',
            "solution": r'''
def first_nan_module(model: nn.Module, x: torch.Tensor) -> str | None:
    found = []

    def make_hook(name):
        def hook(module, inputs, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            if not found and any(isinstance(o, torch.Tensor) and not torch.isfinite(o).all() for o in outs):
                found.append(name)
        return hook

    handles = [m.register_forward_hook(make_hook(name))
               for name, m in model.named_modules() if len(list(m.children())) == 0]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return found[0] if found else None
''',
            "debug": r'''
class Log(nn.Module):
    def forward(self, x):
        return torch.log(x)

model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), Log(), nn.Linear(4, 2))
print(first_nan_module(model, torch.randn(5, 4)))     # expect '2': log(0) = -inf after the ReLU zeros
print(first_nan_module(nn.Sequential(nn.Linear(4, 2)), torch.randn(5, 4)))   # expect None
''',
        },
        {
            "spec": r'''
### Exercise 2: `params_without_grad(model)`

After `loss.backward()`, list the trainable parameters that received **no** gradient.

- returns: a list of parameter names (as in `model.named_parameters()`, in that order) with `requires_grad=True` and `.grad is None`
- frozen parameters (`requires_grad=False`) are **not** problems: leave them out
''',
            "stub": r'''
def params_without_grad(model: nn.Module) -> list[str]:
    pass  # Replace this
''',
            "solution": r'''
def params_without_grad(model: nn.Module) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
''',
            "debug": r'''
class TwoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.used, self.unused = nn.Linear(4, 1), nn.Linear(4, 1)
    def forward(self, x):
        return self.used(x)

m = TwoHeads()
m(torch.randn(3, 4)).sum().backward()
print(params_without_grad(m))     # expect ['unused.weight', 'unused.bias']
''',
        },
        {
            "spec": r'''
### Exercise 3: `diff_report(a, b, rtol=1e-5, atol=1e-8)`

Summarize how two same-shaped tensors differ, like `assert_close` does when it fails.

- returns a dict with:
  - `"max_abs_diff"`: the largest `|a - b|`, as a Python float
  - `"index"`: where it occurs, as a tuple of Python ints (one per dim)
  - `"a"`, `"b"`: the two values at that index, as Python floats
  - `"num_mismatched"`: how many elements have `|a - b| > atol + rtol * |b|`, as a Python int
- a `NaN` in either tensor counts as an **infinite** difference (and as a mismatch)
''',
            "stub": r'''
def diff_report(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> dict:
    pass  # Replace this
''',
            "solution": r'''
def diff_report(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> dict:
    diff = (a - b).abs()
    diff = torch.where(torch.isnan(diff), torch.full_like(diff, float("inf")), diff)
    flat = int(diff.argmax())
    index = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), diff.shape))
    return {
        "max_abs_diff": diff.max().item(),
        "index": index,
        "a": a[index].item(),
        "b": b[index].item(),
        "num_mismatched": int((diff > atol + rtol * b.abs()).sum()),
    }
''',
            "debug": r'''
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[1.0, 2.5], [3.0, 4.0 + 1e-9]])
print(diff_report(a, b))
# expect {'max_abs_diff': 0.5, 'index': (0, 1), 'a': 2.0, 'b': 2.5, 'num_mismatched': 1}
''',
        },
        {
            "spec": r'''
### Exercise 4: `grad_norms_by_name(model)`

The L2 norm of each parameter's gradient, for spotting exploding or dead layers.

- returns: a dict `{name: norm}` with Python floats, for every parameter whose `.grad` is not `None` (in `named_parameters()` order)
''',
            "stub": r'''
def grad_norms_by_name(model: nn.Module) -> dict[str, float]:
    pass  # Replace this
''',
            "solution": r'''
def grad_norms_by_name(model: nn.Module) -> dict[str, float]:
    return {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
''',
            "debug": r'''
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
model(torch.randn(8, 4)).pow(2).sum().backward()
for name, norm in grad_norms_by_name(model).items():
    print(f"{name:<10} {norm:.4f}")
''',
        },
    ],
    # Problem number -> how to use this notebook on it ("Where to use it" section)
    "unlocks": [
        (13, "Feed a GPT-2 block extreme inputs and locate the first non-finite layer with `first_nan_module`"),
        (21, "Print `grad_norms_by_name` before and after clipping"),
        (25, "Check your Flash Attention against plain attention with `diff_report` / `assert_close`"),
        (26, "After one backward, `params_without_grad` should be empty and only the LoRA matrices should have norms"),
        (28, "Route a batch so one expert gets no tokens, then find it with `params_without_grad`"),
    ],
}
