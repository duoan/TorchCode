"""Prerequisite 8: nn.Module anatomy.

Content for templates/00_p08_nn_module.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P8",
    "file": "00_p08_nn_module",
    "task_id": "p08_nn_module",
    "title": "nn.Module Anatomy",
    "imports": "import torch\nfrom torch import nn",
    "intro": r'''
# 🔵 Prerequisite 8: nn.Module Anatomy

Interviewers almost always expect layers written as `nn.Module` subclasses, and many TorchCode problems (Dropout, Embedding, SwiGLU, GPT-2 block, LoRA, MoE, INT8 Linear) are graded on module behavior: which tensors are **parameters**, which are **buffers**, whether submodules are **registered**, and what changes between **train** and **eval** mode.

A module that computes the right numbers can still fail. If its weights are not registered, the optimizer never sees them, `.to(device)` leaves them behind, and `state_dict()` doesn't save them.

### What you'll learn
1. The skeleton: `super().__init__()`, `forward`, calling `model(x)`
2. Parameters vs buffers vs plain tensor attributes
3. Registering submodules: `ModuleList` / `Sequential`, not Python lists
4. `train()` / `eval()` and `self.training`
5. Freezing parameters and counting trainable ones

### Exercises
| # | Function / class | Shows up in |
|---|------------------|-------------|
| 1 | `ScaleShift(dim)` | #03 Linear, #04 LayerNorm (affine), #18 Embedding |
| 2 | `RunningMean(dim, momentum)` | #07 BatchNorm, #17 Dropout, #36 INT8 |
| 3 | `MLPStack(dims)` | #13 GPT-2 Block, #15 SwiGLU, #28 MoE |
| 4 | `freeze_(module)` | #26 LoRA |
| 5 | `num_trainable(module)` | #26 LoRA |

### Rules
- Build modules from `nn.Parameter`, `register_buffer`, `nn.Linear`, `nn.ReLU`, `nn.ModuleList`, and `nn.Sequential`.
- No Python loops over tensor elements.
''',
    "lesson": [
        ("md", r'''
## 1. The skeleton

```python
class MyLayer(nn.Module):
    def __init__(self, ...):
        super().__init__()          # must come first: sets up registration
        ...                         # create parameters / submodules
    def forward(self, x):
        ...
```
Call the module as `layer(x)`, not `layer.forward(x)`. `__call__` runs hooks and then `forward`.
'''),
        ("code", r'''
class Doubler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 2 * x

print(Doubler()(torch.ones(3)))
try:
    class Broken(nn.Module):
        def __init__(self):
            self.w = nn.Parameter(torch.ones(1))   # forgot super().__init__()
    Broken()
except AttributeError as e:
    print("AttributeError:", str(e)[:70])
'''),
        ("md", r'''
## 2. Parameters vs buffers vs plain tensors

| Stored as | Trained by the optimizer? | In `state_dict()`? | Moved by `.to(device)`? |
|-----------|:---:|:---:|:---:|
| `self.w = nn.Parameter(t)` | ✅ | ✅ | ✅ |
| `self.register_buffer("m", t)` | ❌ | ✅ | ✅ |
| `self.t = t` (plain tensor) | ❌ | ❌ | ❌ |

Use a **buffer** for state that is not learned but belongs to the model: running statistics, quantized weights, masks.
'''),
        ("code", r'''
class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.register_buffer("running", torch.zeros(2))
        self.plain = torch.zeros(2)

d = Demo()
print([n for n, _ in d.named_parameters()])   # ['w']
print([n for n, _ in d.named_buffers()])      # ['running']
print(list(d.state_dict().keys()))            # ['w', 'running']: plain is missing
'''),
        ("md", r'''
## 3. Registering submodules

A module assigned as an attribute (`self.fc = nn.Linear(...)`) is registered automatically.
A module inside a **Python list** is **not**: its parameters are invisible to `parameters()`, the optimizer, and `state_dict()`. Use `nn.ModuleList` (indexable, you write the loop) or `nn.Sequential` (runs in order).
'''),
        ("code", r'''
class BadStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(4, 4), nn.Linear(4, 4)]          # plain list

class GoodStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

print(len(list(BadStack().parameters())), len(list(GoodStack().parameters())))   # 0 4
'''),
        ("md", r'''
## 4. `train()` and `eval()`

`model.train()` and `model.eval()` set `self.training` on the module **and every submodule**. Layers like Dropout and BatchNorm check it in `forward` (#17, #07).
`eval()` does **not** turn off gradients. Use `torch.no_grad()` for that.
'''),
        ("code", r'''
drop = nn.Dropout(0.5)
x = torch.ones(8)
drop.train(); print(drop.training, drop(x))   # random zeros, survivors scaled by 2
drop.eval();  print(drop.training, drop(x))   # identity
'''),
        ("md", r'''
## 5. Freezing

`p.requires_grad_(False)` freezes a parameter: no gradient is computed for it, and an optimizer skips it. Freezing a pretrained base and training only small adapters is how LoRA works (#26).
Count trainable parameters with `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
'''),
        ("code", r'''
lin = nn.Linear(10, 5)
print(sum(p.numel() for p in lin.parameters() if p.requires_grad))   # 55
lin.weight.requires_grad_(False)
print(sum(p.numel() for p in lin.parameters() if p.requires_grad))   # 5
'''),
    ],
    "exercises_intro": r'''
---
# ✏️ Exercises

Implement the three classes and two functions below. Each one has a debug cell you can use before you submit.
''',
    "exercises": [
        {
            "spec": r'''
### Exercise 1: `ScaleShift(dim)`

A learnable per-feature affine transform, like the `gamma` / `beta` at the end of LayerNorm.

- parameters: `gamma` of shape `(dim,)` initialized to **ones**, `beta` of shape `(dim,)` initialized to **zeros**
- `forward(x)`: `x` is `(..., dim)`; returns `x * gamma + beta`
- `gamma` and `beta` must be registered parameters, named exactly `gamma` and `beta`
''',
            "stub": r'''
class ScaleShift(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        pass  # create gamma and beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass  # Replace this
''',
            "solution": r'''
class ScaleShift(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma + self.beta
''',
            "debug": r'''
m = ScaleShift(4)
print([n for n, _ in m.named_parameters()])   # expect ['gamma', 'beta']
print(m(torch.randn(2, 3, 4)).shape)          # expect torch.Size([2, 3, 4])
''',
        },
        {
            "spec": r'''
### Exercise 2: `RunningMean(dim, momentum=0.1)`

Center features by their mean, tracking a running mean for evaluation, as BatchNorm does for its mean.

- buffer `running_mean` of shape `(dim,)`, initialized to zeros. It must be a **buffer**, not a parameter.
- `forward(x)` with `x` of shape `(N, dim)`:
  - **training** (`self.training`): compute `batch_mean = x.mean(dim=0)`, update **in place** `running_mean ← (1 - momentum) * running_mean + momentum * batch_mean`, then return `x - batch_mean`
  - **eval**: return `x - running_mean`, with no update
- the running mean must not be part of the autograd graph
''',
            "stub": r'''
class RunningMean(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.1):
        super().__init__()
        pass  # create the running_mean buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass  # Replace this
''',
            "solution": r'''
class RunningMean(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean(dim=0)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
            return x - batch_mean
        return x - self.running_mean
''',
            "debug": r'''
m = RunningMean(2, momentum=0.5)
x = torch.tensor([[1.0, 10.0], [3.0, 30.0]])
print(m(x))                 # expect [[-1., -10.], [1., 10.]]
print(m.running_mean)       # expect tensor([ 1., 10.])  = 0.5 * 0 + 0.5 * [2, 20]
m.eval()
print(m(x))                 # expect x - [1, 10]
print([n for n, _ in m.named_buffers()], list(m.parameters()))   # expect ['running_mean'] []
''',
        },
        {
            "spec": r'''
### Exercise 3: `MLPStack(dims)`

A stack of linear layers with ReLU in between.

- `dims`: a list of at least two ints, e.g. `[16, 32, 32, 4]` → `Linear(16, 32) → ReLU → Linear(32, 32) → ReLU → Linear(32, 4)`
- ReLU goes **between** layers, **not** after the last one
- every `nn.Linear` must be registered, so `parameters()` and `state_dict()` include it
- `forward(x)`: `x` is `(..., dims[0])`; returns `(..., dims[-1])`
''',
            "stub": r'''
class MLPStack(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        pass  # build and register the layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass  # Replace this
''',
            "solution": r'''
class MLPStack(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(d_in, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
''',
            "debug": r'''
m = MLPStack([16, 32, 4])
print(m)
print(sum(p.numel() for p in m.parameters()))   # expect 16*32 + 32 + 32*4 + 4 = 676
print(m(torch.randn(5, 16)).shape)              # expect torch.Size([5, 4])
''',
        },
        {
            "spec": r'''
### Exercise 4: `freeze_(module)`

Freeze every parameter of a module (including its submodules) **in place**, and return the module.

- after the call, no parameter has `requires_grad=True`
- returns: the **same** module object
''',
            "stub": r'''
def freeze_(module: nn.Module) -> nn.Module:
    pass  # Replace this
''',
            "solution": r'''
def freeze_(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module
''',
            "debug": r'''
m = MLPStack([4, 8, 2])
print(freeze_(m) is m, [p.requires_grad for p in m.parameters()])   # expect True [False, False, False, False]
''',
        },
        {
            "spec": r'''
### Exercise 5: `num_trainable(module)`

Count the trainable **elements** (not tensors) of a module: the total `numel()` of the parameters with `requires_grad=True`.

- returns: a Python `int`

*LoRA's selling point (#26) is how small this number gets after freezing the base weights.*
''',
            "stub": r'''
def num_trainable(module: nn.Module) -> int:
    pass  # Replace this
''',
            "solution": r'''
def num_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
''',
            "debug": r'''
lin = nn.Linear(10, 5)
print(num_trainable(lin))       # expect 55
lin.bias.requires_grad_(False)
print(num_trainable(lin))       # expect 50
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (3, "Registering `weight` and `bias` so an optimizer can find them, like `ScaleShift`"),
        (13, "Registered submodules (`nn.Linear`, `nn.LayerNorm`, `nn.Sequential`) inside one block"),
        (14, "A module whose `forward` also returns state (the KV cache)"),
        (15, "Three registered `nn.Linear` projections in one module"),
        (17, "`self.training` decides the behavior, as in `RunningMean`"),
        (18, "One `nn.Parameter` lookup table, like `ScaleShift`'s parameters"),
        (23, "Separate registered projections for queries and keys/values"),
        (26, "`freeze_` the base layer, train only the adapters, check `num_trainable`"),
        (27, "A registered projection applied to every patch"),
        (28, "`nn.ModuleList` of experts, like `MLPStack`'s registered layers"),
        (36, "`register_buffer` for the int8 weight and scale, like `RunningMean`"),
    ],
}
