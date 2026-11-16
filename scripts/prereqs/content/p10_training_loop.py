"""Prerequisite 10: Training loop & randomness.

Content for templates/00_p10_training_loop.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P10",
    "file": "00_p10_training_loop",
    "task_id": "p10_training_loop",
    "title": "Training Loop & Randomness",
    "imports": "import torch\nfrom torch import nn",
    "intro": r'''
# 🔵 Prerequisite 10: Training Loop & Randomness

Most "implement X" problems end with "…and train it". Several more (Dropout, top-k sampling, speculative decoding) are about **randomness**: drawing masks or tokens with the right probabilities, **reproducibly**.

This notebook covers the canonical training loop, how optimizers expose their learning rate, and how to use PyTorch's random number generators so that tests (and experiments) give the same result every run.

### What you'll learn
1. The canonical loop: `zero_grad → forward → loss → backward → step`
2. Optimizer `param_groups` and changing the learning rate
3. Seeds and `torch.Generator` for reproducible randomness
4. Random draws: `rand`, `bernoulli`, `multinomial`, `randperm`
5. Shuffling and mini-batching a dataset
6. Evaluation: `model.eval()` plus `torch.no_grad()`
7. `Dataset`, `DataLoader`, and learning-rate schedulers
8. Many ways to do it: batching and scheduling

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `train_steps(model, optimizer, loss_fn, x, y, steps)` | #40 Linear Regression, #31 Gradient Accumulation |
| 2 | `set_lr_(optimizer, lr)` | #30 Cosine LR Scheduler |
| 3 | `keep_mask(shape, p_drop, generator)` | #17 Dropout |
| 4 | `sample_tokens(probs, n, generator)` | #32 Top-k/p Sampling, #34 Speculative Decoding |
| 5 | `shuffled_batches(X, y, batch_size, generator)` | mini-batch training, #31 |

### Rules
- **No Python loops** over tensor elements. Looping over training steps or batches is fine.
- Use the `generator` you are given for every random draw.
''',
    "lesson": [
        ("md", r'''
## 1. The canonical training loop

```python
for step in range(steps):
    optimizer.zero_grad()        # 1. clear old gradients (they accumulate, see P7)
    pred = model(x)              # 2. forward
    loss = loss_fn(pred, y)      # 3. loss (a scalar)
    loss.backward()              # 4. gradients into p.grad
    optimizer.step()             # 5. update the parameters
```
Log the loss with `loss.item()`. Keeping the tensor `loss` in a list keeps its whole graph alive.
'''),
        ("code", r'''
torch.manual_seed(0)
model = nn.Linear(3, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.randn(32, 3)
y = x @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.3
for step in range(50):
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
print(f"final loss {loss.item():.5f}", model.weight.data, model.bias.data)
'''),
        ("md", r'''
## 2. `param_groups` and the learning rate

An optimizer keeps its hyperparameters in `optimizer.param_groups`, a list of dicts (one per group of parameters). LR schedulers (#30) just write a new `"lr"` into every group before each step.
'''),
        ("code", r'''
opt = torch.optim.SGD([
    {"params": [torch.zeros(2, requires_grad=True)], "lr": 0.1},
    {"params": [torch.zeros(3, requires_grad=True)], "lr": 0.01},
])
print([g["lr"] for g in opt.param_groups])
for g in opt.param_groups:
    g["lr"] = 0.5
print([g["lr"] for g in opt.param_groups])
'''),
        ("md", r'''
## 3. Seeds and generators

`torch.manual_seed(s)` resets the **global** generator, which every random call uses by default.
A `torch.Generator` is a **private** random stream. Pass it as `generator=g`, and your function's randomness no longer depends on what other code drew before it. Tests use this to check that your output is reproducible.
'''),
        ("code", r'''
g1 = torch.Generator().manual_seed(42)
g2 = torch.Generator().manual_seed(42)
torch.manual_seed(0); a = torch.rand(3, generator=g1)
torch.manual_seed(999); b = torch.rand(3, generator=g2)   # global seed doesn't matter
print(a, b, torch.equal(a, b))
'''),
        ("md", r'''
## 4. Random draws

| Want | Use |
|------|-----|
| Uniform in `[0, 1)` | `torch.rand(shape, generator=g)` |
| Coin flips with probability `p` | `torch.rand(shape, generator=g) < p`, or `torch.bernoulli(probs, generator=g)` |
| Categorical samples | `torch.multinomial(probs, n, replacement=True, generator=g)` |
| A random permutation | `torch.randperm(n, generator=g)` |

`multinomial` takes **probabilities**, or any non-negative weights. They don't need to sum to 1, but they must not be logits.
'''),
        ("code", r'''
g = torch.Generator().manual_seed(0)
probs = torch.tensor([0.7, 0.2, 0.1])
draws = torch.multinomial(probs, 10_000, replacement=True, generator=g)
print(torch.bincount(draws, minlength=3) / 10_000)    # close to [0.7, 0.2, 0.1]
print(torch.randperm(8, generator=g))
'''),
        ("md", r'''
## 5. Shuffling and mini-batches

Shuffle once per epoch with a permutation, then slice it into batches. Index `X` and `y` with the **same** permutation so the pairs stay together. `torch.split` slices along dim 0.
'''),
        ("code", r'''
X = torch.arange(10).float()[:, None]
g = torch.Generator().manual_seed(1)
perm = torch.randperm(10, generator=g)
print([chunk.flatten().tolist() for chunk in torch.split(X[perm], 4)])   # sizes 4, 4, 2
'''),
        ("md", r'''
## 6. Evaluation

For evaluation, switch the model's behavior with `model.eval()` (Dropout off, BatchNorm uses running stats; see P8), **and** stop building graphs with `torch.no_grad()`. These are two separate things, and you usually want both.
'''),
        ("code", r'''
model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
model.eval()
with torch.no_grad():
    out = model(torch.randn(2, 4))
print(out.requires_grad, model.training)   # False False
model.train()
'''),
        ("md", r'''
## 7. `Dataset`, `DataLoader`, and LR schedulers

- A **`Dataset`** has `__len__` and `__getitem__(i)`. `TensorDataset(X, y)` wraps tensors that are already in memory.
- A **`DataLoader`** batches, shuffles (`shuffle=True`, with `generator=g` for reproducibility), can drop the last short batch (`drop_last=True`), and can load in parallel (`num_workers`). It stacks samples with a **`collate_fn`**; write your own for variable-length data.
- **`torch.optim.lr_scheduler`** changes the learning rate over time: `LambdaLR`, `CosineAnnealingLR`, `OneCycleLR`, … Call `scheduler.step()` **after** `optimizer.step()`.
'''),
        ("code", r'''
from torch.utils.data import DataLoader, Dataset, TensorDataset

X, y = torch.arange(10.)[:, None], torch.arange(10.)
loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))
print([xb.flatten().tolist() for xb, _ in loader])          # shuffled, sizes 4, 4, 2

class Squares(Dataset):
    def __len__(self): return 5
    def __getitem__(self, i): return torch.tensor(float(i)), torch.tensor(float(i * i))
print(next(iter(DataLoader(Squares(), batch_size=5))))

model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=1.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5 ** step)
lrs = []
for _ in range(4):
    opt.step(); sched.step(); lrs.append(opt.param_groups[0]["lr"])
print(lrs)                                                 # [0.5, 0.25, 0.125, 0.0625]
'''),
        ("md", r'''
## 8. Many ways to do it

| Goal | Equivalent forms | Trade-off |
|------|------------------|-----------|
| Shuffled mini-batches | `torch.randperm` + `torch.split` (Exercise 5), `DataLoader(TensorDataset(X, y), shuffle=True)` | `DataLoader` adds workers, pinning, and custom collation; the manual form is simpler for in-memory tensors |
| Change the LR | write `group["lr"]` yourself (Exercise 2), a built-in `lr_scheduler` | built-ins are tested and save/restore with `state_dict()`; manual is clearer for a one-off schedule like #30 |
| Clear gradients | `optimizer.zero_grad()`, `model.zero_grad()`, `p.grad = None` | the optimizer clears only **its** parameters; the model clears **all** of its parameters |
| Reproducible randomness | `torch.manual_seed(s)`, a `torch.Generator` per component | a generator isolates one component's randomness from everything else |
'''),
        ("code", r'''
X, y = torch.arange(10.)[:, None], torch.arange(10.)
g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
manual = [b.flatten().tolist() for b in torch.split(X[torch.randperm(10, generator=g1)], 4)]
loaded = [xb.flatten().tolist() for xb, _ in DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True, generator=g2)]
print(manual)
print(loaded)    # same batch sizes; the order can differ (DataLoader draws its permutation differently)
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
### Exercise 1: `train_steps(model, optimizer, loss_fn, x, y, steps)`

Run the canonical training loop for `steps` full-batch steps.

- `model`: an `nn.Module`; `optimizer`: built on `model.parameters()`; `loss_fn(pred, y)` returns a scalar
- each step: clear gradients, forward, loss, backward, optimizer step
- returns: a list of `steps` **Python floats**, the loss at each step (computed before that step's update)
''',
            "stub": r'''
def train_steps(model: nn.Module, optimizer, loss_fn, x: torch.Tensor, y: torch.Tensor,
                steps: int) -> list[float]:
    pass  # Replace this
''',
            "solution": r'''
def train_steps(model: nn.Module, optimizer, loss_fn, x: torch.Tensor, y: torch.Tensor,
                steps: int) -> list[float]:
    model.train()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
''',
            "debug": r'''
torch.manual_seed(0)
model = nn.Linear(3, 1)
x = torch.randn(32, 3); y = x.sum(dim=1, keepdim=True)
losses = train_steps(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.MSELoss(), x, y, steps=30)
print(f"{losses[0]:.3f} -> {losses[-1]:.5f}", type(losses[0]))   # expect a falling loss, <class 'float'>
''',
        },
        {
            "spec": r'''
### Exercise 2: `set_lr_(optimizer, lr)`

Set the learning rate of **every** parameter group of an optimizer, in place.

- returns: nothing

*An LR scheduler (#30) computes `lr` for the current step and does exactly this.*
''',
            "stub": r'''
def set_lr_(optimizer, lr: float) -> None:
    pass  # Replace this
''',
            "solution": r'''
def set_lr_(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
''',
            "debug": r'''
opt = torch.optim.SGD([{"params": [torch.zeros(1, requires_grad=True)], "lr": 0.1},
                       {"params": [torch.zeros(1, requires_grad=True)], "lr": 0.2}])
set_lr_(opt, 0.05)
print([g["lr"] for g in opt.param_groups])   # expect [0.05, 0.05]
''',
        },
        {
            "spec": r'''
### Exercise 3: `keep_mask(shape, p_drop, generator)`

Draw a dropout **keep** mask: each entry is `True` with probability `1 - p_drop`, independently.

- `shape`: a tuple of ints
- `p_drop`: float in `[0, 1]`; `p_drop=0` keeps everything, `p_drop=1` keeps nothing
- `generator`: a `torch.Generator`; use it for the random draw
- returns: a `bool` tensor of the given shape

*#17 Dropout multiplies by this mask and rescales by `1 / (1 - p)`.*
''',
            "stub": r'''
def keep_mask(shape: tuple, p_drop: float, generator: torch.Generator) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def keep_mask(shape: tuple, p_drop: float, generator: torch.Generator) -> torch.Tensor:
    return torch.rand(shape, generator=generator) >= p_drop
''',
            "debug": r'''
g = torch.Generator().manual_seed(0)
m = keep_mask((1000, 100), 0.3, g)
print(m.dtype, m.float().mean())   # expect torch.bool, about 0.7
''',
        },
        {
            "spec": r'''
### Exercise 4: `sample_tokens(probs, n, generator)`

Draw `n` token ids, with replacement, from a categorical distribution.

- `probs`: `(V,)` non-negative probabilities summing to 1
- `n`: number of samples
- `generator`: a `torch.Generator`; use it for the draw
- returns: `(n,)` long tensor of token ids in `[0, V)`

*#32 samples the next token from the filtered distribution; #34 re-samples from the adjusted one.*
''',
            "stub": r'''
def sample_tokens(probs: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def sample_tokens(probs: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    return torch.multinomial(probs, n, replacement=True, generator=generator)
''',
            "debug": r'''
g = torch.Generator().manual_seed(0)
draws = sample_tokens(torch.tensor([0.5, 0.0, 0.3, 0.2]), 20_000, g)
print(torch.bincount(draws, minlength=4) / 20_000)   # expect about [0.5, 0.0, 0.3, 0.2]
''',
        },
        {
            "spec": r'''
### Exercise 5: `shuffled_batches(X, y, batch_size, generator)`

Shuffle a dataset once and split it into mini-batches.

- `X`: `(N, ...)`, `y`: `(N, ...)`, paired row by row
- `batch_size`: int; the last batch may be smaller
- `generator`: a `torch.Generator`; use it for the shuffle
- returns: a list of `(X_batch, y_batch)` tuples that together contain **every row exactly once**, in a random order, with `X[i]` still paired with `y[i]`
''',
            "stub": r'''
def shuffled_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                     generator: torch.Generator) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pass  # Replace this
''',
            "solution": r'''
def shuffled_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                     generator: torch.Generator) -> list[tuple[torch.Tensor, torch.Tensor]]:
    perm = torch.randperm(X.shape[0], generator=generator)
    return list(zip(torch.split(X[perm], batch_size), torch.split(y[perm], batch_size)))
''',
            "debug": r'''
X = torch.arange(10).float()[:, None]
y = X.squeeze(1) * 10
g = torch.Generator().manual_seed(0)
for xb, yb in shuffled_batches(X, y, 4, g):
    print(xb.flatten().tolist(), yb.tolist())   # sizes 4, 4, 2; y is always 10 * x
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (17, "`keep_mask`: the random keep/drop decision behind Dropout"),
        (29, "The `zero_grad` / `step` protocol your optimizer must follow"),
        (30, "`set_lr_`: a scheduler writes the new LR into every param group"),
        (31, "The training loop, with micro-batches like `shuffled_batches`"),
        (32, "`sample_tokens`: drawing the next token with `multinomial`"),
        (34, "Random acceptance with `torch.rand`, and re-sampling with `multinomial`"),
        (40, "`train_steps`: the `nn.Linear` + SGD method"),
    ],
}
