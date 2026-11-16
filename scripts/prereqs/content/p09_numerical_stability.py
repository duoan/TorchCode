"""Prerequisite 9: Numerical stability & log-space.

Content for templates/00_p09_numerical_stability.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P9",
    "file": "00_p09_numerical_stability",
    "task_id": "p09_numerical_stability",
    "title": "Numerical Stability & Log-Space",
    "imports": "import torch\nimport torch.nn.functional as F",
    "intro": r'''
# 🔵 Prerequisite 9: Numerical Stability & Log-Space

A formula can be mathematically correct and still return `inf` or `NaN` in float32. Softmax overflows, `log(softmax)` underflows to `-inf`, `log(sigmoid(x))` breaks for very negative `x`, and a fully masked attention row turns into `NaN` that spreads through every gradient.

Language-model losses (cross-entropy, DPO, PPO, GRPO, distillation) all work with **log-probabilities** for exactly this reason. Interviewers often ask "what happens if the logits are 1000?" or "why log-softmax instead of log(softmax)?"

### What you'll learn
1. The limits of float32: where `exp` overflows and `log` gives `-inf`
2. The max-subtraction trick and `logsumexp`
3. `log_softmax` vs `log(softmax)`
4. Stable `log(sigmoid(x))`
5. Working in log-space: sums instead of products, ratios as `exp(a - b)`
6. The `torch.where` NaN-gradient trap, and fully masked rows

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `stable_logsumexp(x, dim)` | #02 Softmax, #16 Cross-Entropy, #25 Flash Attention |
| 2 | `log_softmax(x, dim)` | #16 Cross-Entropy, #41 OPD |
| 3 | `log_sigmoid(x)` | #37 DPO |
| 4 | `kl_from_logits(p_logits, q_logits)` | #41 OPD, distillation |
| 5 | `safe_masked_softmax(scores, mask)` | masked attention with padding |

### Rules
- **No Python loops** over tensor elements.
- Do **not** use `torch.logsumexp`, `torch.softmax`, `F.log_softmax`, `F.logsigmoid`, `F.softplus`, `torch.sigmoid`, or `F.kl_div`. The point is to build them stably yourself. Later exercises may reuse your earlier ones.
''',
    "lesson": [
        ("md", r'''
## 1. The limits of float32

`exp` overflows to `inf` just above `88.7`, and underflows to `0` below about `-103`. `log(0)` is `-inf`.
Once an `inf` appears, `inf - inf` and `0 * inf` give `NaN`, and `NaN` spreads through every later operation.
'''),
        ("code", r'''
print(torch.exp(torch.tensor([88.0, 89.0, -104.0])))       # [1.65e38, inf, 0.]
print(torch.log(torch.tensor(0.0)))                         # -inf
inf = torch.tensor(float("inf"))
print(inf - inf, 0 * inf)                                   # nan nan
'''),
        ("md", r'''
## 2. Max-subtraction and `logsumexp`

Softmax doesn't change if you subtract a constant from every logit, so subtract the **row max**. The largest exponent is then `exp(0) = 1`, and nothing overflows.
The same trick gives a stable `logsumexp`:

$$\log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m}, \qquad m = \max_i x_i$$
'''),
        ("code", r'''
x = torch.tensor([1000.0, 1001.0, 1002.0])
naive = torch.exp(x) / torch.exp(x).sum()
shifted = torch.exp(x - x.max()) / torch.exp(x - x.max()).sum()
print(naive)        # nan: inf / inf
print(shifted)      # [0.0900, 0.2447, 0.6652]
m = x.max()
print(m + torch.log(torch.exp(x - m).sum()), torch.logsumexp(x, dim=0))   # same value
'''),
        ("md", r'''
## 3. `log_softmax` vs `log(softmax)`

A tiny probability can underflow to `0` in `softmax`, and then `log` gives `-inf`. Computing directly in log-space keeps it finite:

$$\log\mathrm{softmax}(x)_i = x_i - \mathrm{logsumexp}(x)$$

That is why cross-entropy is built on `log_softmax`, never on `log(softmax(...))`.
'''),
        ("code", r'''
x = torch.tensor([0.0, -200.0])
print(torch.log(torch.softmax(x, dim=0)))   # [0., -inf]
print(F.log_softmax(x, dim=0))              # [0., -200.]
'''),
        ("md", r'''
## 4. Stable `log(sigmoid(x))`

DPO's loss is `-log σ(z)`. For very negative `z`, `σ(z)` underflows to `0` and the log becomes `-inf`.
A stable form, valid for every `x`:

$$\log\sigma(x) = \min(x, 0) - \log\!\left(1 + e^{-|x|}\right)$$

`exp(-|x|)` is always ≤ 1, so it can't overflow. `torch.log1p(t)` computes `log(1 + t)` accurately for tiny `t`.
'''),
        ("code", r'''
x = torch.tensor([-200.0, 0.0, 200.0])
print(torch.log(torch.sigmoid(x)))                               # [-inf, -0.6931, 0.]
print(torch.minimum(x, torch.zeros_like(x)) - torch.log1p(torch.exp(-x.abs())))   # [-200, -0.6931, 0]
print(F.logsigmoid(x))
'''),
        ("md", r'''
## 5. Work in log-space

- **Products → sums.** The probability of a 1000-token sequence is a product of 1000 numbers below 1, which underflows to `0`. The sum of their logs is fine. Beam search (#33) adds log-probs.
- **Ratios → differences.** PPO's ratio `π_new / π_old` is computed as `exp(logp_new - logp_old)` (#39).
- **Temperature** divides the logits, `logits / T`, before any softmax or log-softmax (#32).
'''),
        ("code", r'''
p = torch.full((1000,), 0.5)
print(p.prod(), p.log().sum())     # 0. (underflow) vs -693.1 (fine)
logp_new, logp_old = torch.tensor(-300.0), torch.tensor(-301.0)
print(torch.exp(logp_new) / torch.exp(logp_old), torch.exp(logp_new - logp_old))   # nan vs 2.718
'''),
        ("md", r'''
## 6. Fully masked rows and the `torch.where` NaN trap

If every key in an attention row is masked with `-inf`, softmax computes `0 / 0 = NaN` for that row. This happens with padding: a padded query that may see nothing.

The obvious fix, `torch.where(denom > 0, e / denom, 0)`, gives the right **values** but `NaN` **gradients**. `torch.where` still computes `e / denom` for every element, and backward through `x / 0` is `NaN`, even for the branch that wasn't selected.
The safe pattern is to make the **input** safe first, then compute: `e / torch.where(denom > 0, denom, 1)`.
'''),
        ("code", r'''
e = torch.tensor([0.0, 0.0], requires_grad=True)
denom = e.sum()
bad = torch.where(denom > 0, e / denom, torch.zeros_like(e))
bad.sum().backward()
print("where(denom > 0, e / denom, 0) grad:", e.grad)            # nan
e.grad = None
good = e / torch.where(denom > 0, denom, torch.ones_like(denom))
good.sum().backward()
print("e / where(denom > 0, denom, 1) grad:", e.grad)            # finite
'''),
    ],
    "exercises_intro": r'''
---
# ✏️ Exercises

Implement the five functions below. Later exercises may call earlier ones. Each one has a debug cell you can use before you submit.
''',
    "exercises": [
        {
            "spec": r'''
### Exercise 1: `stable_logsumexp(x, dim=-1)`

Compute `log(sum(exp(x)))` along `dim` without overflow.

- `x`: any shape, may contain `-inf`
- returns: `x` with `dim` removed
- huge values (e.g. `1000`) must give finite results
- a slice that is **entirely** `-inf` must give `-inf`, not `NaN`
''',
            "stub": r'''
def stable_logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def stable_logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = x.amax(dim=dim, keepdim=True).detach()
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)   # all -inf slice: shift by 0
    return (m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))).squeeze(dim)
''',
            "debug": r'''
x = torch.tensor([[1000.0, 1001.0, 1002.0],
                  [float("-inf")] * 3])
print(stable_logsumexp(x))   # expect tensor([1002.4076, -inf])
''',
        },
        {
            "spec": r'''
### Exercise 2: `log_softmax(x, dim=-1)`

Log-probabilities from logits: `x - logsumexp(x)` along `dim`.

- `x`: any shape
- returns: same shape; `exp(out)` sums to `1` along `dim`
- must stay finite for spreads like `[0, -1000]` (where `log(softmax)` gives `-inf`)
''',
            "stub": r'''
def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x - stable_logsumexp(x, dim=dim).unsqueeze(dim)
''',
            "debug": r'''
x = torch.tensor([0.0, -1000.0, 2.0])
print(log_softmax(x))                  # expect approx [-2.1269, -1002.1269, -0.1269]
print(log_softmax(x).exp().sum())      # expect tensor(1.)
''',
        },
        {
            "spec": r'''
### Exercise 3: `log_sigmoid(x)`

Compute `log(sigmoid(x))` stably for any `x`.

- `x`: any shape
- returns: same shape, finite everywhere (e.g. `log_sigmoid(-200) == -200`)
- gradients must be finite too

*DPO's loss (#37) is `-log_sigmoid(beta * margin).mean()`.*
''',
            "stub": r'''
def log_sigmoid(x: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def log_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.minimum(x, torch.zeros_like(x)) - torch.log1p(torch.exp(-x.abs()))
''',
            "debug": r'''
print(log_sigmoid(torch.tensor([-200.0, -1.0, 0.0, 1.0, 200.0])))
# expect approx [-200.0000, -1.3133, -0.6931, -0.3133, 0.0000]
''',
        },
        {
            "spec": r'''
### Exercise 4: `kl_from_logits(p_logits, q_logits)`

KL divergence `KL(P ‖ Q) = Σ P · (log P − log Q)` between two categorical distributions given as **logits**, along the last dim.

- `p_logits`, `q_logits`: `(..., V)`
- returns: `(...)`, one KL value per distribution
- work in log-space (use your `log_softmax`) so extreme logits stay finite

*#41 OPD minimizes the reverse KL `KL(student ‖ teacher)`. That is `kl_from_logits(student_logits, teacher_logits)`.*
''',
            "stub": r'''
def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    log_p = log_softmax(p_logits, dim=-1)
    log_q = log_softmax(q_logits, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)
''',
            "debug": r'''
p = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
q = torch.tensor([[2.0, 0.0, -1.0], [5.0, 0.0, -5.0]])
print(kl_from_logits(p, q))    # expect tensor([0.0000, 3.9081]): zero when P == Q, positive otherwise
''',
        },
        {
            "spec": r'''
### Exercise 5: `safe_masked_softmax(scores, mask)`

Softmax over the last dim with blocked positions, where a **fully blocked row returns all zeros** instead of `NaN`, and gradients stay finite everywhere.

- `scores`: `(..., S_q, S_k)` float
- `mask`: bool, broadcastable to `scores`; `True` = blocked
- returns: same shape as `scores`
  - rows with at least one visible key: a normal softmax over the visible keys (blocked positions get `0`)
  - rows with no visible key: all zeros

*See lesson §6 for the `torch.where` trap. In this exercise `masked_fill`'s backward pass happens to set the gradient of every blocked position to exactly 0, which hides that trap. Use the safe pattern anyway: the same code without a mask in front would give `NaN` gradients.*
''',
            "stub": r'''
def safe_masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def safe_masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    s = scores.masked_fill(mask, float("-inf"))
    m = s.amax(dim=-1, keepdim=True).detach()
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)          # fully masked row: shift by 0
    e = torch.exp(s - m)                                              # blocked -> exactly 0
    denom = e.sum(dim=-1, keepdim=True)
    return e / torch.where(denom > 0, denom, torch.ones_like(denom))  # safe input, not safe output
''',
            "debug": r'''
scores = torch.zeros(3, 3, requires_grad=True)
mask = torch.tensor([[False, True, True],
                     [False, False, True],
                     [True, True, True]])          # last row: nothing visible
out = safe_masked_softmax(scores, mask)
print(out)            # expect [[1, 0, 0], [0.5, 0.5, 0], [0, 0, 0]]
(out * torch.arange(9.).view(3, 3)).sum().backward()
print(scores.grad)    # expect finite values, no nan
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (2, "Max-subtraction so `exp` never overflows"),
        (16, "`log_softmax` from `stable_logsumexp`, never `log(softmax)`"),
        (25, "Online softmax: rescale the running sum by `exp(old_max - new_max)` when the max changes"),
        (32, "Temperature (`logits / T`), and softmax over logits that include `-inf`"),
        (33, "Scoring sequences by **summing** log-probs instead of multiplying probabilities"),
        (34, "Acceptance ratios between probabilities, guarding against division by ~0"),
        (37, "`log_sigmoid`: the DPO loss is `-log_sigmoid(beta * margin)`"),
        (38, "Adding `eps` to the std before dividing"),
        (39, "The PPO ratio as `exp(new_logps - old_logps)`"),
        (41, "`kl_from_logits(student, teacher)`: the reverse KL, computed in log-space"),
    ],
}
