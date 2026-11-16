"""Bonus 1: Custom autograd functions.

Content for templates/00_b01_custom_autograd.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "B1",
    "file": "00_b01_custom_autograd",
    "task_id": "b01_custom_autograd",
    "title": "Custom Autograd Functions",
    # Shape symbols used in this notebook, shown in the "📐 Notation" box
    "notation": [
        ('N', 'batch size (rows of `x`)'),
        ('in / out', 'input / output features of `LinearFn`'),
        ('V', 'number of classes (softmax over the last dim)'),
    ],
    "covers": "`torch.autograd.Function` (`forward`, `backward`, `ctx`), the backward contract, `gradcheck`, deriving backward rules (element-wise, matmul, reductions, softmax), gradient reversal",
    "intro": r'''
# ⭐ Bonus 1: Custom Autograd Functions

Every TorchCode problem asks for a **forward** pass and lets autograd derive the gradients. A very common interview follow-up is: *"now write the backward pass yourself."* It's also how real code defines custom gradients (straight-through estimators, gradient reversal), saves memory by storing less for backward, or wraps an operation autograd can't see (a custom kernel, NumPy code).

PyTorch's tool for this is `torch.autograd.Function`: you write `forward` **and** `backward`, and autograd calls your `backward` during `.backward()`.

### What you'll learn
1. The `torch.autograd.Function` skeleton: `forward`, `backward`, `ctx`, `.apply`
2. The backward contract: one gradient per input, shaped like that input (or `None`)
3. Checking a backward pass with `torch.autograd.gradcheck`
4. Deriving backward rules: element-wise ops, matmul, reductions, softmax
5. When a custom backward is worth it, e.g. a gradient reversal layer

### Exercises
| # | Class | Follow-up for |
|---|-------|---------------|
| 1 | `ReLUFn` | #01 ReLU |
| 2 | `SigmoidFn` | gates, #37 DPO's `log σ` |
| 3 | `LinearFn` | #03 Linear Layer |
| 4 | `SoftmaxFn` | #02 Softmax, #16 Cross-Entropy |
| 5 | `FakeQuantFn` | #36 INT8 Quantization (quantization-aware training) |

### Rules
- Each class subclasses `torch.autograd.Function` and implements `forward` and `backward` as `@staticmethod`s.
- `forward` may use any tensor ops. `backward` must compute the gradients **by hand**: no `.backward()`, `torch.autograd.grad`, or autograd-enabled recomputation inside it.
- No Python loops over tensor elements.
''',
    "lesson": [
        ("md", r'''
## 1. The skeleton

```python
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):                 # ctx: a scratchpad shared with backward
        ctx.save_for_backward(x)         # tensors needed later
        return x * x

    @staticmethod
    def backward(ctx, grad_out):         # grad_out = d(loss)/d(output)
        (x,) = ctx.saved_tensors
        return grad_out * 2 * x          # d(loss)/d(x) by the chain rule
```

Call it with **`Square.apply(x)`**, never `Square()(x)` or `Square.forward(...)`. Inside `forward`, autograd is switched off: the ops there aren't recorded. That's why your `backward` is the only thing that defines the gradient.
'''),
        ("code", r'''
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * 2 * x

x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
y = Square.apply(x)
print(y, y.grad_fn)              # grad_fn is your class's backward node
y.sum().backward()
print(x.grad)                    # [2., -4., 6.] = 2x
'''),
        ("md", r'''
## 2. The backward contract

- `backward` receives **one `grad_out` per output** of `forward`.
- It must return **one gradient per input** of `forward`, in the same order, each with **the same shape as that input**.
- Return **`None`** for inputs that aren't tensors, or that don't need a gradient. `ctx.needs_input_grad[i]` tells you whether input `i` needs one, so you can skip expensive work.
- Save tensors with **`ctx.save_for_backward(...)`**: autograd then detects if they're modified in place before backward runs. Store anything else (ints, floats, shapes) as plain attributes, e.g. `ctx.factor = factor`.
'''),
        ("code", r'''
class Scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):         # factor is a Python float
        ctx.factor = factor              # non-tensors: plain attributes
        return x * factor

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.factor, None   # one entry per input; None for `factor`

x = torch.ones(3, requires_grad=True)
Scale.apply(x, 2.5).sum().backward()
print(x.grad)                                # [2.5, 2.5, 2.5]

class Broken(Scale):
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.factor         # forgot the None for `factor`
try:
    Broken.apply(torch.ones(3, requires_grad=True), 2.0).sum().backward()
except RuntimeError as e:
    print("RuntimeError:", str(e)[:75])
'''),
        ("md", r'''
## 3. Checking a backward pass: `gradcheck`

`torch.autograd.gradcheck(fn, inputs)` compares your analytic gradient with a numerical one from **finite differences**: it nudges each input by a tiny `eps` and watches the output change.
Use **`float64`** inputs (float32 is too imprecise for finite differences), and keep inputs away from kinks such as ReLU at 0, where the numerical derivative is meaningless.
'''),
        ("code", r'''
x = torch.randn(5, dtype=torch.float64, requires_grad=True)
print(torch.autograd.gradcheck(Square.apply, (x,)))          # True

class WrongSquare(Square):
    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * x                                    # missing the factor 2
print(torch.autograd.gradcheck(WrongSquare.apply, (x,), raise_exception=False))   # False
'''),
        ("md", r'''
## 4. Deriving backward rules

Write `g` for `grad_out`. The gradient always has the **shape of the input**, and a **reduction** in forward becomes a **broadcast** in backward (and the other way around).

| Forward | Backward |
|---------|----------|
| element-wise `y = f(x)` | `dx = g * f'(x)` |
| `y = x @ W.T + b`, with `x: (N, in)` and `W: (out, in)` | `dx = g @ W`, `dW = g.T @ x`, `db = g.sum(0)` |
| `y = x.sum(dim)` | `dx = g.unsqueeze(dim).expand_as(x)` |
| `y = x.mean(dim)` | the same, divided by `x.shape[dim]` |
| `y = x[idx]` (gather) | `dx = zeros_like(x).index_add(0, idx, g)` (scatter back) |
| `y = softmax(x)` | `dx = y * (g - (g * y).sum(-1, keepdim=True))` |

**The softmax rule** comes from its Jacobian, `∂yᵢ/∂xⱼ = yᵢ(δᵢⱼ − yⱼ)`. Multiplying `g` by that matrix simplifies to the formula above, which costs `O(V)` per row instead of building a `V × V` Jacobian. For a 50,000-token vocabulary that's the difference between 50k numbers and 2.5 billion.

**Check a matmul rule by shapes:** `dW` must be `(out, in)`. The only way to combine `g: (N, out)` and `x: (N, in)` into that shape is `g.T @ x`.
'''),
        ("code", r'''
x = torch.randn(4, 3, requires_grad=True)
W = torch.randn(5, 3, requires_grad=True)
g = torch.randn(4, 5)                                  # an upstream gradient
(x @ W.T).backward(g)
print(torch.allclose(x.grad, g @ W), torch.allclose(W.grad, g.T @ x))     # True True

z = torch.randn(2, 6, requires_grad=True)
y = torch.softmax(z, dim=-1)
gz = torch.randn(2, 6)
y.backward(gz)
print(torch.allclose(z.grad, y * (gz - (gz * y).sum(-1, keepdim=True)), atol=1e-6))   # True
'''),
        ("md", r'''
## 5. When a custom backward is worth it

- **Custom gradients:** a straight-through estimator (P7), a *clipped* straight-through estimator for quantization (Exercise 5), or a **gradient reversal layer** (identity forward, `-λ·g` backward), used in domain adaptation.
- **Memory:** save less for backward and recompute the rest, as fused kernels and activation checkpointing do.
- **Numerical stability:** use a stable closed-form backward instead of differentiating through unstable steps.
- **Opaque code:** wrap a custom CUDA/Triton kernel or NumPy code that autograd can't trace.

If ordinary tensor ops already give the right gradient, **don't** write a `Function`: autograd's version is tested and usually just as fast.
'''),
        ("code", r'''
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)                    # identity (a view, so autograd records this node)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lam * grad_out, None

x = torch.ones(3, requires_grad=True)
GradReverse.apply(x, 0.5).sum().backward()
print(x.grad)                                  # [-0.5, -0.5, -0.5]: the gradient is flipped
'''),
    ],
    "exercises_intro": r'''
---
# ✏️ Exercises

Implement the five `torch.autograd.Function` classes below. Each debug cell runs `gradcheck` where it makes sense. Use it before you submit.
''',
    "exercises": [
        {
            "spec": r'''
### Exercise 1: `ReLUFn`

ReLU with a hand-written backward.

- `forward(ctx, x)`: returns `max(x, 0)`, same shape and dtype as `x`
- `backward(ctx, grad_out)`: the gradient is `grad_out` where `x > 0`, and `0` elsewhere (including at exactly `0`)

*Save the least you need: a boolean mask is enough.*
''',
            "stub": r'''
class ReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        pass  # Replace this

    @staticmethod
    def backward(ctx, grad_out):
        pass  # Replace this
''',
            "solution": r'''
class ReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x > 0)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_out):
        (positive,) = ctx.saved_tensors
        return grad_out * positive
''',
            "debug": r'''
x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
y = ReLUFn.apply(x)
y.sum().backward()
print(y, x.grad)        # expect tensor([0., 0., 2.]) and tensor([0., 0., 1.])
xc = torch.tensor([-1.5, -0.3, 0.4, 2.0], dtype=torch.float64, requires_grad=True)   # away from 0
print(torch.autograd.gradcheck(ReLUFn.apply, (xc,)))   # expect True
''',
        },
        {
            "spec": r'''
### Exercise 2: `SigmoidFn`

Sigmoid with a backward that reuses the **output**.

- `forward(ctx, x)`: returns `σ(x) = 1 / (1 + e^{-x})`; it must not overflow for large `|x|` (you may use `torch.sigmoid` here)
- `backward(ctx, grad_out)`: use `σ'(x) = σ(x) · (1 − σ(x))`, computed from the **saved output**, not from `x`
''',
            "stub": r'''
class SigmoidFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        pass  # Replace this

    @staticmethod
    def backward(ctx, grad_out):
        pass  # Replace this
''',
            "solution": r'''
class SigmoidFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.sigmoid(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        return grad_out * y * (1 - y)
''',
            "debug": r'''
x = torch.randn(6, dtype=torch.float64, requires_grad=True)
print(torch.autograd.gradcheck(SigmoidFn.apply, (x,)))      # expect True
big = torch.tensor([-100.0, 0.0, 100.0], requires_grad=True)
SigmoidFn.apply(big).sum().backward()
print(big.grad)                                             # expect [~0, 0.25, ~0], no nan
''',
        },
        {
            "spec": r'''
### Exercise 3: `LinearFn`

A linear layer, `y = x @ W.T + b`, with hand-written gradients for **all three** inputs.

- `forward(ctx, x, W, b)`: `x` is `(N, in)`, `W` is `(out, in)`, `b` is `(out,)`; returns `(N, out)`
- `backward(ctx, grad_out)`: returns `(dx, dW, db)` with the shapes of `x`, `W`, `b` (see lesson §4)
- use `ctx.needs_input_grad` to skip gradients nobody needs (e.g. a frozen `W`)
''',
            "stub": r'''
class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b):
        pass  # Replace this

    @staticmethod
    def backward(ctx, grad_out):
        pass  # Replace this
''',
            "solution": r'''
class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b):
        ctx.save_for_backward(x, W)
        return x @ W.T + b

    @staticmethod
    def backward(ctx, grad_out):
        x, W = ctx.saved_tensors
        dx = grad_out @ W if ctx.needs_input_grad[0] else None
        dW = grad_out.T @ x if ctx.needs_input_grad[1] else None
        db = grad_out.sum(0) if ctx.needs_input_grad[2] else None
        return dx, dW, db
''',
            "debug": r'''
x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
W = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
b = torch.randn(5, dtype=torch.float64, requires_grad=True)
print(LinearFn.apply(x, W, b).shape)                            # expect torch.Size([4, 5])
print(torch.autograd.gradcheck(LinearFn.apply, (x, W, b)))      # expect True
''',
        },
        {
            "spec": r'''
### Exercise 4: `SoftmaxFn`

Softmax over the last dim, with a backward that **never builds the `V × V` Jacobian**.

- `forward(ctx, x)`: `x` is `(..., V)`; a stable softmax (subtract the row max; you may use `torch.softmax`)
- `backward(ctx, grad_out)`: `dx = y * (grad_out - (grad_out * y).sum(-1, keepdim=True))`, using the saved output `y`

*#16 Cross-entropy's gradient, `softmax(x) − one_hot(target)`, is this rule applied to `-log y[target]`.*
''',
            "stub": r'''
class SoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        pass  # Replace this

    @staticmethod
    def backward(ctx, grad_out):
        pass  # Replace this
''',
            "solution": r'''
class SoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        e = torch.exp(x - x.amax(dim=-1, keepdim=True))
        y = e / e.sum(dim=-1, keepdim=True)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        return y * (grad_out - (grad_out * y).sum(dim=-1, keepdim=True))
''',
            "debug": r'''
x = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
print(torch.autograd.gradcheck(SoftmaxFn.apply, (x,)))                      # expect True
print(SoftmaxFn.apply(torch.tensor([[1000.0, 1001.0]])))                   # expect [[0.2689, 0.7311]], no nan
''',
        },
        {
            "spec": r'''
### Exercise 5: `FakeQuantFn`

**Fake quantization** for quantization-aware training (QAT): the forward pass rounds values to the int8 grid and back, so the model *experiences* quantization error while training in float.
`round` has zero gradient almost everywhere, so QAT uses a **clipped straight-through estimator**: pass the gradient through unchanged where the value was inside the int8 range, and block it where the value was clamped.

- `forward(ctx, x, scale)`: `scale` is a Python float; returns `round(x / scale).clamp(-128, 127) * scale` (same shape and dtype as `x`)
- `backward(ctx, grad_out)`: `grad_out` where `-128 <= x / scale <= 127`, `0` elsewhere; return **`None`** for `scale`

*#36 INT8 quantizes once after training. QAT inserts this op during training so the model learns to tolerate it.*
''',
            "stub": r'''
class FakeQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        pass  # Replace this

    @staticmethod
    def backward(ctx, grad_out):
        pass  # Replace this
''',
            "solution": r'''
class FakeQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        q = x / scale
        ctx.save_for_backward((q >= -128) & (q <= 127))   # inside the representable range
        return torch.round(q).clamp(-128, 127) * scale

    @staticmethod
    def backward(ctx, grad_out):
        (inside,) = ctx.saved_tensors
        return grad_out * inside, None
''',
            "debug": r'''
x = torch.tensor([0.04, 0.26, 20.0, -20.0], requires_grad=True)
y = FakeQuantFn.apply(x, 0.1)
y.sum().backward()
print(y)        # expect [0.0, 0.3, 12.7, -12.8]
print(x.grad)   # expect [1., 1., 0., 0.]: blocked where clamped
''',
        },
    ],
    # Problem number -> how to use this notebook on it ("Where to use it" section)
    "unlocks": [
        (1, "Wrap your ReLU in `ReLUFn` and check it with `gradcheck`"),
        (2, "`SoftmaxFn`: the O(V) backward instead of the Jacobian"),
        (3, "`LinearFn`: derive `dx`, `dW`, `db` by hand; check shapes, then `gradcheck`"),
        (16, "Show that the gradient of cross-entropy w.r.t. the logits is `softmax − one_hot`"),
        (19, "Write GELU's backward: `Φ(x) + x·φ(x)` for the exact `erf` form"),
        (36, "Insert `FakeQuantFn` before the matmul to train with quantization (QAT)"),
    ],
}
