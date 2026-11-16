"""Bonus 1: Custom autograd functions (multi-exercise task)."""

TASK = {
    "title": "Bonus 1: Custom Autograd Functions",
    "difficulty": "Bonus",
    "order": 1,
    "function_name": "ReLUFn",
    "function_names": [
        "ReLUFn",
        "SigmoidFn",
        "LinearFn",
        "SoftmaxFn",
        "FakeQuantFn",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `ReLUFn`: save the mask `x > 0` in forward; backward is `grad_out * mask`.\n"
        "- `SigmoidFn`: save the output `y`; backward is `grad_out * y * (1 - y)`.\n"
        "- `LinearFn`: save `x` and `W`; return `(grad_out @ W, grad_out.T @ x, grad_out.sum(0))`.\n"
        "- `SoftmaxFn`: save the output `y`; backward is `y * (grad_out - (grad_out * y).sum(-1, keepdim=True))`.\n"
        "- `FakeQuantFn`: save the mask `-128 <= x / scale <= 127`; return `(grad_out * mask, None)`. One entry per forward input."
    ),
    "tests": [
        {
            "name": "ReLUFn: values, gradient, gradcheck",
            "code": """
import torch
assert isinstance(ReLUFn, type) and issubclass(ReLUFn, torch.autograd.Function), 'ReLUFn must subclass torch.autograd.Function'
x = torch.tensor([-2.0, -0.0, 0.0, 0.5, 3.0], requires_grad=True)
y = ReLUFn.apply(x)
assert isinstance(y, torch.Tensor) and torch.equal(y.detach(), torch.tensor([0., 0., 0., 0.5, 3.])), f'Wrong forward values: {y}'
assert 'ReLUFn' in type(y.grad_fn).__name__, 'The output should come from your Function (call ReLUFn.apply)'
y.sum().backward()
assert torch.equal(x.grad, torch.tensor([0., 0., 0., 1., 1.])), f'Gradient should be 1 where x > 0 and 0 elsewhere (including 0), got {x.grad}'
xc = torch.tensor([-1.5, -0.3, 0.4, 2.0, -0.7], dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(ReLUFn.apply, (xc,), raise_exception=False), 'gradcheck failed: backward does not match finite differences'
""",
        },
        {
            "name": "SigmoidFn: gradcheck and extreme inputs",
            "code": """
import torch
assert issubclass(SigmoidFn, torch.autograd.Function), 'SigmoidFn must subclass torch.autograd.Function'
x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
assert torch.allclose(SigmoidFn.apply(x), torch.sigmoid(x)), 'Wrong forward values'
assert torch.autograd.gradcheck(SigmoidFn.apply, (x,), raise_exception=False), 'gradcheck failed: use sigma * (1 - sigma) from the saved output'
big = torch.tensor([-200.0, 0.0, 200.0], requires_grad=True)
y = SigmoidFn.apply(big)
y.sum().backward()
assert torch.isfinite(y).all() and torch.isfinite(big.grad).all(), f'No inf/nan for large |x|: y={y}, grad={big.grad}'
assert torch.allclose(big.grad, torch.tensor([0.0, 0.25, 0.0]), atol=1e-6), f'Expected gradients [0, 0.25, 0], got {big.grad}'
""",
        },
        {
            "name": "LinearFn: gradcheck for x, W and b",
            "code": """
import torch
assert issubclass(LinearFn, torch.autograd.Function), 'LinearFn must subclass torch.autograd.Function'
torch.manual_seed(0)
x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
W = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
b = torch.randn(5, dtype=torch.float64, requires_grad=True)
out = LinearFn.apply(x, W, b)
assert isinstance(out, torch.Tensor) and out.shape == (4, 5), f'Expected shape (N, out) = (4, 5), got {getattr(out, "shape", out)}'
assert torch.allclose(out, torch.nn.functional.linear(x, W, b)), 'Wrong forward values'
assert torch.autograd.gradcheck(LinearFn.apply, (x, W, b), raise_exception=False), 'gradcheck failed: dx = g @ W, dW = g.T @ x, db = g.sum(0)'
""",
        },
        {
            "name": "LinearFn: frozen weight, single row",
            "code": """
import torch
x = torch.randn(1, 6, requires_grad=True)
W = torch.randn(2, 6)                         # frozen: no grad needed
b = torch.zeros(2, requires_grad=True)
g = torch.randn(1, 2)
LinearFn.apply(x, W, b).backward(g)
assert W.grad is None, 'W does not require grad, so it must not get one'
assert torch.allclose(x.grad, g @ W, atol=1e-6) and torch.allclose(b.grad, g.sum(0), atol=1e-6), 'Wrong gradients for x or b'
""",
        },
        {
            "name": "SoftmaxFn: gradcheck and stability",
            "code": """
import torch
assert issubclass(SoftmaxFn, torch.autograd.Function), 'SoftmaxFn must subclass torch.autograd.Function'
x = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
assert torch.allclose(SoftmaxFn.apply(x), torch.softmax(x, -1)), 'Wrong forward values (softmax over the last dim)'
assert torch.autograd.gradcheck(SoftmaxFn.apply, (x,), raise_exception=False), 'gradcheck failed'
y = SoftmaxFn.apply(torch.tensor([[1000.0, 1001.0, 0.0]]))
assert torch.isfinite(y).all() and torch.allclose(y.sum(), torch.tensor(1.0)), f'Unstable forward for large logits: {y}'
""",
        },
        {
            "name": "SoftmaxFn: backward never builds the V x V Jacobian",
            "code": """
import torch
from torch.utils._python_dispatch import TorchDispatchMode
class _Shapes(TorchDispatchMode):   # dispatch level: also sees ops run inside a custom backward
    def __init__(self): super().__init__(); self.shapes = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor): self.shapes.append(tuple(out.shape))
        return out
V = 64
x = torch.randn(3, V, requires_grad=True)
y = SoftmaxFn.apply(x)
g = torch.randn(3, V)
with _Shapes() as rec:
    y.backward(g)
assert not any(len(s) >= 2 and s[-2:] == (V, V) for s in rec.shapes), f'Backward built a ({V}, {V}) Jacobian: use y * (g - (g * y).sum(-1, keepdim=True))'
ys = torch.softmax(x.detach(), -1)
assert torch.allclose(x.grad, ys * (g - (g * ys).sum(-1, keepdim=True)), atol=1e-5), 'Wrong gradient'
""",
        },
        {
            "name": "FakeQuantFn: values, clipped straight-through gradient",
            "code": """
import torch
assert issubclass(FakeQuantFn, torch.autograd.Function), 'FakeQuantFn must subclass torch.autograd.Function'
x = torch.tensor([0.04, 0.26, -0.33, 12.7, 20.0, -20.0], dtype=torch.float64, requires_grad=True)
y = FakeQuantFn.apply(x, 0.1)
expected = torch.tensor([0.0, 0.3, -0.3, 12.7, 12.7, -12.8], dtype=torch.float64)
assert isinstance(y, torch.Tensor) and y.dtype == torch.float64, 'Output must keep the input dtype'
assert torch.allclose(y, expected), f'Expected {expected.tolist()}, got {y.tolist()}'
y.backward(torch.tensor([1., 2., 3., 4., 5., 6.], dtype=torch.float64))
assert torch.equal(x.grad, torch.tensor([1., 2., 3., 4., 0., 0.], dtype=torch.float64)), f'Pass grad_out through inside the int8 range, 0 where clamped; got {x.grad.tolist()}'
""",
        },
    ],
}
