"""Prerequisite 6: Matmul & einsum (multi-exercise task)."""

# Shared by the tests that must not materialize a large intermediate matrix:
# records the shape of every matrix product computed while the user's function runs.
_RECORDER = """
import torch
from torch.overrides import TorchFunctionMode

class _ProductShapes(TorchFunctionMode):
    PRODUCTS = {'matmul', '__matmul__', '__rmatmul__', 'mm', 'bmm', 'einsum'}
    def __init__(self):
        super().__init__()
        self.shapes = []
    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if getattr(func, '__name__', '') in self.PRODUCTS and isinstance(out, torch.Tensor):
            self.shapes.append(tuple(out.shape))
        return out
"""

TASK = {
    "title": "Prereq 6: Matmul & Einsum",
    "difficulty": "Prereq",
    "order": 6,
    "function_name": "linear_forward",
    "function_names": [
        "linear_forward",
        "attention_scores",
        "patch_conv",
        "qkv_reassociated",
        "lora_forward",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `linear_forward`: `x @ W.T + b`. `W` is `(out, in)`.\n"
        "- `attention_scores`: `Q @ K.transpose(-2, -1) / math.sqrt(d)`. `K.T` reverses ALL dims on batched tensors.\n"
        "- `patch_conv`: `torch.einsum('bchwij,ocij->bohw', patches, weight)`: `c, i, j` are summed because they're missing from the output.\n"
        "- `qkv_reassociated`: compute `K.transpose(-2, -1) @ V` first (shape `(B, d_k, d_v)`), then `Q @` that.\n"
        "- `lora_forward`: `x @ W.T + (x @ A.T) @ B.T * scaling`; never compute `B @ A`."
    ),
    "tests": [
        {
            "name": "linear_forward: values and layout",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(2, 5, 4)
W, b = torch.randn(3, 4), torch.randn(3)
out = linear_forward(x, W, b)
assert isinstance(out, torch.Tensor) and out.shape == (2, 5, 3), f'Expected (..., out_features) = (2, 5, 3), got {getattr(out, "shape", out)}'
assert torch.allclose(out, torch.nn.functional.linear(x, W, b), atol=1e-5), 'Wrong values'
Wsq = torch.randn(4, 4)
assert torch.allclose(linear_forward(x, Wsq, torch.zeros(4)), x @ Wsq.T, atol=1e-5), 'Square W: use W.T, not W'
""",
        },
        {
            "name": "attention_scores: 3-D and 4-D, S_q != S_k",
            "code": """
import torch, math
torch.manual_seed(0)
Q, K = torch.randn(2, 5, 8), torch.randn(2, 7, 8)
out = attention_scores(Q, K)
assert isinstance(out, torch.Tensor) and out.shape == (2, 5, 7), f'Expected (B, S_q, S_k) = (2, 5, 7), got {getattr(out, "shape", out)}'
ref = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(8)
if torch.allclose(out, ref * math.sqrt(8), atol=1e-5):
    raise AssertionError('Missing the 1/sqrt(d) scaling')
if torch.allclose(out, ref * math.sqrt(8) / 8, atol=1e-5):
    raise AssertionError('Scale by 1/sqrt(d), not 1/d')
assert torch.allclose(out, ref, atol=1e-5), 'Wrong values'
Q4, K4 = torch.randn(2, 3, 5, 16), torch.randn(2, 3, 6, 16)
out4 = attention_scores(Q4, K4)
assert out4.shape == (2, 3, 5, 6), f'4-D (B, H, S_q, d): expected (2, 3, 5, 6), got {out4.shape}. K.T reverses all dims; use transpose(-2, -1)'
assert torch.allclose(out4, torch.einsum('bhqd,bhkd->bhqk', Q4, K4) / 4.0, atol=1e-5), 'Wrong values for 4-D input'
""",
        },
        {
            "name": "patch_conv: matches conv2d",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(2, 3, 7, 6)
w = torch.randn(4, 3, 3, 2)
for stride in (1, 2):
    patches = x.unfold(2, 3, stride).unfold(3, 2, stride)
    out = patch_conv(patches, w)
    ref = torch.nn.functional.conv2d(x, w, stride=stride)
    assert isinstance(out, torch.Tensor) and out.shape == ref.shape, f'stride={stride}: expected {tuple(ref.shape)}, got {getattr(out, "shape", out)}'
    assert torch.allclose(out, ref, atol=1e-4), f'stride={stride}: values do not match conv2d'
""",
        },
        {
            "name": "qkv_reassociated: values",
            "code": """
import torch
torch.manual_seed(0)
Q, K, V = torch.randn(2, 9, 4), torch.randn(2, 9, 4), torch.randn(2, 9, 3)
out = qkv_reassociated(Q, K, V)
assert isinstance(out, torch.Tensor) and out.shape == (2, 9, 3), f'Expected (B, S, d_v) = (2, 9, 3), got {getattr(out, "shape", out)}'
assert torch.allclose(out, (Q @ K.transpose(-2, -1)) @ V, atol=1e-4), 'Wrong values'
""",
        },
        {
            "name": "qkv_reassociated: never builds the (S, S) matrix",
            "code": _RECORDER + """
S = 50
Q, K, V = torch.randn(1, S, 4), torch.randn(1, S, 4), torch.randn(1, S, 3)
with _ProductShapes() as rec:
    out = qkv_reassociated(Q, K, V)
assert isinstance(out, torch.Tensor), 'Return a tensor'
big = [s for s in rec.shapes if len(s) >= 2 and s[-2:] == (S, S)]
assert not big, f'Built an (S, S) = ({S}, {S}) matrix: compute K^T @ V first. Product shapes seen: {rec.shapes}'
""",
        },
        {
            "name": "lora_forward: values and gradients",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(2, 5, 20)
W = torch.randn(12, 20)
A = torch.randn(2, 20, requires_grad=True)
B = torch.randn(12, 2, requires_grad=True)
out = lora_forward(x, W, A, B, 0.5)
assert isinstance(out, torch.Tensor) and out.shape == (2, 5, 12), f'Expected (..., out) = (2, 5, 12), got {getattr(out, "shape", out)}'
assert torch.allclose(out, x @ (W + 0.5 * B @ A).T, atol=1e-3), 'Wrong values'
out.sum().backward()
assert A.grad is not None and B.grad is not None, 'Gradients must flow to A and B'
""",
        },
        {
            "name": "lora_forward: never builds B @ A",
            "code": _RECORDER + """
x = torch.randn(3, 20)
W, A, B = torch.randn(12, 20), torch.randn(2, 20), torch.randn(12, 2)
with _ProductShapes() as rec:
    out = lora_forward(x, W, A, B, 1.0)
assert isinstance(out, torch.Tensor), 'Return a tensor'
full = [s for s in rec.shapes if s in ((12, 20), (20, 12))]
assert not full, f'Built a full (out, in) matrix from A and B: go through the rank-r bottleneck. Product shapes seen: {rec.shapes}'
""",
        },
    ],
}
