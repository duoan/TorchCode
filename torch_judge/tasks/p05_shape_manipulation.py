"""Prerequisite 5: Shape manipulation (multi-exercise task)."""

TASK = {
    "title": "Prereq 5: Shape Manipulation",
    "difficulty": "Prereq",
    "order": 5,
    "function_name": "split_heads",
    "function_names": [
        "split_heads",
        "merge_heads",
        "repeat_kv",
        "interleave",
        "patchify",
    ],
    "hint": (
        "split_heads: view(B, S, H, d) then transpose(1, 2). merge_heads: transpose back, then "
        ".contiguous().view(...) (or reshape). repeat_kv: repeat_interleave on the head dim. "
        "interleave: stack on a new last dim, then flatten(-2). patchify: reshape to "
        "(B, C, H/p, p, W/p, p), permute to (B, H/p, W/p, C, p, p), reshape."
    ),
    "tests": [
        {
            "name": "split_heads: shape and head layout",
            "code": """
import torch
B, S, D, H = 2, 5, 12, 3
x = torch.randn(B, S, D)
out = split_heads(x, H)
assert out.shape == (B, H, S, D // H), f'Shape mismatch: {out.shape}, expected (B, H, S, d) = {(B, H, S, D // H)}'
d = D // H
for h in range(H):
    assert torch.equal(out[:, h], x[:, :, h * d:(h + 1) * d]), f'Head {h} should hold features {h*d}:{(h+1)*d}'
""",
        },
        {
            "name": "merge_heads: inverse of split_heads",
            "code": """
import torch
x = torch.randn(3, 7, 16)
for H in (1, 2, 4, 8):
    out = merge_heads(split_heads(x, H))
    assert out.shape == x.shape, f'Shape mismatch for H={H}: {out.shape}'
    assert torch.equal(out, x), f'merge_heads(split_heads(x)) != x for H={H}'
""",
        },
        {
            "name": "merge_heads: works on non-contiguous input",
            "code": """
import torch
x = torch.randn(2, 6, 4, 3).transpose(1, 2)   # (B, H=4, S=6, d=3), non-contiguous
out = merge_heads(x)
assert out.shape == (2, 6, 12), f'Shape mismatch: {out.shape}'
assert torch.equal(out[:, :, 3:6], x[:, 1]), 'Head 1 should occupy features 3:6'
""",
        },
        {
            "name": "repeat_kv: order and shape",
            "code": """
import torch
x = torch.randn(2, 3, 5, 4)
out = repeat_kv(x, 2)
assert out.shape == (2, 6, 5, 4), f'Shape mismatch: {out.shape}'
for h in range(6):
    assert torch.equal(out[:, h], x[:, h // 2]), f'Query head {h} should use KV head {h // 2}'
assert torch.equal(repeat_kv(x, 1), x), 'n_rep=1 should be a no-op'
""",
        },
        {
            "name": "interleave",
            "code": """
import torch
a = torch.tensor([1, 2, 3]); b = torch.tensor([10, 20, 30])
assert torch.equal(interleave(a, b), torch.tensor([1, 10, 2, 20, 3, 30])), f'Got {interleave(a, b)}'
a = torch.randn(2, 4, 5); b = torch.randn(2, 4, 5)
out = interleave(a, b)
assert out.shape == (2, 4, 10), f'Shape mismatch: {out.shape}'
assert torch.equal(out[..., 0::2], a) and torch.equal(out[..., 1::2], b), 'Even slots must be a, odd slots b'
""",
        },
        {
            "name": "interleave: gradient flows",
            "code": """
import torch
a = torch.randn(3, 4, requires_grad=True); b = torch.randn(3, 4, requires_grad=True)
w = torch.arange(8).float()
(interleave(a, b) * w).sum().backward()
assert torch.equal(a.grad[0], w[0::2]) and torch.equal(b.grad[0], w[1::2]), 'Wrong gradients'
""",
        },
        {
            "name": "patchify: layout",
            "code": """
import torch
img = torch.arange(2 * 3 * 4 * 6).float().view(2, 3, 4, 6)
p = 2
out = patchify(img, p)
n_h, n_w = 4 // p, 6 // p
assert out.shape == (2, n_h * n_w, 3 * p * p), f'Shape mismatch: {out.shape}'
for r in range(n_h):
    for c in range(n_w):
        expected = img[:, :, r*p:(r+1)*p, c*p:(c+1)*p].reshape(2, -1)
        assert torch.equal(out[:, r * n_w + c], expected), f'Patch (row={r}, col={c}) is wrong'
""",
        },
        {
            "name": "patchify: patch_size == image size",
            "code": """
import torch
img = torch.randn(1, 2, 4, 4)
out = patchify(img, 4)
assert out.shape == (1, 1, 32), f'Shape mismatch: {out.shape}'
assert torch.equal(out[0, 0], img.flatten()), 'A single patch should equal the flattened image'
""",
        },
    ],
}
