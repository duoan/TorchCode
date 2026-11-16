"""Bonus 4: Data & batching for sequences (multi-exercise task)."""

TASK = {
    "title": "Bonus 4: Sequence Batching",
    "difficulty": "Bonus",
    "order": 4,
    "function_name": "pad_batch",
    "function_names": [
        "pad_batch",
        "padding_mask",
        "attention_mask_padded",
        "next_token_batch",
        "masked_token_loss",
    ],
    # One bullet per exercise (rendered as markdown in the web UI)
    "hint": (
        "- `pad_batch`: `torch.full((B, S_max), pad_id)`, then copy each sequence into its row (or `pad_sequence`).\n"
        "- `padding_mask`: `torch.arange(S)[None, :] >= lengths[:, None]`.\n"
        "- `attention_mask_padded`: the padding mask as `(B, 1, S)` (it's about **keys**), OR the causal `(1, S, S)` mask.\n"
        "- `next_token_batch`: `targets = padded[:, 1:].clone()`; target `t` is padding when `t >= length - 1`.\n"
        "- `masked_token_loss`: replace ignored targets with 0 before `gather`, then average the NLL over valid positions only (`clamp_min(1)` on the count)."
    ),
    "tests": [
        {
            "name": "pad_batch: values, dtype, empty sequence",
            "code": """
import torch
seqs = [torch.tensor([5, 6, 7]), torch.tensor([8]), torch.tensor([], dtype=torch.long), torch.tensor([1, 2])]
res = pad_batch(seqs, pad_id=99)
assert isinstance(res, tuple) and len(res) == 2, 'Return (padded, lengths)'
padded, lengths = res
assert padded.dtype == torch.long and lengths.dtype == torch.long, f'Both outputs must be long, got {padded.dtype}, {lengths.dtype}'
expected = torch.tensor([[5, 6, 7], [8, 99, 99], [99, 99, 99], [1, 2, 99]])
assert torch.equal(padded, expected), f'Got\\n{padded}'
assert torch.equal(lengths, torch.tensor([3, 1, 0, 2])), f'Got lengths {lengths}'
""",
        },
        {
            "name": "padding_mask",
            "code": """
import torch
m = padding_mask(torch.tensor([3, 1, 0, 4]), 4)
assert isinstance(m, torch.Tensor) and m.dtype == torch.bool and m.shape == (4, 4), f'Expected a (4, 4) bool tensor, got {getattr(m, "dtype", type(m))} {getattr(m, "shape", "")}'
expected = torch.tensor([[0, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
assert torch.equal(m, expected), f'True must mark padding (position >= length). Got\\n{m.int()}'
""",
        },
        {
            "name": "attention_mask_padded: matches a brute-force mask",
            "code": """
import torch
lengths = torch.tensor([4, 2, 1]); S = 5
for causal in (True, False):
    m = attention_mask_padded(lengths, S, causal=causal)
    assert isinstance(m, torch.Tensor) and m.dtype == torch.bool and m.shape == (3, S, S), f'Expected a (B, S, S) bool tensor, got {getattr(m, "shape", m)}'
    ref = torch.zeros(3, S, S, dtype=torch.bool)
    for b in range(3):
        for i in range(S):
            for j in range(S):
                ref[b, i, j] = (j >= lengths[b]) or (causal and j > i)
    if not torch.equal(m, ref):
        if causal and torch.equal(m, ref.transpose(1, 2)):
            raise AssertionError('Mask is transposed: padding is about KEYS (the last dim), the causal mask blocks j > i')
        raise AssertionError(f'causal={causal}: wrong mask for sequence lengths {lengths.tolist()}. Padding blocks KEYS (the last dim): every query must ignore padded keys')
""",
        },
        {
            "name": "next_token_batch",
            "code": """
import torch
padded = torch.tensor([[5, 6, 7, 8], [9, 3, 0, 0], [4, 0, 0, 0]])
before = padded.clone()
res = next_token_batch(padded, torch.tensor([4, 2, 1]), ignore_index=-1)
assert isinstance(res, tuple) and len(res) == 2, 'Return (inputs, targets)'
inputs, targets = res
assert torch.equal(padded, before), 'Do not modify padded in place: clone before overwriting targets'
assert torch.equal(inputs, padded[:, :-1]), 'inputs should be padded[:, :-1]'
expected = torch.tensor([[6, 7, 8], [3, -1, -1], [-1, -1, -1]])
assert torch.equal(targets, expected), f'Expected targets\\n{expected}\\ngot\\n{targets}. Target t is token t + 1'
""",
        },
        {
            "name": "masked_token_loss: matches cross_entropy with ignore_index",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(0)
logits = torch.randn(3, 5, 11, requires_grad=True)
targets = torch.randint(0, 11, (3, 5))
targets[0, 3:] = -100; targets[2, 1:] = -100
loss = masked_token_loss(logits, targets)
ref = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
assert isinstance(loss, torch.Tensor) and loss.ndim == 0, 'Return a scalar tensor'
assert torch.allclose(loss, ref, atol=1e-5), f'Expected {ref.item():.5f}, got {loss.item():.5f}. Average over VALID tokens only'
loss.backward()
assert torch.all(logits.grad[0, 3:] == 0) and torch.all(logits.grad[2, 1:] == 0), 'Ignored positions must get zero gradient'
""",
        },
        {
            "name": "masked_token_loss: all ignored, custom ignore_index, huge logits",
            "code": """
import torch
import torch.nn.functional as F
z = masked_token_loss(torch.randn(2, 3, 7), torch.full((2, 3), -100))
assert torch.isfinite(z) and z == 0, f'No valid targets should give 0, got {z}'
logits = torch.randn(2, 4, 6) * 1000
t = torch.tensor([[1, 2, -1, -1], [0, 5, 4, -1]])
out = masked_token_loss(logits, t, ignore_index=-1)
assert torch.isfinite(out) and torch.allclose(out, F.cross_entropy(logits.flatten(0, 1), t.flatten(), ignore_index=-1), rtol=1e-4), 'Use a stable log_softmax and respect ignore_index'
""",
        },
    ],
}
