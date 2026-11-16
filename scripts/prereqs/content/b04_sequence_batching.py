"""Bonus 4: Data & batching for sequences.

Content for templates/00_b04_sequence_batching.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "B4",
    "file": "00_b04_sequence_batching",
    "task_id": "b04_sequence_batching",
    "title": "Sequence Batching",
    "covers": "Padding and lengths, padding + causal masks, `DataLoader` with `collate_fn`, bucketing, next-token targets with `ignore_index`, packing with block-diagonal masks",
    "imports": "import torch\nimport torch.nn.functional as F",
    "intro": r'''
# ⭐ Bonus 4: Data & Batching for Sequences

The attention problems all assume neat `(B, S, D)` tensors where every sequence has the same length. Real text doesn't: a batch mixes a 7-token sentence with a 300-token paragraph. Getting from a list of variable-length sequences to tensors a model can use, **without the padding leaking into attention or the loss**, is everyday work in LLM code and a common interview topic.

### What you'll learn
1. Padding a batch, and keeping track of the real lengths
2. Padding masks, and combining them with a causal mask
3. Batching with a `DataLoader` and a custom `collate_fn`; bucketing by length
4. Next-token prediction: shifting inputs and targets, ignoring padding in the loss
5. Packing several short documents into one row, with a block-diagonal mask

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `pad_batch(seqs, pad_id)` | every batched sequence model |
| 2 | `padding_mask(lengths, S)` | #09, #13, #14 with real (padded) batches |
| 3 | `attention_mask_padded(lengths, S, causal)` | #09 Causal Attention, #13 GPT-2 Block |
| 4 | `next_token_batch(padded, lengths)` | language-model training |
| 5 | `masked_token_loss(logits, targets)` | #16 Cross-Entropy, #41 OPD |

### Rules
- No Python loops over tensor **elements**. Looping over the list of sequences in Exercise 1 is fine.
- Do **not** use `F.cross_entropy` or `F.nll_loss` in Exercise 5.
''',
    "lesson": [
        ("md", r'''
## 1. Padding a batch

Stack sequences of different lengths by **padding** the short ones to the longest length with a `pad_id`, and keep the true **lengths** next to the tensor.
`torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)` does the padding in one call.
'''),
        ("code", r'''
from torch.nn.utils.rnn import pad_sequence
seqs = [torch.tensor([5, 6, 7]), torch.tensor([8]), torch.tensor([9, 10])]
padded = pad_sequence(seqs, batch_first=True, padding_value=0)
lengths = torch.tensor([len(s) for s in seqs])
print(padded)       # [[5, 6, 7], [8, 0, 0], [9, 10, 0]]
print(lengths)      # [3, 1, 2]
'''),
        ("md", r'''
## 2. Padding masks, and combining them with causal masks

A **padding mask** marks the positions that hold padding: `position >= length`. That's an outer comparison (P2).
In attention, a query must not attend to **keys** that are padding. Combine that with the causal mask by **broadcasting**: the padding mask is per sequence `(B, 1, S_k)`, the causal mask is shared `(1, S_q, S_k)`, and `|` gives `(B, S_q, S_k)`.

Queries at padding positions still produce *some* output; it's simply ignored later by the loss. As long as every row can see at least one real key, softmax stays well-defined (otherwise, see P9's `safe_masked_softmax`).
'''),
        ("code", r'''
lengths = torch.tensor([3, 1])
S = 4
pad = torch.arange(S)[None, :] >= lengths[:, None]          # (B, S): True = padding
causal = torch.triu(torch.ones(S, S, dtype=torch.bool), 1)  # (S, S): True = future
blocked = pad[:, None, :] | causal[None, :, :]              # (B, S, S)
print(pad)
print(blocked[1].int())    # sequence of length 1: only key 0 is ever visible
'''),
        ("md", r'''
## 3. `DataLoader`, `collate_fn`, and bucketing

A `DataLoader` builds a batch by calling `collate_fn(list_of_samples)`. The default one stacks tensors, which fails for different lengths, so sequence code supplies its own `collate_fn` that pads and returns lengths or masks.

Padding wastes compute. **Bucketing** (grouping sequences of similar length into the same batch, e.g. by sorting by length first) keeps the padding small.
'''),
        ("code", r'''
from torch.utils.data import DataLoader

data = [torch.randint(1, 50, (n,)) for n in [3, 9, 4, 8, 2, 7]]

def collate(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0), torch.tensor([len(s) for s in batch])

for padded, lengths in DataLoader(data, batch_size=3, collate_fn=collate):
    print(padded.shape, lengths.tolist(), "padding:", (padded == 0).sum().item())

data_sorted = sorted(data, key=len)                       # bucket by length
for padded, lengths in DataLoader(data_sorted, batch_size=3, collate_fn=collate):
    print(padded.shape, lengths.tolist(), "padding:", (padded == 0).sum().item())
'''),
        ("md", r'''
## 4. Next-token prediction and ignoring padding in the loss

A language model predicts token `t+1` from tokens `0…t`. From one padded batch:
- **inputs** = `padded[:, :-1]`, **targets** = `padded[:, 1:]` (shifted left by one)
- targets that fall on padding are set to an **ignore index** (by convention `-100`), so the loss skips them

`F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)` averages only over real tokens. Exercise 5 rebuilds that by hand.
'''),
        ("code", r'''
padded = torch.tensor([[5, 6, 7, 8],
                       [9, 3, 0, 0]])                     # second sequence has length 2
lengths = torch.tensor([4, 2])
inputs, targets = padded[:, :-1], padded[:, 1:].clone()
targets[torch.arange(3)[None, :] >= (lengths - 1)[:, None]] = -100
print(inputs)
print(targets)       # [[6, 7, 8], [3, -100, -100]]
logits = torch.randn(2, 3, 20)
print(F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100))
'''),
        ("md", r'''
## 5. Packing documents into rows

Instead of padding, pretraining code often **packs** many short documents back to back into fixed-length rows, so no compute is wasted on padding. Each token then carries a **document id**, and attention must not cross document boundaries: the mask is **block-diagonal** (same document) combined with causal.
'''),
        ("code", r'''
doc = torch.tensor([0, 0, 0, 1, 1, 2])                   # three documents packed into one row of 6
same_doc = doc[:, None] == doc[None, :]
causal = torch.ones(6, 6, dtype=torch.bool).tril()
print((same_doc & causal).int())                         # allowed: same document AND not in the future
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
### Exercise 1: `pad_batch(seqs, pad_id)`

Pad a list of 1-D token tensors into one batch.

- `seqs`: a list of `B` 1-D `long` tensors with possibly different lengths (some may be empty)
- returns: `(padded, lengths)`
  - `padded`: `(B, S_max)` `long`, row `i` is `seqs[i]` followed by `pad_id`
  - `lengths`: `(B,)` `long`, the original lengths
''',
            "stub": r'''
def pad_batch(seqs: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    pass  # Replace this
''',
            "solution": r'''
def pad_batch(seqs: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    S = int(lengths.max()) if len(seqs) else 0
    padded = torch.full((len(seqs), S), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return padded, lengths
''',
            "debug": r'''
padded, lengths = pad_batch([torch.tensor([5, 6, 7]), torch.tensor([8]), torch.tensor([], dtype=torch.long)], pad_id=0)
print(padded)    # expect [[5, 6, 7], [8, 0, 0], [0, 0, 0]]
print(lengths)   # expect tensor([3, 1, 0])
''',
        },
        {
            "spec": r'''
### Exercise 2: `padding_mask(lengths, S)`

- `lengths`: `(B,)` long
- `S`: the padded length
- returns: `(B, S)` bool, `True` where position `j >= lengths[b]` (padding)
''',
            "stub": r'''
def padding_mask(lengths: torch.Tensor, S: int) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def padding_mask(lengths: torch.Tensor, S: int) -> torch.Tensor:
    return torch.arange(S, device=lengths.device)[None, :] >= lengths[:, None]
''',
            "debug": r'''
print(padding_mask(torch.tensor([3, 1, 0]), 4).int())
# expect [[0, 0, 0, 1],
#         [0, 1, 1, 1],
#         [1, 1, 1, 1]]
''',
        },
        {
            "spec": r'''
### Exercise 3: `attention_mask_padded(lengths, S, causal=True)`

The full attention mask for a padded batch, with `True` = **blocked**.

- returns: `(B, S, S)` bool where `out[b, i, j]` is `True` if key `j` is padding for sequence `b`, **or** (when `causal`) `j > i`
- build it by broadcasting a `(B, 1, S)` padding mask against an `(S, S)` causal mask; no loops
''',
            "stub": r'''
def attention_mask_padded(lengths: torch.Tensor, S: int, causal: bool = True) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def attention_mask_padded(lengths: torch.Tensor, S: int, causal: bool = True) -> torch.Tensor:
    pos = torch.arange(S, device=lengths.device)
    blocked = (pos[None, :] >= lengths[:, None])[:, None, :]            # (B, 1, S): key is padding
    if causal:
        blocked = blocked | (pos[None, :] > pos[:, None])[None, :, :]   # (1, S, S): key in the future
    return blocked.expand(len(lengths), S, S)
''',
            "debug": r'''
m = attention_mask_padded(torch.tensor([3, 2]), 4)
print(m.shape)        # expect torch.Size([2, 4, 4])
print(m[1].int())     # expect key 2 and 3 always blocked; above the diagonal blocked
''',
        },
        {
            "spec": r'''
### Exercise 4: `next_token_batch(padded, lengths, ignore_index=-100)`

Turn a padded batch into language-model inputs and targets.

- `padded`: `(B, S)` long; `lengths`: `(B,)` long
- returns: `(inputs, targets)`, each `(B, S - 1)`
  - `inputs = padded[:, :-1]`
  - `targets = padded[:, 1:]`, except positions whose target token is padding are set to `ignore_index`
- do **not** modify `padded` in place
''',
            "stub": r'''
def next_token_batch(padded: torch.Tensor, lengths: torch.Tensor,
                     ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    pass  # Replace this
''',
            "solution": r'''
def next_token_batch(padded: torch.Tensor, lengths: torch.Tensor,
                     ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = padded[:, :-1]
    targets = padded[:, 1:].clone()
    T = targets.shape[1]
    # target t is token t + 1, which is padding when t + 1 >= length
    targets[torch.arange(T, device=padded.device)[None, :] >= (lengths - 1)[:, None]] = ignore_index
    return inputs, targets
''',
            "debug": r'''
padded = torch.tensor([[5, 6, 7, 8], [9, 3, 0, 0]])
inputs, targets = next_token_batch(padded, torch.tensor([4, 2]))
print(inputs)     # expect [[5, 6, 7], [9, 3, 0]]
print(targets)    # expect [[6, 7, 8], [3, -100, -100]]
''',
        },
        {
            "spec": r'''
### Exercise 5: `masked_token_loss(logits, targets, ignore_index=-100)`

Mean cross-entropy over the tokens whose target is **not** `ignore_index`, written by hand.

- `logits`: `(B, T, V)`; `targets`: `(B, T)` long, with `ignore_index` at positions to skip
- returns: a scalar tensor; equal to `F.cross_entropy(..., ignore_index=-100)` when at least one target is valid
- if **no** target is valid, return `0.0` (not `NaN`)
- use a stable `log_softmax` (P9); `ignore_index` is not a valid class index, so make it safe before indexing
''',
            "stub": r'''
def masked_token_loss(logits: torch.Tensor, targets: torch.Tensor,
                      ignore_index: int = -100) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def masked_token_loss(logits: torch.Tensor, targets: torch.Tensor,
                      ignore_index: int = -100) -> torch.Tensor:
    valid = targets != ignore_index
    safe_targets = targets.masked_fill(~valid, 0)                       # any valid class index
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
    return (nll * valid).sum() / valid.sum().clamp_min(1)
''',
            "debug": r'''
torch.manual_seed(0)
logits = torch.randn(2, 3, 10)
targets = torch.tensor([[6, 7, 8], [3, -100, -100]])
print(masked_token_loss(logits, targets))
print(F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100))   # expect the same
print(masked_token_loss(logits, torch.full((2, 3), -100)))                           # expect tensor(0.)
''',
        },
    ],
    # Problem number -> how to use this notebook on it ("Where to use it" section)
    "unlocks": [
        (9, "Run causal attention on a padded batch with `attention_mask_padded`"),
        (13, "Feed a GPT-2 block a padded batch; check that padding never changes the real tokens' outputs"),
        (14, "Prefill a KV cache from a padded prompt batch using the padding mask"),
        (16, "`masked_token_loss` is cross-entropy with `ignore_index`"),
        (41, "Average the per-token distillation loss over real tokens only"),
    ],
}
