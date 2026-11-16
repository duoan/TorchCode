"""Shape-symbol notation for every notebook, and the "📐 Notation" box that explains it.

Letters such as B, S, H, V mean different things in different notebooks (H is image height in
a convolution but the number of heads in attention), so each notebook lists its own symbols.

- Generated prerequisite / bonus notebooks: a "notation" field in scripts/prereqs/content/*.py
- Hand-written problem notebooks: PROBLEMS below, applied by scripts/sync_notation.py

Run `make prereqs` after editing; `make prereqs-check` fails if a box is missing or stale.
"""

from __future__ import annotations

import re

HEADING = "### 📐 Notation"

# Hand-written problem notebooks (file stem -> symbols). Only notebooks that use shape letters
# or abbreviated dimension names appear here.
PROBLEMS: dict[str, list[tuple[str, str]]] = {
    "05_attention": [
        ("batch", "batch size"),
        ("seq_q", "number of queries (query sequence length)"),
        ("seq_k", "number of keys / values (key sequence length)"),
        ("d_k", "dimension of each query and key vector"),
        ("d_v", "dimension of each value vector"),
    ],
    "06_multihead_attention": [
        ("B", "batch size"),
        ("seq_q", "number of queries"),
        ("seq_k", "number of keys / values"),
        ("d_model", "model (embedding) dimension"),
        ("num_heads", "number of attention heads"),
        ("d_k", "dimension per head, `d_model // num_heads`"),
    ],
    "07_batchnorm": [
        ("N", "batch size (number of rows)"),
        ("D", "number of features"),
    ],
    "09_causal_attention": [
        ("batch", "batch size"),
        ("seq", "sequence length (queries and keys are the same tokens)"),
        ("d", "dimension of each query / key vector (`d_k`)"),
        ("d_v", "dimension of each value vector"),
    ],
    "10_gqa": [
        ("d_model", "model (embedding) dimension"),
        ("num_heads", "number of query heads"),
        ("num_kv_heads", "number of key/value heads; each is shared by `num_heads // num_kv_heads` query heads"),
        ("d_k", "dimension per head, `d_model // num_heads`"),
    ],
    "11_sliding_window": [
        ("batch", "batch size"),
        ("seq", "sequence length"),
        ("d", "dimension of each query / key vector (`d_k`)"),
        ("d_v", "dimension of each value vector"),
        ("w", "window size (`window_size`): how far each position may look"),
    ],
    "12_linear_attention": [
        ("B", "batch size"),
        ("S", "sequence length"),
        ("D_k", "dimension of each query / key vector"),
        ("D_v", "dimension of each value vector"),
        ("D", "feature dimension in the complexity bounds (`D_k` ≈ `D_v`)"),
    ],
    "13_gpt2_block": [
        ("d_model", "model (embedding) dimension, also written `d`"),
        ("num_heads", "number of attention heads"),
        ("d_k", "dimension per head, `d_model // num_heads`"),
        ("4d", "hidden size of the MLP: `4 * d_model`"),
    ],
    "14_kv_cache": [
        ("B", "batch size"),
        ("S_new", "number of new tokens in this call (1 when decoding)"),
        ("S_past", "number of tokens already in the cache"),
        ("S", "total sequence length, `S_past + S_new`"),
        ("D", "model dimension (`d_model`)"),
        ("num_heads", "number of attention heads"),
        ("d_k", "dimension per head, `d_model // num_heads`"),
    ],
    "15_mlp": [
        ("d_model", "model (embedding) dimension"),
        ("d_ff", "hidden size of the feed-forward layer"),
    ],
    "16_cross_entropy": [
        ("B", "batch size"),
        ("C", "number of classes"),
    ],
    "20_weight_init": [
        ("C_out", "output channels (or output features for a linear layer)"),
        ("C_in", "input channels (or input features)"),
        ("kH", "kernel height"),
        ("kW", "kernel width"),
        ("fan_in", "number of inputs feeding each output unit"),
    ],
    "22_conv2d": [
        ("B", "batch size"),
        ("C_in", "input channels"),
        ("C_out", "output channels (number of filters)"),
        ("H", "input height"),
        ("W", "input width"),
        ("kH", "kernel height"),
        ("kW", "kernel width"),
        ("H_out", "output height"),
        ("W_out", "output width"),
    ],
    "23_cross_attention": [
        ("B", "batch size"),
        ("S_q", "number of query tokens (e.g. the decoder sequence)"),
        ("S_kv", "number of key/value tokens (e.g. the encoder sequence)"),
        ("D", "model dimension (`d_model`)"),
        ("d_model", "model (embedding) dimension"),
    ],
    "24_rope": [
        ("B", "batch size"),
        ("S", "sequence length (number of positions)"),
        ("D", "vector dimension; must be even, since dimensions are rotated in pairs"),
    ],
    "25_flash_attention": [
        ("B", "batch size"),
        ("S", "sequence length"),
        ("D", "dimension of each query / key / value vector"),
        ("block_size", "number of queries (and keys) processed per block"),
        ("bs_q", "number of queries in the current block (the last block may be smaller)"),
    ],
    "27_vit_patch": [
        ("B", "batch size"),
        ("C", "input channels (3 for RGB)"),
        ("H", "image height"),
        ("W", "image width"),
        ("P", "patch size (`p` in the code)"),
        ("N", "number of patches, `(H / P) * (W / P)`"),
        ("n_h", "patches per column, `H // P`"),
        ("n_w", "patches per row, `W // P`"),
        ("embed_dim", "output dimension of each patch embedding"),
    ],
    "28_moe": [
        ("B", "batch size"),
        ("S", "sequence length"),
        ("D", "model dimension (`d_model`)"),
        ("d_model", "model (embedding) dimension"),
        ("d_ff", "hidden size of each expert's feed-forward layer"),
        ("num_experts", "number of experts"),
        ("top_k", "number of experts each token is routed to"),
    ],
    "34_speculative_decoding": [
        ("K", "number of draft tokens proposed (not the key tensor)"),
        ("V", "vocabulary size (not the value tensor)"),
    ],
    "40_linear_regression": [
        ("N", "number of examples"),
        ("D", "number of features"),
        ("D+1", "features plus the column of ones for the bias"),
    ],
    "41_opd_loss": [
        ("V", "vocabulary size"),
        ("T", "number of teachers (not a sequence length)"),
        ("...", "any leading dimensions, e.g. `(batch, seq)`"),
    ],
    "42_batchnorm2d": [
        ("N", "batch size (PyTorch's image convention)"),
        ("C", "channels"),
        ("H", "image height"),
        ("W", "image width"),
        ("D", "number of features in #07's `(N, D)` version"),
    ],
}


def used_entries(entries: list[tuple[str, str]], text: str) -> list[tuple[str, str]]:
    """The entries whose symbol (or one of its " / "-separated spellings) appears in text."""
    def appears(token: str) -> bool:
        return re.search(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", text) is not None
    return [(s, m) for s, m in entries if any(appears(t.strip()) for t in s.split(" / "))]


def render_box(entries: list[tuple[str, str]]) -> str:
    rows = "\n".join(f"| `{sym}` | {meaning} |" for sym, meaning in entries)
    return f"{HEADING}\n\n| Symbol | Meaning |\n|:---:|---|\n{rows}"


def strip_box(markdown: str) -> str:
    """Remove an existing Notation section (from its heading to the next ### heading or the end)."""
    i = markdown.find(HEADING)
    if i < 0:
        return markdown
    j = markdown.find("\n### ", i + len(HEADING))
    rest = markdown[j:] if j >= 0 else ""
    return (markdown[:i].rstrip("\n") + ("\n" + rest if rest else "")).rstrip("\n")


def with_box(markdown: str, entries: list[tuple[str, str]]) -> str:
    """The markdown with exactly one Notation box at the end (or none if entries is empty)."""
    base = strip_box(markdown).rstrip("\n")
    return f"{base}\n\n{render_box(entries)}\n" if entries else base + "\n"
