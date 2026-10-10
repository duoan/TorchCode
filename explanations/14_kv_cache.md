# KV Cache Attention

This implements **causal multi-head self-attention with a key/value (KV) cache**. Each new token can attend to earlier tokens and itself. The cache stores earlier keys and values so they don’t need to be recomputed on every generation step.

**The input and dimensions**

```python
x.shape == [B, S_new, d_model]
```

- `B`: batch size.
- `S_new`: number of tokens being processed in this call.
- `d_model`: size of each token’s representation.
- `num_heads`: number of attention heads.
- `d_k = d_model // num_heads`: size of each head.

For example, with `d_model=512` and `num_heads=8`, each head uses 64 dimensions. This implementation requires `d_model` to be divisible by `num_heads`.

**Creating queries, keys, and values**

The three learned linear layers create different representations of each token:

- **Query (`q`)**: what information this token is looking for.
- **Key (`k`)**: what other tokens can match against.
- **Value (`v`)**: the information contributed when a token receives attention.

Each projection initially has shape `[B, S_new, d_model]`. This operation splits it into heads:

```python
.view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
```

The shape changes as follows:

```text
[B, S_new, d_model]
          ↓ split into heads
[B, S_new, num_heads, d_k]
          ↓ swap sequence and head dimensions
[B, num_heads, S_new, d_k]
```

Each head computes attention separately.

**Appending the cache**

```python
if cache is not None:
    k = torch.cat([cache[0], k], dim=2)
    v = torch.cat([cache[1], v], dim=2)
```

`cache` contains `(past_keys, past_values)`. Dimension `2` is the sequence dimension, so these lines append the new tokens after the previous tokens.

Afterward:

```text
q: [B, num_heads, S_new,   d_k]
k: [B, num_heads, S_total, d_k]
v: [B, num_heads, S_total, d_k]

S_total = S_past + S_new
```

**Queries cover only the new tokens; keys and values cover the entire sequence.** Earlier queries aren’t needed because this call computes outputs only for the new tokens.

`new_cache = (k, v)` saves the expanded history for the next call. The module doesn’t store the cache internally—you pass it back in.

**Computing attention scores**

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```

For every new query, this computes a dot product with every key:



$$

\text{score}_{i,j} = \frac{q_i \cdot k_j}{\sqrt{d_k}}

$$



The resulting shape is `[B, num_heads, S_new, S_total]`. `torch.matmul` performs these matrix multiplications separately across batch and head dimensions. [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)

Dividing by `sqrt(d_k)` keeps dot products from growing too large as the head dimension increases, helping prevent excessively sharp softmax weights.

**Blocking attention to future tokens**

The causal mask ensures that new query `i` can attend only to key positions:

```text
j <= S_past + i
```

For example, suppose `A B C` are cached and `D E` arrive together:

```text
             Keys
          A B C D E
Query D   ✓ ✓ ✓ ✓ ✗
Query E   ✓ ✓ ✓ ✓ ✓
```

This is why the code uses:

```python
diagonal=S_past + 1
```

`torch.triu` marks positions on and above that offset diagonal—exactly the future positions to block. [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.triu.html)

Those scores become `-inf`, which gives them zero probability after softmax.

When `S_new == 1`, there are no future tokens in the supplied keys: everything is either past history or the current token. No causal mask is needed.

The mask also works for multiple new tokens appended to an existing cache, despite the comment mentioning only prefill.

**Combining the values and producing the output**

```python
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
```

Softmax turns each query’s scores into weights over key positions. Multiplying by `v` creates a weighted sum of the values:

```text
attn: [B, num_heads, S_new, d_k]
```

Finally, the code swaps dimensions back, combines the heads, and applies the learned output projection `W_o`:

```text
[B, num_heads, S_new, d_k]
                 ↓
[B, S_new, d_model]
```

`.contiguous()` arranges the transposed tensor in memory so the following `.view()` can combine the heads.

The return values are:

```python
out        # [B, S_new, d_model]
new_cache  # two tensors, each [B, num_heads, S_total, d_k]
```

Typical use:

```python
attention = KVCacheAttention(d_model=512, num_heads=8).eval()

with torch.inference_mode():
    # Prefill: process the prompt.
    out, cache = attention(torch.randn(1, 3, 512))

    # Decode: process only the next token, reusing the prompt's K/V.
    out, cache = attention(torch.randn(1, 1, 512), cache)
```

Caching saves recomputation of past keys and values, but each new query still attends over the growing history. This is an attention layer; positional encoding, padding masks, and the other parts of a Transformer block would need to be supplied elsewhere.
