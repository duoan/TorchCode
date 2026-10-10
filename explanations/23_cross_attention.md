# Multi-head cross-attention

This implements **multi-head cross-attention**: each token in `x_q` gathers information from the tokens in `x_kv`.

The inputs are:

- `x_q`: query sequence, shape **`(B, S_q, d_model)`**
- `x_kv`: key/value sequence, shape **`(B, S_kv, d_model)`**

Here, `B` is the batch size, and the two sequences can have different lengths.

**1. Create queries, keys, and values**

```python
q = self.W_q(x_q)
k = self.W_k(x_kv)
v = self.W_v(x_kv)
```

These are learned linear transformations. Intuitively:

- **Query:** what a token is looking for.
- **Key:** what another token can be matched against.
- **Value:** the information that token contributes.

Queries come from `x_q`; keys and values come from `x_kv`. That makes this *cross-attention*.

**2. Split into attention heads**

```python
.view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
```

Each `d_model` vector is split into `num_heads` smaller vectors, each of size:

```python
d_k = d_model // num_heads
```

After reshaping:

| Tensor | Shape |
|---|---|
| `q` | `(B, num_heads, S_q, d_k)` |
| `k`, `v` | `(B, num_heads, S_kv, d_k)` |

Each head can learn different matching patterns.

**3. Measure how well queries match keys**

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```

For each head, this computes a dot product between every query and every key.

The result has shape **`(B, num_heads, S_q, S_kv)`**. Each entry describes how strongly one query matches one key. Dividing by `sqrt(d_k)` keeps score magnitudes under control, helping softmax behave well.

**4. Convert scores into attention weights**

```python
weights = torch.softmax(scores, dim=-1)
```

Softmax runs across the key/value positions. For every query and head, the weights sum to 1.

For example, weights `[0.1, 0.7, 0.2]` mean that query puts most of its attention on the second key/value token.

**5. Gather information from the values**

```python
attn = torch.matmul(weights, v)
```

Each query receives a weighted sum of the value vectors:



$$

\text{output}_i = \sum_j \text{weight}_{ij}\,v_j

$$



The shape is **`(B, num_heads, S_q, d_k)`**.

**6. Combine the heads and project the result**

```python
attn.transpose(1, 2).contiguous().view(B, S_q, -1)
```

This concatenates the heads back into `d_model` features. `.contiguous()` ensures the transposed tensor has a memory layout compatible with `.view()`.

Finally, `W_o` learns how to mix the heads’ combined features.

The returned shape is **`(B, S_q, d_model)`**: one updated vector for every query token.

Two details to watch:

- `d_model` must be divisible by `num_heads`; the code should validate this.
- There is no attention mask, so every query can attend to every key/value position, including padding if present.
