# Softmax Attention

This computes **scaled dot-product attention**: each query compares itself with the keys, then uses those similarities to take a weighted average of the values.

Think of:

- **Q (queries):** what you’re looking for.
- **K (keys):** what each item can be matched against.
- **V (values):** the information each item provides.

Here’s each step:

```python
d_k = K.size(-1)
```
Gets the number of features in each key vector.

```python
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
```
Computes a dot product between every query and every key. Larger scores mean stronger matches.

- `K.transpose(1, 2)` swaps the key and feature dimensions so matrix multiplication works.
- `torch.bmm` performs matrix multiplication separately for each batch item.
- Dividing by `sqrt(d_k)` keeps scores from growing too large as the feature count increases, helping softmax avoid overly concentrated weights.

```python
weights = torch.softmax(scores, dim=-1)
```
Converts each query’s scores into nonnegative weights that sum to **1 across the keys**.

```python
return torch.bmm(weights, V)
```
Uses those weights to mix the value vectors. For example, weights `[0.7, 0.2, 0.1]` produce:

```text
output = 0.7 × V₁ + 0.2 × V₂ + 0.1 × V₃
```

The tensor shapes are:

| Tensor | Shape |
|---|---|
| `Q` | `[batch, num_queries, d_k]` |
| `K` | `[batch, num_keys, d_k]` |
| `V` | `[batch, num_keys, d_v]` |
| `scores`, `weights` | `[batch, num_queries, num_keys]` |
| Output | `[batch, num_queries, d_v]` |

So **each query gets one output vector containing the information it attended to**. This implementation allows every query to attend to every key; it has no masking.

## Try it interactively

```torchcode-visual
attention
```
