# Multi-Head Attention

This implements **multi-head scaled dot-product attention**. Each query gets an output by taking a weighted combination of the value vectors. Its similarity to the keys determines those weights.

The inputs are:

- **Q — queries:** what each position is looking for.
- **K — keys:** what each position can be matched against.
- **V — values:** the information each position contributes.

For **self-attention**, you typically pass the same tensor as `Q`, `K`, and `V`. The learned projections then turn it into different query, key, and value representations.

Let `B` be the batch size, `H` the number of heads, and `D = d_model`.

**1. Create the learned projections**

```python
self.d_k = d_model // num_heads
self.W_q = nn.Linear(d_model, d_model)
self.W_k = nn.Linear(d_model, d_model)
self.W_v = nn.Linear(d_model, d_model)
self.W_o = nn.Linear(d_model, d_model)
```

Each head uses `d_k = D / H` features. For example, with `d_model=512` and `num_heads=8`, each head has 64 features.

`W_q`, `W_k`, and `W_v` learn how to produce queries, keys, and values. `W_o` learns how to mix the heads’ outputs.

**2. Project the inputs and split them into heads**

```python
q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
```

The shape changes like this:

```text
[B, S_q, D] → [B, S_q, H, d_k] → [B, H, S_q, d_k]
```

The code does the same for `k` and `v`, using `S_k` positions. Putting the head dimension second lets PyTorch compute all heads independently in one operation.

**3. Compute query–key similarity scores**

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```

For every head, each query is compared with every key using a dot product:

```text
[B, H, S_q, d_k] @ [B, H, d_k, S_k]
                  ↓
          [B, H, S_q, S_k]
```

So `scores[b, h, i, j]` measures how strongly query `i` matches key `j` in head `h`.

Dividing by `sqrt(d_k)` keeps the scores from growing too large as the head dimension increases, which helps avoid overly sharp softmax outputs.

**4. Turn scores into weights and combine values**

```python
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
```

Softmax runs over the **key positions**. For each query in each head, its weights sum to 1.

The next multiplication computes a weighted sum of the values:



$$

\text{attn}_{b,h,i}=\sum_j \text{weights}_{b,h,i,j}\,v_{b,h,j}

$$



The result has shape `[B, H, S_q, d_k]`.

**5. Join the heads and apply the output projection**

```python
out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)
return self.W_o(out)
```

The shapes change back:

```text
[B, H, S_q, d_k] → [B, S_q, H, d_k] → [B, S_q, D]
```

`.contiguous()` makes the transposed tensor’s memory layout suitable for `.view()`.

The final output has shape **`[B, S_q, d_model]`**: one output vector per query position.

A few implementation details matter:

- `d_model` must be divisible by `num_heads`.
- `K` and `V` must have the same sequence length here.
- To integrate properly with PyTorch training, declare `class MultiHeadAttention(nn.Module)` and call `super().__init__()` in the constructor. That registers the layers and enables methods such as `.parameters()` and `.to()`.
- There is no attention mask, so every query can attend to every key, including padding or future positions.
