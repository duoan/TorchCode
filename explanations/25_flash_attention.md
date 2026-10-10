# Blockwise attention with online softmax

This computes **scaled dot-product attention**, processing small blocks at a time so it doesn’t need to build the full attention matrix.

Mathematically, it computes:



$$

\text{output}=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}\right)V

$$



The ordinary implementation would be:

```python
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)
weights = torch.softmax(scores, dim=-1)
output = torch.bmm(weights, V)
```

Your code computes the same mathematical result, up to floating-point rounding, using a **running softmax**.

**The inputs**

`Q`, `K`, and `V` have shape `(B, S, D)`:

- `B`: batch size.
- `S`: sequence length.
- `D`: vector dimension.

Each query in `Q` is compared with every key in `K`. Those comparison scores determine how much each corresponding value in `V` contributes to the output.

**1. Select a block of queries**

```python
for i in range(0, S, block_size):
    qi = Q[:, i:i+block_size]
```

`qi` has shape `(B, bs_q, D)`. Usually `bs_q == block_size`; the final block can be smaller.

For these queries, the code creates three running quantities:

| Variable | Meaning | Shape |
|---|---|---|
| `row_max` | Largest score seen so far for each query | `(B, bs_q, 1)` |
| `row_sum` | Sum of shifted exponentials seen so far | `(B, bs_q, 1)` |
| `acc` | Sum of values weighted by those exponentials | `(B, bs_q, D)` |

They start at `-inf`, zero, and zero because no keys have been processed yet.

**2. Compare these queries with every key block**

```python
for j in range(0, S, block_size):
    kj = K[:, j:j+block_size]
    vj = V[:, j:j+block_size]
    scores = torch.bmm(qi, kj.transpose(1, 2)) / math.sqrt(D)
```

If both blocks contain 32 positions, `scores` has shape `(B, 32, 32)`.

Each entry is:



$$

s_{q,k}=\frac{q\cdot k}{\sqrt D}

$$



Larger scores give that key’s value more influence. Dividing by $\sqrt D$ helps control the scale of the scores.

Every query block visits **all** key blocks, so attention covers the entire sequence.

**3. Compute stable exponentials**

Softmax can be written as:



$$

p_k=\frac{e^{s_k-m}}{\sum_t e^{s_t-m}}

$$



Subtracting the same $m$ from every score leaves the result unchanged. Choosing the maximum score avoids overflowing the exponential.

```python
block_max = scores.max(dim=-1, keepdim=True).values
new_max = torch.maximum(row_max, block_max)
exp_scores = torch.exp(scores - new_max)
```

Here, `new_max` is the largest score across both previously processed keys and the current block.

**4. Rescale the previous contributions**

This is the crucial step:

```python
correction = torch.exp(row_max - new_max)
```

Previous contributions used `row_max` as their reference. If the maximum increases, they must be converted to the new reference:



$$

e^{s-m_{\text{old}}}
\underbrace{e^{m_{\text{old}}-m_{\text{new}}}}_{\text{correction}}
=e^{s-m_{\text{new}}}

$$



For example, if the maximum rises from 5 to 7, all previous contributions are multiplied by $e^{-2}$.

Then the code adds the current block’s contributions:

```python
acc = acc * correction + torch.bmm(exp_scores, vj)
row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
row_max = new_max
```

For each query, the running state now represents:



$$

\text{acc}=\sum_{\text{processed }k}e^{s_k-m}v_k,
\qquad
\text{row\_sum}=\sum_{\text{processed }k}e^{s_k-m}

$$



On the first block, `correction` is zero because the old maximum is `-inf`.

**5. Normalize once all keys have been processed**

```python
output[:, i:i+block_size] = acc / row_sum
```

This produces:



$$

\frac{\sum_k e^{s_k-m}v_k}{\sum_k e^{s_k-m}}
=\sum_k \operatorname{softmax}(s)_k\,v_k

$$



You cannot independently softmax each block and add the results: every block must share the normalization across **all keys**.

This illustrates FlashAttention’s block processing and running-softmax idea. Production FlashAttention uses GPU implementations designed to reduce memory transfers; these Python loops do not provide those performance guarantees. [FlashAttention paper](https://arxiv.org/abs/2205.14135)

The computation still requires $O(BS^2D)$ arithmetic. It has no causal mask, and its default accumulator dtypes can cause a mismatch with FP16/BF16 inputs.
