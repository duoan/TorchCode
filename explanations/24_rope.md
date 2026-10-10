# Rotary positional embeddings

This implements **Rotary Positional Embeddings (RoPE)**. It adds position information to attention queries (`q`) and keys (`k`) by rotating pairs of their coordinates.

Both inputs have shape `[B, S, D]`:

- `B`: batch size
- `S`: sequence length
- `D`: vector dimension, which must be even

Here’s how it works:

**1. Create positions and rotation frequencies**

```python
pos = torch.arange(S).unsqueeze(1).float()
```

Creates positions `0, 1, ..., S-1`, with shape `[S, 1]`.

```python
dim = torch.arange(0, D, 2).float()
freqs = 1.0 / (10000.0 ** (dim / D))
```

Creates one frequency for each coordinate pair: `(0, 1)`, `(2, 3)`, etc. Earlier pairs rotate faster; later pairs rotate more slowly.

For example, with `D = 4`:

```text
coordinate pair:  (0, 1)   (2, 3)
frequency:           1       0.01
```

**2. Calculate rotation angles**

```python
angles = pos * freqs
```

Broadcasting produces shape `[S, D/2]`. Each angle is:



$$

\theta_{p,i} = p \cdot 10000^{-2i/D}

$$



So at position `2`, the example above gives angles `[2, 0.02]` radians. At position `0`, every angle is zero, so the vectors stay unchanged.

**3. Rotate each coordinate pair**

```python
x1, x2 = x[..., 0::2], x[..., 1::2]
```

Splits the vector into even and odd coordinates. Each pair then undergoes a standard 2D rotation:



$$

\begin{aligned}
x'_1 &= x_1\cos\theta - x_2\sin\theta \\
x'_2 &= x_1\sin\theta + x_2\cos\theta
\end{aligned}

$$



The cosine and sine tensors broadcast across the batch.

```python
torch.stack([...], dim=-1).flatten(-2)
```

Interleaves the rotated coordinates back into their original order:

```text
[x′₀, x′₁, x′₂, x′₃, ...]
```

The result retains shape `[B, S, D]`.

**4. Apply the rotations to both queries and keys**

```python
return rotate(q), rotate(k)
```

Why rotate both? Attention compares queries and keys using dot products. These rotations make those comparisons depend on the **relative distance between token positions**: the interaction between positions `p` and `t` involves a rotation proportional to `t - p`.

`math` is unused in this code.

## Try it interactively

```torchcode-visual
rope
```
