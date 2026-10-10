# Linear Self-Attention

This implements **linear attention**: each query produces a weighted average of the values, without building the full token-by-token attention matrix.

Assume the inputs have these shapes:

- `Q`: `(B, S, D_k)` — queries
- `K`: `(B, S, D_k)` — keys
- `V`: `(B, S, D_v)` — values

Here, `B` is batch size, `S` is sequence length, and `D_k` / `D_v` are feature dimensions.

1. **Make queries and keys positive**

   ```python
   Q_prime = F.elu(Q) + 1
   K_prime = F.elu(K) + 1
   ```

   This applies a feature map $\phi(x)=\mathrm{ELU}(x)+1$. Attention scores will be dot products between these transformed vectors:



$$

   w_{ij}=\phi(q_i)^\top\phi(k_j)

$$



   The positive features make these weights nonnegative.

2. **Combine keys and values into one shared summary**

   ```python
   KV = torch.bmm(K_prime.transpose(1, 2), V)
   ```

   `torch.bmm` performs matrix multiplication separately for each batch item. This computes:



$$

   KV=\sum_j \phi(k_j)v_j^\top

$$



   Its shape is `(B, D_k, D_v)`. It summarizes all key–value pairs.

3. **Sum the keys for normalization**

   ```python
   Z = K_prime.sum(dim=1, keepdim=True)
   ```

   This gives $\sum_j\phi(k_j)$, with shape `(B, 1, D_k)`.

4. **Use each query to read the summary**

   ```python
   num = torch.bmm(Q_prime, KV)
   den = torch.bmm(Q_prime, Z.transpose(1, 2))
   ```

   For query $i$, these compute:



$$

   \text{num}_i=\sum_j w_{ij}v_j,
   \qquad
   \text{den}_i=\sum_j w_{ij}

$$



5. **Return the normalized weighted average**

   ```python
   return num / (den + 1e-6)
   ```

   The denominator broadcasts across value features. The small constant helps avoid division by zero. The output shape is `(B, S, D_v)`.

The key trick is changing the multiplication order:



$$

(Q'K'^\top)V = Q'(K'^\top V)

$$



The left side creates an `(S, S)` attention matrix. The code computes the right side, creating only a `(D_k, D_v)` summary. With fixed feature dimensions, computation grows **linearly with sequence length**.

This uses a different attention rule from softmax attention; it is not an exact replacement for `softmax(Q @ K.T) @ V`. As written, every query can attend to every key, so it is not causal.
