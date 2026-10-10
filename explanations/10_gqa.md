# Grouped Query Attention

This implements **grouped-query self-attention (GQA)**. Each token gets several query heads, but multiple query heads share the same key and value heads.

The input and output both have shape **`(B, S, d_model)`**:

- `B`: batch size
- `S`: number of tokens
- `d_model`: features per token

Here’s how it works.

1. **Set the size of each head**

   ```python
   self.d_k = d_model // num_heads
   ```

   For example, with `d_model=512` and `num_heads=8`, each head has 64 features.

2. **Create queries, keys, and values**

   ```python
   q = self.W_q(x) ...
   k = self.W_k(x) ...
   v = self.W_v(x) ...
   ```

   The linear layers learn three representations of each token:

   - **Query:** what information this token is looking for.
   - **Key:** what information this token can be matched against.
   - **Value:** the information this token contributes when attended to.

   The reshaping separates heads, and `transpose(1, 2)` moves the head dimension before the sequence dimension:

   | Tensor | Shape |
   |---|---|
   | `q` | `(B, num_heads, S, d_k)` |
   | `k` | `(B, num_kv_heads, S, d_k)` |
   | `v` | `(B, num_kv_heads, S, d_k)` |

   Queries have more heads than keys and values. That is the defining feature of GQA.

3. **Share each key/value head across a group of query heads**

   ```python
   repeats = self.num_heads // self.num_kv_heads
   k = k.repeat_interleave(repeats, dim=1)
   v = v.repeat_interleave(repeats, dim=1)
   ```

   With 8 query heads and 2 key/value heads:

   ```text
   Query heads:     Q0 Q1 Q2 Q3 | Q4 Q5 Q6 Q7
   Shared KV head:  KV0         | KV1
   ```

   Repeating makes `k` and `v` match the query head count. The copies within each group contain identical data; the queries remain different.

4. **Calculate how strongly each token attends to every token**

   ```python
   scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
   weights = torch.softmax(scores, dim=-1)
   ```

   Each query is compared with every key using a dot product:



$$

   \text{score}_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_k}}

$$



   Scaling helps keep the scores from becoming too large. Softmax converts each row into attention weights that sum to 1 across the key tokens.

   Both `scores` and `weights` have shape **`(B, num_heads, S, S)`**.

5. **Gather information and combine the heads**

   ```python
   attn = torch.matmul(weights, v)
   ```

   Each token receives a weighted sum of the value vectors. The result has shape `(B, num_heads, S, d_k)`.

   ```python
   out = attn.transpose(1, 2).contiguous().view(B, S, -1)
   return self.W_o(out)
   ```

   This concatenates the head outputs into `(B, S, d_model)`. `.contiguous()` makes the transposed tensor suitable for `.view()`, and `W_o` learns how to mix the heads’ results.

A few practical details matter:

- Require `d_model % num_heads == 0` and `num_heads % num_kv_heads == 0`.
- For normal PyTorch training, inherit from `nn.Module` and call `super().__init__()` so the layers are registered.
- There is **no causal or padding mask**, so every token can attend to every position.
- GQA reduces key/value projection sizes and can reduce KV-cache memory. This implementation explicitly copies K/V heads with `repeat_interleave`, so it loses some of that memory advantage during attention computation.
