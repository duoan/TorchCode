# Causal Self-Attention

This computes **causal attention**: each token combines information from itself and earlier tokens, while ignoring future tokens. This is used in models that generate text one token at a time.

The inputs are:

- **Q (queries):** what each token is looking for.
- **K (keys):** what each token offers for matching.
- **V (values):** the information each token contributes.

Assume `Q` and `K` have shape `(batch, sequence_length, d_k)`, and `V` has shape `(batch, sequence_length, d_v)`.

1. **Calculate attention scores**
   ```python
   d_k = K.size(-1)
   scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
   ```
   `torch.bmm` performs matrix multiplication separately for each batch. Each query is compared with every key using a dot product.

   `scores` has shape `(batch, sequence_length, sequence_length)`. Entry `[b, i, j]` measures how strongly token `i` matches token `j`. Dividing by `√d_k` keeps scores from growing too large as the vector dimension increases.

2. **Identify future tokens**
   ```python
   S = scores.size(-1)
   mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
   ```
   The mask is `True` above the main diagonal:
   ```text
   False  True   True
   False  False  True
   False  False  False
   ```
   Rows represent querying tokens; columns represent tokens they might attend to.

3. **Block attention to the future**
   ```python
   scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
   ```
   Future-token scores become `−∞`. `unsqueeze(0)` lets the same mask apply to every batch.

4. **Convert scores into weights**
   ```python
   weights = torch.softmax(scores, dim=-1)
   ```
   Each row becomes weights that sum to 1. Masked positions receive weight **0**, because `exp(−∞) = 0`.

5. **Combine the values**
   ```python
   return torch.bmm(weights, V)
   ```
   Each output token is a weighted sum of the allowed value vectors. The output shape is `(batch, sequence_length, d_v)`.

For example, the third token can attend to tokens **1, 2, and 3**, but cannot attend to token **4** or anything after it.

## Try it interactively

```torchcode-visual
attention-mask
```
