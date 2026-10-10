# Sliding Window Attention

This function computes **sliding-window attention**: each token combines information from nearby tokens, using attention weights to decide how much each neighbor contributes.

Assume the tensors have these shapes:

- `Q` (queries): `[batch, sequence_length, d_k]`
- `K` (keys): `[batch, sequence_length, d_k]`
- `V` (values): `[batch, sequence_length, d_v]`

Here’s what each step does:

1. **Calculate attention scores**
   ```python
   d_k = K.size(-1)
   scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
   ```
   `torch.bmm` performs matrix multiplication separately for each batch item. Each query is compared with every key using a dot product.

   The result has shape `[batch, sequence_length, sequence_length]`. `scores[b, i, j]` measures how strongly token `i` matches token `j`. Dividing by `sqrt(d_k)` helps keep scores from growing too large.

2. **Identify tokens outside the window**
   ```python
   S = Q.size(1)
   idx = torch.arange(S, device=Q.device)
   mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > window_size
   ```
   `idx` contains token positions: `[0, 1, ..., S-1]`. Broadcasting the two reshaped copies creates a matrix of distances between positions.

   Thus:
   ```python
   mask[i, j] = abs(i - j) > window_size
   ```
   `True` means token `i` is forbidden from attending to token `j`.

3. **Mask those scores**
   ```python
   scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
   ```
   The mask is shared across all batch items. Forbidden scores become negative infinity, so they receive zero weight after softmax.

4. **Convert scores to attention weights**
   ```python
   weights = torch.softmax(scores, dim=-1)
   ```
   For each query token, softmax produces weights over the allowed key positions. These weights sum to 1.

5. **Combine the values**
   ```python
   return torch.bmm(weights, V)
   ```
   Each output token is a weighted sum of nearby value vectors. The output shape is `[batch, sequence_length, d_v]`.

For example, with `window_size = 1`:

```text
Query token    Can attend to
0              0, 1
1              0, 1, 2
2              1, 2, 3
3              2, 3, 4
4              3, 4
```

**Two important details:** `window_size` is a radius, so an interior token can attend to up to `2 * window_size + 1` positions, including itself. The window includes both past and future tokens, so this is **not causal attention**. Also, the code still computes the full score matrix before masking, so its computation and score storage remain quadratic in sequence length.

## Try it interactively

```torchcode-visual
attention-mask
```
