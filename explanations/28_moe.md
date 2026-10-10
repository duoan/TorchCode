# Mixture of Experts

This is a **Mixture-of-Experts (MoE) layer**. It contains several neural networks called *experts*. For each input token, a learned *router* selects `top_k` experts, runs the token through them, and combines their outputs.

Different tokens can select different experts.

The constructor defines:

- `d_model`: the size of each input and output vector.
- `d_ff`: the hidden size inside each expert.
- `num_experts`: how many expert networks exist.
- `top_k`: how many experts process each token, defaulting to `2`.

```python
self.router = nn.Linear(d_model, num_experts)
```

The router produces one score per expert for each token.

```python
self.experts = nn.ModuleList([
    nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model)
    )
    for _ in range(num_experts)
])
```

Each expert is an independent feed-forward network:

```text
d_model → d_ff → ReLU → d_model
```

All experts have the same architecture but separate learned parameters. `ModuleList` registers them as part of the model so PyTorch tracks their parameters. [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)

The `forward()` method works in five steps.

1. **Flatten the tokens.**

   ```python
   orig_shape = x.shape
   if x.dim() == 3:
       B, S, D = x.shape
       x_flat = x.reshape(-1, D)
   else:
       x_flat = x
   ```

   A sequence input has shape `[B, S, D]`: batch size, sequence length, and vector dimension. It becomes `[B*S, D]`, with one token per row. A two-dimensional input `[N, D]` is used directly.

   Tokens are processed independently; this layer does not mix information between sequence positions.

2. **Score every expert and select the best ones.**

   ```python
   logits = self.router(x_flat)
   top_vals, top_idx = logits.topk(self.top_k, dim=-1)
   ```

   For `N` tokens and `E` experts:

   | Tensor | Shape | Meaning |
   |---|---|---|
   | `logits` | `[N, E]` | Every token’s score for every expert |
   | `top_vals` | `[N, top_k]` | Selected scores |
   | `top_idx` | `[N, top_k]` | Selected expert indices |

   `topk` returns the largest scores and their indices along the last dimension. [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.topk.html)

3. **Turn the selected scores into mixture weights.**

   ```python
   weights = torch.softmax(top_vals, dim=-1)
   ```

   The weights are positive and sum to `1` for each token. Softmax is applied **only to the selected experts**. [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)

   For example:

   ```text
   Router scores:    [1, 3, 0, 2]
   Selected experts: [1, 3]       # zero-based indices
   Selected scores:  [3, 2]
   Mixture weights:  [0.731, 0.269]
   ```

4. **Run the selected experts and accumulate their contributions.**

   ```python
   output = torch.zeros_like(x_flat)

   for k in range(self.top_k):
       for e in range(len(self.experts)):
           mask = (top_idx[:, k] == e)
           if mask.any():
               output[mask] += (
                   weights[mask, k:k+1]
                   * self.experts[e](x_flat[mask])
               )
   ```

   Here, `k` is the selection position—first choice, second choice, etc.—and `e` is an expert index.

   `mask` identifies tokens whose choice at position `k` is expert `e`. Only those tokens are passed to that expert.

   `weights[mask, k:k+1]` has shape `[number_of_selected_tokens, 1]`. Keeping that final dimension allows each token’s scalar weight to multiply every feature of its expert output.

   In the example above, one token’s result is:



$$

   y = 0.731\,\mathrm{Expert}_1(x)
       + 0.269\,\mathrm{Expert}_3(x)

$$



   The `+=` combines contributions from all selected experts.

5. **Restore the original shape.**

   ```python
   return output.reshape(orig_shape)
   ```

   An input shaped `[B, S, d_model]` produces an output with the same shape.

This provides many expert parameters while evaluating only `top_k` experts per token. It is a simple implementation: the Python loops can be slow at scale, and there is no mechanism here to encourage balanced expert usage. For normal operation, use `1 <= top_k <= num_experts`.
