# Implement LayerNorm

This implements **layer normalization**: it normalizes the values along the last dimension of `x`, then applies a scale (`gamma`) and shift (`beta`).

For example, if `x` has shape `(batch, sequence_length, hidden_size)`, each token’s `hidden_size` values are normalized independently.

Line by line:

- **`mean = x.mean(dim=-1, keepdim=True)`**\
  Computes the average along the last dimension. `keepdim=True` preserves that dimension with size 1, so subtraction broadcasts correctly.

- **`var = x.var(dim=-1, keepdim=True, unbiased=False)`**\
  Computes variance along the same dimension. `unbiased=False` divides the sum of squared deviations by the number of values, $N$.

- **`x_norm = (x - mean) / torch.sqrt(var + eps)`**\
  Subtracts the mean and divides by the standard deviation, giving approximately zero mean and unit variance. The small `eps` keeps the denominator positive, including when all values are identical.

- **`return gamma * x_norm + beta`**\
  Scales and shifts the normalized values. `gamma` and `beta` are typically learnable parameters, often shaped `(hidden_size,)`. This function uses them; they must be registered as parameters elsewhere to be learned.

The formula is:



$$

y_i = \gamma_i \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i

$$



For `x = [1, 2, 3]`, the mean is `2` and variance is `2/3`. The normalized values are approximately `[-1.225, 0, 1.225]`. With `gamma = 1` and `beta = 0`, those are also the output.

## Try it interactively

```torchcode-visual
normalization
```
