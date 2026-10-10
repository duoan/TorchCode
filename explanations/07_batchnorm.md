# Implement BatchNorm

This function implements **batch normalization**: it normalizes each feature, then applies a learned scale (`gamma`) and shift (`beta`). It assumes `x` has shape **`(batch_size, num_features)`**.

The calculation is:



$$

y = \gamma \frac{x-\text{mean}}{\sqrt{\text{variance}+\epsilon}}+\beta

$$



Here’s how the code works:

1. **During training, compute statistics from the current batch.**

   ```python
   batch_mean = x.mean(dim=0)
   batch_var = x.var(dim=0, unbiased=False)
   ```

   `dim=0` reduces across examples, producing one mean and variance per feature. `unbiased=False` divides the squared deviations by the batch size, $N$, rather than $N-1$.

2. **Update the stored running statistics.**

   ```python
   running_mean.mul_(1 - momentum).add_(momentum * batch_mean.detach())
   ```

   This means:

   ```text
   new_running_mean = (1 - momentum) * old_running_mean
                      + momentum * batch_mean
   ```

   The variance is updated the same way. With `momentum=0.1`, the stored estimate keeps **90% of its previous value and incorporates 10% of the current batch value**.

   The trailing underscores in `mul_` and `add_` mean these operations modify the original tensors. `.detach()` prevents these stored updates from tracking gradients; gradients still flow through the batch statistics used to normalize `x`.

3. **Choose statistics based on the mode.**

   - `training=True`: normalize using the **current batch’s** mean and variance.
   - `training=False`: normalize using the **stored running** mean and variance, without updating them.

4. **Normalize, scale, and shift.**

   ```python
   x_norm = (x - mean) / torch.sqrt(var + eps)
   return gamma * x_norm + beta
   ```

   `eps` keeps the denominator from becoming zero. `gamma` and `beta` let the model adjust the normalized values and are typically trainable tensors with one value per feature.

Typically, `running_mean` starts at zeros, `running_var` at ones, `gamma` at ones, and `beta` at zeros.

The repeated `import torch` is harmless; you only need it once.

## Try it interactively

```torchcode-visual
normalization
```
