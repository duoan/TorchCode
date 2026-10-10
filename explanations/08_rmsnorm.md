# Implement RMSNorm

This implements **RMS normalization (RMSNorm)**: it normalizes the magnitude of each vector along `x`’s last dimension, then applies a scale called `weight`.

```python
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
```

This computes the root mean square:

1. `x.pow(2)` squares every element.
2. `.mean(dim=-1)` averages those squares over the **last dimension**.
3. `+ eps` adds a small number to prevent division by zero.
4. `torch.sqrt(...)` takes the square root.
5. `keepdim=True` keeps that dimension with size `1`, allowing the division below to broadcast correctly.

```python
return x / rms * weight
```

- `x / rms` normalizes each vector’s magnitude.
- `* weight` scales each feature. In a neural network, `weight` is usually a learned parameter with one value per feature.

Mathematically, for a vector with $d$ elements:



$$

y_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}\;w_i

$$



For example, with `x = [3, 4]` and `weight = [1, 1]`:

```text
rms    = sqrt((9 + 16) / 2) ≈ 3.536
output ≈ [0.849, 1.131]
```

If `x` has shape `[batch, sequence, features]`, each token’s feature vector is normalized independently, and the output has the same shape.

Unlike **LayerNorm**, RMSNorm does **not subtract the mean**—it only rescales the values.

## Try it interactively

```torchcode-visual
normalization
```
