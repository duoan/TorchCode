# Dropout

This implements **dropout**: during training, it randomly sets some input values to zero to help reduce overfitting. During evaluation, it leaves the input unchanged. This follows the behavior of [PyTorch’s `nn.Dropout`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).

Let’s walk through it:

```python
class MyDropout(nn.Module):
```

Inheriting from `nn.Module` makes this a PyTorch layer.

```python
def __init__(self, p=0.5):
    super().__init__()
    self.p = p
```

- `super().__init__()` initializes the underlying PyTorch module.
- `p` is the probability of dropping each element. With `p=0.5`, each element has roughly a 50% chance of becoming zero.

```python
def forward(self, x):
```

Defines the operation performed when you call the layer, such as `dropout(x)`.

```python
if not self.training or self.p == 0:
    return x
```

Returns the input unchanged when the layer is in evaluation mode or dropout is disabled. Calling `.train()` enables training mode; `.eval()` enables evaluation mode. See [module modes](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train).

```python
mask = (torch.rand_like(x) > self.p).float()
```

This builds a random mask:

1. `torch.rand_like(x)` generates random numbers in `[0, 1)` with the same shape and device as `x`. See [`rand_like`](https://docs.pytorch.org/docs/stable/generated/torch.rand_like.html).
2. `> self.p` produces `True` for elements to keep and `False` for elements to drop.
3. `.float()` converts those booleans into `1.0` and `0.0`.

For example, with `p=0.5`:

```text
Random values: [0.8, 0.2, 0.7, 0.1]
Mask:          [1.0, 0.0, 1.0, 0.0]
```

```python
return x * mask / (1 - self.p)
```

Multiplication zeroes out dropped elements. Division scales up the surviving elements:

```text
Input:           [2, 4, 6, 8]
After masking:   [2, 0, 6, 0]
Divide by 0.5:   [4, 0, 12, 0]
```

**Why scale up?** Each value survives with probability `1-p`. Scaling by `1/(1-p)` preserves its expected value:



$$

\mathbb{E}[\text{output}_i]
= (1-p)\frac{x_i}{1-p}+p\cdot0
= x_i.

$$



That preserves values *on average over many random masks*, not necessarily for one call. A fresh mask is generated each training call.

One edge case: this implementation needs `0 <= p < 1`. With `p=1`, it divides by zero; it also lacks validation for invalid probabilities.

## Try it interactively

```torchcode-visual
dropout
```
