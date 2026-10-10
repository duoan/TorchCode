# Implement ReLU

This implements **ReLU** (Rectified Linear Unit): positive values stay unchanged, while negative values become zero.

```python
return x * (x > 0).float()
```

It works element by element:

1. **`x > 0`** creates a Boolean mask: `True` for positive values, `False` otherwise.
2. **`.float()`** converts that mask to numbers: `True → 1.0`, `False → 0.0`.
3. **`x * ...`** multiplies each value by its mask: positive values multiply by 1; other values multiply by 0.

For example:

```text
x:       [-2,  0,  3]
mask:    [ 0,  0,  1]
result:  [ 0,  0,  3]
```

The `torch.Tensor` annotations indicate that the function expects and returns a tensor. PyTorch also provides a built-in version:

```python
return torch.relu(x)
```
