# Simple Linear Layer

This implements a **fully connected linear layer**: each output is a weighted sum of the inputs, plus a bias.



$$

y = xW^\top + b

$$



`__init__` creates the layer’s weights and biases:

- `in_features`: number of values in each input.
- `out_features`: number of values in each output.
- For example, `SimpleLinear(3, 2)` turns three input values into two output values.

```python
self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
```

Creates a weight matrix with shape `(out_features, in_features)`. Each row contains the weights for one output.

`torch.randn` samples random numbers from a normal distribution with mean 0 and standard deviation 1. Multiplying by `1 / sqrt(in_features)` makes the initial weights smaller as the input size grows, helping control the scale of the outputs.

```python
self.weight.requires_grad_(True)
```

Enables gradient tracking for the weights. After calculating a loss and calling `loss.backward()`, PyTorch stores their gradients in `self.weight.grad`. The trailing `_` means this method changes the tensor in place.

```python
self.bias = torch.zeros(out_features, requires_grad=True)
```

Creates one bias per output, initially zero, with gradient tracking enabled.

`forward` computes the output:

```python
return x @ self.weight.T + self.bias
```

- `@` performs matrix multiplication.
- `.T` transposes the weight matrix so the dimensions match.
- Adding `bias` applies the same bias vector to every input in the batch.

For a layer with three inputs and two outputs:

| Tensor | Shape |
|---|---|
| `x` | `(batch_size, 3)` |
| `weight` | `(2, 3)` |
| `weight.T` | `(3, 2)` |
| Output | `(batch_size, 2)` |

For example:

```python
layer = SimpleLinear(3, 2)
x = torch.randn(4, 3)       # Four examples, three features each
y = layer.forward(x)
print(y.shape)              # torch.Size([4, 2])
```

There is **no activation function** here. Gradient tracking supports training, but an optimizer must still update the weights and biases. Because this is a plain Python class rather than a `torch.nn.Module`, it also lacks conveniences such as `layer.parameters()` and calling the layer as `layer(x)`.
