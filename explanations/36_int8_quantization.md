# INT8 Quantized Linear Layer

`Int8Linear` is a linear layer that **stores its weights as 8-bit integers**, then converts them back to floating point to compute:

```python
output = x @ weight.T + bias
```

This is the same operation as [`nn.Linear`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html), using an approximation of the original weights.

Assume `weight` has shape `[out_features, in_features]`: each row contains the weights for one output feature.

**1. Calculate one scale per row**

```python
scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
```

- `abs()` takes the magnitude of each weight.
- `amax(dim=1)` finds the largest magnitude in each row.
- Dividing by `127` makes that largest magnitude correspond to roughly `127` or `-127` when quantized.
- `keepdim=True` gives `scale` shape `[out_features, 1]`, so it broadcasts across each row.

Using a separate scale for each output feature is called **per-channel quantization**.

**2. Convert the weights to integers**

```python
torch.round(weight / (scale + 1e-10)).clamp(-128, 127).to(torch.int8)
```

This divides each row by its scale, rounds to integers, restricts values to the int8 range, and stores them as `torch.int8`. The small `1e-10` protects against division by zero for all-zero rows in typical float32 use.

For example:

```text
Original row:     [0.2, -0.5, 1.0]
Scale:           1.0 / 127 ≈ 0.007874
Integer weights: [25, -64, 127]
Reconstructed:   [0.19685, -0.50394, 1.0]
```

The reconstruction is close, but rounding loses some precision.

**3. Store weights, scales, and bias**

```python
self.register_buffer('weight_int8', ...)
self.register_buffer('scale', scale)
```

Buffers belong to the module, are included in its `state_dict`, and move with it when calling `.to(device)`. They are excluded from `model.parameters()`, so a normal optimizer does not update them. [PyTorch buffer documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)

```python
self.bias = nn.Parameter(bias.clone()) if bias is not None else None
```

The bias is copied and registered as a **trainable parameter**, if provided.

**4. Run the forward pass**

```python
w = self.weight_int8.float() * self.scale
```

This *dequantizes* the weights: integer values are multiplied by their row scales to recover approximate floating-point weights.

```python
out = x @ w.T
```

For input shape `[batch_size, in_features]`, the output has shape `[batch_size, out_features]`. The optional bias is then added across the batch.

**The main tradeoff:** int8 weight storage uses about one-quarter the space of float32 weights, plus the scales. However, this implementation recreates floating-point weights on every forward pass and performs floating-point matrix multiplication. It therefore **does not provide an int8 computation speedup automatically**, and peak runtime memory savings may be limited.
