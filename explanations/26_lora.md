# LoRA linear layer

This implements **LoRA (Low-Rank Adaptation)**: keep a linear layer fixed and learn a small correction using two trainable matrices. This reduces the number of parameters you need to train. [Original LoRA paper](https://arxiv.org/abs/2106.09685)

The output combines two paths:

```text
x ──→ frozen linear layer ──────────────────→ add → output
 └──→ trainable A ──→ trainable B ──→ scale ──↗
```

**The base layer**

```python
self.linear = nn.Linear(in_features, out_features)
self.linear.weight.requires_grad_(False)
self.linear.bias.requires_grad_(False)
```

`nn.Linear` computes `x @ weight.T + bias`. Setting `requires_grad` to `False` freezes its weight and bias, so ordinary gradient-based training leaves them unchanged. [PyTorch Linear documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.Linear.html)

**The trainable correction**

```python
self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
```

`nn.Parameter` registers these tensors as model parameters, with gradients enabled by default. [PyTorch Parameter documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)

| Matrix | Shape | Purpose |
|---|---|---|
| `A` | `(rank, in_features)` | Projects input features into a smaller space |
| `B` | `(out_features, rank)` | Projects that smaller representation to output features |

Typically, `rank` is much smaller than either feature dimension. For example, with 1,024 input features and `rank=8`, `A` produces an 8-dimensional intermediate representation.

**The forward pass**

Your expression can be expanded as:

```python
base = self.linear(x)
compressed = x @ self.lora_A.T
correction = compressed @ self.lora_B.T
return base + correction * self.scaling
```

Here, `@` means matrix multiplication and `.T` means transpose. For a batch of inputs:

```text
x:          (batch_size, in_features)
compressed: (batch_size, rank)
correction: (batch_size, out_features)
output:     (batch_size, out_features)
```

`self.scaling = alpha / rank` controls the size of the correction.

Mathematically, this is equivalent to using an effective weight:



$$

W_{\text{effective}} = W + \frac{\alpha}{r}BA

$$



The update `BA` has rank at most `r`, which explains the name **low-rank adaptation**. The original `W` stays fixed while `A` and `B` learn.

**Why initialize `B` to zero?**

Initially, the correction is zero, so this module produces exactly the base layer’s output. `B` can receive a nonzero gradient on the first backward pass; `A`’s loss gradient initially equals zero because `B` is zero. Once `B` changes, `A` can learn too.

For a 1,024 × 1,024 layer with `rank=8`, you train only:

```text
A + B: 8×1024 + 1024×8 = 16,384 parameters
Full weight:             1,048,576 parameters
```

That is **64 times fewer trainable weight parameters**.

One important detail: **this code creates a randomly initialized base layer and immediately freezes it**. For typical LoRA fine-tuning, you would first load or copy an existing pretrained layer’s weights and bias.
