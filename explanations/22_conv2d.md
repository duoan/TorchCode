# Two-dimensional convolution

This implements a **2D convolution** by extracting small windows from the input and taking a weighted sum of each window. It matches `F.conv2d` for integer stride and padding, with `groups=1` and `dilation=1`.

The tensors have these shapes:

- `x`: `(B, C_in, H, W)` — batch size, input channels, height, width.
- `weight`: `(C_out, C_in, kH, kW)` — one filter per output channel.
- `bias`: `(C_out,)` — optional bias per output channel.

**1. Add padding**

```python
if padding > 0:
    x = F.pad(x, [padding] * 4)
```

Adds zeros to the left, right, top, and bottom. For example, `padding=1` turns a `5×5` input into `7×7`.

**2. Calculate output dimensions**

```python
B, C_in, H, W = x.shape
C_out, _, kH, kW = weight.shape
H_out = (H - kH) // stride + 1
W_out = (W - kW) // stride + 1
```

Here, `H` and `W` already include padding. Each filter moves `stride` pixels at a time.

`H_out` and `W_out` describe the output size, although these variables are not used later.

**3. Extract sliding windows**

```python
patches = x.unfold(2, kH, stride).unfold(3, kW, stride)
```

- The first `unfold` extracts windows along height, dimension `2`.
- The second extracts windows along width, dimension `3`.

The resulting shape is:

```text
(B, C_in, H_out, W_out, kH, kW)
```

So `patches[b, :, h, w, :, :]` contains the entire input window needed to compute output position `(h, w)` for batch item `b`.

**4. Apply every filter to every window**

```python
out = torch.einsum('bihwjk,oijk->bohw', patches, weight)
```

The letters label dimensions:

| Letter | Meaning |
|---|---|
| `b` | Batch |
| `i` | Input channel |
| `h`, `w` | Output position |
| `j`, `k` | Position inside the filter |
| `o` | Output channel |

Because `i`, `j`, and `k` disappear from the output expression, `einsum` sums over them:

```text
out[b, o, h, w] =
    sum over i, j, k:
        patches[b, i, h, w, j, k] * weight[o, i, j, k]
```

The output shape is `(B, C_out, H_out, W_out)`.

**5. Add bias**

```python
out = out + bias.view(1, -1, 1, 1)
```

Reshapes the bias to `(1, C_out, 1, 1)`, letting broadcasting add the same channel-specific bias at every spatial position and for every batch item.

For example, an input shaped `(2, 3, 32, 32)` with weights shaped `(16, 3, 3, 3)`, `padding=1`, and `stride=1` produces `(2, 16, 32, 32)`.

Technically, this computes **cross-correlation**, because the filters are not flipped—the same convention PyTorch uses for convolution.
