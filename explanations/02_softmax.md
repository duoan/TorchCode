# Implement Softmax

This function computes **softmax**: it turns a tensor’s values into nonnegative weights that sum to 1 along the chosen dimension. Larger inputs receive larger weights.

The formula is:



$$

\operatorname{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}

$$



Here’s what each line does:

1. **Find the largest value**
   ```python
   x_max = x.max(dim=dim, keepdim=True).values
   ```
   `dim=-1` means the last dimension. For a 2D tensor, this finds the maximum in each row. `keepdim=True` preserves that dimension with size 1 so subtraction broadcasts correctly.

2. **Subtract the maximum, then exponentiate**
   ```python
   e_x = torch.exp(x - x_max)
   ```
   Subtracting the same constant from every value leaves the mathematical softmax result unchanged. It makes the largest exponent input zero, preventing overflow from large positive values.

3. **Normalize**
   ```python
   return e_x / e_x.sum(dim=dim, keepdim=True)
   ```
   Divide each exponential by the sum of exponentials along that dimension. The resulting weights sum to 1.

For example:

```python
x = torch.tensor([1.0, 2.0, 3.0])
my_softmax(x)
# approximately tensor([0.0900, 0.2447, 0.6652])
```

The output has the same shape as `x`. PyTorch provides this operation directly as `torch.softmax(x, dim=dim)`.
