# Gradient norm clipping

This function **limits the overall size of the gradients** by scaling them down when their combined L2 norm exceeds `max_norm`. This helps prevent excessively large updates during training.

Here’s what each part does:

1. **Keep parameters that have gradients**
   ```python
   parameters = [p for p in parameters if p.grad is not None]
   ```
   Some parameters may not have participated in the computation, so their `.grad` is `None`.

2. **Calculate the combined gradient norm**
   ```python
   total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in parameters))
   ```
   This treats all gradient values across all parameters as one large vector:


$$

   \text{total norm} = \sqrt{\sum_i \|g_i\|_2^2}

$$



3. **Calculate a scaling factor**
   ```python
   clip_coef = max_norm / (total_norm + 1e-6)
   ```
   The small `1e-6` prevents division by zero.

4. **Scale down gradients if necessary**
   ```python
   if clip_coef < 1:
       for p in parameters:
           p.grad.mul_(clip_coef)
   ```
   `mul_` modifies each gradient **in place**. Every gradient gets the same scaling factor, preserving their combined direction. Gradients below the limit stay unchanged.

5. **Return the original norm**
   ```python
   return total_norm.item()
   ```
   `.item()` converts the scalar tensor to a Python number. The returned norm is measured **before clipping**.

For example, gradients `[3, 4]` have norm `5`. With `max_norm=2`, the scaling factor is approximately `0.4`, giving gradients `[1.2, 1.6]` with norm approximately `2`.

Use it between backpropagation and the optimizer update:

```python
loss.backward()
original_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
optimizer.step()
```

One edge case: if no parameters have gradients, this implementation errors because `sum(...)` returns the integer `0`, which `torch.sqrt` cannot accept.
