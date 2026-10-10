# Adam Optimizer

This is a simple implementation of **Adam**, an optimizer that updates a model’s parameters using their gradients. Adam keeps a moving average of recent gradients and squared gradients to adjust each parameter’s update.

**1. Initialization: `__init__`**

- `params`: the tensors to optimize, usually `model.parameters()`. Converting them to a list lets the optimizer reuse them.
- `lr`: the learning rate, which controls the overall update size.
- `beta1`, `beta2`: how much past gradient information to retain.
- `eps`: a small number that prevents division by zero.
- `t`: the number of calls to `step()`.
- `m`: a moving average of gradients, initially zero.
- `v`: a moving average of squared gradients, initially zero.

`torch.zeros_like(p)` creates a zero tensor with the same shape, device, and data type as parameter `p`.

**2. Updating parameters: `step()`**

Each call increments `t`, then processes every parameter.

```python
with torch.no_grad():
```

This prevents PyTorch from tracking the optimizer’s calculations as part of the computation graph.

```python
if p.grad is None:
    continue
```

If a parameter has no gradient, it is skipped. Otherwise, let its current gradient be $g_t$.

The optimizer updates its two moving averages:



$$

m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t

$$





$$

v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2

$$



- `m` smooths the gradient direction.
- `v` tracks gradient magnitude through squared gradients.

Because both start at zero, their early values are biased toward zero. These lines correct that:

```python
m_hat = self.m[i] / (1 - self.beta1 ** self.t)
v_hat = self.v[i] / (1 - self.beta2 ** self.t)
```

Finally:

```python
p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

This updates the parameter **in place**:



$$

p \leftarrow p - \mathrm{lr}\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}

$$



All operations are elementwise. Each element’s update is scaled by its own gradient history.

**3. Clearing gradients: `zero_grad()`**

PyTorch accumulates gradients across calls to `backward()`. This method sets existing gradients to zero so the next training step starts fresh.

Typical usage:

```python
optimizer = MyAdam(model.parameters())

for inputs, targets in loader:
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)
    loss.backward()   # Compute gradients
    optimizer.step()  # Update parameters
```

One subtlety: `t` is shared across all parameters. If some parameters frequently have `grad=None`, their bias correction uses the total number of optimizer steps rather than the number of updates they actually received.
