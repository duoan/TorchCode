# Linear Regression in Three Ways

This class shows **three ways to fit the same linear regression model**:



$$

\hat y = Xw+b

$$



It learns weights `w` and a bias `b` so that predictions are close to the targets `y`.

| Tensor | Shape | Meaning |
|---|---|---|
| `X` | `(N, D)` | `N` examples, each with `D` features |
| `y` | `(N,)` | One target value per example |
| `w` | `(D,)` | One weight per feature |
| `b` | `()` | A scalar bias |

For example, with two features, each prediction is:



$$

\hat y_i=w_1x_{i1}+w_2x_{i2}+b

$$



All three methods aim to minimize mean squared error:



$$

L=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2

$$



**`closed_form`: solve the least-squares problem directly.**

```python
X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)
```

This adds a column of ones to `X`. That lets the bias become another coefficient:

```text
Original row:   [x1, x2]
Augmented row:  [x1, x2, 1]
Coefficients:   [w1, w2, b]
```

Multiplying the augmented row by the coefficients gives `x1*w1 + x2*w2 + b`.

```python
theta = torch.linalg.lstsq(X_aug, y).solution
```

This finds coefficients minimizing the squared prediction error, using a numerical least-squares solver. There is no learning rate or training loop. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.lstsq.html)

```python
w = theta[:D]
b = theta[D]
```

The first `D` coefficients are weights; the last is the bias. `.detach()` returns tensors disconnected from autograd.

The “normal equation” comment describes the mathematical condition for the solution, but the code calls `lstsq` without explicitly constructing $X_{\text{aug}}^T X_{\text{aug}}$.

**`gradient_descent`: calculate gradients yourself and repeatedly update the parameters.**

It starts with every weight and the bias equal to zero:

```python
w = torch.zeros(D)
b = torch.tensor(0.0)
```

Each iteration predicts all examples and calculates their errors:

```python
pred = X @ w + b
error = pred - y
```

Here, `@` means matrix multiplication. The scalar `b` is automatically added to every prediction.

The gradients of MSE are:



$$

\nabla_w L=\frac{2}{N}X^T(\hat y-y),
\qquad
\frac{\partial L}{\partial b}=\frac{2}{N}\sum_i(\hat y_i-y_i)

$$



These correspond to:

```python
grad_w = (2.0 / N) * (X.T @ error)
grad_b = (2.0 / N) * error.sum()
```

The factor `2` comes from differentiating a square; dividing by `N` comes from averaging the loss.

Then the parameters move opposite the gradient:

```python
w = w - lr * grad_w
b = b - lr * grad_b
```

`lr` controls the update size, and `steps` controls how many updates happen. A large learning rate can cause divergence; a small one may require more steps.

**`nn_linear`: let PyTorch calculate the gradients and apply the updates.**

```python
layer = nn.Linear(D, 1)
```

This layer contains the same weights and bias. Its weight shape is `(1, D)` and its bias shape is `(1,)`. Unlike your manual implementation, it initializes them randomly. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html)

The loop follows the usual PyTorch training sequence:

```python
optimizer.zero_grad()        # Clear previous parameter gradients
pred = layer(X).squeeze(-1)   # Convert predictions from (N, 1) to (N,)
loss = loss_fn(pred, y)       # Calculate MSE
loss.backward()              # Autograd calculates gradients
optimizer.step()             # SGD updates weights and bias
```

`nn.MSELoss()` averages squared errors by default, matching the loss used in your manual gradients. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html)

Although the optimizer is called SGD, this code uses the **entire dataset for each update**, so it performs full-batch gradient descent. With identical initialization, these two training methods would produce essentially the same updates.

A few practical details matter:

- Keep `y` shaped `(N,)`; `(N, 1)` can cause unintended broadcasting in the training methods.
- Create tensors and the layer with `X`’s dtype and device, for example `X.new_ones(N, 1)`, `X.new_zeros(D)`, and `nn.Linear(D, 1).to(X)`.
- Prefer `layer.weight.detach().squeeze(0)` and `layer.bias.detach().squeeze(0)` over `.data`.

Each method returns `w, b`, which you can use to predict new examples:

```python
model = LinearRegression()
w, b = model.closed_form(X, y)
predictions = X_new @ w + b
```
