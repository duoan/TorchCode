"""Linear Regression Three Ways task."""

TASK = {
    "title": "Linear Regression",
    "difficulty": "Medium",
    "function_name": "LinearRegression",
    "hint": "Closed-form: augment X with ones column, solve w = (X^T X)^{-1} X^T y via torch.linalg.lstsq. Gradient descent: grad_w = (2/N) * X^T @ (pred - y), update w -= lr * grad_w. nn.Linear: create nn.Linear(D, 1), use MSELoss + optimizer.step() loop.",
    "tests": [
        {
            "name": "Closed-form returns correct shapes",
            "code": """
import torch
torch.manual_seed(42)
X = torch.randn(50, 3)
y = X @ torch.tensor([2.0, -1.0, 0.5]) + 3.0 + torch.randn(50) * 0.01
model = {fn}()
w, b = model.closed_form(X, y)
assert w.shape == (3,), f'w shape: {w.shape}, expected (3,)'
assert b.shape == (), f'b shape: {b.shape}, expected scalar'
""",
        },
        {
            "name": "Closed-form finds correct weights",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.closed_form(X, y)
assert torch.allclose(w, true_w, atol=1e-4), f'w: {w} vs true: {true_w}'
assert torch.allclose(b, torch.tensor(true_b), atol=1e-4), f'b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "Gradient descent converges",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.gradient_descent(X, y, lr=0.05, steps=2000)
assert torch.allclose(w, true_w, atol=0.1), f'GD w: {w} vs true: {true_w}'
assert abs(b.item() - true_b) < 0.1, f'GD b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "nn.Linear approach works",
            "code": """
import torch
torch.manual_seed(42)
true_w = torch.tensor([2.0, -1.0, 0.5])
true_b = 3.0
X = torch.randn(100, 3)
y = X @ true_w + true_b
model = {fn}()
w, b = model.nn_linear(X, y, lr=0.05, steps=2000)
assert torch.allclose(w, true_w, atol=0.1), f'nn w: {w} vs true: {true_w}'
assert abs(b.item() - true_b) < 0.1, f'nn b: {b.item():.4f} vs true: {true_b}'
""",
        },
        {
            "name": "All three methods agree",
            "code": """
import torch
torch.manual_seed(0)
X = torch.randn(200, 2)
true_w = torch.tensor([1.5, -2.0])
y = X @ true_w + 1.0 + torch.randn(200) * 0.1
model = {fn}()
w_cf, b_cf = model.closed_form(X, y)
w_gd, b_gd = model.gradient_descent(X, y, lr=0.05, steps=3000)
w_nn, b_nn = model.nn_linear(X, y, lr=0.05, steps=3000)
assert torch.allclose(w_cf, w_gd, atol=0.15), f'CF vs GD: max diff {(w_cf - w_gd).abs().max():.4f}'
assert torch.allclose(w_cf, w_nn, atol=0.15), f'CF vs NN: max diff {(w_cf - w_nn).abs().max():.4f}'
assert abs(b_cf.item() - b_gd.item()) < 0.15, f'Bias CF vs GD: {b_cf.item():.4f} vs {b_gd.item():.4f}'
assert abs(b_cf.item() - b_nn.item()) < 0.15, f'Bias CF vs NN: {b_cf.item():.4f} vs {b_nn.item():.4f}'
""",
        },
        {
            "name": "Closed-form uses no autograd",
            "code": """
import torch
X = torch.randn(30, 2)
y = X @ torch.tensor([1.0, 2.0]) + 0.5
model = {fn}()
w, b = model.closed_form(X, y)
assert not w.requires_grad, 'Closed-form w should not require grad'
""",
        },
    ],
}
