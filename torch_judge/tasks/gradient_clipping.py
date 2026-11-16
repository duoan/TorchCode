"""Gradient Norm Clipping task."""

TASK = {
    "title": "Gradient Norm Clipping",
    "difficulty": "Easy",
    "function_name": "clip_grad_norm",
    "hint": "Total norm = sqrt(sum(p.grad.norm()^2)). If total > max_norm, scale all grads by max_norm/total. Return original total norm.",
    "tests": [
        {
            "name": "Clips to max_norm",
            "code": "\nimport torch\np1 = torch.randn(10, requires_grad=True)\np2 = torch.randn(10, requires_grad=True)\n(p1 * 10).sum().backward()\n(p2 * 10).sum().backward()\n{fn}([p1, p2], max_norm=1.0)\nnew_norm = torch.sqrt(p1.grad.norm()**2 + p2.grad.norm()**2).item()\nassert new_norm <= 1.0 + 1e-5, f'Clipped norm {new_norm:.4f} > 1.0'\n"
        },
        {
            "name": "Returns original norm",
            "code": "\nimport torch\np = torch.randn(10, requires_grad=True)\n(p * 3).sum().backward()\nexpected = p.grad.norm().item()\nreturned = {fn}([p], max_norm=100.0)\nassert abs(returned - expected) < 1e-4, f'Returned {returned:.4f}, expected {expected:.4f}'\n"
        },
        {
            "name": "No change when norm < max_norm",
            "code": "\nimport torch\np = torch.randn(4, requires_grad=True)\n(p * 0.001).sum().backward()\ngrad_before = p.grad.clone()\ntotal = {fn}([p], max_norm=100.0)\nassert torch.equal(p.grad, grad_before), 'Should not change when norm < max_norm'\nassert total is not None and abs(float(total) - grad_before.norm().item()) < 1e-4, f'Should return the total gradient norm {grad_before.norm().item():.6f}, got {total}'\n"
        },
        {
            "name": "Preserves direction",
            "code": "\nimport torch\ntorch.manual_seed(0)\np = torch.randn(100, requires_grad=True)\n(p * 10).sum().backward()\ndir_before = p.grad / p.grad.norm()\n{fn}([p], max_norm=1.0)\nassert p.grad.norm().item() <= 1.0 + 1e-4, f'Gradient should be clipped to norm <= 1.0, got {p.grad.norm().item():.4f}'\ndir_after = p.grad / p.grad.norm()\nassert torch.allclose(dir_before, dir_after, atol=1e-5), 'Should preserve direction'\n"
        }
    ]
}
