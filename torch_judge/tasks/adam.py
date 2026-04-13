"""Adam Optimizer task."""

TASK = {
    "title": "Adam Optimizer",
    "difficulty": "Medium",
    "function_name": "MyAdam",
    "hint": "Track $m$ (1st moment) and $v$ (2nd moment). $m = \\beta_1 m + (1-\\beta_1)\\nabla$, $v = \\beta_2 v + (1-\\beta_2)\\nabla^2$. Bias correct: $\\hat{m} = m/(1-\\beta_1^t)$, $\\hat{v} = v/(1-\\beta_2^t)$. Update: $p \\leftarrow p - \\text{lr} \\cdot \\hat{m} / (\\sqrt{\\hat{v}} + \\epsilon)$.",
    "tests": [
        {
            "name": "Parameters change after step",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw = torch.randn(4, 3, requires_grad=True)\nopt = {fn}([w], lr=0.01)\n(w ** 2).sum().backward()\nw_before = w.data.clone()\nopt.step()\nassert not torch.equal(w.data, w_before), 'Should change after step'\n"
        },
        {
            "name": "Matches torch.optim.Adam",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw1 = torch.randn(8, 4, requires_grad=True)\nw2 = w1.data.clone().requires_grad_(True)\nopt1 = {fn}([w1], lr=0.001, betas=(0.9, 0.999), eps=1e-8)\nopt2 = torch.optim.Adam([w2], lr=0.001, betas=(0.9, 0.999), eps=1e-8)\nfor _ in range(5):\n    (w1 ** 2).sum().backward()\n    opt1.step(); opt1.zero_grad()\n    (w2 ** 2).sum().backward()\n    opt2.step(); opt2.zero_grad()\nassert torch.allclose(w1.data, w2.data, atol=1e-5), f'Max diff: {(w1.data-w2.data).abs().max():.6f}'\n"
        },
        {
            "name": "zero_grad works",
            "code": "\nimport torch\nw = torch.randn(4, requires_grad=True)\nopt = {fn}([w], lr=0.01)\n(w ** 2).sum().backward()\nassert w.grad.abs().sum() > 0\nopt.zero_grad()\nassert w.grad.abs().sum() == 0, 'zero_grad should zero all gradients'\n"
        }
    ]
}
