"""GRPO (Group Relative Policy Optimization) Loss task."""

TASK = {
    "title": "GRPO (Group Relative Policy Optimization) Loss",
    "difficulty": "Hard",
    "function_name": "grpo_loss",
    "hint": (
        "Per group, normalize rewards: A_i = (r_i - mean_g) / (std_g + eps). "
        "Detach A_i from graph, then return -mean(A_i * logps)."
    ),
    "tests": [
        {
            "name": "Basic shape & type",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "logps = torch.randn(6, requires_grad=True)\n"
            "rewards = torch.randn(6)\n"
            "group_ids = torch.tensor([0, 0, 0, 1, 1, 1])\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be scalar Tensor'\n"
        },
        {
            "name": "Higher reward lowers loss",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([0.0, 0.0, 0.0])\n"
            "rewards_hi = torch.tensor([2.0, 1.0, 0.0])\n"
            "rewards_lo = torch.tensor([0.0, 1.0, 2.0])\n"
            "group_ids = torch.tensor([0, 0, 0])\n"
            "loss_hi = {fn}(logps, rewards_hi, group_ids)\n"
            "loss_lo = {fn}(logps, rewards_lo, group_ids)\n"
            "assert loss_hi < loss_lo, 'Better rewards should yield smaller loss when logps are fixed'\n"
        },
        {
            "name": "Gradient flows to logps only",
            "code": "\n"
            "import torch\n"
            "logps = torch.randn(4, requires_grad=True)\n"
            "rewards = torch.randn(4, requires_grad=True)\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "loss.backward()\n"
            "assert logps.grad is not None and rewards.grad is None, 'Gradients should flow only through logps'\n"
        },
        {
            "name": "Group-wise normalization",
            "code": "\n"
            "import torch\n"
            "logps = torch.zeros(4, requires_grad=True)\n"
            "rewards = torch.tensor([0.0, 1.0, 10.0, 11.0])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss = {fn}(logps, rewards, group_ids)\n"
            "loss.backward()\n"
            "# Since each group has rewards [0,1] and [10,11], the normalized advantages\n"
            "# should be identical across groups, leading to identical gradients per position.\n"
            "assert torch.allclose(logps.grad[:2], logps.grad[2:]), 'Groups should be treated independently but symmetrically'\n"
        },
    ],
}

