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
            "name": "Numeric check vs reference",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "\n"
            "def _reference_grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor, eps: float = 1e-5) -> Tensor:\n"
            "    # Same semantics as the reference solution: per-group z-score then -E[A_i * logp_i].\n"
            "    logps = logps.view(-1)\n"
            "    rewards = rewards.view(-1)\n"
            "    group_ids = group_ids.view(-1)\n"
            "    unique_ids = group_ids.unique()\n"
            "    advantages = torch.empty_like(rewards)\n"
            "    for gid in unique_ids:\n"
            "        mask = group_ids == gid\n"
            "        r_g = rewards[mask]\n"
            "        mean_g = r_g.mean()\n"
            "        std_g = r_g.std(unbiased=False)\n"
            "        advantages[mask] = (r_g - mean_g) / (std_g + eps)\n"
            "    advantages_detached = advantages.detach()\n"
            "    return -(advantages_detached * logps).mean()\n"
            "\n"
            "logps = torch.tensor([0.0, -0.5, -1.0, -1.5])\n"
            "rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "loss_student = {fn}(logps, rewards, group_ids)\n"
            "loss_ref = _reference_grpo_loss(logps, rewards, group_ids)\n"
            "assert torch.allclose(loss_student, loss_ref, atol=1e-5, rtol=1e-5), 'Loss should match reference implementation numerically on a fixed example'\n"
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

