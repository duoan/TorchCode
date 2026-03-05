"""PPO (Proximal Policy Optimization) clipped loss task."""

TASK = {
    "title": "PPO (Proximal Policy Optimization) Clipped Loss",
    "difficulty": "Hard",
    "function_name": "ppo_loss",
    "hint": (
        "Compute ratio r = exp(new_logps - old_logps_detached). "
        "Form unclipped = r * adv_detached and clipped = clamp(r, 1-clip, 1+clip) * adv_detached. "
        "Return the negative mean of min(unclipped, clipped). "
        "Gradients should flow only through new_logps."
    ),
    "tests": [
        {
            "name": "Basic shape & type",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "new_logps = torch.randn(16, requires_grad=True)\n"
            "old_logps = torch.randn(16)\n"
            "advantages = torch.randn(16)\n"
            "loss = {fn}(new_logps, old_logps, advantages)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be scalar Tensor'\n"
        },
        {
            "name": "Numeric check vs reference",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "\n"
            "def _reference_ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,\n"
            "                        clip_ratio: float = 0.2) -> Tensor:\n"
            "    new_logps = new_logps.view(-1)\n"
            "    old_logps = old_logps.view(-1)\n"
            "    advantages = advantages.view(-1)\n"
            "    # Treat old_logps and advantages as constants\n"
            "    old_logps_detached = old_logps.detach()\n"
            "    adv_detached = advantages.detach()\n"
            "    ratios = torch.exp(new_logps - old_logps_detached)\n"
            "    unclipped = ratios * adv_detached\n"
            "    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_detached\n"
            "    return -torch.min(unclipped, clipped).mean()\n"
            "\n"
            "new_logps = torch.tensor([0.0, -0.2, -0.4, -0.6])\n"
            "old_logps = torch.tensor([0.0, -0.1, -0.5, -0.5])\n"
            "advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])\n"
            "loss_student = {fn}(new_logps, old_logps, advantages, clip_ratio=0.2)\n"
            "loss_ref = _reference_ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2)\n"
            "assert torch.allclose(loss_student, loss_ref, atol=1e-5, rtol=1e-5), 'Loss should match reference implementation numerically on a fixed example'\n"
        },
        {
            "name": "Gradient flows to new_logps only",
            "code": "\n"
            "import torch\n"
            "new_logps = torch.randn(8, requires_grad=True)\n"
            "old_logps = torch.randn(8, requires_grad=True)\n"
            "advantages = torch.randn(8, requires_grad=True)\n"
            "loss = {fn}(new_logps, old_logps, advantages)\n"
            "loss.backward()\n"
            "assert new_logps.grad is not None, 'Gradients should flow through new_logps'\n"
            "assert old_logps.grad is None, 'Gradients should not flow through old_logps (treat as constant baseline)'\n"
            "assert advantages.grad is None, 'Gradients should not flow through advantages (treat as constant advantages)'\n"
        },
    ],
}

