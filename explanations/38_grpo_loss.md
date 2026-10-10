# Group Relative Policy Optimization Loss

This function builds a loss that **encourages responses with above-average rewards for their prompt and discourages responses with below-average rewards**.

Each position in the tensors describes one sampled response:

- `logps[i]`: the model’s log-probability of that response.
- `rewards[i]`: its reward; higher means better.
- `group_ids[i]`: which prompt it belongs to.
- `B`: the total number of responses across all prompts.

The function does three things.

**1. Compare rewards within each group**

```python
unique_ids = group_ids.unique()
advantages = torch.empty_like(rewards)
```

`unique()` finds the distinct groups. `empty_like()` allocates space for the advantages, using the shape, dtype, and device of `rewards`. The loop fills every position.

```python
mask = group_ids == gid
r_g = rewards[mask]
```

The Boolean mask selects only responses belonging to the current group.

```python
mean_g = r_g.mean()
std_g = r_g.std(unbiased=False)
advantages[mask] = (r_g - mean_g) / (std_g + eps)
```

For response $i$ in group $g$, this calculates:



$$

A_i = \frac{r_i-\mu_g}{\sigma_g+\varepsilon}

$$



The **advantage** measures how much better or worse the response is than its group average:

- Positive: above-average reward.
- Negative: below-average reward.
- Zero: exactly average.

Dividing by the standard deviation makes reward scales more comparable across groups. `unbiased=False` uses the population standard deviation, dividing the variance by the group size rather than group size minus one. `eps` prevents division by zero. [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.std.html)

For example, suppose one prompt has three responses:

| Reward | Difference from mean $2$ | Advantage, approximately |
|---:|---:|---:|
| 1 | −1 | −1.225 |
| 2 | 0 | 0 |
| 3 | +1 | +1.225 |

Another prompt with rewards `[10, 20, 30]` gets approximately the same advantages. The comparison is relative to each prompt’s responses.

**2. Treat advantages as fixed weights**

```python
advantages_detached = advantages.detach()
```

`detach()` removes the advantages from the computation graph. Backpropagation through this loss can affect the policy through `logps`, but cannot affect the reward calculations through the advantages. [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)

**3. Compute the policy loss**

```python
return -(advantages_detached * logps).mean()
```

Mathematically:



$$

L=-\frac{1}{B}\sum_{i=1}^{B} A_i\log\pi_\theta(y_i\mid x_i)

$$



The minus sign turns the objective into something an optimizer can **minimize**. Its derivative with respect to each log-probability is:



$$

\frac{\partial L}{\partial \text{logps}_i}=-\frac{A_i}{B}

$$



So a positive advantage pushes toward increasing that response’s log-probability; a negative advantage pushes toward decreasing it. The function returns a scalar tensor; calling `loss.backward()` computes gradients, and the optimizer performs the update.

A few details matter:

- If every reward in a group is equal—including a group with only one response—all its advantages are zero, so it contributes no learning signal.
- `rewards` should be floating point, and `logps` must remain connected to the model’s computation graph.
- `.mean()` weights responses equally, so larger groups have more total weight.
- `F` is imported but unused.

This is a **simplified policy-gradient loss using GRPO’s group-relative advantages**. The original GRPO objective also includes current/old-policy probability ratios, clipping, and KL regularization toward a reference policy. Those terms are absent here. [Original GRPO paper](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS1)
