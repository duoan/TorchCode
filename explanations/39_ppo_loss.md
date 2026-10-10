# PPO Clipped Surrogate Loss

This computes the **policy loss for PPO**. It encourages actions that performed better than expected and discourages actions that performed worse, while limiting the incentive to make large policy changes.

Each element corresponds to one sampled action:

- `new_logps`: its log-probability under the current policy.
- `old_logps`: its log-probability under the policy that collected the data.
- `advantages`: how much better or worse that action was than expected. Positive means good; negative means bad.

**1. Keep the old policy and advantages fixed**

```python
old_logps_detached = old_logps.detach()
adv_detached = advantages.detach()
```

`detach()` preserves the values but removes them from gradient computation. Training gradients flow through `new_logps`.

**2. Calculate how the action’s probability changed**

```python
ratios = torch.exp(new_logps - old_logps_detached)
```

Because these are log-probabilities:



$$

r = \exp(\log \pi_{\text{new}}-\log \pi_{\text{old}})
  = \frac{\pi_{\text{new}}}{\pi_{\text{old}}}

$$



For example, `r = 1.3` means the action is now 30% more likely; `r = 0.7` means it is 30% less likely.

**3. Compute the ordinary and clipped objectives**

```python
unclipped = ratios * adv_detached
clipped = torch.clamp(ratios, 0.8, 1.2) * adv_detached
```

The bounds are `[0.8, 1.2]` when `clip_ratio=0.2`.

Maximizing `ratio * advantage` encourages increasing the probability of good actions and decreasing the probability of bad actions.

**4. Choose the conservative objective and turn it into a loss**

```python
return -torch.min(unclipped, clipped).mean()
```

The minimum prevents the objective from rewarding excessively large changes in the beneficial direction:

| Advantage | Ratio | Unclipped | Clipped | Selected |
|---|---:|---:|---:|---:|
| `+1` (good action) | `1.3` | `1.3` | `1.2` | `1.2` |
| `−1` (bad action) | `0.7` | `−0.7` | `−0.8` | `−0.8` |

For a good action, increasing its probability beyond `1.2 ×` receives no additional reward. For a bad action, decreasing its probability below `0.8 ×` receives no additional reward. Changes in the harmful direction still receive a penalty.

Finally, `.mean()` averages across the batch, and the minus sign converts an objective to **maximize** into a loss to **minimize**.

Clipping does **not** enforce a hard limit on probability changes; it limits the incentive supplied by this loss.

## Try it interactively

```torchcode-visual
ppo
```
