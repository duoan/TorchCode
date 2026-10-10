# Direct Preference Optimization Loss

This computes the **Direct Preference Optimization (DPO) loss**, used to train a model to favor a preferred answer over a rejected answer for the same prompt. It measures that preference relative to a fixed reference model. [Original DPO paper](https://arxiv.org/html/2305.18290v3#S4)

The inputs are usually tensors of shape `[batch_size]`:

- `policy_chosen_logps`: the model being trained’s log probabilities for preferred answers.
- `policy_rejected_logps`: its log probabilities for rejected answers.
- `ref_chosen_logps`, `ref_rejected_logps`: the reference model’s corresponding log probabilities.

For language models, each value usually sums the answer tokens’ log probabilities. **Higher log probability means more likely**—for example, `-2` is higher than `-5`.

1. **Compute a score relative to the reference.**

   ```python
   chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
   rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
   ```

   Subtracting log probabilities gives a log probability ratio:



$$

   \log p_{\text{policy}}(y)-\log p_{\text{ref}}(y)
   =\log\frac{p_{\text{policy}}(y)}{p_{\text{ref}}(y)}

$$



   A positive score means the policy assigns that answer greater probability than the reference does. These are *implicit reward scores*. `beta` scales them.

2. **Compare the preferred and rejected scores.**

   ```python
   margin = chosen_rewards - rejected_rewards
   ```

   A positive margin means the policy favors the chosen answer **more strongly than the reference does**.

3. **Convert that margin into a loss.**

   ```python
   -F.logsigmoid(margin).mean()
   ```

   For each pair, the loss is:



$$

   -\log\sigma(m),\qquad \sigma(m)=\frac{1}{1+e^{-m}}

$$



   | Margin $m$ | Loss |
   |---:|---:|
   | −2 | 2.127 |
   | 0 | 0.693 |
   | +2 | 0.127 |

   Minimizing the loss encourages a larger margin. `F.logsigmoid` computes this stably, and `.mean()` averages across the batch.

The function returns a single loss tensor. Calling `loss.backward()` supplies gradients that push up the chosen answer’s log probability and push down the rejected answer’s. The reference model should be frozen; the policy log probabilities must retain their gradients.
