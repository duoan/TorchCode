# Speculative Decoding Acceptance and Rejection

This implements the **accept/reject part of speculative decoding**: a fast draft model proposes tokens, and a target model checks them. The function keeps accepted tokens until the first rejection, replaces that token, and stops.

The inputs are expected to be:

- `target_probs`: shape `[K, V]`, containing the target model’s token probabilities.
- `draft_probs`: shape `[K, V]`, containing the draft model’s token probabilities.
- `draft_tokens`: shape `[K]`, containing token IDs sampled from the draft model.

Here, `K` is the number of proposed tokens and `V` is the vocabulary size.

**1. Check each proposed token**

```python
t = draft_tokens[i].item()
ratio = target_probs[i, t] / max(draft_probs[i, t].item(), 1e-10)
```

`.item()` converts a single-element tensor into a Python number. The ratio compares how likely the proposed token is under the two models:



$$

\text{ratio} = \frac{p_{\text{target}}(t)}{p_{\text{draft}}(t)}

$$



The `1e-10` floor protects against division by zero.

```python
if torch.rand(1).item() < min(1.0, ratio.item()):
    accepted.append(t)
```

This accepts the token with probability:



$$

P(\text{accept }t)=\min\left(1,\frac{p_{\text{target}}(t)}
                                       {p_{\text{draft}}(t)}\right)

$$



For example:

- Target probability `0.6`, draft probability `0.3` → always accept.
- Target probability `0.2`, draft probability `0.5` → accept with probability `0.4`.

This rule reduces how often tokens that the draft model overproduces are kept.

**2. If rejected, sample a replacement**

```python
adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
```

For every vocabulary token, this computes:



$$

\text{adjusted}(x)=\max(p_{\text{target}}(x)-p_{\text{draft}}(x),0)

$$



It gives weight to tokens that the target model wants more often than the draft model proposes.

For example:

```text
                  A      B      C
target_probs    0.50   0.30   0.20
draft_probs     0.20   0.60   0.20
difference      0.30  -0.30   0.00
clamped         0.30   0.00   0.00
normalized      1.00   0.00   0.00
```

If the draft proposed `B`, it would be accepted with probability `0.30 / 0.60 = 0.5`. If rejected, its replacement would always be `A`.

```python
s = adjusted.sum()
if s > 0:
    adjusted = adjusted / s
```

Normalization makes the weights sum to one. If all weights are zero, the code falls back to a uniform distribution.

```python
accepted.append(torch.multinomial(adjusted, 1).item())
return accepted
```

`torch.multinomial` samples one replacement token. The function then stops because later draft tokens were generated assuming the rejected token was part of the sequence; that context has changed.

**3. If every token passes, return them all**

The final `return accepted` returns all `K` draft tokens. The result is a **Python list of token IDs**, including a replacement if rejection occurred.

The acceptance rule and correction distribution together preserve the target model’s sampling distribution under the algorithm’s assumptions. See the [original speculative-decoding paper](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).

For that guarantee, the inputs must be normalized probabilities, each draft token must actually be sampled from its corresponding draft distribution, and both models’ rows must use the same preceding context. The uniform fallback is only a safeguard; after a genuine rejection, it should be unnecessary with valid distributions and exact arithmetic.

This function only handles verification and correction. A full implementation also runs both models and usually samples one extra target token when all draft tokens are accepted.
