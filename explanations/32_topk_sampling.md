# Top-k and Top-p Sampling

This function **randomly chooses a token index** from a 1D tensor of model scores (`logits`). Temperature controls randomness; top-k and top-p restrict which tokens can be chosen.

It works in four steps:

1. **Apply temperature**
   ```python
   logits = logits / max(temperature, 1e-8)
   ```
   - `temperature < 1`: favors higher-scoring tokens more strongly.
   - `temperature > 1`: makes probabilities more evenly spread.
   - `temperature = 1`: leaves scores unchanged.

   The `max` prevents division by zero.

2. **Keep the top-k candidates**
   ```python
   top_k_val = logits.topk(top_k).values[-1]
   logits[logits < top_k_val] = float('-inf')
   ```
   `top_k_val` is the kth-highest score. Scores below it become `-inf`, which gives them probability zero after softmax.

   For example, `top_k=3` keeps the three highest-scoring tokens, plus any tokens tied at the cutoff. `top_k=0` disables this step.

3. **Apply top-p, also called nucleus sampling**

   The code sorts scores from highest to lowest, converts them to probabilities, then computes their cumulative sums.

   ```python
   mask = (cumsum - probs) > top_p
   ```
   `cumsum - probs` is the total probability of the tokens **before** each token. A token is removed when that preceding total already exceeds `top_p`. This preserves the token that pushes the total past the cutoff.

   For example:

   | Token | Probability | Probability before it | Keep with `top_p=0.7`? |
   |---|---:|---:|---|
   | A | 0.50 | 0.00 | Yes |
   | B | 0.30 | 0.50 | Yes |
   | C | 0.15 | 0.80 | No |
   | D | 0.05 | 0.95 | No |

   `scatter_` puts the filtered scores back into their original token-index order. `top_p=1.0` disables this step.

4. **Normalize and sample**
   ```python
   probs = torch.softmax(logits, dim=-1)
   return torch.multinomial(probs, 1).item()
   ```
   Softmax renormalizes the remaining candidates. `multinomial` randomly draws one index according to those probabilities, and `.item()` returns it as a Python integer.

If both filters are enabled, **top-k runs first**, and top-p uses probabilities normalized over the surviving candidates.

The function expects a single 1D logits vector, and `top_k` must not exceed its length.

## Try it interactively

```torchcode-visual
sampling
```
