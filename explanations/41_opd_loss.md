# On-Policy Distillation Reverse-KL Loss

This function computes a **training loss that encourages a student model’s token probabilities to match one or more teacher models**. It compares their full vocabulary distributions, combines the teachers’ losses, and averages across the selected token positions.

The central calculation is **reverse KL divergence**:



$$

D_{\mathrm{KL}}(p_{\text{student}}\|p_{\text{teacher}})
=
\sum_{v=1}^{V}
p_{\text{student}}(v)
\left[\log p_{\text{student}}(v)-\log p_{\text{teacher}}(v)\right]

$$



Here, $V$ is the vocabulary size. Let’s follow the code.

1. **Give every teacher tensor a teacher dimension.**

   Suppose the student logits have shape `(B, L, V)`:

   - `B`: batch size
   - `L`: sequence length
   - `V`: vocabulary size

   A single teacher also has shape `(B, L, V)`. Calling `unsqueeze(0)` changes it to `(1, B, L, V)`. Multiple teachers already have shape `(T, B, L, V)`.

   This lets the remaining code handle both cases identically.

2. **Stop gradients flowing into the teachers.**

   ```python
   teacher_logits = teacher_logits.detach()
   ```

   The teachers provide fixed targets for this loss. Backpropagation still flows through the student logits, but this teacher branch is disconnected from autograd. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html)

3. **Convert logits into probability distributions.**

   ```python
   student_logp = F.log_softmax(student_logits / t, dim=-1)
   student_prob = student_logp.exp()
   teacher_logp = F.log_softmax(teacher_logits / t, dim=-1)
   ```

   Logits are raw scores. Softmax turns them into probabilities that sum to one across the vocabulary; `log_softmax` directly computes their logarithms using a numerically stable formulation. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.log_softmax.html)

   Temperature controls how concentrated the probabilities are:

   - `t = 1`: ordinary softmax.
   - `t > 1`: softer, more evenly spread probabilities.
   - `0 < t < 1`: sharper probabilities.

4. **Compute reverse KL for every teacher and token position.**

   ```python
   per_teacher_kl = (
       student_prob.unsqueeze(0)
       * (student_logp.unsqueeze(0) - teacher_logp)
   ).sum(dim=-1)
   ```

   This implements the equation above. `unsqueeze(0)` lets the student distribution broadcast across all teachers. Summing over `dim=-1` removes the vocabulary dimension, producing shape `(T, B, L)`.

   The **student probabilities weight the comparison**. Consequently, assigning substantial student probability to a token the teacher considers very unlikely incurs a strong penalty. For one teacher, identical distributions give KL zero.

5. **Combine the teachers’ losses.**

   Without supplied weights, all teachers contribute equally. Otherwise:

   ```python
   weights = weights / weights.sum()
   ```

   For example, `[2, 1]` becomes `[2/3, 1/3]`.

   `weights.view(view_shape)` reshapes these weights to `(T, 1, 1)` in this example, so each teacher’s weight applies at every position. Summing across teachers produces `per_token_kl` with shape `(B, L)`.

   This computes a **weighted average of individual KL divergences**. Computing KL against an averaged teacher probability distribution would generally give a different result.

6. **Average the selected positions and scale the loss.**

   With a binary mask:

   ```python
   loss = (per_token_kl * mask).sum() / mask.sum().clamp_min(1.0)
   ```

   Positions marked `1` contribute; positions marked `0`, such as padding, are ignored. The denominator counts included positions. Clamping prevents division by zero; an all-zero mask returns zero when the per-token values are finite.

   Without a mask, `.mean()` averages every position.

   Finally, `t ** 2` applies the conventional distillation scaling intended to compensate for smaller gradients at higher temperatures. It does **not** make the loss or gradients exactly temperature-independent. [Distillation paper](https://arxiv.org/html/1503.02531v1)

One subtlety: **“on-policy” depends on where the inputs came from.** The surrounding training loop must generate sequences using the student and evaluate both models on those same prefixes. This function computes the distribution loss; it does not perform that generation. [On-policy distillation paper](https://arxiv.org/abs/2306.13649)

The code assumes a positive temperature, matching token positions and vocabulary, and nonnegative teacher weights with a positive sum; it does not validate all those conditions.
