# Cross-entropy loss

This computes the **average cross-entropy loss** for a batch of multiclass predictions. It measures how much probability the model assigns to the correct classes: higher probability means lower loss.

Inputs:

- `logits`: raw prediction scores with shape `[batch_size, num_classes]`.
- `targets`: correct class indices with shape `[batch_size]`, such as `[2, 0, 1]`.

```python
log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
```

This converts scores into **log probabilities**, equivalent to applying `log_softmax`:



$$

\log p_c = z_c - \log\left(\sum_j e^{z_j}\right)

$$



`dim=-1` sums across classes. `keepdim=True` preserves shape `[batch_size, 1]` so subtraction broadcasts across every class. `logsumexp` computes the normalization in a numerically stable way.

```python
log_probs[torch.arange(targets.shape[0]), targets]
```

This selects the log probability of the correct class for each example. For targets `[2, 0, 1]`, it selects:

```python
log_probs[0, 2]
log_probs[1, 0]
log_probs[2, 1]
```

Finally:

```python
return -(...).mean()
```

It negates these log probabilities and averages them:



$$

\text{loss} = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,\text{correct class}}

$$



For example, assigning the correct class probability `0.9` gives loss ≈ `0.105`; assigning it `0.1` gives loss ≈ `2.303`.

For these input shapes, this matches the default behavior of `torch.nn.functional.cross_entropy(logits, targets)`.
