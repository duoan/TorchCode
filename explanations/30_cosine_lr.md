# Cosine Learning Rate Schedule with Warmup

This function computes the **learning rate at a given training step**. The learning rate controls how large the model’s parameter updates are.

It has three phases:

1. **Linear warmup** (`step < warmup_steps`)\
   The learning rate rises from `0` toward `max_lr`:
   ```python
   max_lr * step / warmup_steps
   ```
   Halfway through warmup, it is half of `max_lr`.

2. **Cosine decay** (`warmup_steps <= step < total_steps`)\
   `progress` measures how far you are through the decay phase, from `0` to `1`:
   ```python
   progress = (step - warmup_steps) / (total_steps - warmup_steps)
   ```
   As progress increases, `cos(π × progress)` decreases from `1` to `-1`. Therefore:
   ```python
   0.5 * (1.0 + math.cos(math.pi * progress))
   ```
   decreases smoothly from `1` to `0`, bringing the learning rate from `max_lr` down to `min_lr`.

3. **After training** (`step >= total_steps`)\
   The function returns `min_lr`.

For example, with `total_steps=1000`, `warmup_steps=100`, `max_lr=0.001`, and `min_lr=0`:

| Step | Learning rate |
|---:|---:|
| 0 | 0 |
| 50 | 0.0005 |
| 100 | 0.001 |
| 550 | 0.0005 |
| 1000+ | 0 |

The cosine portion falls slowly at first, faster in the middle, then slowly again near the end.
