# GELU activation

This function implements **GELU** (Gaussian Error Linear Unit), an activation function used in neural networks.

For each element of the PyTorch tensor `x`, it computes:



$$

\operatorname{GELU}(x)=x\,\Phi(x)

$$



Here, $\Phi(x)$ is the standard normal cumulative distribution function—the probability that a standard normal random variable is at most $x$.

The code builds that probability using the **error function**, `erf`:

```python
0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

Then it multiplies the result by `x`. This smoothly scales each input by a factor between 0 and 1:

- Large positive inputs are almost unchanged.
- Large negative inputs are pushed close to zero.
- Inputs near zero are roughly halved.

For example:

```text
input:  [-2.000, -1.000, 0.000, 1.000, 2.000]
output: [-0.046, -0.159, 0.000, 0.841, 1.954]
```

All operations are elementwise, so the output has the same shape as `x`, and PyTorch can compute gradients through it.
