# GPT-2 Transformer Block

This implements **one GPT-2-style Transformer block**. It updates each token’s representation in two stages:

1. **Causal self-attention:** gather information from the current token and earlier tokens.
2. **An MLP:** transform the resulting features for each token independently.

The input and output have the same shape:

```text
[B, S, d_model]

B       = batch size
S       = number of tokens per sequence
d_model = number of features representing each token
```

**The constructor creates the learned layers.**

```python
self.num_heads = num_heads
self.d_k = d_model // num_heads
```

Attention uses several heads, each with `d_k` features. For example, `d_model=768` and `num_heads=12` gives 64 features per head. **`d_model` must be divisible by `num_heads`.**

`ln1` and `ln2` normalize each token across its feature dimension, helping stabilize training. The four linear layers produce queries, keys, values, and the final attention output. Inheriting from `nn.Module` registers these layers so PyTorch can train their parameters.

**`_attn` first constructs queries, keys, and values.**

```python
q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
```

The same operations create `k` and `v`. Intuitively:

- **Query:** what information this token is looking for.
- **Key:** what information another token can match against.
- **Value:** the information that token contributes if attended to.

Each projection starts with shape `[B, S, d_model]`. Reshaping and transposing separate the heads:

```text
[B, S, d_model]
       ↓ view
[B, S, num_heads, d_k]
       ↓ transpose
[B, num_heads, S, d_k]
```

Each head can learn different patterns of relationships between tokens.

**Attention scores measure how strongly tokens match.**

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```

For each head, this calculates every query’s dot product with every key:



$$

\text{score}_{i,j}=\frac{q_i\cdot k_j}{\sqrt{d_k}}

$$



The result has shape `[B, num_heads, S, S]`. Entry `(i, j)` measures how strongly token `i` attends to token `j`. Dividing by `sqrt(d_k)` controls the scale of the scores, helping prevent softmax from becoming overly concentrated.

**The mask blocks access to future tokens.**

```python
mask = torch.triu(torch.ones(S, S, ...), diagonal=1)
scores = scores.masked_fill(mask, float('-inf'))
```

For four tokens, the mask is:

```text
        key position
          0  1  2  3
query 0   F  T  T  T
      1   F  F  T  T
      2   F  F  F  T
      3   F  F  F  F
```

`T` means “blocked.” Thus, token 1 can attend to tokens 0 and 1, but cannot see tokens 2 or 3. The same mask broadcasts across all batches and heads.

```python
weights = torch.softmax(scores, dim=-1)
attn = torch.matmul(weights, v)
```

Softmax converts each row into attention weights that sum to 1. Masked positions receive weight zero. Multiplying by `v` gives each token a **weighted sum of the allowed value vectors**.

**The heads are concatenated and mixed.**

```python
return self.W_o(
    attn.transpose(1, 2).contiguous().view(B, S, -1)
)
```

The shape changes back:

```text
[B, num_heads, S, d_k]
       ↓ transpose and reshape
[B, S, d_model]
```

`.contiguous()` ensures the memory layout supports `.view()`. `W_o` then learns how to combine information from the different heads.

**The MLP transforms each token’s features.**

```text
d_model → 4 × d_model → GELU → d_model
```

The first linear layer expands the feature space, GELU adds nonlinearity, and the second layer reduces it back. The same MLP is applied independently at every token position.

**`forward` connects everything with residual additions.**

```python
x = x + self._attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
```

Equivalently:



$$

h=x+\operatorname{Attention}(\operatorname{LN}_1(x))

$$




$$

y=h+\operatorname{MLP}(\operatorname{LN}_2(h))

$$



Each sublayer learns an update that is added to the existing representation. These **residual connections** preserve information and help gradients flow during training. Because normalization happens before each sublayer, this is called a **pre-LayerNorm block**.

A complete language model would also need token and position embeddings, multiple blocks, and an output layer that produces vocabulary scores.
