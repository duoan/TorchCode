# Embedding lookup

This defines an **embedding layer**: it maps each integer ID—for example, a word’s ID—to a trainable vector.

1. **Create a PyTorch module**
   ```python
   class MyEmbedding(nn.Module):
   ```
   Inheriting from `nn.Module` lets this layer work inside PyTorch models. `super().__init__()` initializes the module’s machinery.

2. **Create the embedding table**
   ```python
   self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
   ```
   - `num_embeddings`: how many IDs have vectors.
   - `embedding_dim`: how many numbers each vector contains.
   - `torch.randn(...)`: initializes the table with random values from a standard normal distribution.

   The table’s shape is `[num_embeddings, embedding_dim]`; each **row is one embedding vector**.

   `nn.Parameter` registers the table as a trainable parameter, so it appears in `model.parameters()` and can be passed to an optimizer. [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)

3. **Look up vectors**
   ```python
   def forward(self, indices):
       return self.weight[indices]
   ```
   Each integer in `indices` selects the corresponding row of the table.

For example:

```python
embedding = MyEmbedding(num_embeddings=5, embedding_dim=3)
# embedding.weight has shape [5, 3]

indices = torch.tensor([2, 0, 2], dtype=torch.long)
vectors = embedding(indices)  # Calls forward()

print(vectors.shape)  # torch.Size([3, 3])
```

The output contains row **2**, then row **0**, then row **2** again. The first and third vectors are identical because they use the same ID.

For integer tensor inputs, the output shape is the input shape with `embedding_dim` appended: `[batch_size, sequence_length]` becomes `[batch_size, sequence_length, embedding_dim]`. This is the basic embedding lookup behavior. [PyTorch documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.embedding.html)

During training, backpropagation computes gradients for the selected rows, and an optimizer can update their values. **The vectors are learned; the integer IDs are not.**

## Try it interactively

```torchcode-visual
embedding
```
