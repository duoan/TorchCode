# Gradient Accumulation

This function performs **one training update using gradient accumulation**: it processes several small batches, combines their gradients, then updates the model once. This saves memory compared with processing one large batch.

The arguments are:

- `model`: the neural network.
- `optimizer`: updates the model’s parameters.
- `loss_fn`: measures prediction error.
- `micro_batches`: a collection of `(inputs, targets)` pairs.

Here’s what each part does:

```python
optimizer.zero_grad()
```
Clears gradients left over from previous training updates.

```python
total_loss = 0.0
n = len(micro_batches)
```
Starts a loss counter and counts the small batches.

```python
for x, y in micro_batches:
    loss = loss_fn(model(x), y) / n
```
Computes predictions and their loss. Dividing by `n` makes each batch contribute an equal share to the average loss.

```python
    loss.backward()
```
Computes gradients and **adds them to the existing parameter gradients**. The model’s weights do not change yet.

```python
    total_loss += loss.item()
```
Converts the scaled loss into a Python number and adds it to the counter. The final value is the average of the micro-batch losses.

```python
optimizer.step()
return total_loss
```
Updates the weights once using all accumulated gradients, then returns the loss for logging.

For example, **4 micro-batches of 8 examples** produce one update with an effective batch size of **32**.

Assuming equal-sized micro-batches and a loss averaged over examples, this matches the gradient of one combined batch when the model’s computation is independent of batch composition. Layers such as BatchNorm can make the results differ.
