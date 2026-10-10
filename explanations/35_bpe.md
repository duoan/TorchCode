# Byte-Pair Encoding

This is a simple **Byte Pair Encoding (BPE) tokenizer**. It starts with individual characters and learns which neighboring tokens frequently occur together, then combines them into larger tokens.

For example:

```text
l o w </w>
→ lo w </w>
→ low </w>
→ low</w>
```

`</w>` marks the end of a word.

## `__init__`: store the learned merges

```python
self.merges = []
```

This list records pairs in the order they were learned:

```python
[('l', 'o'), ('lo', 'w'), ('low', '</w>')]
```

The order matters because later merges can depend on earlier ones.

## `train`: learn frequent pairs

**1. Build a vocabulary of words and their frequencies.**

```python
symbols = tuple(word) + ('</w>',)
vocab[symbols] = vocab.get(symbols, 0) + 1
```

For this corpus:

```python
["low", "low", "lower"]
```

the initial vocabulary is:

```python
{
    ('l', 'o', 'w', '</w>'): 2,
    ('l', 'o', 'w', 'e', 'r', '</w>'): 1
}
```

Repeated words count more heavily during training.

**2. Count every adjacent pair.**

```python
pairs[pair] = pairs.get(pair, 0) + freq
```

Here, `('l', 'o')` appears three times overall: twice in `"low"` and once in `"lower"`.

**3. Choose the most frequent pair.**

```python
best = max(pairs, key=pairs.get)
self.merges.append(best)
```

If pairs tie, Python chooses the first one encountered.

**4. Replace that pair everywhere in the vocabulary.**

```python
new_word.append(word[i] + word[i + 1])
i += 2
```

For example, merging `('l', 'o')` changes:

```text
('l', 'o', 'w', '</w>')
→ ('lo', 'w', '</w>')
```

`i += 2` skips both consumed symbols. Otherwise, the code copies one symbol and advances by one.

This process repeats up to `num_merges` times, stopping early if no adjacent pairs remain.

## `encode`: apply the learned merges

Encoding begins by splitting the text into words, then splitting each word into characters:

```python
symbols = list(word) + ['</w>']
```

It applies every learned merge **in training order**:

```python
for a, b in self.merges:
```

Whenever neighboring symbols match a learned pair, it replaces them with their concatenation:

```python
symbols = symbols[:i] + [a + b] + symbols[i + 2:]
```

Finally, it puts all words’ tokens into one flat list.

## Complete example

```python
bpe = SimpleBPE()
bpe.train(["low", "low", "lower"], num_merges=3)

print(bpe.merges)
# [('l', 'o'), ('lo', 'w'), ('low', '</w>')]

print(bpe.encode("low lower"))
# ['low</w>', 'low', 'e', 'r', '</w>']
```

The frequent word `"low"` becomes a single token. `"lower"` shares the learned `"low"` token, but its remaining characters stay separate.

A few details: `train()` expects an iterable of **words**, encoding discards original whitespace through `split()`, and this implementation operates on **characters**, despite the “Byte Pair Encoding” name.
