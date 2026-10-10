# Beam Search Decoding

This implements **beam search**: it builds a token sequence while keeping the `beam_width` highest-scoring partial sequences at each step.

Unlike greedy decoding, which chooses one next token, beam search keeps several possibilities so a promising alternative can win later.

Each beam is a pair:

```python
(score, sequence)
```

The score is the **sum of the tokens’ log probabilities**. Higher is better: `-1.0` is better than `-3.0`.

Here’s the flow:

1. **Start with one sequence.**
   ```python
   beams = [(0.0, [start_token])]
   ```
   Nothing has been generated yet, so its score is zero. `completed` holds sequences that reach the end token.

2. **Generate up to `max_len` additional tokens.**
   ```python
   for _ in range(max_len):
   ```
   Each iteration extends the current sequences by one token.

3. **Expand each unfinished beam.**
   ```python
   log_probs = log_prob_fn(torch.tensor(seq))
   topk_lp, topk_idx = log_probs.topk(beam_width)
   ```
   `log_prob_fn` takes the current sequence and returns a log probability for every possible next token. `topk` selects the best `beam_width` token choices and their scores.

   Each choice creates a new candidate:
   ```python
   (score + next_token_log_prob, seq + [next_token])
   ```

   For example, if a beam’s score is `-0.7` and a next token has log probability `-0.2`, the extended sequence gets score `-0.9`.

4. **Keep the best candidates across all beams.**
   ```python
   candidates.sort(key=lambda x: x[0], reverse=True)
   beams = candidates[:beam_width]
   ```
   There can be up to `beam_width²` candidates, but only the best `beam_width` survive.

   A sequence ending in `eos_token` goes into `completed` and is no longer extended. If nothing can be extended, the loop stops.

5. **Return the highest-scoring sequence.**
   ```python
   all_seqs = completed + beams
   ```
   It compares finished sequences with those still active, then returns the best sequence—including the start token and EOS token, if present.

A few details matter:

- `log_prob_fn` must return a **1D vector of log probabilities**, with at least `beam_width` entries.
- `max_len` counts newly generated tokens, so the returned sequence can contain `max_len + 1` tokens.
- The result may lack EOS if the length limit is reached.
- Summing log probabilities often favors shorter sequences; there is no length normalization here.
- Beam search can discard a path that would eventually be best, so it does not guarantee the globally optimal sequence.
