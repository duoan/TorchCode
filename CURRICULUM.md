# 🗺️ TorchCode Curriculum

The 41 problems test **ML concepts** (attention, normalization, RL losses, …), but almost all of them quietly assume the same ~10 **PyTorch skills**: broadcasting, reductions with `dim`/`keepdim`, fancy indexing, the split/merge-heads reshape, `detach`, `nn.Module` structure, and so on.
When those skills are shaky, a learner can understand attention perfectly and still spend an hour on a transposed view.

This curriculum adds a **Track 0: Prerequisites** that teaches exactly those skills. It also orders the existing problems into modules, so each problem only uses skills already practiced.

## How prerequisite notebooks work

- File names: `templates/00_pXX_<topic>.ipynb` (solutions in `solutions/00_pXX_<topic>_solution.ipynb`). Judge task id: `pXX_<topic>`.
- Each notebook is a **short lesson** (runnable demo cells) followed by **3–5 mini-exercises**. Each exercise is a small function taken from a real TorchCode problem.
- One judge task per notebook: `check("p05_shape_manipulation")` tests every exercise. Each test is labelled with the exercise it checks.
- Task definitions use `difficulty: "Prereq"`, so `status()` and the web UI list them before the Easy problems.

## Track 0: Prerequisites

| # | Notebook | Covers | Exercises | Unlocks | Status |
|---|----------|--------|-----------|---------|:------:|
| P1 | Tensor basics | Creation (`tensor`/`zeros`/`full`/`arange`/`*_like`), dtype (float/bool/int8), device, `.item()`, `.to()` | `bool_to_float_mask`, `make_like` | 01, 19, 20, 36 | planned |
| P2 | [Broadcasting](templates/00_p02_broadcasting.ipynb) | Right-alignment rule, inserting size-1 dims, outer ops, the `(N,)` vs `(N,1)` bug | `add_channel_bias`, `scale_rows`, `abs_distance_matrix`, `outer_angles`, `pairwise_sq_dists` | 04, 07, 08, 11, 22, 24, 28 | ✅ |
| P3 | Reductions along dims | `dim`, `keepdim`, `.values/.indices`, `mean`/`var(unbiased=False)`/`std`, `amax`, `norm`, `cumsum` | `row_normalize`, `masked_mean`, `global_l2_norm` | 02, 04, 07, 08, 16, 21, 36, 38 | planned |
| P4 | Indexing & masking | Slicing and `::2` strides, `x[arange, idx]`, boolean read/write, `masked_fill(-inf)`, `triu`/`tril`, `where`, `topk`, `sort`, `scatter`/`gather` | `pick_per_row`, `causal_mask`, `keep_topk`, `normalize_per_group` | 09, 11, 14, 16, 18, 24, 28, 32, 38 | planned |
| P5 | [Shape manipulation](templates/00_p05_shape_manipulation.ipynb) | `view` vs `reshape`, contiguity, `transpose`/`permute`, `squeeze`/`flatten`, `cat`/`stack`, `repeat_interleave` | `split_heads`, `merge_heads`, `repeat_kv`, `interleave`, `patchify` | 06, 10, 13, 14, 22, 23, 24, 27 | ✅ |
| P6 | Matmul & einsum | `@`/`mm`/`bmm`/`matmul` batching, `transpose(-2,-1)`, the `xWᵀ + b` convention, einsum | `batched_scores`, `linear_forward`, `bmm_as_einsum` | 03, 05, 06, 12, 22, 26, 40 | planned |
| P7 | Autograd essentials | `requires_grad`, leaf tensors, grad accumulation, `zero_grad`, `no_grad`, `detach` as stop-gradient, in-place updates on leaves | `manual_sgd_step`, `stop_grad_loss`, `accumulated_grad` | 20, 21, 29, 31, 37–41 | planned |
| P8 | nn.Module anatomy | `super().__init__`, Parameter vs buffer vs plain tensor, `ModuleList`, `Sequential`, `train()`/`eval()`, freezing | `ScaleShift`, `RunningCounter` (buffer), `FrozenBackboneHead` | 13–18, 23, 26, 27, 28, 36 | planned |
| P9 | Numerical stability & log-space | Overflow, subtracting the max, `logsumexp`, `log_softmax`, `logsigmoid`, eps placement, rows that are entirely `-inf`, `exp(logp − logq)`, temperature | `stable_logsumexp`, `kl_from_logits`, `safe_masked_softmax` | 02, 16, 25, 32, 34, 37–39, 41 | planned |
| P10 | Training loop & randomness | forward → loss → backward → step → zero_grad, optimizer/lr, `manual_seed`, `rand`/`bernoulli`/`multinomial` | `train_tiny_regressor`, `sample_from_probs`, `bernoulli_mask` | 17, 29–34, 40 | planned |

## Main path: the 41 problems by module

Problem numbers stay the same. This is only a recommended **order**.

| Module | Problems, in order | Requires |
|--------|--------------------|----------|
| 1. Element-wise | 01 ReLU → 19 GELU → 02 Softmax | P1–P3, P9 |
| 2. Losses & norms | 16 Cross-Entropy → 04 LayerNorm → 08 RMSNorm → 07 BatchNorm | P3, P4, P9 (07 also P7) |
| 3. Layers | 03 Linear → 20 Kaiming Init → 18 Embedding → 17 Dropout → 15 SwiGLU → 22 Conv2d → 27 ViT Patch | P5, P6, P8 |
| 4. Training | 40 Linear Regression → 21 Grad Clipping → 31 Grad Accumulation → 29 Adam → 30 Cosine LR | P7, P10 |
| 5. Attention | 05 SDPA → 09 Causal → 11 Sliding Window → 06 MHA → 23 Cross-Attn → 10 GQA → 24 RoPE → 12 Linear Attn → 25 Flash Attn | P4–P6, P9 |
| 6. Architecture & efficiency | 13 GPT-2 Block → 14 KV Cache → 26 LoRA → 28 MoE → 36 INT8 Quantization | Modules 3 and 5 |
| 7. Decoding | 32 Top-k/p Sampling → 33 Beam Search → 34 Speculative Decoding → 35 BPE | P4, P9, P10 (35 is plain Python) |
| 8. Alignment / RL | 37 DPO → 39 PPO → 38 GRPO → 41 OPD | P7, P9 |

## Notes on existing problems

- **#03 Linear** and **#06 MHA** reference solutions are plain Python classes, not `nn.Module` subclasses, but the README describes #03 as an `nn.Module`. Once P8 exists, these two should be aligned with it.
- **#40 Linear Regression** is the gentlest end-to-end problem (closed form, manual GD, then `nn.Linear` plus an optimizer). It is a good bridge from Track 0 into the main set, even though it is labelled Medium.
- **#30 Cosine LR** and **#35 BPE** use almost no tensor operations and need no prerequisites.

## Adding a prerequisite notebook

1. Add `torch_judge/tasks/pXX_<topic>.py` with `difficulty: "Prereq"`, `order: XX`, and every exercise listed in `function_names` (`function_name` is the first entry). Tests reference the user functions by name.
2. Add `templates/00_pXX_<topic>.ipynb`: lesson cells, then for each exercise a spec cell, a `# ✏️ YOUR IMPLEMENTATION HERE` cell, and a debug cell, ending with `check("pXX_<topic>")`.
3. Add the matching `solutions/00_pXX_<topic>_solution.ipynb`. It should pass the judge.
4. Run `python scripts/add_colab_badges.py`, then update the table above, the README, and `00_welcome.ipynb`.

> **Colab note:** Colab installs `torch-judge` from PyPI, so new prerequisite tasks only work there after the next PyPI release.
