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
| P2 | [Broadcasting](templates/00_p02_broadcasting.ipynb) | Right-alignment rule, inserting size-1 dims, outer ops, the `(N,)` vs `(N,1)` bug | `add_channel_bias`, `scale_rows`, `abs_distance_matrix`, `relative_position`, `outer_angles`, `pairwise_sq_dists` | 04, 07, 08, 09, 11, 13, 22, 24, 28, 40 | ✅ |
| P3 | Reductions along dims | `dim`, `keepdim`, `.values/.indices`, `mean`/`var(unbiased=False)`/`std`, `amax`, `norm`, `cumsum` | `row_normalize`, `masked_mean`, `global_l2_norm` | 02, 04, 07, 08, 16, 21, 36, 38 | planned |
| P4 | Indexing & masking | Slicing and `::2` strides, `x[arange, idx]`, boolean read/write, `masked_fill(-inf)`, `triu`/`tril`, `where`, `topk`, `sort`, `scatter`/`gather` | `pick_per_row`, `attention_mask(n, kind, w)` (causal / Longformer sliding / Mistral causal-sliding, built on P2's `relative_position`, applied with `masked_fill`), `keep_topk`, `normalize_per_group` | 09, 11, 13, 14, 16, 18, 24, 28, 32, 38 | planned |
| P5 | [Shape manipulation](templates/00_p05_shape_manipulation.ipynb) | `view` vs `reshape`, contiguity, `transpose`/`permute`, `squeeze`/`flatten`, `cat`/`stack`, `repeat_interleave` | `split_heads`, `merge_heads`, `repeat_kv`, `interleave`, `patchify` | 06, 10, 13, 14, 23, 24, 27, 28 | ✅ |
| P6 | Matmul & einsum | `@`/`mm`/`bmm`/`matmul` batching, `transpose(-2,-1)`, the `xWᵀ + b` convention, einsum | `batched_scores`, `linear_forward`, `bmm_as_einsum` | 03, 05, 06, 12, 22, 26, 40 | planned |
| P7 | Autograd essentials | `requires_grad`, leaf tensors, grad accumulation, `zero_grad`, `no_grad`, `detach` as stop-gradient, in-place updates on leaves | `manual_sgd_step`, `stop_grad_loss`, `accumulated_grad` | 07, 20, 21, 29, 31, 37–41 | planned |
| P8 | nn.Module anatomy | `super().__init__`, Parameter vs buffer vs plain tensor, `ModuleList`, `Sequential`, `train()`/`eval()`, freezing | `ScaleShift`, `RunningCounter` (buffer), `FrozenBackboneHead` | 03, 13, 14, 15, 17, 18, 23, 26, 27, 28, 36 | planned |
| P9 | Numerical stability & log-space | Overflow, subtracting the max, `logsumexp`, `log_softmax`, `logsigmoid`, eps placement, rows that are entirely `-inf`, `exp(logp − logq)`, temperature | `stable_logsumexp`, `kl_from_logits`, `safe_masked_softmax` | 02, 16, 25, 32, 34, 37–39, 41 | planned |
| P10 | Training loop & randomness | forward → loss → backward → step → zero_grad, optimizer/lr, `manual_seed`, `rand`/`bernoulli`/`multinomial` | `train_tiny_regressor`, `sample_from_probs`, `bernoulli_mask` | 17, 29–34, 40 | planned |

## Main path: the 41 problems by module

The recommended order lives in **[LEARNING_PATH.md](LEARNING_PATH.md)**. It groups the problems into 8 modules (Element-wise → Losses & norms → Layers → Training → Attention → Architecture & efficiency → Decoding → Alignment / RL). Each problem has template, solution, and Colab links, plus the prerequisites it builds on.

## Notes on existing problems

Issues found while building this curriculum (a toolbar bug, statement inaccuracies, inconsistent difficulty labels, and more) are tracked in **[IMPROVEMENTS.md](IMPROVEMENTS.md)**.

## Adding a prerequisite notebook

The prerequisite notebooks and `LEARNING_PATH.md` are **generated** from Python definitions in [`scripts/prereqs/`](scripts/prereqs/). Edit the definitions, not the generated files; the next build overwrites the generated files.

| File | What it holds |
|------|---------------|
| `scripts/prereqs/content/pXX_<topic>.py` | One `NOTEBOOK` dict: intro, lesson cells, exercises (spec, stub, solution, debug cell), and the "What this unlocks" rows |
| `scripts/prereqs/curriculum.py` | The P1–P10 list (with planned unlocks for prerequisites without a notebook yet) and the module order of the 41 problems |
| `scripts/prereqs/build.py` | Renders the template and solution notebooks and `LEARNING_PATH.md` |
| `scripts/prereqs/verify.py` | Runs the judge: solutions must pass every test, blank stubs must pass none |

Steps:

1. Write the judge task `torch_judge/tasks/pXX_<topic>.py` by hand, with `difficulty: "Prereq"`, `order: XX`, and every exercise in `function_names` (`function_name` is the first entry). Tests call the user functions by name. Any timing test must also check that the outputs are tensors, so blank stubs don't pass it.
2. Add `scripts/prereqs/content/pXX_<topic>.py` (copy an existing one). Set `prereq_id` to match the entry in `curriculum.py`.
3. Run `make prereqs` to generate `templates/00_pXX_<topic>.ipynb`, `solutions/00_pXX_<topic>_solution.ipynb`, and the updated `LEARNING_PATH.md`.
4. Run `make prereqs-verify` (needs PyTorch; a CPU build in `.venv` is enough). It fails if any generated file is stale, a solution fails a test, or a blank stub passes one.
5. Update the Track 0 table above, the README's prerequisites table, and `00_welcome.ipynb`. These are still edited by hand.

> **Colab note:** Colab installs `torch-judge` from PyPI, so new prerequisite tasks only work there after the next PyPI release.
