# 🗺️ TorchCode Curriculum

The 42 problems test **ML concepts** (attention, normalization, RL losses, …), but almost all of them quietly assume the same ~10 **PyTorch skills**: broadcasting, reductions with `dim`/`keepdim`, fancy indexing, the split/merge-heads reshape, `detach`, `nn.Module` structure, and so on.
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
| P1 | [Tensor basics](templates/00_p01_tensor_basics.ipynb) | Creation (`tensor`/`zeros`/`full`/`arange`/`*_like`/`new_*`), dtype (float/bool/int8, casting and overflow), shape, device, `.item()`, in-place ops, element-wise math reference (`erf`, `sin`/`cos`, `F.silu`/`F.elu`/`F.gelu`) | `positive_mask`, `running_max_init`, `scale_inplace_`, `quantize_int8`, `fan_in_out` | 01, 07, 12, 13, 15, 19, 20, 21, 24, 25, 29, 36 | ✅ |
| P2 | [Broadcasting](templates/00_p02_broadcasting.ipynb) | Right-alignment rule, inserting size-1 dims, outer ops, the `(N,)` vs `(N,1)` bug | `add_channel_bias`, `scale_rows`, `abs_distance_matrix`, `relative_position`, `outer_angles`, `pairwise_sq_dists` | 04, 07, 08, 09, 11, 13, 22, 24, 28, 40, 42 | ✅ |
| P3 | [Reductions along dims](templates/00_p03_reductions.ipynb) | `dim`, `keepdim`, `.values/.indices`, `amax`/`argmax`, `mean`/`var(unbiased=False)`/`std`, multi-dim reductions, norms across a list of tensors, cumulative sums (inclusive and exclusive); more: `std_mean`, `median`/`quantile`, `logcumsumexp`, `count_nonzero`, `isclose`; many-ways box | `stable_shift`, `normalize_rows`, `channel_stats`, `masked_mean`, `global_l2_norm`, `exclusive_cumsum` | 02, 04, 07, 08, 12, 21, 25, 32, 36, 38, 41, 42 | ✅ |
| P4 | [Indexing & masking](templates/00_p04_indexing_masking.ipynb) | Slicing and `::2` strides, `weight[idx]` and `x[rows, cols]`, boolean read/write, `torch.where`, `masked_fill(-inf)`, `triu`, `topk`/`sort`, `gather`/`scatter`, grouping by id (`unique`, `bincount`, `index_add`); more: `F.one_hot`, `nonzero` (and GPU sync), `searchsorted` (inverse-CDF sampling), `take_along_dim`, `index_select`, `scatter_add`; many-ways box | `pick_per_row`, `attention_mask`, `apply_mask`, `keep_topk`, `unsort`, `group_mean` | 09, 11, 13, 14, 16, 18, 24, 28, 32, 38 | ✅ |
| P5 | [Shape manipulation](templates/00_p05_shape_manipulation.ipynb) | `view` vs `reshape`, contiguity, `transpose`/`permute`, `squeeze`/`flatten`, `cat`/`stack`, `repeat_interleave`, sliding windows (`F.pad`, `unfold`, output-size formula); more: `expand` vs `repeat` (view vs copy), `unflatten`, `movedim`, `roll`, `*_as`; many-ways box | `split_heads`, `merge_heads`, `repeat_kv`, `interleave`, `patchify`, `sliding_windows` | 06, 10, 13, 14, 22, 23, 24, 27, 28 | ✅ |
| P6 | [Matmul & einsum](templates/00_p06_matmul_einsum.ipynb) | `x @ W.T + b` layout, `@`/`mm`/`bmm`/`matmul` batching, `transpose(-2, -1)`, einsum, associativity and cost, solving linear systems (`linalg.solve`, `lstsq`, conditioning) | `linear_forward`, `attention_scores`, `patch_conv`, `qkv_reassociated`, `lora_forward` | 03, 05, 06, 09, 12, 22, 23, 26, 36, 40 | ✅ |
| P7 | [Autograd essentials](templates/00_p07_autograd.ipynb) | Leaf tensors and `.grad`, gradient accumulation and zeroing, `torch.no_grad()` updates, `detach` as stop-gradient, `torch.autograd.grad`, straight-through estimator; more: `inference_mode`, tensor hooks, `gradcheck`, `detect_anomaly`; ways to stop gradients compared | `grad_of`, `sgd_step_`, `stop_gradient_mse`, `accumulate_grads`, `straight_through_round` | 07, 20, 21, 29, 31, 37, 38, 39, 40, 41, 42 | ✅ |
| P8 | [nn.Module anatomy](templates/00_p08_nn_module.ipynb) | `super().__init__()`, Parameter vs buffer vs plain tensor, `ModuleList`/`Sequential` registration, `train()`/`eval()`, freezing; more: `state_dict` save/load, `apply` for init, forward hooks, `ModuleDict`/`add_module` | `ScaleShift`, `RunningMean`, `MLPStack`, `freeze_`, `num_trainable` | 03, 13, 14, 15, 17, 18, 23, 26, 27, 28, 36 | ✅ |
| P9 | [Numerical stability & log-space](templates/00_p09_numerical_stability.ipynb) | float32 limits, max-subtraction, `logsumexp`, `log_softmax` vs `log(softmax)`, stable `log σ`, log-space sums and ratios, the `torch.where` NaN-gradient trap, fully masked rows, online softmax (merging block statistics) | `stable_logsumexp`, `log_softmax`, `log_sigmoid`, `kl_from_logits`, `safe_masked_softmax`, `merge_softmax_stats` | 02, 16, 25, 32, 33, 34, 37, 38, 39, 41 | ✅ |
| P10 | [Training loop & randomness](templates/00_p10_training_loop.ipynb) | The canonical loop, `param_groups` and the LR, seeds and `torch.Generator`, `rand`/`bernoulli`/`multinomial`/`randperm`, shuffling and mini-batches, `eval()` + `no_grad()`; more: `Dataset`/`DataLoader`, `lr_scheduler`; many-ways box | `train_steps`, `set_lr_`, `keep_mask`, `sample_tokens`, `shuffled_batches` | 17, 29, 30, 31, 32, 34, 40 | ✅ |

## Track B: Bonus (optional)

Notebooks beyond the problems: techniques interviewers ask about and real code uses, but that no problem requires. They work exactly like prerequisites (content in `scripts/prereqs/content/bNN_*.py`, `difficulty: "Bonus"`, same build and verify), except that they end with **🔁 Where to use it** (problems to re-solve with the technique) instead of "What this unlocks", and they don't appear in the problems' "Builds on" column.

<!-- bonus-table:start -->
| # | Notebook | Covers | Exercises | Try it on | Status |
|---|----------|--------|-----------|-----------|:------:|
| B1 | [Custom autograd functions](templates/00_b01_custom_autograd.ipynb) | `torch.autograd.Function` (`forward`, `backward`, `ctx`), the backward contract, `gradcheck`, deriving backward rules (element-wise, matmul, reductions, softmax), gradient reversal | `ReLUFn`, `SigmoidFn`, `LinearFn`, `SoftmaxFn`, `FakeQuantFn` | 01, 02, 03, 16, 19, 36 | ✅ |
| B2 | [Debugging & testing tensors](templates/00_b02_debugging.ipynb) | The common error messages decoded, `torch.testing.assert_close`, finding the first NaN with forward hooks, gradient health (missing and exploding gradients), a debugging workflow | `first_nan_module`, `params_without_grad`, `diff_report`, `grad_norms_by_name` | 13, 21, 25, 26, 28 | ✅ |
| B3 | [Mixed precision & devices](templates/00_b03_mixed_precision.ipynb) | float32 vs float16 vs bfloat16 (`torch.finfo`), overflow / underflow / accumulation error, what to keep in float32, `torch.autocast`, loss scaling (what `GradScaler` does), device-agnostic code and GPU timing | `count_overflow`, `softmax_lowp`, `unscale_and_check`, `to_device_like` | 02, 08, 25, 29, 31, 36 | ✅ |
| B4 | [Sequence batching](templates/00_b04_sequence_batching.ipynb) | Padding and lengths, padding + causal masks, `DataLoader` with `collate_fn`, bucketing, next-token targets with `ignore_index`, packing with block-diagonal masks | `pad_batch`, `padding_mask`, `attention_mask_padded`, `next_token_batch`, `masked_token_loss` | 09, 13, 14, 16, 41 | ✅ |
| B5 | [Vectorizing with torch.func](templates/00_b05_torch_func.ipynb) | `vmap` (in_dims, nesting), functional `grad`, `functional_call`, per-sample gradients, `jacrev` / `hessian`, model ensembles with `stack_module_state` | `batched_dot`, `pairwise_fn`, `per_sample_grads`, `jacobian_of`, `ensemble_forward` | 05, 21, 28, 40 | ✅ |
| B6 | [Memory & performance](templates/00_b06_performance.ipynb) | Views vs copies (shared storage), in-place ops and autograd, measuring speed with `torch.utils.benchmark`, hidden syncs (`.item()`), peak memory: chunking, `no_grad`, activation checkpointing | `is_view_of`, `moving_average`, `chunked_attention`, `checkpointed_mlp` | 05, 13, 14, 22, 25 | ✅ |
<!-- bonus-table:end -->

## Main path: the 42 problems by module

The recommended order lives in **[LEARNING_PATH.md](LEARNING_PATH.md)**. It groups the problems into 8 modules (Element-wise → Losses & norms → Layers → Training → Attention → Architecture & efficiency → Decoding → Alignment / RL). Each problem has template, solution, and Colab links, plus the prerequisites it builds on.

## Notes on existing problems

Issues found while building this curriculum (a toolbar bug, statement inaccuracies, inconsistent difficulty labels, and more) are tracked in **[IMPROVEMENTS.md](IMPROVEMENTS.md)**.

## Notation

Shapes are written with short symbols. Most follow PyTorch's docs and the papers the problems come from, so **some letters mean different things in different notebooks**. Rather than rename them, every notebook that uses shape symbols has a **📐 Notation** box in its first cell listing the symbols *that file* uses and what they mean there. The boxes come from `scripts/notation.py` (problems) and the `notation` field of each `scripts/prereqs/content/*.py` file (prerequisites and bonuses). `make prereqs` writes them, and `make prereqs-check` fails if one is missing, stale, or lists a symbol the notebook never uses.

| Symbol | Usual meaning | Also means | Where the other meaning appears |
|--------|---------------|------------|---------------------------------|
| `B` | batch size | | |
| `N` | number of rows / examples | **batch size** in image notation `(N, C, H, W)` (PyTorch's convention); number of **patches** | #07, #42, P8; P5, #27 |
| `S` | sequence length | | `T` (B4) and `L` (P2) are also used for lengths |
| `S_q`, `S_k`, `S_kv` | number of queries / keys / key-value tokens | | |
| `S_past`, `S_new` | tokens already cached / new tokens (KV cache) | | |
| `D` | model / feature dimension (`d_model`) | | |
| `d`, `d_k`, `d_v` | per-head dimension, `D // H`; query-key / value dimension | | |
| `H` | number of attention **heads** | image **height** | P2, P3, P5, P6, #22, #27, #42 (P5 and P6 use both) |
| `W` | image **width** | a **weight matrix** `(out, in)` | P6, #03 |
| `C` | **channels** | number of **classes** | P4, #16 |
| `V` | the **value** tensor in `Q, K, V` | **vocabulary** size `(..., V)` | P4, P9, P10, B1, B4, #34, #41 |
| `K` | the **key** tensor | number of **draft tokens** | #34 |
| `T` | number of target positions (B4) | number of **teachers** | #41 |
| `k` | top-k / kernel or window size | | `kH`, `kW` are kernel height / width |
| `p` | patch size | probability (top-p), padding, dropout rate (`p_drop`) | P5, #27; #32; B6; P10 |
| `M`, `F`, `G`, `E`, `r` | second set size, frequencies, groups, ensemble size, LoRA rank | | P2, P2, P4, B5, P6 |

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
