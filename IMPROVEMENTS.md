# 🛠️ Improvements Backlog

Issues and suggestions found while reviewing all 41 problems (templates, reference solutions, and judge tests) to design the [prerequisite curriculum](CURRICULUM.md).
Each item says what is wrong, why it matters, and a suggested fix. Items are grouped by how much they affect learners today.

**Status:** items marked ✅ are fixed on the `fix/problem-quick-fixes` branch (also merged into `feat/prereq-curriculum`), which can be sent upstream as its own PR. Unmarked items are still open.

## 🐛 Bugs

### ✅ JupyterLab Check/Hint buttons fail on #06 Multi-Head Attention and on solution notebooks
`labextension/src/index.ts` infers the task id from the file name (`06_multihead_attention.ipynb` → `multihead_attention`), but the judge task is registered as `mha`. Clicking **Check** or **Hint** in the toolbar runs `check("multihead_attention")` and prints *Unknown task*. The submit cell inside the notebook works because it hard-codes `check("mha")`. The web API already handles this case with `TASK_ID_ALIASES` in `api/parser.py`.
The same function also turned every solution notebook's name into a task id with a `_solution` suffix (e.g. `relu_solution`), so the buttons failed on all 41 solution notebooks too.
**Fixed:** the extension now strips `_solution` and uses the same alias map as `api/parser.py`. Verified that all 82 template and solution notebooks map to a real judge task, and type-checked against JupyterLab 4.

### ✅ Tests that pass against an unimplemented stub
The "Performance" test in **#01 ReLU** and the timing test in **#12 Linear Attention** only measure elapsed time. A blank template (`pass`, returning `None`) runs instantly and gets a ✅, so learners see a misleading partial pass before writing any code.
Running every blank template through the judge found three more: **#21 Gradient Clipping** ("No change when norm < max_norm" and "Preserves direction" both hold when the function does nothing) and **#36 INT8 Quantization** ("Weight is buffer not parameter" only checked that the weight was *not* a parameter).
**Fixed:** timing tests now require a tensor output; #21 also checks the returned total norm and that clipping actually happens; #36 requires `weight_int8` and `scale` to be registered buffers. All 41 blank templates now pass 0 tests, and every reference solution still passes all of its tests.

## 📐 Accuracy of problem statements

### ✅ #11 Sliding Window Attention: Longformer vs Mistral
The statement says the problem is "used in Longformer, Mistral, etc." and defines the window as two-sided: position *i* attends to every *j* with `abs(i - j) <= w`. That is the **Longformer-style (bidirectional)** window. **Mistral's** sliding window is **causal**: position *i* attends only to `[i - w, i]`, which is the mask `(i - j < 0) | (i - j > w)`. The solution and tests implement only the bidirectional version, so the Mistral reference is misleading.
**Fix (pick one):**
- Reword the statement to "bidirectional sliding window (Longformer-style)" and drop or qualify the Mistral reference.
- Add a `causal: bool` argument (or a follow-up problem) for the Mistral variant. It combines naturally with #09 Causal Attention and #14 KV Cache.

**Fixed (first option):** the statement now says "bidirectional (Longformer-style)" and adds a note that Mistral's window is causal, suggesting a causal variant as an extension. The README's key concepts no longer say "Mistral-style". A `causal` argument is still open as a possible follow-up problem.

**Prerequisite coverage (done in P2 and P4):** P2 Broadcasting now teaches both grids. `abs_distance_matrix(n)` (`abs(i - j)`) matches #11 as written, and the signed `relative_position(n)` (`R[i, j] = i - j`) keeps the direction, so every attention mask is one comparison on it:

| Mask | From `R` |
|------|----------|
| Causal (#09) | `R < 0` |
| Bidirectional sliding window (#11) | `R.abs() > w` |
| Causal sliding window (Mistral) | `(R < 0) \| (R > w)` |

P4 Indexing & masking's `attention_mask(n, kind, w)` builds all three from `R` (`"causal"`, `"sliding"`, `"causal_sliding"`) and `apply_mask` applies them with `masked_fill`. That is where the Longformer vs Mistral distinction is taught in full.

### ✅ #24 RoPE: two pairing conventions
The statement says RoPE is "used in LLaMA, GPT-NeoX" and rotates **adjacent pairs** `(x[2k], x[2k+1])`. This interleaved convention matches the original RoFormer paper and Meta's LLaMA reference code. GPT-NeoX and the Hugging Face LLaMA implementation instead pair `x[k]` with `x[k + D/2]` (`rotate_half`). Both encode relative position, but they are **not interchangeable** when loading pretrained weights. Interviewers often ask about this.
**Fixed:** the statement now has a convention note naming both styles. A `rotate_half` follow-up is still open.

### ✅ #07 BatchNorm: running variance differs from PyTorch
The solution and judge updated `running_var` with the **biased** batch variance, while the statement said "update running stats like PyTorch". `nn.BatchNorm1d` normalizes with the biased variance but updates `running_var` with the **unbiased** one. A learner who matched PyTorch failed the "running stats update" test.
Both conventions exist in practice: Flax and Keras use the biased batch variance for the running estimate too (checked in their source), so the old behavior was a convention, not nonsense, but the statement contradicted it.
**Fixed:** #07 now follows PyTorch. The statement explains both variances and why the running estimate uses Bessel's correction, the reference and test use `unbiased=True` for `running_var` (the old update fails with a targeted message), and a new test compares against `nn.BatchNorm1d` over several steps.
**Follow-up added:** a new problem, **#42 BatchNorm2d**, applies the same logic per channel over `(N, H, W)`: `(1, C, 1, 1)` broadcasting, and a Bessel correction over `N·H·W` samples. Its tests compare against `nn.BatchNorm2d`.

### ✅ #38 GRPO: simplified objective
The statement describes GRPO as baseline-subtracted REINFORCE with group-normalized advantages, and it is honest about that. The GRPO objective from the DeepSeekMath paper also includes the **PPO-style clipped importance ratio** and a **KL penalty** against a reference policy. Interview questions about GRPO usually expect those terms.
**Fix:** name this "GRPO advantage + policy-gradient core" in the statement, or add a follow-up that combines it with #39 PPO (clipped ratio) and a KL term. The statement's "RLAIF" framing could also mention verifiable rewards (RLVR), where GRPO is most often used.
**Fixed (wording):** a scope note now says what the full objective adds (clipped ratio, KL penalty), and the framing says RL fine-tuning / RLVR. A combined follow-up problem is still open.

### ✅ #03 Linear Layer: init is not Kaiming
The README's key concepts for #03 list "Kaiming init", but the template asks for `randn * (1/√in_features)`. That is LeCun-style variance scaling. Kaiming would be `√(2/in_features)`, which is #20.
**Fixed:** the README now says "`1/√fan_in` init".

### ✅ #20 Kaiming Init: `fan_in` only correct for 2-D weights
The statement fixes `fan_in = weight.shape[1]`. That is correct for `nn.Linear` weights but not for conv weights `(C_out, C_in, kH, kW)`, where `fan_in = C_in * kH * kW`.
**Fixed (statement):** a note now explains the conv case and `weight[0].numel()`. A 4-D test is still open.

## 🧩 Consistency

### #03 Linear and #06 MHA are not `nn.Module`s
Both reference solutions are plain Python classes (`self.weight` tensors with `requires_grad`, a `forward` method, no `super().__init__()`), while #13–#18, #23 and later problems subclass `nn.Module`. The README describes #03 as an `nn.Module`. Interviewers almost always expect `nn.Module`.
**Fix:** convert both to `nn.Module` subclasses using `nn.Parameter` and update the tests (e.g. check `list(model.parameters())`). **Now unblocked:** P8 nn.Module anatomy exists, so learners are taught registration before reaching #03. Converting changes what the judge accepts for two existing problems, so it needs a maintainer decision.

### Difficulty labels
- **#05 Scaled Dot-Product Attention** (4 lines) is **Hard**, while **#23 Cross-Attention**, a full multi-head module, is **Medium**.
- **#40 Linear Regression** is **Medium** but is the gentlest end-to-end problem (closed form, manual gradient descent, then `nn.Linear` with an optimizer). In [LEARNING_PATH.md](LEARNING_PATH.md) it opens Module 4 and works well as the bridge from Track 0 into the main set.

**Fix:** re-grade #05 to Medium and #23 to Hard (or at least Medium → Hard for #23), and consider #40 → Easy.

## 🗺️ Curriculum notes

- **#30 Cosine LR** and **#35 BPE** use almost no tensor operations (pure math and pure Python respectively), so they need no prerequisites. They can be done at any point.
- Problems are numbered in the order they were added, not by learning order. [LEARNING_PATH.md](LEARNING_PATH.md) gives a recommended order without renumbering, which would break existing links and progress files.

## 🔧 Prerequisite-track follow-ups

- **PyPI release:** Colab installs `torch-judge` from PyPI, so `check("p02_broadcasting")` and the other prerequisite tasks only work in Colab after the next release.
- ~~**Notebook generator:** commit the scratch generator.~~ Done: `scripts/prereqs/` with `make prereqs` / `make prereqs-verify`.
- **Web UI shows no lessons:** the standalone web UI (`api/parser.py`) shows only a notebook's first Markdown cell and its ✏️ stubs. For prerequisites, the lesson and demo cells, the debug cells, and the "What this unlocks" section are missing, and step 1 of the "How to use this notebook" box doesn't apply. Fix: either render the lesson cells as part of the description, or link to the notebook from the web UI.
- **Web UI:** the new "Prereq" difficulty group and badge color in `web/src/app/page.tsx` have not been checked in a browser yet.
