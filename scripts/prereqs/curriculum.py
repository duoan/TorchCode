"""Curriculum structure: the Track 0 prerequisites and the module order of the 41 problems.

Used by build.py to generate LEARNING_PATH.md. For prerequisites that already have a
notebook (scripts/prereqs/content/), the unlock list comes from the notebook's own
"unlocks" entries, so ``planned_unlocks`` below only matters while a prerequisite is planned.
"""

# (id, title, planned_unlocks)
PREREQS = [
    ("P1", "Tensor basics", None),
    ("P2", "Broadcasting", None),
    ("P3", "Reductions along dims", None),
    ("P4", "Indexing & masking", None),
    ("P5", "Shape manipulation", None),
    ("P6", "Matmul & einsum", None),
    ("P7", "Autograd essentials", None),
    ("P8", "nn.Module anatomy", None),
    ("P9", "Numerical stability & log-space", [2, 16, 25, 32, 34, 37, 38, 39, 41]),
    ("P10", "Training loop & randomness", [17, 29, 30, 31, 32, 33, 34, 40]),
]

# (title, description, problem numbers in recommended order). Must cover every problem exactly once.
MODULES = [
    ("1. Element-wise", "Single-tensor ops; get comfortable with autograd-friendly math.", [1, 19, 2]),
    ("2. Losses & norms", "Reductions, `keepdim`, and numerically stable log-space math.", [16, 4, 8, 7]),
    ("3. Layers", "Parameters, initialization, and the building blocks of every network.", [3, 20, 18, 17, 15, 22, 27]),
    ("4. Training", "What happens between `loss.backward()` and the next step.", [40, 21, 31, 29, 30]),
    ("5. Attention", "From one head to production variants. Each problem builds on the previous one.", [5, 9, 11, 6, 23, 10, 24, 12, 25]),
    ("6. Architecture & efficiency", "Combine layers and attention into real model components.", [13, 14, 26, 28, 36]),
    ("7. Decoding", "Turning logits into text.", [32, 33, 34, 35]),
    ("8. Alignment / RL", "Preference and policy-gradient losses used to post-train LLMs.", [37, 39, 38, 41]),
]
