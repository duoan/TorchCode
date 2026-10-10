# Reference solution explanations

These 41 guides explain the implementations checked into `solutions/`. The practice dashboard serves these files in its **Explanation** tab, beside the reference code. They are generated ahead of time; opening a guide does not generate new text or record a practice attempt.

Each explanation was generated in a separate GPT-6.1 Sol session with **extra-high reasoning effort (`xhigh`)**, using only this prompt:

```text
<solution_code>
explain it
```

`<solution_code>` is the exact reference implementation shown in the **Solution Code** tab, including its imports. Notebook setup cells, demos, and judge calls are excluded. No old explanation or extra writing instructions are supplied to the model.

The generated prose is preserved. Presentation changes add the problem title, convert LaTeX math delimiters for the viewer, and retain the existing interactive examples. [generation.json](generation.json) records the model, effort, prompt template, and per-file source and output hashes.

Read a guide alongside its **Solution Code** tab. Code excerpts explain individual operations; example snippets may assume that the complete reference implementation and its imports have already been run. Small examples, formulas, shape tables, and tensor diagrams appear where the generated explanation uses them.

Tables and tensor diagrams make the intermediate steps visible. Selected guides also have interactive examples in the dashboard: adjust a dropout probability, choose embedding IDs, change an attention query or mask, compare normalization axes, rotate feature pairs, explore sampling cutoffs, or inspect PPO clipping. These small examples illustrate the reference operations; they do not modify your practice code or record submissions.

Interactive examples are embedded with a `torchcode-visual` fenced block naming the visual. Plain Markdown readers show that identifier; the dashboard renders the controls alongside the guide's static examples.

When a reference implementation changes, regenerate its matching explanation with the same prompt and check its examples too.

| # | Problem | Explanation | Reference notebook |
|---|---|---|---|
| 1 | Implement ReLU | [01_relu.md](01_relu.md) | [01_relu_solution.ipynb](../solutions/01_relu_solution.ipynb) |
| 2 | Implement Softmax | [02_softmax.md](02_softmax.md) | [02_softmax_solution.ipynb](../solutions/02_softmax_solution.ipynb) |
| 3 | Simple Linear Layer | [03_linear.md](03_linear.md) | [03_linear_solution.ipynb](../solutions/03_linear_solution.ipynb) |
| 4 | Implement LayerNorm | [04_layernorm.md](04_layernorm.md) | [04_layernorm_solution.ipynb](../solutions/04_layernorm_solution.ipynb) |
| 5 | Softmax Attention | [05_attention.md](05_attention.md) | [05_attention_solution.ipynb](../solutions/05_attention_solution.ipynb) |
| 6 | Multi-Head Attention | [06_multihead_attention.md](06_multihead_attention.md) | [06_multihead_attention_solution.ipynb](../solutions/06_multihead_attention_solution.ipynb) |
| 7 | Implement BatchNorm | [07_batchnorm.md](07_batchnorm.md) | [07_batchnorm_solution.ipynb](../solutions/07_batchnorm_solution.ipynb) |
| 8 | Implement RMSNorm | [08_rmsnorm.md](08_rmsnorm.md) | [08_rmsnorm_solution.ipynb](../solutions/08_rmsnorm_solution.ipynb) |
| 9 | Causal Self-Attention | [09_causal_attention.md](09_causal_attention.md) | [09_causal_attention_solution.ipynb](../solutions/09_causal_attention_solution.ipynb) |
| 10 | Grouped Query Attention | [10_gqa.md](10_gqa.md) | [10_gqa_solution.ipynb](../solutions/10_gqa_solution.ipynb) |
| 11 | Sliding Window Attention | [11_sliding_window.md](11_sliding_window.md) | [11_sliding_window_solution.ipynb](../solutions/11_sliding_window_solution.ipynb) |
| 12 | Linear Self-Attention | [12_linear_attention.md](12_linear_attention.md) | [12_linear_attention_solution.ipynb](../solutions/12_linear_attention_solution.ipynb) |
| 13 | GPT-2 Transformer Block | [13_gpt2_block.md](13_gpt2_block.md) | [13_gpt2_block_solution.ipynb](../solutions/13_gpt2_block_solution.ipynb) |
| 14 | KV Cache Attention | [14_kv_cache.md](14_kv_cache.md) | [14_kv_cache_solution.ipynb](../solutions/14_kv_cache_solution.ipynb) |
| 15 | SwiGLU MLP | [15_mlp.md](15_mlp.md) | [15_mlp_solution.ipynb](../solutions/15_mlp_solution.ipynb) |
| 16 | Cross-Entropy Loss | [16_cross_entropy.md](16_cross_entropy.md) | [16_cross_entropy_solution.ipynb](../solutions/16_cross_entropy_solution.ipynb) |
| 17 | Implement Dropout | [17_dropout.md](17_dropout.md) | [17_dropout_solution.ipynb](../solutions/17_dropout_solution.ipynb) |
| 18 | Embedding Layer | [18_embedding.md](18_embedding.md) | [18_embedding_solution.ipynb](../solutions/18_embedding_solution.ipynb) |
| 19 | GELU Activation | [19_gelu.md](19_gelu.md) | [19_gelu_solution.ipynb](../solutions/19_gelu_solution.ipynb) |
| 20 | Kaiming Initialization | [20_weight_init.md](20_weight_init.md) | [20_weight_init_solution.ipynb](../solutions/20_weight_init_solution.ipynb) |
| 21 | Gradient Norm Clipping | [21_gradient_clipping.md](21_gradient_clipping.md) | [21_gradient_clipping_solution.ipynb](../solutions/21_gradient_clipping_solution.ipynb) |
| 22 | 2D Convolution | [22_conv2d.md](22_conv2d.md) | [22_conv2d_solution.ipynb](../solutions/22_conv2d_solution.ipynb) |
| 23 | Multi-Head Cross-Attention | [23_cross_attention.md](23_cross_attention.md) | [23_cross_attention_solution.ipynb](../solutions/23_cross_attention_solution.ipynb) |
| 24 | Rotary Position Embedding (RoPE) | [24_rope.md](24_rope.md) | [24_rope_solution.ipynb](../solutions/24_rope_solution.ipynb) |
| 25 | Flash Attention (Tiled) | [25_flash_attention.md](25_flash_attention.md) | [25_flash_attention_solution.ipynb](../solutions/25_flash_attention_solution.ipynb) |
| 26 | LoRA (Low-Rank Adaptation) | [26_lora.md](26_lora.md) | [26_lora_solution.ipynb](../solutions/26_lora_solution.ipynb) |
| 27 | ViT Patch Embedding | [27_vit_patch.md](27_vit_patch.md) | [27_vit_patch_solution.ipynb](../solutions/27_vit_patch_solution.ipynb) |
| 28 | Mixture of Experts (MoE) | [28_moe.md](28_moe.md) | [28_moe_solution.ipynb](../solutions/28_moe_solution.ipynb) |
| 29 | Adam Optimizer | [29_adam.md](29_adam.md) | [29_adam_solution.ipynb](../solutions/29_adam_solution.ipynb) |
| 30 | Cosine LR Scheduler with Warmup | [30_cosine_lr.md](30_cosine_lr.md) | [30_cosine_lr_solution.ipynb](../solutions/30_cosine_lr_solution.ipynb) |
| 31 | Gradient Accumulation | [31_gradient_accumulation.md](31_gradient_accumulation.md) | [31_gradient_accumulation_solution.ipynb](../solutions/31_gradient_accumulation_solution.ipynb) |
| 32 | Top-k / Top-p Sampling | [32_topk_sampling.md](32_topk_sampling.md) | [32_topk_sampling_solution.ipynb](../solutions/32_topk_sampling_solution.ipynb) |
| 33 | Beam Search Decoding | [33_beam_search.md](33_beam_search.md) | [33_beam_search_solution.ipynb](../solutions/33_beam_search_solution.ipynb) |
| 34 | Speculative Decoding | [34_speculative_decoding.md](34_speculative_decoding.md) | [34_speculative_decoding_solution.ipynb](../solutions/34_speculative_decoding_solution.ipynb) |
| 35 | Byte-Pair Encoding (BPE) | [35_bpe.md](35_bpe.md) | [35_bpe_solution.ipynb](../solutions/35_bpe_solution.ipynb) |
| 36 | INT8 Quantized Linear | [36_int8_quantization.md](36_int8_quantization.md) | [36_int8_quantization_solution.ipynb](../solutions/36_int8_quantization_solution.ipynb) |
| 37 | DPO (Direct Preference Optimization) Loss | [37_dpo_loss.md](37_dpo_loss.md) | [37_dpo_loss_solution.ipynb](../solutions/37_dpo_loss_solution.ipynb) |
| 38 | GRPO (Group Relative Policy Optimization) Loss | [38_grpo_loss.md](38_grpo_loss.md) | [38_grpo_loss_solution.ipynb](../solutions/38_grpo_loss_solution.ipynb) |
| 39 | PPO (Proximal Policy Optimization) Clipped Loss | [39_ppo_loss.md](39_ppo_loss.md) | [39_ppo_loss_solution.ipynb](../solutions/39_ppo_loss_solution.ipynb) |
| 40 | Linear Regression | [40_linear_regression.md](40_linear_regression.md) | [40_linear_regression_solution.ipynb](../solutions/40_linear_regression_solution.ipynb) |
| 41 | OPD (On-Policy Distillation) Loss | [41_opd_loss.md](41_opd_loss.md) | [41_opd_loss_solution.ipynb](../solutions/41_opd_loss_solution.ipynb) |
