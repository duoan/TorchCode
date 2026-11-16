# 🧭 Learning Path

A recommended order for working through TorchCode. Start with the **prerequisites** (PyTorch skills), then work through the modules from top to bottom. Within a module, each problem builds on the previous one.

Problem numbers are unchanged; this is only an ordering. For the reasoning behind it and how prerequisite notebooks work, see [CURRICULUM.md](CURRICULUM.md).

> 🔵 = prerequisite available · ⚪ = prerequisite planned

## Track 0: Prerequisites

| # | Prerequisite | Unlocks | Links |
|:---:|---|---|---|
| ⚪ P1 | Tensor basics *(planned)* | [#1](templates/01_relu.ipynb), [#19](templates/19_gelu.ipynb), [#20](templates/20_weight_init.ipynb), [#36](templates/36_int8_quantization.ipynb) | — |
| 🔵 P2 | **Broadcasting** | [#4](templates/04_layernorm.ipynb), [#7](templates/07_batchnorm.ipynb), [#8](templates/08_rmsnorm.ipynb), [#9](templates/09_causal_attention.ipynb), [#11](templates/11_sliding_window.ipynb), [#22](templates/22_conv2d.ipynb), [#24](templates/24_rope.ipynb), [#28](templates/28_moe.ipynb), [#40](templates/40_linear_regression.ipynb) | [Template](templates/00_p02_broadcasting.ipynb) · [Solution](solutions/00_p02_broadcasting_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/00_p02_broadcasting.ipynb) |
| ⚪ P3 | Reductions along dims *(planned)* | [#2](templates/02_softmax.ipynb), [#4](templates/04_layernorm.ipynb), [#7](templates/07_batchnorm.ipynb), [#8](templates/08_rmsnorm.ipynb), [#16](templates/16_cross_entropy.ipynb), [#21](templates/21_gradient_clipping.ipynb), [#36](templates/36_int8_quantization.ipynb), [#38](templates/38_grpo_loss.ipynb) | — |
| ⚪ P4 | Indexing & masking *(planned)* | [#9](templates/09_causal_attention.ipynb), [#11](templates/11_sliding_window.ipynb), [#14](templates/14_kv_cache.ipynb), [#16](templates/16_cross_entropy.ipynb), [#18](templates/18_embedding.ipynb), [#24](templates/24_rope.ipynb), [#28](templates/28_moe.ipynb), [#32](templates/32_topk_sampling.ipynb), [#38](templates/38_grpo_loss.ipynb) | — |
| 🔵 P5 | **Shape manipulation** | [#6](templates/06_multihead_attention.ipynb), [#10](templates/10_gqa.ipynb), [#13](templates/13_gpt2_block.ipynb), [#14](templates/14_kv_cache.ipynb), [#23](templates/23_cross_attention.ipynb), [#24](templates/24_rope.ipynb), [#27](templates/27_vit_patch.ipynb), [#28](templates/28_moe.ipynb) | [Template](templates/00_p05_shape_manipulation.ipynb) · [Solution](solutions/00_p05_shape_manipulation_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/00_p05_shape_manipulation.ipynb) |
| ⚪ P6 | Matmul & einsum *(planned)* | [#3](templates/03_linear.ipynb), [#5](templates/05_attention.ipynb), [#6](templates/06_multihead_attention.ipynb), [#12](templates/12_linear_attention.ipynb), [#22](templates/22_conv2d.ipynb), [#26](templates/26_lora.ipynb), [#40](templates/40_linear_regression.ipynb) | — |
| ⚪ P7 | Autograd essentials *(planned)* | [#7](templates/07_batchnorm.ipynb), [#20](templates/20_weight_init.ipynb), [#21](templates/21_gradient_clipping.ipynb), [#29](templates/29_adam.ipynb), [#31](templates/31_gradient_accumulation.ipynb), [#37](templates/37_dpo_loss.ipynb), [#38](templates/38_grpo_loss.ipynb), [#39](templates/39_ppo_loss.ipynb), [#40](templates/40_linear_regression.ipynb), [#41](templates/41_opd_loss.ipynb) | — |
| ⚪ P8 | nn.Module anatomy *(planned)* | [#3](templates/03_linear.ipynb), [#13](templates/13_gpt2_block.ipynb), [#14](templates/14_kv_cache.ipynb), [#15](templates/15_mlp.ipynb), [#17](templates/17_dropout.ipynb), [#18](templates/18_embedding.ipynb), [#23](templates/23_cross_attention.ipynb), [#26](templates/26_lora.ipynb), [#27](templates/27_vit_patch.ipynb), [#28](templates/28_moe.ipynb), [#36](templates/36_int8_quantization.ipynb) | — |
| ⚪ P9 | Numerical stability & log-space *(planned)* | [#2](templates/02_softmax.ipynb), [#16](templates/16_cross_entropy.ipynb), [#25](templates/25_flash_attention.ipynb), [#32](templates/32_topk_sampling.ipynb), [#34](templates/34_speculative_decoding.ipynb), [#37](templates/37_dpo_loss.ipynb), [#38](templates/38_grpo_loss.ipynb), [#39](templates/39_ppo_loss.ipynb), [#41](templates/41_opd_loss.ipynb) | — |
| ⚪ P10 | Training loop & randomness *(planned)* | [#17](templates/17_dropout.ipynb), [#29](templates/29_adam.ipynb), [#30](templates/30_cosine_lr.ipynb), [#31](templates/31_gradient_accumulation.ipynb), [#32](templates/32_topk_sampling.ipynb), [#33](templates/33_beam_search.ipynb), [#34](templates/34_speculative_decoding.ipynb), [#40](templates/40_linear_regression.ipynb) | — |

## Module 1. Element-wise

Single-tensor ops; get comfortable with autograd-friendly math.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 1 | **ReLU** | 🟢 Easy | ⚪ P1 | [Template](templates/01_relu.ipynb) · [Solution](solutions/01_relu_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/01_relu.ipynb) |
| 2 | 19 | **GELU** | 🟢 Easy | ⚪ P1 | [Template](templates/19_gelu.ipynb) · [Solution](solutions/19_gelu_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/19_gelu.ipynb) |
| 3 | 2 | **Softmax** | 🟢 Easy | ⚪ P3, ⚪ P9 | [Template](templates/02_softmax.ipynb) · [Solution](solutions/02_softmax_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/02_softmax.ipynb) |

## Module 2. Losses & norms

Reductions, `keepdim`, and numerically stable log-space math.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 16 | **Cross-Entropy Loss** | 🟢 Easy | ⚪ P3, ⚪ P4, ⚪ P9 | [Template](templates/16_cross_entropy.ipynb) · [Solution](solutions/16_cross_entropy_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/16_cross_entropy.ipynb) |
| 2 | 4 | **LayerNorm** | 🟡 Medium | 🔵 [P2](templates/00_p02_broadcasting.ipynb), ⚪ P3 | [Template](templates/04_layernorm.ipynb) · [Solution](solutions/04_layernorm_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/04_layernorm.ipynb) |
| 3 | 8 | **RMSNorm** | 🟡 Medium | 🔵 [P2](templates/00_p02_broadcasting.ipynb), ⚪ P3 | [Template](templates/08_rmsnorm.ipynb) · [Solution](solutions/08_rmsnorm_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/08_rmsnorm.ipynb) |
| 4 | 7 | **BatchNorm** | 🟡 Medium | 🔵 [P2](templates/00_p02_broadcasting.ipynb), ⚪ P3, ⚪ P7 | [Template](templates/07_batchnorm.ipynb) · [Solution](solutions/07_batchnorm_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/07_batchnorm.ipynb) |

## Module 3. Layers

Parameters, initialization, and the building blocks of every network.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 3 | **Linear Layer** | 🟡 Medium | ⚪ P6, ⚪ P8 | [Template](templates/03_linear.ipynb) · [Solution](solutions/03_linear_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/03_linear.ipynb) |
| 2 | 20 | **Kaiming Init** | 🟢 Easy | ⚪ P1, ⚪ P7 | [Template](templates/20_weight_init.ipynb) · [Solution](solutions/20_weight_init_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/20_weight_init.ipynb) |
| 3 | 18 | **Embedding** | 🟢 Easy | ⚪ P4, ⚪ P8 | [Template](templates/18_embedding.ipynb) · [Solution](solutions/18_embedding_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/18_embedding.ipynb) |
| 4 | 17 | **Dropout** | 🟢 Easy | ⚪ P8, ⚪ P10 | [Template](templates/17_dropout.ipynb) · [Solution](solutions/17_dropout_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/17_dropout.ipynb) |
| 5 | 15 | **SwiGLU MLP** | 🟡 Medium | ⚪ P8 | [Template](templates/15_mlp.ipynb) · [Solution](solutions/15_mlp_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/15_mlp.ipynb) |
| 6 | 22 | **Conv2d** | 🟡 Medium | 🔵 [P2](templates/00_p02_broadcasting.ipynb), ⚪ P6 | [Template](templates/22_conv2d.ipynb) · [Solution](solutions/22_conv2d_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/22_conv2d.ipynb) |
| 7 | 27 | **ViT Patch Embedding** | 🟡 Medium | 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), ⚪ P8 | [Template](templates/27_vit_patch.ipynb) · [Solution](solutions/27_vit_patch_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/27_vit_patch.ipynb) |

## Module 4. Training

What happens between `loss.backward()` and the next step.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 40 | **Linear Regression** | 🟡 Medium | 🔵 [P2](templates/00_p02_broadcasting.ipynb), ⚪ P6, ⚪ P7, ⚪ P10 | [Template](templates/40_linear_regression.ipynb) · [Solution](solutions/40_linear_regression_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/40_linear_regression.ipynb) |
| 2 | 21 | **Gradient Clipping** | 🟢 Easy | ⚪ P3, ⚪ P7 | [Template](templates/21_gradient_clipping.ipynb) · [Solution](solutions/21_gradient_clipping_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/21_gradient_clipping.ipynb) |
| 3 | 31 | **Gradient Accumulation** | 🟢 Easy | ⚪ P7, ⚪ P10 | [Template](templates/31_gradient_accumulation.ipynb) · [Solution](solutions/31_gradient_accumulation_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/31_gradient_accumulation.ipynb) |
| 4 | 29 | **Adam Optimizer** | 🟡 Medium | ⚪ P7, ⚪ P10 | [Template](templates/29_adam.ipynb) · [Solution](solutions/29_adam_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/29_adam.ipynb) |
| 5 | 30 | **Cosine LR Scheduler** | 🟡 Medium | ⚪ P10 | [Template](templates/30_cosine_lr.ipynb) · [Solution](solutions/30_cosine_lr_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/30_cosine_lr.ipynb) |

## Module 5. Attention

From one head to production variants. Each problem builds on the previous one.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 5 | **Scaled Dot-Product Attention** | 🔴 Hard | ⚪ P6 | [Template](templates/05_attention.ipynb) · [Solution](solutions/05_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/05_attention.ipynb) |
| 2 | 9 | **Causal Self-Attention** | 🔴 Hard | 🔵 [P2](templates/00_p02_broadcasting.ipynb), ⚪ P4 | [Template](templates/09_causal_attention.ipynb) · [Solution](solutions/09_causal_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/09_causal_attention.ipynb) |
| 3 | 11 | **Sliding Window Attention** | 🔴 Hard | 🔵 [P2](templates/00_p02_broadcasting.ipynb), ⚪ P4 | [Template](templates/11_sliding_window.ipynb) · [Solution](solutions/11_sliding_window_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/11_sliding_window.ipynb) |
| 4 | 6 | **Multi-Head Attention** | 🔴 Hard | 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), ⚪ P6 | [Template](templates/06_multihead_attention.ipynb) · [Solution](solutions/06_multihead_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/06_multihead_attention.ipynb) |
| 5 | 23 | **Cross-Attention** | 🟡 Medium | 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), ⚪ P8 | [Template](templates/23_cross_attention.ipynb) · [Solution](solutions/23_cross_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/23_cross_attention.ipynb) |
| 6 | 10 | **Grouped Query Attention** | 🔴 Hard | 🔵 [P5](templates/00_p05_shape_manipulation.ipynb) | [Template](templates/10_gqa.ipynb) · [Solution](solutions/10_gqa_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/10_gqa.ipynb) |
| 7 | 24 | **RoPE** | 🔴 Hard | 🔵 [P2](templates/00_p02_broadcasting.ipynb), ⚪ P4, 🔵 [P5](templates/00_p05_shape_manipulation.ipynb) | [Template](templates/24_rope.ipynb) · [Solution](solutions/24_rope_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/24_rope.ipynb) |
| 8 | 12 | **Linear Attention** | 🔴 Hard | ⚪ P6 | [Template](templates/12_linear_attention.ipynb) · [Solution](solutions/12_linear_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/12_linear_attention.ipynb) |
| 9 | 25 | **Flash Attention** | 🔴 Hard | ⚪ P9 | [Template](templates/25_flash_attention.ipynb) · [Solution](solutions/25_flash_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/25_flash_attention.ipynb) |

## Module 6. Architecture & efficiency

Combine layers and attention into real model components.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 13 | **GPT-2 Block** | 🔴 Hard | 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), ⚪ P8 | [Template](templates/13_gpt2_block.ipynb) · [Solution](solutions/13_gpt2_block_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/13_gpt2_block.ipynb) |
| 2 | 14 | **KV Cache Attention** | 🔴 Hard | ⚪ P4, 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), ⚪ P8 | [Template](templates/14_kv_cache.ipynb) · [Solution](solutions/14_kv_cache_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/14_kv_cache.ipynb) |
| 3 | 26 | **LoRA** | 🟡 Medium | ⚪ P6, ⚪ P8 | [Template](templates/26_lora.ipynb) · [Solution](solutions/26_lora_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/26_lora.ipynb) |
| 4 | 28 | **Mixture of Experts** | 🔴 Hard | 🔵 [P2](templates/00_p02_broadcasting.ipynb), ⚪ P4, 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), ⚪ P8 | [Template](templates/28_moe.ipynb) · [Solution](solutions/28_moe_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/28_moe.ipynb) |
| 5 | 36 | **INT8 Quantization** | 🔴 Hard | ⚪ P1, ⚪ P3, ⚪ P8 | [Template](templates/36_int8_quantization.ipynb) · [Solution](solutions/36_int8_quantization_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/36_int8_quantization.ipynb) |

## Module 7. Decoding

Turning logits into text.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 32 | **Top-k / Top-p Sampling** | 🟡 Medium | ⚪ P4, ⚪ P9, ⚪ P10 | [Template](templates/32_topk_sampling.ipynb) · [Solution](solutions/32_topk_sampling_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/32_topk_sampling.ipynb) |
| 2 | 33 | **Beam Search** | 🟡 Medium | ⚪ P10 | [Template](templates/33_beam_search.ipynb) · [Solution](solutions/33_beam_search_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/33_beam_search.ipynb) |
| 3 | 34 | **Speculative Decoding** | 🔴 Hard | ⚪ P9, ⚪ P10 | [Template](templates/34_speculative_decoding.ipynb) · [Solution](solutions/34_speculative_decoding_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/34_speculative_decoding.ipynb) |
| 4 | 35 | **BPE Tokenizer** | 🔴 Hard | — | [Template](templates/35_bpe.ipynb) · [Solution](solutions/35_bpe_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/35_bpe.ipynb) |

## Module 8. Alignment / RL

Preference and policy-gradient losses used to post-train LLMs.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 37 | **DPO Loss** | 🔴 Hard | ⚪ P7, ⚪ P9 | [Template](templates/37_dpo_loss.ipynb) · [Solution](solutions/37_dpo_loss_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/37_dpo_loss.ipynb) |
| 2 | 39 | **PPO Loss** | 🔴 Hard | ⚪ P7, ⚪ P9 | [Template](templates/39_ppo_loss.ipynb) · [Solution](solutions/39_ppo_loss_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/39_ppo_loss.ipynb) |
| 3 | 38 | **GRPO Loss** | 🔴 Hard | ⚪ P3, ⚪ P4, ⚪ P7, ⚪ P9 | [Template](templates/38_grpo_loss.ipynb) · [Solution](solutions/38_grpo_loss_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/38_grpo_loss.ipynb) |
| 4 | 41 | **OPD Loss** | 🔴 Hard | ⚪ P7, ⚪ P9 | [Template](templates/41_opd_loss.ipynb) · [Solution](solutions/41_opd_loss_solution.ipynb) · [Colab](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/41_opd_loss.ipynb) |
