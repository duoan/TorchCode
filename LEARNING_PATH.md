# 🧭 Learning Path

A recommended order for working through TorchCode. Start with the **prerequisites** (PyTorch skills), then work through the modules from top to bottom. Within a module, each problem builds on the previous one.

Problem numbers are unchanged; this is only an ordering. For the reasoning behind it and how prerequisite notebooks work, see [CURRICULUM.md](CURRICULUM.md).

## Track 0: Prerequisites

| # | Prerequisite | Unlocks | Links |
|:---:|---|---|---|
| 🔵 P1 | **Tensor basics** | [#1](templates/01_relu.ipynb), [#7](templates/07_batchnorm.ipynb), [#12](templates/12_linear_attention.ipynb), [#13](templates/13_gpt2_block.ipynb), [#15](templates/15_mlp.ipynb), [#19](templates/19_gelu.ipynb), [#20](templates/20_weight_init.ipynb), [#21](templates/21_gradient_clipping.ipynb), [#24](templates/24_rope.ipynb), [#25](templates/25_flash_attention.ipynb), [#29](templates/29_adam.ipynb), [#36](templates/36_int8_quantization.ipynb) | [Template](templates/00_p01_tensor_basics.ipynb) · [Solution](solutions/00_p01_tensor_basics_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_p01_tensor_basics.ipynb) |
| 🔵 P2 | **Broadcasting** | [#4](templates/04_layernorm.ipynb), [#7](templates/07_batchnorm.ipynb), [#8](templates/08_rmsnorm.ipynb), [#9](templates/09_causal_attention.ipynb), [#11](templates/11_sliding_window.ipynb), [#13](templates/13_gpt2_block.ipynb), [#22](templates/22_conv2d.ipynb), [#24](templates/24_rope.ipynb), [#28](templates/28_moe.ipynb), [#40](templates/40_linear_regression.ipynb), [#42](templates/42_batchnorm2d.ipynb) | [Template](templates/00_p02_broadcasting.ipynb) · [Solution](solutions/00_p02_broadcasting_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_p02_broadcasting.ipynb) |
| 🔵 P3 | **Reductions along dims** | [#2](templates/02_softmax.ipynb), [#4](templates/04_layernorm.ipynb), [#7](templates/07_batchnorm.ipynb), [#8](templates/08_rmsnorm.ipynb), [#12](templates/12_linear_attention.ipynb), [#21](templates/21_gradient_clipping.ipynb), [#25](templates/25_flash_attention.ipynb), [#32](templates/32_topk_sampling.ipynb), [#36](templates/36_int8_quantization.ipynb), [#38](templates/38_grpo_loss.ipynb), [#41](templates/41_opd_loss.ipynb), [#42](templates/42_batchnorm2d.ipynb) | [Template](templates/00_p03_reductions.ipynb) · [Solution](solutions/00_p03_reductions_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_p03_reductions.ipynb) |
| 🔵 P4 | **Indexing & masking** | [#9](templates/09_causal_attention.ipynb), [#11](templates/11_sliding_window.ipynb), [#13](templates/13_gpt2_block.ipynb), [#14](templates/14_kv_cache.ipynb), [#16](templates/16_cross_entropy.ipynb), [#18](templates/18_embedding.ipynb), [#24](templates/24_rope.ipynb), [#28](templates/28_moe.ipynb), [#32](templates/32_topk_sampling.ipynb), [#38](templates/38_grpo_loss.ipynb) | [Template](templates/00_p04_indexing_masking.ipynb) · [Solution](solutions/00_p04_indexing_masking_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_p04_indexing_masking.ipynb) |
| 🔵 P5 | **Shape manipulation** | [#6](templates/06_multihead_attention.ipynb), [#10](templates/10_gqa.ipynb), [#13](templates/13_gpt2_block.ipynb), [#14](templates/14_kv_cache.ipynb), [#22](templates/22_conv2d.ipynb), [#23](templates/23_cross_attention.ipynb), [#24](templates/24_rope.ipynb), [#27](templates/27_vit_patch.ipynb), [#28](templates/28_moe.ipynb) | [Template](templates/00_p05_shape_manipulation.ipynb) · [Solution](solutions/00_p05_shape_manipulation_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_p05_shape_manipulation.ipynb) |
| 🔵 P6 | **Matmul & einsum** | [#3](templates/03_linear.ipynb), [#5](templates/05_attention.ipynb), [#6](templates/06_multihead_attention.ipynb), [#9](templates/09_causal_attention.ipynb), [#12](templates/12_linear_attention.ipynb), [#22](templates/22_conv2d.ipynb), [#23](templates/23_cross_attention.ipynb), [#26](templates/26_lora.ipynb), [#36](templates/36_int8_quantization.ipynb), [#40](templates/40_linear_regression.ipynb) | [Template](templates/00_p06_matmul_einsum.ipynb) · [Solution](solutions/00_p06_matmul_einsum_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_p06_matmul_einsum.ipynb) |
| 🔵 P7 | **Autograd essentials** | [#7](templates/07_batchnorm.ipynb), [#20](templates/20_weight_init.ipynb), [#21](templates/21_gradient_clipping.ipynb), [#29](templates/29_adam.ipynb), [#31](templates/31_gradient_accumulation.ipynb), [#37](templates/37_dpo_loss.ipynb), [#38](templates/38_grpo_loss.ipynb), [#39](templates/39_ppo_loss.ipynb), [#40](templates/40_linear_regression.ipynb), [#41](templates/41_opd_loss.ipynb), [#42](templates/42_batchnorm2d.ipynb) | [Template](templates/00_p07_autograd.ipynb) · [Solution](solutions/00_p07_autograd_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_p07_autograd.ipynb) |
| 🔵 P8 | **nn.Module anatomy** | [#3](templates/03_linear.ipynb), [#13](templates/13_gpt2_block.ipynb), [#14](templates/14_kv_cache.ipynb), [#15](templates/15_mlp.ipynb), [#17](templates/17_dropout.ipynb), [#18](templates/18_embedding.ipynb), [#23](templates/23_cross_attention.ipynb), [#26](templates/26_lora.ipynb), [#27](templates/27_vit_patch.ipynb), [#28](templates/28_moe.ipynb), [#36](templates/36_int8_quantization.ipynb) | [Template](templates/00_p08_nn_module.ipynb) · [Solution](solutions/00_p08_nn_module_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_p08_nn_module.ipynb) |
| 🔵 P9 | **Numerical stability & log-space** | [#2](templates/02_softmax.ipynb), [#16](templates/16_cross_entropy.ipynb), [#25](templates/25_flash_attention.ipynb), [#32](templates/32_topk_sampling.ipynb), [#33](templates/33_beam_search.ipynb), [#34](templates/34_speculative_decoding.ipynb), [#37](templates/37_dpo_loss.ipynb), [#38](templates/38_grpo_loss.ipynb), [#39](templates/39_ppo_loss.ipynb), [#41](templates/41_opd_loss.ipynb) | [Template](templates/00_p09_numerical_stability.ipynb) · [Solution](solutions/00_p09_numerical_stability_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_p09_numerical_stability.ipynb) |
| 🔵 P10 | **Training loop & randomness** | [#17](templates/17_dropout.ipynb), [#29](templates/29_adam.ipynb), [#30](templates/30_cosine_lr.ipynb), [#31](templates/31_gradient_accumulation.ipynb), [#32](templates/32_topk_sampling.ipynb), [#34](templates/34_speculative_decoding.ipynb), [#40](templates/40_linear_regression.ipynb) | [Template](templates/00_p10_training_loop.ipynb) · [Solution](solutions/00_p10_training_loop_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_p10_training_loop.ipynb) |

## Module 1. Element-wise

Single-tensor ops; get comfortable with autograd-friendly math.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 1 | **ReLU** | 🟢 Easy | 🔵 [P1](templates/00_p01_tensor_basics.ipynb) | [Template](templates/01_relu.ipynb) · [Solution](solutions/01_relu_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/01_relu.ipynb) |
| 2 | 19 | **GELU** | 🟢 Easy | 🔵 [P1](templates/00_p01_tensor_basics.ipynb) | [Template](templates/19_gelu.ipynb) · [Solution](solutions/19_gelu_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/19_gelu.ipynb) |
| 3 | 2 | **Softmax** | 🟢 Easy | 🔵 [P3](templates/00_p03_reductions.ipynb), 🔵 [P9](templates/00_p09_numerical_stability.ipynb) | [Template](templates/02_softmax.ipynb) · [Solution](solutions/02_softmax_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/02_softmax.ipynb) |

## Module 2. Losses & norms

Reductions, `keepdim`, and numerically stable log-space math.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 16 | **Cross-Entropy Loss** | 🟢 Easy | 🔵 [P4](templates/00_p04_indexing_masking.ipynb), 🔵 [P9](templates/00_p09_numerical_stability.ipynb) | [Template](templates/16_cross_entropy.ipynb) · [Solution](solutions/16_cross_entropy_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/16_cross_entropy.ipynb) |
| 2 | 4 | **LayerNorm** | 🟡 Medium | 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P3](templates/00_p03_reductions.ipynb) | [Template](templates/04_layernorm.ipynb) · [Solution](solutions/04_layernorm_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/04_layernorm.ipynb) |
| 3 | 8 | **RMSNorm** | 🟡 Medium | 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P3](templates/00_p03_reductions.ipynb) | [Template](templates/08_rmsnorm.ipynb) · [Solution](solutions/08_rmsnorm_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/08_rmsnorm.ipynb) |
| 4 | 7 | **BatchNorm** | 🟡 Medium | 🔵 [P1](templates/00_p01_tensor_basics.ipynb), 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P3](templates/00_p03_reductions.ipynb), 🔵 [P7](templates/00_p07_autograd.ipynb) | [Template](templates/07_batchnorm.ipynb) · [Solution](solutions/07_batchnorm_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/07_batchnorm.ipynb) |
| 5 | 42 | **BatchNorm2d** | 🟡 Medium | 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P3](templates/00_p03_reductions.ipynb), 🔵 [P7](templates/00_p07_autograd.ipynb) | [Template](templates/42_batchnorm2d.ipynb) · [Solution](solutions/42_batchnorm2d_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/42_batchnorm2d.ipynb) |

## Module 3. Layers

Parameters, initialization, and the building blocks of every network.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 3 | **Linear Layer** | 🟡 Medium | 🔵 [P6](templates/00_p06_matmul_einsum.ipynb), 🔵 [P8](templates/00_p08_nn_module.ipynb) | [Template](templates/03_linear.ipynb) · [Solution](solutions/03_linear_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/03_linear.ipynb) |
| 2 | 20 | **Kaiming Init** | 🟢 Easy | 🔵 [P1](templates/00_p01_tensor_basics.ipynb), 🔵 [P7](templates/00_p07_autograd.ipynb) | [Template](templates/20_weight_init.ipynb) · [Solution](solutions/20_weight_init_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/20_weight_init.ipynb) |
| 3 | 18 | **Embedding** | 🟢 Easy | 🔵 [P4](templates/00_p04_indexing_masking.ipynb), 🔵 [P8](templates/00_p08_nn_module.ipynb) | [Template](templates/18_embedding.ipynb) · [Solution](solutions/18_embedding_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/18_embedding.ipynb) |
| 4 | 17 | **Dropout** | 🟢 Easy | 🔵 [P8](templates/00_p08_nn_module.ipynb), 🔵 [P10](templates/00_p10_training_loop.ipynb) | [Template](templates/17_dropout.ipynb) · [Solution](solutions/17_dropout_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/17_dropout.ipynb) |
| 5 | 15 | **SwiGLU MLP** | 🟡 Medium | 🔵 [P1](templates/00_p01_tensor_basics.ipynb), 🔵 [P8](templates/00_p08_nn_module.ipynb) | [Template](templates/15_mlp.ipynb) · [Solution](solutions/15_mlp_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/15_mlp.ipynb) |
| 6 | 22 | **Conv2d** | 🟡 Medium | 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), 🔵 [P6](templates/00_p06_matmul_einsum.ipynb) | [Template](templates/22_conv2d.ipynb) · [Solution](solutions/22_conv2d_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/22_conv2d.ipynb) |
| 7 | 27 | **ViT Patch Embedding** | 🟡 Medium | 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), 🔵 [P8](templates/00_p08_nn_module.ipynb) | [Template](templates/27_vit_patch.ipynb) · [Solution](solutions/27_vit_patch_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/27_vit_patch.ipynb) |

## Module 4. Training

What happens between `loss.backward()` and the next step.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 40 | **Linear Regression** | 🟡 Medium | 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P6](templates/00_p06_matmul_einsum.ipynb), 🔵 [P7](templates/00_p07_autograd.ipynb), 🔵 [P10](templates/00_p10_training_loop.ipynb) | [Template](templates/40_linear_regression.ipynb) · [Solution](solutions/40_linear_regression_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/40_linear_regression.ipynb) |
| 2 | 21 | **Gradient Clipping** | 🟢 Easy | 🔵 [P1](templates/00_p01_tensor_basics.ipynb), 🔵 [P3](templates/00_p03_reductions.ipynb), 🔵 [P7](templates/00_p07_autograd.ipynb) | [Template](templates/21_gradient_clipping.ipynb) · [Solution](solutions/21_gradient_clipping_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/21_gradient_clipping.ipynb) |
| 3 | 31 | **Gradient Accumulation** | 🟢 Easy | 🔵 [P7](templates/00_p07_autograd.ipynb), 🔵 [P10](templates/00_p10_training_loop.ipynb) | [Template](templates/31_gradient_accumulation.ipynb) · [Solution](solutions/31_gradient_accumulation_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/31_gradient_accumulation.ipynb) |
| 4 | 29 | **Adam Optimizer** | 🟡 Medium | 🔵 [P1](templates/00_p01_tensor_basics.ipynb), 🔵 [P7](templates/00_p07_autograd.ipynb), 🔵 [P10](templates/00_p10_training_loop.ipynb) | [Template](templates/29_adam.ipynb) · [Solution](solutions/29_adam_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/29_adam.ipynb) |
| 5 | 30 | **Cosine LR Scheduler** | 🟢 Easy | 🔵 [P10](templates/00_p10_training_loop.ipynb) | [Template](templates/30_cosine_lr.ipynb) · [Solution](solutions/30_cosine_lr_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/30_cosine_lr.ipynb) |

## Module 5. Attention

From one head to production variants. Each problem builds on the previous one.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 5 | **Scaled Dot-Product Attention** | 🟡 Medium | 🔵 [P6](templates/00_p06_matmul_einsum.ipynb) | [Template](templates/05_attention.ipynb) · [Solution](solutions/05_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/05_attention.ipynb) |
| 2 | 9 | **Causal Self-Attention** | 🔴 Hard | 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P4](templates/00_p04_indexing_masking.ipynb), 🔵 [P6](templates/00_p06_matmul_einsum.ipynb) | [Template](templates/09_causal_attention.ipynb) · [Solution](solutions/09_causal_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/09_causal_attention.ipynb) |
| 3 | 11 | **Sliding Window Attention** | 🔴 Hard | 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P4](templates/00_p04_indexing_masking.ipynb) | [Template](templates/11_sliding_window.ipynb) · [Solution](solutions/11_sliding_window_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/11_sliding_window.ipynb) |
| 4 | 6 | **Multi-Head Attention** | 🔴 Hard | 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), 🔵 [P6](templates/00_p06_matmul_einsum.ipynb) | [Template](templates/06_multihead_attention.ipynb) · [Solution](solutions/06_multihead_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/06_multihead_attention.ipynb) |
| 5 | 23 | **Cross-Attention** | 🔴 Hard | 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), 🔵 [P6](templates/00_p06_matmul_einsum.ipynb), 🔵 [P8](templates/00_p08_nn_module.ipynb) | [Template](templates/23_cross_attention.ipynb) · [Solution](solutions/23_cross_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/23_cross_attention.ipynb) |
| 6 | 10 | **Grouped Query Attention** | 🔴 Hard | 🔵 [P5](templates/00_p05_shape_manipulation.ipynb) | [Template](templates/10_gqa.ipynb) · [Solution](solutions/10_gqa_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/10_gqa.ipynb) |
| 7 | 24 | **RoPE** | 🔴 Hard | 🔵 [P1](templates/00_p01_tensor_basics.ipynb), 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P4](templates/00_p04_indexing_masking.ipynb), 🔵 [P5](templates/00_p05_shape_manipulation.ipynb) | [Template](templates/24_rope.ipynb) · [Solution](solutions/24_rope_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/24_rope.ipynb) |
| 8 | 12 | **Linear Attention** | 🔴 Hard | 🔵 [P1](templates/00_p01_tensor_basics.ipynb), 🔵 [P3](templates/00_p03_reductions.ipynb), 🔵 [P6](templates/00_p06_matmul_einsum.ipynb) | [Template](templates/12_linear_attention.ipynb) · [Solution](solutions/12_linear_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/12_linear_attention.ipynb) |
| 9 | 25 | **Flash Attention** | 🔴 Hard | 🔵 [P1](templates/00_p01_tensor_basics.ipynb), 🔵 [P3](templates/00_p03_reductions.ipynb), 🔵 [P9](templates/00_p09_numerical_stability.ipynb) | [Template](templates/25_flash_attention.ipynb) · [Solution](solutions/25_flash_attention_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/25_flash_attention.ipynb) |

## Module 6. Architecture & efficiency

Combine layers and attention into real model components.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 13 | **GPT-2 Block** | 🔴 Hard | 🔵 [P1](templates/00_p01_tensor_basics.ipynb), 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P4](templates/00_p04_indexing_masking.ipynb), 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), 🔵 [P8](templates/00_p08_nn_module.ipynb) | [Template](templates/13_gpt2_block.ipynb) · [Solution](solutions/13_gpt2_block_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/13_gpt2_block.ipynb) |
| 2 | 14 | **KV Cache Attention** | 🔴 Hard | 🔵 [P4](templates/00_p04_indexing_masking.ipynb), 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), 🔵 [P8](templates/00_p08_nn_module.ipynb) | [Template](templates/14_kv_cache.ipynb) · [Solution](solutions/14_kv_cache_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/14_kv_cache.ipynb) |
| 3 | 26 | **LoRA** | 🟡 Medium | 🔵 [P6](templates/00_p06_matmul_einsum.ipynb), 🔵 [P8](templates/00_p08_nn_module.ipynb) | [Template](templates/26_lora.ipynb) · [Solution](solutions/26_lora_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/26_lora.ipynb) |
| 4 | 28 | **Mixture of Experts** | 🔴 Hard | 🔵 [P2](templates/00_p02_broadcasting.ipynb), 🔵 [P4](templates/00_p04_indexing_masking.ipynb), 🔵 [P5](templates/00_p05_shape_manipulation.ipynb), 🔵 [P8](templates/00_p08_nn_module.ipynb) | [Template](templates/28_moe.ipynb) · [Solution](solutions/28_moe_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/28_moe.ipynb) |
| 5 | 36 | **INT8 Quantization** | 🔴 Hard | 🔵 [P1](templates/00_p01_tensor_basics.ipynb), 🔵 [P3](templates/00_p03_reductions.ipynb), 🔵 [P6](templates/00_p06_matmul_einsum.ipynb), 🔵 [P8](templates/00_p08_nn_module.ipynb) | [Template](templates/36_int8_quantization.ipynb) · [Solution](solutions/36_int8_quantization_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/36_int8_quantization.ipynb) |

## Module 7. Decoding

Turning logits into text.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 32 | **Top-k / Top-p Sampling** | 🟡 Medium | 🔵 [P3](templates/00_p03_reductions.ipynb), 🔵 [P4](templates/00_p04_indexing_masking.ipynb), 🔵 [P9](templates/00_p09_numerical_stability.ipynb), 🔵 [P10](templates/00_p10_training_loop.ipynb) | [Template](templates/32_topk_sampling.ipynb) · [Solution](solutions/32_topk_sampling_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/32_topk_sampling.ipynb) |
| 2 | 33 | **Beam Search** | 🟡 Medium | 🔵 [P9](templates/00_p09_numerical_stability.ipynb) | [Template](templates/33_beam_search.ipynb) · [Solution](solutions/33_beam_search_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/33_beam_search.ipynb) |
| 3 | 34 | **Speculative Decoding** | 🔴 Hard | 🔵 [P9](templates/00_p09_numerical_stability.ipynb), 🔵 [P10](templates/00_p10_training_loop.ipynb) | [Template](templates/34_speculative_decoding.ipynb) · [Solution](solutions/34_speculative_decoding_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/34_speculative_decoding.ipynb) |
| 4 | 35 | **BPE Tokenizer** | 🔴 Hard | — | [Template](templates/35_bpe.ipynb) · [Solution](solutions/35_bpe_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/35_bpe.ipynb) |

## Module 8. Alignment / RL

Preference and policy-gradient losses used to post-train LLMs.

| Step | # | Problem | Difficulty | Builds on | Links |
|:---:|:---:|---|:---:|---|---|
| 1 | 37 | **DPO Loss** | 🔴 Hard | 🔵 [P7](templates/00_p07_autograd.ipynb), 🔵 [P9](templates/00_p09_numerical_stability.ipynb) | [Template](templates/37_dpo_loss.ipynb) · [Solution](solutions/37_dpo_loss_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/37_dpo_loss.ipynb) |
| 2 | 39 | **PPO Loss** | 🔴 Hard | 🔵 [P7](templates/00_p07_autograd.ipynb), 🔵 [P9](templates/00_p09_numerical_stability.ipynb) | [Template](templates/39_ppo_loss.ipynb) · [Solution](solutions/39_ppo_loss_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/39_ppo_loss.ipynb) |
| 3 | 38 | **GRPO Loss** | 🔴 Hard | 🔵 [P3](templates/00_p03_reductions.ipynb), 🔵 [P4](templates/00_p04_indexing_masking.ipynb), 🔵 [P7](templates/00_p07_autograd.ipynb), 🔵 [P9](templates/00_p09_numerical_stability.ipynb) | [Template](templates/38_grpo_loss.ipynb) · [Solution](solutions/38_grpo_loss_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/38_grpo_loss.ipynb) |
| 4 | 41 | **OPD Loss** | 🔴 Hard | 🔵 [P3](templates/00_p03_reductions.ipynb), 🔵 [P7](templates/00_p07_autograd.ipynb), 🔵 [P9](templates/00_p09_numerical_stability.ipynb) | [Template](templates/41_opd_loss.ipynb) · [Solution](solutions/41_opd_loss_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/41_opd_loss.ipynb) |

## Track B: Bonus (optional)

Techniques beyond the problems that come up in interviews and real codebases. No problem requires them; each one lists problems to re-solve with it.

| # | Bonus | Try it on | Links |
|:---:|---|---|---|
| ⭐ B1 | **Custom autograd functions** | [#1](templates/01_relu.ipynb), [#2](templates/02_softmax.ipynb), [#3](templates/03_linear.ipynb), [#16](templates/16_cross_entropy.ipynb), [#19](templates/19_gelu.ipynb), [#36](templates/36_int8_quantization.ipynb) | [Template](templates/00_b01_custom_autograd.ipynb) · [Solution](solutions/00_b01_custom_autograd_solution.ipynb) · [Colab](https://colab.research.google.com/github/lalo/TorchCode/blob/master/templates/00_b01_custom_autograd.ipynb) |
