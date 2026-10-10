# Vision Transformer patch embedding

This module **splits each image into non-overlapping patches and converts each patch into a learned vector**. This is commonly used to prepare images for a Vision Transformer.

The input `x` has shape `[B, C, H, W]`:

- `B`: batch size—number of images.
- `C`: number of channels, usually 3 for RGB.
- `H, W`: image height and width.

The constructor sets the patch size and creates a linear layer:

```python
self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
```

Each patch contains `C × patch_size × patch_size` pixel values. This layer transforms those values into a vector of length `embed_dim`. Its weights are learned during training, and the same layer processes every patch.

The `forward` method does this in three steps:

1. **Split the height and width into patch positions and positions within each patch.**

   ```python
   x = x.reshape(B, C, n_h, p, n_w, p)
   ```

   Here, `n_h = H // p` and `n_w = W // p` count the patch rows and columns. The two `p` dimensions describe the pixels inside each patch.

2. **Group each patch’s channels and pixels together, then flatten them.**

   ```python
   x = x.permute(0, 2, 4, 1, 3, 5)
   # Shape: [B, n_h, n_w, C, p, p]

   x = x.reshape(B, n_h * n_w, C * p * p)
   ```

   Now each image is a sequence of patches. Each patch is represented by one flat vector containing all its pixel values. Patches are ordered left to right, then top to bottom.

3. **Project each patch into an embedding.**

   ```python
   return self.proj(x)
   ```

   The output shape is **`[B, number_of_patches, embed_dim]`**.

For example, with RGB images of size `32 × 32`, patches of size `8 × 8`, and `embed_dim=64`:

```text
Input:          [B, 3, 32, 32]
Flattened:      [B, 16, 192]     # 4 × 4 patches; 3 × 8 × 8 values each
Output:         [B, 16, 64]      # 16 patch embeddings per image
```

Two details matter: `H` and `W` must both be divisible by `patch_size`, and `self.num_patches` is only stored metadata—the forward pass calculates the actual patch count from the input. This module does not add positional information or perform attention.
