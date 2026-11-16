"""INT8 Quantized Linear task."""

TASK = {
    "title": "INT8 Quantized Linear",
    "difficulty": "Hard",
    "function_name": "Int8Linear",
    "hint": "Per-channel scale = abs(weight).max(dim=1) / 127. Quantize: round(weight/scale).clamp(-128,127).to(int8). Forward: dequantize and matmul.",
    "tests": [
        {
            "name": "Weight is int8",
            "code": "\nimport torch, torch.nn as nn\nw = torch.randn(32, 16)\nq = {fn}(w)\nassert isinstance(q, nn.Module)\nassert q.weight_int8.dtype == torch.int8, f'dtype: {q.weight_int8.dtype}'\n"
        },
        {
            "name": "Values in [-128, 127]",
            "code": "\nimport torch\nq = {fn}(torch.randn(64, 32) * 10)\nassert q.weight_int8.min() >= -128 and q.weight_int8.max() <= 127\n"
        },
        {
            "name": "Dequantized close to original",
            "code": "\nimport torch\ntorch.manual_seed(0)\nw = torch.randn(16, 8)\nq = {fn}(w)\nw_recon = q.weight_int8.float() * q.scale\nassert (w - w_recon).abs().max() < 0.1, 'Quantization error too large'\n"
        },
        {
            "name": "Forward output shape",
            "code": "\nimport torch\nq = {fn}(torch.randn(8, 4), torch.randn(8))\nout = q(torch.randn(2, 4))\nassert out.shape == (2, 8), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Weight is buffer not parameter",
            "code": "\nimport torch\nq = {fn}(torch.randn(4, 4))\nparam_names = [n for n, _ in q.named_parameters()]\nbuffer_names = [n for n, _ in q.named_buffers()]\nassert 'weight_int8' in buffer_names and 'scale' in buffer_names, f'Register weight_int8 and scale with register_buffer; found buffers {buffer_names}'\nassert 'weight_int8' not in param_names, 'weight_int8 should be a buffer'\nassert 'scale' not in param_names, 'scale should be a buffer'\n"
        }
    ]
}
