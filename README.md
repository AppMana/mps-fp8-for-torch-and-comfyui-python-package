Registers FP8 (float8_e4m3fn, float8_e5m2) and FP4 (float4_e2m1fn_x2) support for PyTorch's MPS backend on Apple Silicon. Once installed, `import torch` auto-loads the extension via the `torch.backends` entry point, enabling `tensor.to(torch.float8_e4m3fn)`, `torch._scaled_mm`, and `tensor.copy_` to work transparently on MPS through Metal shader kernels dispatched via `torch.mps.compile_shader`. The FP8 encode is tested byte-for-byte against all 254 representable values and their midpoints to match CPU PyTorch exactly; FP4 decode is verified exhaustively against all 256 packed byte patterns. 80 tests run on macOS MPS hardware in CI.

```
pip install fp4-fp8-for-torch-mps
```
