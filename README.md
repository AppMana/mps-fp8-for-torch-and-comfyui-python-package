FP8/FP4 support for PyTorch MPS on Apple Silicon.

Installing this lets ComfyUI use FP8 and FP4 weight checkpoints on MPS
automatically.

Fast paths use Metal kernels. Fallback paths round-trip through CPU Torch for
exact PyTorch-compatible bytes.

```
pip install fp4-fp8-for-torch-mps
```

## Decode backend

FP8 e4m3fn and FP4 decode can use `lut`, `lut4`, `arithmetic`,
`arithmetic4`, or `auto`. Default: `lut`.

```python
import fp4_fp8_for_torch_mps as mps_fp
mps_fp.set_decode_backend("lut")
```

Or:

```
FP4_FP8_MPS_DECODE_BACKEND=lut
```
