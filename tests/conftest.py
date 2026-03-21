import pytest
import torch


requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)

requires_fp8 = pytest.mark.skipif(
    not hasattr(torch, 'float8_e4m3fn'),
    reason="FP8 dtypes not available"
)
