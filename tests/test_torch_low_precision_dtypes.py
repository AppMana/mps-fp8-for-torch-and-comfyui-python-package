"""Coverage for every low-precision dtype exposed by this Torch build."""

import pytest
import torch

from conftest import requires_mps


FP8_DTYPE_NAMES = [
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
]


def _available_dtype_names(names):
    return [name for name in names if hasattr(torch, name)]


def _finite_equal(actual, expected):
    actual = actual.cpu()
    expected = expected.cpu()
    nan_mask = expected.isnan()
    if nan_mask.any():
        assert actual[nan_mask].isnan().all()
    valid = ~nan_mask
    assert torch.equal(actual[valid], expected[valid])


@requires_mps
@pytest.mark.parametrize("dtype_name", _available_dtype_names(FP8_DTYPE_NAMES))
def test_fp8_float_to_mps_bytes_match_cpu(dtype_name):
    """MPS float -> FP8 conversion should produce the same raw bytes as CPU Torch."""
    dtype = getattr(torch, dtype_name)
    values = torch.tensor(
        [-16.0, -2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 16.0],
        dtype=torch.float32,
    )

    expected = values.to(dtype).view(torch.uint8)
    actual = values.to("mps").to(dtype).view(torch.uint8).cpu()

    assert torch.equal(actual, expected), dtype_name


@requires_mps
@pytest.mark.parametrize("dtype_name", _available_dtype_names(FP8_DTYPE_NAMES))
def test_fp8_cpu_to_mps_preserves_bytes(dtype_name):
    """Moving already-encoded FP8 tensors CPU -> MPS should preserve storage exactly."""
    dtype = getattr(torch, dtype_name)
    raw = torch.arange(256, dtype=torch.uint8)

    actual = raw.view(dtype).to("mps").view(torch.uint8).cpu()

    assert torch.equal(actual, raw), dtype_name


@requires_mps
@pytest.mark.parametrize("dtype_name", _available_dtype_names(FP8_DTYPE_NAMES))
def test_fp8_mps_to_float_matches_cpu(dtype_name):
    """MPS FP8 -> float conversion should match CPU Torch for all bit patterns."""
    dtype = getattr(torch, dtype_name)
    raw = torch.arange(256, dtype=torch.uint8)
    expected = raw.view(dtype).to(torch.float32)

    actual = raw.view(dtype).to("mps").to(torch.float32)

    _finite_equal(actual, expected)


@requires_mps
@pytest.mark.parametrize("dtype_name", _available_dtype_names(FP8_DTYPE_NAMES))
def test_fp8_copy_float_to_mps_bytes_match_cpu(dtype_name):
    """copy_ into MPS FP8 tensors should encode like CPU Torch."""
    dtype = getattr(torch, dtype_name)
    values = torch.tensor([-16.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 16.0])
    expected = values.to(dtype).view(torch.uint8)
    actual = torch.empty(values.shape, dtype=dtype, device="mps")

    actual.copy_(values.to("mps"))

    assert torch.equal(actual.view(torch.uint8).cpu(), expected), dtype_name


@requires_mps
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="FP4 dtype not available")
def test_fp4_float_to_mps_roundtrip_shape_and_values():
    """The Torch FP4 dtype should still convert and dequantize on MPS."""
    dtype = torch.float4_e2m1fn_x2
    values = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0])

    encoded = values.to("mps").to(dtype)
    decoded = encoded.to(torch.float32)

    assert encoded.dtype == dtype
    assert encoded.device.type == "mps"
    assert torch.allclose(decoded.cpu(), values, atol=0.0, rtol=0.0)


@requires_mps
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="FP4 dtype not available")
def test_fp4_cpu_to_mps_preserves_bytes():
    """Moving packed FP4 tensors CPU -> MPS should preserve raw byte storage."""
    dtype = torch.float4_e2m1fn_x2
    raw = torch.arange(16, dtype=torch.uint8)

    actual = raw.view(dtype).to("mps").view(torch.uint8).cpu()

    assert torch.equal(actual, raw)
