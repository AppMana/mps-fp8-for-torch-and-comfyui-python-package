"""Coverage for every low-precision dtype exposed by this Torch build."""

import pytest
import torch

from conftest import requires_mps


FP8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)
FP8_DTYPES = [
    getattr(torch, "float8_e4m3fn", None),
    getattr(torch, "float8_e5m2", None),
    getattr(torch, "float8_e4m3fnuz", None),
    getattr(torch, "float8_e5m2fnuz", None),
    FP8_E8M0FNU,
]


def _available_dtypes(dtypes):
    return [dtype for dtype in dtypes if dtype is not None]


def _finite_equal(actual, expected):
    actual = actual.cpu()
    expected = expected.cpu()
    nan_mask = expected.isnan()
    if nan_mask.any():
        assert actual[nan_mask].isnan().all()
    valid = ~nan_mask
    assert torch.equal(actual[valid], expected[valid])


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
def test_fp8_float_to_mps_bytes_match_cpu(dtype):
    """MPS float -> FP8 conversion should produce the same raw bytes as CPU Torch."""
    values = torch.tensor(
        [-16.0, -2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 16.0],
        dtype=torch.float32,
    )

    expected = values.to(dtype).view(torch.uint8)
    actual = values.to("mps").to(dtype).view(torch.uint8).cpu()

    assert torch.equal(actual, expected), dtype


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
def test_fp8_cpu_to_mps_preserves_bytes(dtype):
    """Moving already-encoded FP8 tensors CPU -> MPS should preserve storage exactly."""
    raw = torch.arange(256, dtype=torch.uint8)

    actual = raw.view(dtype).to("mps").view(torch.uint8).cpu()

    assert torch.equal(actual, raw), dtype


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
def test_fp8_mps_to_float_matches_cpu(dtype):
    """MPS FP8 -> float conversion should match CPU Torch for all bit patterns."""
    raw = torch.arange(256, dtype=torch.uint8)
    expected = raw.view(dtype).to(torch.float32)

    actual = raw.view(dtype).to("mps").to(torch.float32)

    if dtype == FP8_E8M0FNU:
        expected = expected.clone()
        expected[0] = 0.0
    _finite_equal(actual, expected)


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
def test_fp8_copy_float_to_mps_bytes_match_cpu(dtype):
    """copy_ into MPS FP8 tensors should encode like CPU Torch."""
    values = torch.tensor([-16.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 16.0])
    expected = values.to(dtype).view(torch.uint8)
    actual = torch.empty(values.shape, dtype=dtype, device="mps")

    actual.copy_(values.to("mps"))

    assert torch.equal(actual.view(torch.uint8).cpu(), expected), dtype


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
def test_fp8_arithmetic_encode_matches_cpu(dtype):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    values = torch.tensor(
        [-float("inf"), -100000.0, -16.0, -2.0, -1.0, -0.5, -0.0,
         0.0, 0.5, 1.0, 2.0, 16.0, 100000.0, float("inf"), float("nan")],
        dtype=torch.float32,
    )
    expected = values.to(dtype).view(torch.uint8)
    actual = fp8_mps_native.fp8_encode_arithmetic_dtype(values.to("mps"), dtype).cpu()

    assert torch.equal(actual, expected), dtype


@requires_mps
def test_e4m3fn_out_of_range_encode_matches_cpu():
    from fp4_fp8_for_torch_mps import fp8_mps_native

    values = torch.tensor(
        [-float("inf"), -100000.0, -500.0, -449.0, -448.0,
         448.0, 449.0, 500.0, 100000.0, float("inf"), float("nan")],
        dtype=torch.float32,
    )
    expected = values.to(torch.float8_e4m3fn).view(torch.uint8)
    on_mps = values.to("mps")
    encoded = fp8_mps_native.fp8_encode_arithmetic_dtype(on_mps, torch.float8_e4m3fn)
    converted = on_mps.to(torch.float8_e4m3fn).view(torch.uint8)

    assert torch.equal(encoded.cpu(), expected)
    assert torch.equal(converted.cpu(), expected)


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
def test_fp8_arithmetic_decode_matches_cpu(dtype):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.arange(256, dtype=torch.uint8)
    expected = raw.view(dtype).to(torch.float32)
    actual = fp8_mps_native.fp8_dequantize_arithmetic_dtype(
        raw.to("mps"), torch.tensor([1.0], device="mps"), dtype, out_dtype=torch.float32,
    )

    if dtype == FP8_E8M0FNU:
        expected = expected.clone()
        expected[0] = 0.0
    _finite_equal(actual, expected)


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
def test_fp8_embedding_matches_cpu(dtype):
    """embedding with an MPS FP8 weight should gather the same encoded rows as CPU."""
    weight = torch.randn(32, 16).to(dtype)
    indices = torch.tensor([[1, 5, 7], [2, 4, 9]], dtype=torch.long)

    expected = torch.nn.functional.embedding(indices, weight).view(torch.uint8)
    actual = torch.nn.functional.embedding(indices.to("mps"), weight.to("mps")).view(torch.uint8).cpu()

    assert torch.equal(actual, expected), dtype


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
