"""Typed FP8 matmul must decode each operand using its own format."""

import pytest
import torch

from conftest import requires_mps


@requires_mps
@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
def test_reported_example_all_matrix_ops(dtype):
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    a = x.to(dtype)
    expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]])
    one = torch.tensor(1.0, device="mps")
    bias = torch.full((2, 2), 2.0, device="mps")

    for result in (torch.mm(a, a), torch.matmul(a, a), a @ a,
                   torch._scaled_mm(a, a, one, one)):
        assert result.dtype == torch.float32
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.addmm(bias, a, a).cpu(), expected + 2, rtol=0, atol=0)
    torch.testing.assert_close(torch.addmm(bias, a, a, beta=2, alpha=3).cpu(),
                               expected * 3 + 4, rtol=0, atol=0)


@requires_mps
@pytest.mark.parametrize("shape", [(1, 3, 2), (2, 3, 4), (32, 64, 24)])
@pytest.mark.parametrize("dtypes", [
    (torch.float8_e5m2, torch.float8_e5m2),
    (torch.float8_e5m2, torch.float8_e4m3fn),
    (torch.float8_e4m3fn, torch.float8_e5m2),
])
def test_e5m2_matmul_matches_dequantized_mps(shape, dtypes):
    m, k, n = shape
    # Includes negative values, non-unit scales, and both vecmat and matmul kernels.
    a = (torch.arange(m * k, device="mps").reshape(m, k).float() % 9 - 4) / 2
    b = (torch.arange(k * n, device="mps").reshape(k, n).float() % 7 - 3) / 2
    a, b = a.to(dtypes[0]), b.to(dtypes[1])
    expected = a.float() @ b.float()
    result = torch.mm(a, b)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    scale_a = torch.tensor(0.5, device="mps")
    scale_b = torch.tensor(1.5, device="mps")
    scaled = torch._scaled_mm(a, b, scale_a, scale_b)
    torch.testing.assert_close(scaled, expected * 0.75, rtol=1e-4, atol=1e-4)


@requires_mps
def test_e5m2_special_range_preserves_float32_result():
    a = torch.tensor([[57344.0, -256.0, 2**-16], [0.0, 1.0, -2.0]], device="mps").to(torch.float8_e5m2)
    b = torch.tensor([[2.0, 0.0], [1.0, -3.0], [0.0, 4.0]], device="mps").to(torch.float8_e5m2)
    assert a.view(torch.uint8)[0, 0].item() == 0x7B
    torch.testing.assert_close(a @ b, a.float() @ b.float(), rtol=1e-5, atol=1e-5)


@requires_mps
def test_e5m2_max_finite_rounds_like_cpu():
    values = torch.tensor([57344.0, 60000.0, 61440.0, -57344.0])
    expected = values.to(torch.float8_e5m2).view(torch.uint8)
    actual = values.to("mps").to(torch.float8_e5m2).view(torch.uint8).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_mps
def test_e5m2_all_bit_patterns_decode_in_matmul():
    raw = torch.arange(256, dtype=torch.uint8)
    a = raw.to("mps").view(torch.float8_e5m2).reshape(256, 1)
    one = torch.tensor([[1.0]], device="mps").to(torch.float8_e5m2)
    actual = torch.mm(a, one).cpu().reshape(-1)
    expected = raw.view(torch.float8_e5m2).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@requires_mps
@pytest.mark.parametrize("dtype", [
    dtype for dtype in (
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
        getattr(torch, "float8_e8m0fnu", None),
    ) if dtype is not None
])
def test_other_fp8_formats_use_their_own_decoder(dtype):
    x = torch.tensor([[1.0, 2.0], [4.0, 8.0]], device="mps").to(dtype)
    torch.testing.assert_close(x @ x, x.float() @ x.float(), rtol=0, atol=0)


@requires_mps
def test_e4m3fn_quantize_max_stays_finite():
    from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_quantize

    torch.manual_seed(80)
    q, _ = fp8_quantize(torch.randn(64, 256))
    assert not ((q == 0x7F) | (q == 0xFF)).any().item()
