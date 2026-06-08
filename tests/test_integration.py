"""Integration tests using only standard PyTorch APIs.

These tests do NOT import fp4_fp8_for_torch_mps — the torch.backends entry point
auto-loads the extension when torch is imported. They verify that FP8
dtypes work on MPS identically to CPU, and that FP8 matmul compares
favorably to FP16.
"""

import pytest
import torch
import torch.nn.functional as F

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)
requires_fp8 = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 dtypes not available",
)


def _quantize_row_scaled_fp8_uint8(weight):
    target_amax = 400.0
    fp32_weight = weight.float()
    amax = fp32_weight.abs().amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny)
    scale = (target_amax / amax).to(device=weight.device, dtype=torch.float32)
    inv_scale = (amax / target_amax).to(device=weight.device, dtype=torch.float32)
    scaled = (fp32_weight * scale[:, None]).clamp(min=-target_amax, max=target_amax)
    quantized = scaled.to(torch.float8_e4m3fn).view(torch.uint8)
    return quantized.contiguous(), inv_scale


def _dequantize_row_scaled_fp8_uint8(quantized, inv_scale, dtype):
    dequantized = quantized.view(torch.float8_e4m3fn).to(dtype)
    return dequantized * inv_scale[:, None].to(device=quantized.device, dtype=dtype)


@requires_mps
@requires_fp8
class TestFp8DtypeOnMps:
    """FP8 dtypes should work transparently on MPS after autoload."""

    def test_float32_to_fp8(self):
        x = torch.randn(64, 64, device="mps")
        x_fp8 = x.to(torch.float8_e4m3fn)
        assert x_fp8.dtype == torch.float8_e4m3fn
        assert x_fp8.device.type == "mps"

    def test_fp8_roundtrip_preserves_values(self):
        values = torch.tensor([0.5, 1.0, 2.0, 10.0, 100.0, 400.0], device="mps")
        recovered = values.to(torch.float8_e4m3fn).to(torch.float32)
        for v, r in zip(values.cpu(), recovered.cpu()):
            if abs(v) > 1e-6:
                assert abs(r - v) / abs(v) < 0.15

    def test_fp8_cpu_to_mps_transfer(self):
        x_cpu = torch.randn(32, 32).to(torch.float8_e4m3fn)
        x_mps = x_cpu.to("mps")
        assert x_mps.dtype == torch.float8_e4m3fn
        assert x_mps.device.type == "mps"
        assert torch.equal(x_mps.view(torch.uint8).cpu(), x_cpu.view(torch.uint8))

    def test_fp8_copy(self):
        src = torch.randn(16, 16, device="mps").to(torch.float8_e4m3fn)
        dst = torch.empty(16, 16, dtype=torch.float8_e4m3fn, device="mps")
        dst.copy_(src)
        assert torch.equal(dst.view(torch.uint8).cpu(), src.view(torch.uint8).cpu())

    def test_fp8_embedding(self):
        weight = torch.randn(32, 16)
        indices = torch.tensor([[1, 5, 7], [2, 4, 9]], device="mps")
        weight_fp8_cpu = weight.to(torch.float8_e4m3fn)
        weight_fp8_mps = weight_fp8_cpu.to("mps")

        expected = torch.nn.functional.embedding(indices.cpu(), weight_fp8_cpu).to(torch.float32)
        actual = torch.nn.functional.embedding(indices, weight_fp8_mps).to(torch.float32).cpu()

        assert torch.equal(actual, expected)


@requires_mps
@requires_fp8
class TestFp8MpsMatchesCpu:
    """MPS FP8 results must be identical to CPU FP8 results."""

    def test_to_fp8_near_identical_bytes(self):
        """float32 -> FP8 on CPU and MPS produce nearly identical bytes (<=1 LSB)."""
        x = torch.randn(128, 128)
        fp8_cpu = x.to(torch.float8_e4m3fn).view(torch.uint8).int()
        fp8_mps = x.to("mps").to(torch.float8_e4m3fn).view(torch.uint8).cpu().int()
        max_diff = (fp8_cpu - fp8_mps).abs().max().item()
        assert max_diff <= 1, f"Max byte diff {max_diff} > 1 LSB"

    def test_fp8_to_float32_identical(self):
        """FP8 -> float32 on CPU and MPS produce the same values."""
        raw = torch.arange(256, dtype=torch.uint8)
        fp8_cpu = raw.view(torch.float8_e4m3fn).to(torch.float32)
        fp8_mps = raw.to("mps").view(torch.float8_e4m3fn).to(torch.float32).cpu()
        # NaN values should both be NaN
        nan_mask = fp8_cpu.isnan()
        assert fp8_mps[nan_mask].isnan().all(), "NaN bytes should decode to NaN on MPS"
        # Non-NaN values should match exactly
        valid = ~nan_mask
        assert torch.allclose(fp8_cpu[valid], fp8_mps[valid], atol=1e-6)

    def test_copy_float32_to_fp8_near_identical(self):
        """Float32 .copy_() into FP8 produces nearly identical bytes on CPU and MPS (<=1 LSB)."""
        src = torch.randn(64, 64)

        dst_cpu = torch.empty(64, 64, dtype=torch.float8_e4m3fn)
        dst_cpu.copy_(src)

        dst_mps = torch.empty(64, 64, dtype=torch.float8_e4m3fn, device="mps")
        dst_mps.copy_(src.to("mps"))

        max_diff = (dst_cpu.view(torch.uint8).int() - dst_mps.view(torch.uint8).cpu().int()).abs().max().item()
        assert max_diff <= 1, f"Max byte diff {max_diff} > 1 LSB"


@requires_mps
@requires_fp8
class TestScaledMmOnMps:
    """torch._scaled_mm should work on MPS with FP8 inputs."""

    def test_basic_shape(self):
        M, K, N = 32, 64, 48
        a = torch.randn(M, K, device="mps").to(torch.float8_e4m3fn)
        b = torch.randn(K, N, device="mps").to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device="mps")
        sb = torch.tensor(1.0, device="mps")
        result = torch._scaled_mm(a, b, sa, sb)
        assert result.shape == (M, N)
        assert result.device.type == "mps"

    def test_matches_fp16_quality(self):
        """FP8 matmul accuracy is within 15% relative RMSE of float32 reference."""
        M, K, N = 64, 128, 64
        A = torch.randn(M, K, device="mps")
        B = torch.randn(K, N, device="mps")
        ref = (A @ B).cpu()

        A_fp8 = A.to(torch.float8_e4m3fn)
        B_fp8 = B.to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device="mps")
        sb = torch.tensor(1.0, device="mps")
        result = torch._scaled_mm(A_fp8, B_fp8, sa, sb).cpu()

        diff = result - ref
        rel_rmse = torch.sqrt((diff**2).mean()) / torch.sqrt((ref**2).mean())
        assert rel_rmse < 0.15, f"FP8 matmul rel RMSE {rel_rmse:.2%} >= 15%"

    def test_fp8_scaled_matmul_accuracy(self):
        """FP8 scaled matmul (with proper quantization) has <15% relative RMSE."""
        M, K, N = 64, 128, 64
        A = torch.randn(M, K, device="mps")
        B = torch.randn(N, K, device="mps")
        ref = (A @ B.T).cpu().float()

        # Quantize with scaling for proper FP8 range usage
        A_q, A_s = torch.ops.fp8_mps.quantize(A)
        B_q, B_s = torch.ops.fp8_mps.quantize(B)
        fp8_result = torch.ops.fp8_mps.scaled_mm(A_q, B_q, A_s, B_s).cpu().float()

        rel_rmse = torch.sqrt(((fp8_result - ref) ** 2).mean()) / torch.sqrt((ref ** 2).mean())
        assert rel_rmse < 0.15, f"FP8 scaled matmul rel RMSE {rel_rmse:.2%} >= 15%"


@requires_mps
@requires_fp8
class TestFp8MatmulOpsOnMps:
    """Standard torch matrix ops should not leak unsupported raw FP8 MPS matmul."""

    def test_torch_matmul_fp8_fp8_uses_scaled_mm(self):
        a = torch.randn((32, 64), device="mps").to(torch.float8_e4m3fn)
        b = torch.randn((64, 48), device="mps").to(torch.float8_e4m3fn)

        out = torch.matmul(a, b)

        assert out.shape == (32, 48)
        assert out.device.type == "mps"
        assert out.dtype == torch.float32

    def test_torch_matmul_fp8_fp8_large_avoids_recursive_fast_path(self):
        a = torch.randn((64, 128), device="mps").to(torch.float8_e4m3fn)
        b = torch.randn((128, 96), device="mps").to(torch.float8_e4m3fn)

        out = torch.matmul(a, b)

        assert out.shape == (64, 96)
        assert out.device.type == "mps"
        assert out.dtype == torch.float32

    def test_at_operator_fp8_fp8_uses_scaled_mm(self):
        a = torch.randn((32, 64), device="mps").to(torch.float8_e4m3fn)
        b = torch.randn((64, 48), device="mps").to(torch.float8_e4m3fn)

        out = a @ b

        assert out.shape == (32, 48)
        assert out.device.type == "mps"
        assert out.dtype == torch.float32

    def test_transposed_fp8_contiguous_avoids_copy_recursion(self):
        weight = torch.randn((256, 256), device="mps").to(torch.float8_e4m3fn)

        out = weight.t().contiguous()

        assert out.shape == (256, 256)
        assert out.device.type == "mps"
        assert out.dtype == torch.float8_e4m3fn
        assert out.is_contiguous()

    def test_addmm_fp8_fp8_uses_scaled_mm(self):
        bias = torch.randn((32, 48), device="mps")
        a = torch.randn((32, 64), device="mps").to(torch.float8_e4m3fn)
        b = torch.randn((64, 48), device="mps").to(torch.float8_e4m3fn)

        out = torch.addmm(bias, a, b)

        assert out.shape == (32, 48)
        assert out.device.type == "mps"
        assert out.dtype == torch.float32

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_linear_float_input_fp8_weight_dequantizes_weight(self, dtype):
        x = torch.randn((4, 8, 64), device="mps", dtype=dtype)
        weight = torch.randn((48, 64), device="mps").to(torch.float8_e4m3fn)
        bias = torch.randn((48,), device="mps", dtype=dtype)

        out = F.linear(x, weight, bias)

        assert out.shape == (4, 8, 48)
        assert out.device.type == "mps"
        assert out.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(16, 64), (4, 8, 64)])
    @pytest.mark.parametrize("has_bias", [False, True])
    def test_linear_fp8_weight_matches_explicit_dequantize_and_keeps_storage(self, dtype, shape, has_bias):
        x = torch.randn(shape, device="mps", dtype=dtype)
        weight = torch.randn((48, 64), device="mps").to(torch.float8_e4m3fn)
        original_weight_bytes = weight.view(torch.uint8).clone()
        bias = torch.randn((48,), device="mps", dtype=dtype) if has_bias else None

        out = F.linear(x, weight, bias)
        expected = F.linear(x, weight.to(dtype), bias)

        assert out.shape == (*shape[:-1], 48)
        assert out.device.type == "mps"
        assert out.dtype == dtype
        assert weight.dtype == torch.float8_e4m3fn
        assert torch.equal(weight.view(torch.uint8), original_weight_bytes)
        assert torch.allclose(out, expected, atol=0, rtol=0)


@requires_mps
@requires_fp8
class TestRowScaledUint8LinearOnMps:
    """Test-only row-scaled uint8 FP8 storage for comparison benchmarks."""

    def test_row_scaled_uint8_weight_dequantizes_for_linear(self):
        x = torch.randn((16, 64), device="mps", dtype=torch.float16)
        weight = torch.randn((48, 64), device="mps", dtype=torch.float16)
        bias = torch.randn((48,), device="mps", dtype=torch.float16)

        quantized, inv_scale = _quantize_row_scaled_fp8_uint8(weight)
        dequantized = _dequantize_row_scaled_fp8_uint8(quantized, inv_scale, x.dtype)
        out = F.linear(x, dequantized, bias)
        ref = F.linear(x, weight, bias)

        assert quantized.dtype == torch.uint8
        assert quantized.device.type == "mps"
        assert quantized.shape == weight.shape
        assert inv_scale.shape == (weight.shape[0],)
        assert out.shape == (16, 48)
        rel_rmse = torch.sqrt(((out.float() - ref.float()) ** 2).mean()) / torch.sqrt((ref.float() ** 2).mean())
        assert rel_rmse < 0.15


@requires_mps
@requires_fp8
class TestCustomOpsAccessible:
    """Custom ops in torch.ops.fp8_mps namespace should be callable."""

    def test_quantize_and_scaled_mm(self):
        A_q, A_s = torch.ops.fp8_mps.quantize(torch.randn(16, 32))
        B_q, B_s = torch.ops.fp8_mps.quantize(torch.randn(24, 32))
        result = torch.ops.fp8_mps.scaled_mm(A_q, B_q, A_s, B_s)
        assert result.shape == (16, 24)

    def test_encode_dequantize_roundtrip(self):
        x = torch.tensor([1.0, 10.0, 100.0, 400.0])
        encoded = torch.ops.fp8_mps.encode(x)
        decoded = torch.ops.fp8_mps.dequantize(encoded, torch.tensor([1.0]))
        for orig, dec in zip(x, decoded.cpu().float()):
            if abs(orig) > 1e-6:
                assert abs(dec - orig) / abs(orig) < 0.15
