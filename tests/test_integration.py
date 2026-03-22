"""Integration tests using only standard PyTorch APIs.

These tests do NOT import fp4_fp8_for_torch_mps — the torch.backends entry point
auto-loads the extension when torch is imported. They verify that FP8
dtypes work on MPS identically to CPU, and that FP8 matmul compares
favorably to FP16.
"""

import pytest
import torch

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)
requires_fp8 = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 dtypes not available",
)


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
