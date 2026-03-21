"""FP8 Metal kernel accuracy and correctness tests."""

import pytest
import torch

from conftest import requires_mps, requires_fp8

FP8_E4M3FN_RELATIVE_TOLERANCE = 0.15


def _fp8_e4m3fn_decode_reference(bits: int) -> float:
    """Pure Python reference decode for e4m3fn format."""
    if (bits & 0x7F) == 0x7F:
        return float('nan')
    sign = (bits >> 7) & 1
    exp_bits = (bits >> 3) & 0xF
    mant_bits = bits & 0x7
    if exp_bits == 0:
        value = (mant_bits / 8.0) * (2.0 ** -6)
    else:
        mantissa = 1.0 + mant_bits / 8.0
        exponent = exp_bits - 7
        value = mantissa * (2.0 ** exponent)
    return -value if sign else value


class TestExhaustiveFp8Decode:
    """Test all 256 FP8 bit patterns against reference."""

    def test_all_256_patterns(self):
        """Native decode of every uint8 matches the Python reference."""
        from fp8_mps_metal import fp8_mps_native

        all_bits = torch.arange(256, dtype=torch.uint8)
        scale = torch.tensor([1.0])
        decoded = fp8_mps_native.fp8_dequantize(all_bits, scale).cpu().float()
        ref = torch.tensor([_fp8_e4m3fn_decode_reference(i) for i in range(256)])

        # NaN values should both be NaN
        nan_mask = ref.isnan()
        assert decoded[nan_mask].isnan().all(), "NaN bytes should decode to NaN"

        # Non-NaN values should match
        valid = ~nan_mask
        max_abs_err = (decoded[valid] - ref[valid]).abs().max().item()
        assert max_abs_err < 0.5, f"Max absolute decode error {max_abs_err} >= 0.5"


class TestMatmulAccuracyNative:
    """Test FP8 scaled matmul via native Metal shaders."""

    def test_fused_kernel(self):
        """Fused kernel relative RMSE stays below 15%."""
        from fp8_mps_metal import fp8_mps_native

        M, K, N = 64, 256, 128
        A_f32 = torch.randn(M, K)
        B_f32 = torch.randn(N, K)
        ref = A_f32 @ B_f32.T

        A_q, A_scale = fp8_mps_native.fp8_quantize(A_f32)
        B_q, B_scale = fp8_mps_native.fp8_quantize(B_f32)
        result = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_scale, B_scale).cpu().float()
        diff = result - ref
        rel_rmse = torch.sqrt((diff ** 2).mean()).item() / torch.sqrt((ref ** 2).mean()).item()
        assert rel_rmse < 0.15, f"Fused kernel rel RMSE {rel_rmse:.4%} >= 15%"

    def test_fast_path(self):
        """Fast (dequant + native matmul) relative RMSE stays below 15%."""
        from fp8_mps_metal import fp8_mps_native

        M, K, N = 64, 256, 128
        A_f32 = torch.randn(M, K)
        B_f32 = torch.randn(N, K)
        ref = A_f32 @ B_f32.T

        A_q, A_scale = fp8_mps_native.fp8_quantize(A_f32)
        B_q, B_scale = fp8_mps_native.fp8_quantize(B_f32)
        result = fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_scale, B_scale).cpu().float()
        diff = result - ref
        rel_rmse = torch.sqrt((diff ** 2).mean()).item() / torch.sqrt((ref ** 2).mean()).item()
        assert rel_rmse < 0.15, f"Fast path rel RMSE {rel_rmse:.4%} >= 15%"

    def test_auto_selector_shape(self):
        """Auto selector produces correct output shape."""
        from fp8_mps_metal import fp8_mps_native

        M, K, N = 64, 256, 128
        A_q, A_scale = fp8_mps_native.fp8_quantize(torch.randn(M, K))
        B_q, B_scale = fp8_mps_native.fp8_quantize(torch.randn(N, K))
        result = fp8_mps_native.fp8_scaled_mm_auto(A_q, B_q, A_scale, B_scale)
        assert result.shape == (M, N)


class TestQuantizeRoundtrip:
    """Test quantize then dequantize roundtrip."""

    def test_roundtrip_max_error(self):
        """Roundtrip error is bounded."""
        from fp8_mps_metal import fp8_mps_native

        x = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 448.0])
        q, scale = fp8_mps_native.fp8_quantize(x)
        d = fp8_mps_native.fp8_dequantize(q, scale).cpu().float()
        max_err = (d - x).abs().max().item()
        assert max_err < 50.0, f"Roundtrip max error {max_err} >= 50.0"


class TestVecmat:
    """Test M=1 vecmat kernel path."""

    def test_vecmat_accuracy(self):
        """Vecmat relative RMSE stays below 15%."""
        from fp8_mps_metal import fp8_mps_native

        K, N = 512, 256
        x = torch.randn(1, K)
        W = torch.randn(N, K)
        ref = x @ W.T

        x_q, x_s = fp8_mps_native.fp8_quantize(x)
        W_q, W_s = fp8_mps_native.fp8_quantize(W)
        result = fp8_mps_native.fp8_scaled_mm(x_q, W_q, x_s, W_s).cpu().float()

        diff = result - ref
        rel_rmse = torch.sqrt((diff ** 2).mean()).item() / torch.sqrt((ref ** 2).mean()).item()
        assert rel_rmse < 0.15, f"Vecmat rel RMSE {rel_rmse:.4%} >= 15%"
        assert result.shape == (1, N)


@requires_mps
@requires_fp8
class TestFp8Conversion:
    """Test Float8_e4m3fn dtype conversion on MPS."""

    def test_basic_float32_to_fp8(self):
        """float32 -> float8_e4m3fn on MPS preserves shape and device."""
        x = torch.randn(4, 8, device="mps")
        x_fp8 = x.to(torch.float8_e4m3fn)
        assert x_fp8.dtype == torch.float8_e4m3fn
        assert x_fp8.device.type == "mps"
        assert x_fp8.shape == x.shape

    def test_cpu_to_mps_fp8(self):
        """CPU float32 -> MPS float8_e4m3fn."""
        x_cpu = torch.randn(4, 8)
        x_fp8 = x_cpu.to("mps", dtype=torch.float8_e4m3fn)
        assert x_fp8.dtype == torch.float8_e4m3fn
        assert x_fp8.device.type == "mps"

    @pytest.mark.parametrize("src_dtype", [torch.float32, torch.float16])
    def test_conversion_from_various_dtypes(self, src_dtype):
        """Conversion from float32 and float16 to FP8."""
        x = torch.randn(4, 4, dtype=src_dtype, device="mps")
        x_fp8 = x.to(torch.float8_e4m3fn)
        assert x_fp8.dtype == torch.float8_e4m3fn

    def test_empty_tensor(self):
        """Empty tensor converts to FP8."""
        empty = torch.empty(0, device="mps")
        empty_fp8 = empty.to(torch.float8_e4m3fn)
        assert empty_fp8.numel() == 0

    def test_single_element(self):
        """Single-element tensor converts to FP8."""
        single = torch.tensor([3.14], device="mps")
        single_fp8 = single.to(torch.float8_e4m3fn)
        assert single_fp8.shape == single.shape

    def test_large_tensor(self):
        """128x256 tensor converts to FP8."""
        large = torch.randn(128, 256, device="mps")
        large_fp8 = large.to(torch.float8_e4m3fn)
        assert large_fp8.shape == large.shape

    def test_fp8_encoding_valid_uint8(self):
        """Encoded FP8 values are valid uint8."""
        from fp8_mps_metal.fp8_mps_native import fp8_dequantize

        x = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 448.0], device="mps")
        x_fp8 = x.to(torch.float8_e4m3fn)
        x_u8 = x_fp8.view(torch.uint8).cpu()
        assert x_u8.min() >= 0 and x_u8.max() <= 255

    def test_fp8_cpu_to_mps_bytes_preserved(self):
        """FP8 tensor transferred CPU -> MPS preserves raw bytes."""
        x_u8_cpu = torch.randint(0, 255, (4, 8), dtype=torch.uint8)
        x_fp8_cpu = x_u8_cpu.view(torch.float8_e4m3fn)
        x_fp8_mps = x_fp8_cpu.to("mps")
        assert x_fp8_mps.dtype == torch.float8_e4m3fn
        assert x_fp8_mps.device.type == "mps"
        assert torch.equal(x_fp8_mps.view(torch.uint8).cpu(), x_u8_cpu)

    def test_fp8_in_matmul_pipeline(self):
        """Converted FP8 tensors can be used in scaled matmul."""
        from fp8_mps_metal.fp8_mps_native import fp8_scaled_mm_auto

        A = torch.randn(16, 32, device="mps").to(torch.float8_e4m3fn).view(torch.uint8)
        B = torch.randn(32, 32, device="mps").to(torch.float8_e4m3fn).view(torch.uint8)
        scale_a = torch.tensor([1.0])
        scale_b = torch.tensor([1.0])
        result = fp8_scaled_mm_auto(A, B.t().contiguous(), scale_a, scale_b)
        assert result.shape == (16, 32)
        assert result.device.type == "mps"

    def test_fp8_copy_preserves_bytes(self):
        """FP8 .copy_() preserves raw bytes."""
        fp8_source = torch.randint(0, 255, (4, 8), dtype=torch.uint8).view(torch.float8_e4m3fn)
        fp8_dest = torch.empty(4, 8, dtype=torch.float8_e4m3fn, device="mps")
        fp8_dest.copy_(fp8_source)
        assert torch.equal(fp8_dest.view(torch.uint8).cpu(), fp8_source.view(torch.uint8))

    def test_float32_to_fp8_via_copy(self):
        """Float32 -> FP8 via .copy_() preserves values within tolerance."""
        f32_source = torch.tensor(
            [[1.0, 2.5, -3.0, 0.5], [10.0, -8.0, 0.0, 100.0]],
            device="mps", dtype=torch.float32,
        )
        fp8_dest = torch.empty(2, 4, dtype=torch.float8_e4m3fn, device="mps")
        fp8_dest.copy_(f32_source)
        result_f32 = fp8_dest.to(torch.float32)

        for i in range(2):
            for j in range(4):
                expected = f32_source[i, j].item()
                actual = result_f32[i, j].item()
                if abs(expected) > 1e-6:
                    rel_error = abs(actual - expected) / abs(expected)
                    assert rel_error < FP8_E4M3FN_RELATIVE_TOLERANCE, (
                        f"[{i},{j}]: expected {expected}, got {actual} (rel_error={rel_error:.2%})"
                    )
                else:
                    assert abs(actual - expected) < 0.1, (
                        f"[{i},{j}]: expected {expected}, got {actual}"
                    )


@requires_mps
@requires_fp8
class TestFp8ValuePreservation:
    """Ensure FP8 conversions preserve value semantics (no automatic scaling)."""

    def test_small_values(self):
        """Small values survive FP8 roundtrip."""
        vals = torch.tensor([0.1, 0.5, 1.0, 2.0], device="mps")
        decoded = vals.to(torch.float8_e4m3fn).to(torch.float32)
        for orig, dec in zip(vals.cpu(), decoded.cpu()):
            if abs(orig) > 1e-6:
                assert abs(dec - orig) / abs(orig) < 0.15

    def test_medium_values(self):
        """Medium values survive FP8 roundtrip."""
        vals = torch.tensor([10.0, 50.0, 100.0, 200.0], device="mps")
        decoded = vals.to(torch.float8_e4m3fn).to(torch.float32)
        for orig, dec in zip(vals.cpu(), decoded.cpu()):
            if abs(orig) > 1e-6:
                assert abs(dec - orig) / abs(orig) < 0.15

    def test_mixed_range(self):
        """Mixed-range values survive FP8 roundtrip."""
        vals = torch.tensor([0.1, 1.0, 10.0, 100.0], device="mps")
        decoded = vals.to(torch.float8_e4m3fn).to(torch.float32)
        for orig, dec in zip(vals.cpu(), decoded.cpu()):
            if abs(orig) > 1e-6:
                assert abs(dec - orig) / abs(orig) < 0.15

    def test_no_automatic_scaling(self):
        """Converting 1.0 to FP8 and back yields ~1.0, not ~448.0."""
        test_value = torch.tensor([1.0], device="mps")
        decoded = test_value.to(torch.float8_e4m3fn).to(torch.float32)
        assert abs(decoded.cpu().item() - 1.0) < 0.2, (
            f"Value was scaled: expected ~1.0, got {decoded.cpu().item()}"
        )

    def test_copy_preserves_values(self):
        """.copy_() from float32 to FP8 preserves values."""
        src = torch.tensor([1.0, 5.0, 10.0, 50.0], device="mps", dtype=torch.float32)
        dst = torch.empty(4, dtype=torch.float8_e4m3fn, device="mps")
        dst.copy_(src)
        decoded = dst.to(torch.float32)
        for orig, dec in zip(src.cpu(), decoded.cpu()):
            if abs(orig) > 1e-6:
                assert abs(dec - orig) / abs(orig) < 0.15
