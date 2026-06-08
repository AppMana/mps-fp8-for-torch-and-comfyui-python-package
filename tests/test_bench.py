"""Microbenchmarks for FP8 Metal kernels."""

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_mps


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
class TestBenchDequantize:

    @pytest.fixture
    def data_1m(self):
        return torch.randint(0, 255, (1024, 1024), dtype=torch.uint8, device="mps")

    def test_dequantize(self, benchmark, data_1m):
        from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_dequantize
        scale = torch.tensor([1.0], device="mps")
        benchmark(fp8_dequantize, data_1m, scale)

    def test_encode(self, benchmark):
        from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_encode
        x = torch.randn(1024, 1024, device="mps")
        benchmark(fp8_encode, x)

    def test_quantize(self, benchmark):
        from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_quantize
        x = torch.randn(1024, 1024, device="mps")
        benchmark(fp8_quantize, x)


@requires_mps
class TestBenchMatmul:

    def test_scaled_mm_64x256x128(self, benchmark):
        from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_quantize, fp8_scaled_mm
        A_q, A_s = fp8_quantize(torch.randn(64, 256))
        B_q, B_s = fp8_quantize(torch.randn(128, 256))

        def run():
            result = fp8_scaled_mm(A_q, B_q, A_s, B_s)
            torch.mps.synchronize()
            return result
        benchmark(run)

    def test_scaled_mm_auto_64x256x128(self, benchmark):
        from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_quantize, fp8_scaled_mm_auto
        A_q, A_s = fp8_quantize(torch.randn(64, 256))
        B_q, B_s = fp8_quantize(torch.randn(128, 256))

        def run():
            result = fp8_scaled_mm_auto(A_q, B_q, A_s, B_s)
            torch.mps.synchronize()
            return result
        benchmark(run)

    def test_vecmat_1x512x256(self, benchmark):
        from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_quantize, fp8_scaled_mm
        x_q, x_s = fp8_quantize(torch.randn(1, 512))
        W_q, W_s = fp8_quantize(torch.randn(256, 512))

        def run():
            result = fp8_scaled_mm(x_q, W_q, x_s, W_s)
            torch.mps.synchronize()
            return result
        benchmark(run)

    def test_fast_path_64x256x128(self, benchmark):
        from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_quantize, fp8_scaled_mm_fast
        A_q, A_s = fp8_quantize(torch.randn(64, 256))
        B_q, B_s = fp8_quantize(torch.randn(128, 256))

        def run():
            result = fp8_scaled_mm_fast(A_q, B_q, A_s, B_s)
            torch.mps.synchronize()
            return result
        benchmark(run)

    def test_native_fp16_matmul_64x256x128(self, benchmark):
        """Baseline: native FP16 matmul for comparison."""
        A = torch.randn(64, 256, dtype=torch.float16, device="mps")
        B = torch.randn(128, 256, dtype=torch.float16, device="mps")

        def run():
            result = A @ B.T
            torch.mps.synchronize()
            return result
        benchmark(run)


@requires_mps
class TestBenchLinearStorageFormats:
    """Compare resident FP8 storage against row-scaled uint8 storage."""

    @pytest.mark.parametrize(
        ("m", "k", "n"),
        [
            (256, 256, 256),
            (1024, 1024, 1024),
            (256, 4096, 4096),
        ],
        ids=["256x256x256", "1024x1024x1024", "256x4096x4096"],
    )
    @pytest.mark.parametrize(
        "storage",
        ["native_fp16", "fp8_storage", "row_scaled_uint8"],
    )
    @pytest.mark.benchmark(group="linear_weight_storage")
    def test_linear_weight_storage(self, benchmark, m, k, n, storage):
        x = torch.randn((m, k), dtype=torch.float16, device="mps")
        weight = torch.randn((n, k), dtype=torch.float16, device="mps")
        bias = torch.randn((n,), dtype=torch.float16, device="mps")

        if storage == "native_fp16":
            def run():
                result = F.linear(x, weight, bias)
                torch.mps.synchronize()
                return result
        elif storage == "fp8_storage":
            fp8_weight = weight.to(torch.float8_e4m3fn)

            def run():
                result = F.linear(x, fp8_weight, bias)
                torch.mps.synchronize()
                return result
        else:
            quantized, inv_scale = _quantize_row_scaled_fp8_uint8(weight)

            def run():
                dequantized = _dequantize_row_scaled_fp8_uint8(quantized, inv_scale, x.dtype)
                result = F.linear(x, dequantized, bias)
                torch.mps.synchronize()
                return result

        benchmark(run)


@requires_mps
class TestBenchAttentionBlock:
    """Benchmark the realistic path: FP8 weight projections -> FP16 SDPA."""

    def test_fp8_weights_projection_then_sdpa(self, benchmark):
        """FP8 linear projections (Q/K/V) via _scaled_mm, then native FP16 SDPA."""
        B, S, D, H = 1, 256, 512, 8
        head_dim = D // H

        x = torch.randn(B * S, D, dtype=torch.float16, device="mps")

        Wq_q, Wq_s = torch.ops.fp8_mps.quantize(torch.randn(D, D))
        Wk_q, Wk_s = torch.ops.fp8_mps.quantize(torch.randn(D, D))
        Wv_q, Wv_s = torch.ops.fp8_mps.quantize(torch.randn(D, D))

        x_q, x_s = torch.ops.fp8_mps.quantize(x.float())

        def run():
            q = torch._scaled_mm(x_q, Wq_q.t().contiguous(), x_s, Wq_s).half()
            k = torch._scaled_mm(x_q, Wk_q.t().contiguous(), x_s, Wk_s).half()
            v = torch._scaled_mm(x_q, Wv_q.t().contiguous(), x_s, Wv_s).half()
            q = q.view(B, S, H, head_dim).transpose(1, 2)
            k = k.view(B, S, H, head_dim).transpose(1, 2)
            v = v.view(B, S, H, head_dim).transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.mps.synchronize()
            return out
        benchmark(run)

    def test_fp16_weights_projection_then_sdpa(self, benchmark):
        """Baseline: FP16 linear projections (Q/K/V), then native FP16 SDPA."""
        B, S, D, H = 1, 256, 512, 8
        head_dim = D // H

        x = torch.randn(B * S, D, dtype=torch.float16, device="mps")
        Wq = torch.randn(D, D, dtype=torch.float16, device="mps")
        Wk = torch.randn(D, D, dtype=torch.float16, device="mps")
        Wv = torch.randn(D, D, dtype=torch.float16, device="mps")

        def run():
            q = (x @ Wq.T).view(B, S, H, head_dim).transpose(1, 2)
            k = (x @ Wk.T).view(B, S, H, head_dim).transpose(1, 2)
            v = (x @ Wv.T).view(B, S, H, head_dim).transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.mps.synchronize()
            return out
        benchmark(run)
