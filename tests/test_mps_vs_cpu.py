"""MPS vs CPU validation tests.

Ensures the MPS Metal GPU implementation produces identical results
to the CPU fallback for FP8 encoding, decoding, roundtrip, dtype
conversion, and matmul.
"""

import pytest
import torch

from conftest import requires_mps, requires_fp8

DECODE_TOLERANCE = 1e-6
ENCODE_TOLERANCE = 0
MATMUL_TOLERANCE = 1e-4


@requires_mps
class TestEncodeMpsVsCpu:
    """Compare FP8 encoding on MPS vs CPU."""

    def test_encoding_identical(self):
        """MPS and CPU encoding produce identical byte results."""
        from fp8_mps_metal.fp8_mps_native import fp8_quantize

        test_values = [
            0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 50.0,
            100.0, 200.0, 300.0, 400.0, 448.0,
            -0.001, -0.1, -1.0, -10.0, -100.0, -448.0,
        ]
        t_cpu = torch.tensor(test_values, dtype=torch.float32)
        t_mps = t_cpu.to("mps")

        encoded_cpu, _ = fp8_quantize(t_cpu)
        encoded_mps, _ = fp8_quantize(t_mps)

        assert torch.equal(encoded_cpu.cpu(), encoded_mps.cpu()), (
            "MPS and CPU encoding differ"
        )


@requires_mps
class TestDecodeMpsVsCpu:
    """Compare FP8 decoding on MPS vs CPU."""

    def test_all_256_values_identical(self):
        """MPS and CPU decoding of all 256 byte values are identical."""
        from fp8_mps_metal.fp8_mps_native import fp8_dequantize

        all_bytes = torch.arange(256, dtype=torch.uint8)
        scale = torch.tensor([1.0], dtype=torch.float32)

        decoded_cpu = fp8_dequantize(all_bytes, scale).cpu().float()
        decoded_mps = fp8_dequantize(all_bytes.to("mps"), scale).cpu().float()

        # NaN values should both be NaN
        valid = ~decoded_cpu.isnan()
        assert decoded_cpu.isnan().equal(decoded_mps.isnan()), "NaN pattern mismatch"
        diff = (decoded_cpu[valid] - decoded_mps[valid]).abs()
        assert diff.max().item() <= DECODE_TOLERANCE, (
            f"Max decode diff {diff.max().item()} > tolerance {DECODE_TOLERANCE}"
        )


@requires_mps
class TestRoundtripMpsVsCpu:
    """Compare encode-decode roundtrip on MPS vs CPU."""

    def test_roundtrip_identical(self):
        """CPU and MPS roundtrip decoded values are identical."""
        from fp8_mps_metal.fp8_mps_native import fp8_quantize, fp8_dequantize

        test_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]
        t_cpu = torch.tensor(test_values, dtype=torch.float32)
        t_mps = t_cpu.to("mps")

        enc_cpu, sc_cpu = fp8_quantize(t_cpu)
        dec_cpu = fp8_dequantize(enc_cpu, sc_cpu).cpu().float()

        enc_mps, sc_mps = fp8_quantize(t_mps)
        dec_mps = fp8_dequantize(enc_mps, sc_mps).cpu().float()

        cross_diff = (dec_cpu - dec_mps).abs().max().item()
        assert cross_diff <= DECODE_TOLERANCE, (
            f"Roundtrip cross diff {cross_diff} > tolerance {DECODE_TOLERANCE}"
        )


@requires_mps
@requires_fp8
class TestDtypeConversionMpsVsCpu:
    """Compare .to(float8_e4m3fn) on MPS vs CPU."""

    def test_byte_identical(self):
        """CPU and MPS .to(float8_e4m3fn) produce identical bytes."""
        test_values = [0.5, 1.0, 2.0, 10.0, 100.0]
        t = torch.tensor(test_values, dtype=torch.float32)

        fp8_cpu_bytes = t.to(torch.float8_e4m3fn).view(torch.uint8)
        fp8_mps_bytes = t.to("mps").to(torch.float8_e4m3fn).view(torch.uint8).cpu()

        assert torch.equal(fp8_cpu_bytes, fp8_mps_bytes), (
            "CPU and MPS .to() produce different bytes"
        )


@requires_mps
class TestMatmulMpsVsCpu:
    """Compare FP8 matmul on MPS vs CPU."""

    def test_matmul_consistent(self):
        """CPU and MPS FP8 matmul results are within tolerance."""
        from fp8_mps_metal.fp8_mps_native import fp8_quantize, fp8_scaled_mm

        M, K, N = 16, 32, 24
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(N, K, dtype=torch.float32)

        A_q_cpu, A_s_cpu = fp8_quantize(A)
        B_q_cpu, B_s_cpu = fp8_quantize(B)
        result_cpu = fp8_scaled_mm(A_q_cpu, B_q_cpu, A_s_cpu, B_s_cpu).cpu()

        A_q_mps, A_s_mps = fp8_quantize(A.to("mps"))
        B_q_mps, B_s_mps = fp8_quantize(B.to("mps"))
        result_mps = fp8_scaled_mm(A_q_mps, B_q_mps, A_s_mps, B_s_mps).cpu()

        max_diff = (result_cpu - result_mps).abs().max().item()
        assert max_diff <= MATMUL_TOLERANCE, (
            f"Matmul max diff {max_diff} > tolerance {MATMUL_TOLERANCE}"
        )
