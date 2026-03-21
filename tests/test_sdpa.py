"""Scaled dot product attention with FP8 tensors on MPS."""

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_mps, requires_fp8


@requires_mps
@requires_fp8
class TestSdpaFp8:

    def test_fp8_to_fp16_sdpa(self):
        """FP8 weights dequantized to FP16 produce valid SDPA output."""
        B, H, S, D = 1, 8, 16, 64
        q = torch.randn(B, H, S, D, device="mps")
        k = torch.randn(B, H, S, D, device="mps")
        v = torch.randn(B, H, S, D, device="mps")

        q_f16 = q.to(torch.float8_e4m3fn).to(torch.float16)
        k_f16 = k.to(torch.float8_e4m3fn).to(torch.float16)
        v_f16 = v.to(torch.float8_e4m3fn).to(torch.float16)

        out = F.scaled_dot_product_attention(q_f16, k_f16, v_f16)
        assert out.shape == (B, H, S, D)
        assert out.dtype == torch.float16
        assert not out.isnan().any()

    def test_fp8_sdpa_vs_float32_reference(self):
        """FP8->FP16 SDPA output is close to float32 reference."""
        B, H, S, D = 1, 4, 32, 64
        q = torch.randn(B, H, S, D, device="mps")
        k = torch.randn(B, H, S, D, device="mps")
        v = torch.randn(B, H, S, D, device="mps")

        ref = F.scaled_dot_product_attention(q, k, v).float()

        q_f16 = q.to(torch.float8_e4m3fn).to(torch.float16)
        k_f16 = k.to(torch.float8_e4m3fn).to(torch.float16)
        v_f16 = v.to(torch.float8_e4m3fn).to(torch.float16)
        out = F.scaled_dot_product_attention(q_f16, k_f16, v_f16).float()

        rel_err = (out - ref).abs().mean() / ref.abs().mean()
        assert rel_err < 0.2, f"Mean relative error {rel_err:.2%} >= 20%"

    def test_fp8_sdpa_larger_sequence(self):
        """SDPA works with FP8 at realistic sequence lengths."""
        B, H, S, D = 2, 12, 256, 64
        q = torch.randn(B, H, S, D, device="mps")
        k = torch.randn(B, H, S, D, device="mps")
        v = torch.randn(B, H, S, D, device="mps")

        q_f16 = q.to(torch.float8_e4m3fn).to(torch.float16)
        k_f16 = k.to(torch.float8_e4m3fn).to(torch.float16)
        v_f16 = v.to(torch.float8_e4m3fn).to(torch.float16)

        out = F.scaled_dot_product_attention(q_f16, k_f16, v_f16)
        assert out.shape == (B, H, S, D)
        assert not out.isnan().any()

    def test_fp8_cpu_weights_to_mps_sdpa(self):
        """FP8 weights loaded on CPU, transferred to MPS, used in SDPA."""
        B, H, S, D = 1, 4, 16, 32

        # Simulate loading pre-quantized FP8 weights from disk
        q_fp8_cpu = torch.randn(B, H, S, D).to(torch.float8_e4m3fn)
        k_fp8_cpu = torch.randn(B, H, S, D).to(torch.float8_e4m3fn)
        v_fp8_cpu = torch.randn(B, H, S, D).to(torch.float8_e4m3fn)

        # Transfer to MPS and dequantize
        q = q_fp8_cpu.to("mps").to(torch.float16)
        k = k_fp8_cpu.to("mps").to(torch.float16)
        v = v_fp8_cpu.to("mps").to(torch.float16)

        out = F.scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, S, D)
        assert out.device.type == "mps"
