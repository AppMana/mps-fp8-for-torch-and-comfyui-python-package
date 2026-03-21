"""FP4 (float4_e2m1fn_x2) support on MPS."""

import pytest
import torch

from conftest import requires_mps

requires_fp4 = pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="FP4 dtype not available",
)


@requires_mps
@requires_fp4
class TestFp4Encode:

    def test_encode_known_values(self):
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="mps")
        encoded = torch.ops.fp8_mps.fp4_encode(x)
        assert encoded.dtype == torch.uint8
        assert encoded.shape == (4,)

    def test_encode_roundtrip(self):
        x = torch.tensor([0.0, 1.0, -1.0, 3.0, -6.0, 0.5, -0.5, 4.0], device="mps")
        encoded = torch.ops.fp8_mps.fp4_encode(x)
        decoded = torch.ops.fp8_mps.fp4_dequantize(encoded, torch.tensor([1.0], device="mps"))
        for orig, dec in zip(x.cpu(), decoded.cpu().float()):
            assert abs(dec - orig) < 0.5, f"orig={orig.item()}, decoded={dec.item()}"

    def test_encode_clamps_to_range(self):
        x = torch.tensor([100.0, -100.0, 0.001, -0.001, 0.0, 0.0, 0.0, 0.0], device="mps")
        encoded = torch.ops.fp8_mps.fp4_encode(x)
        decoded = torch.ops.fp8_mps.fp4_dequantize(encoded, torch.tensor([1.0], device="mps"))
        assert decoded[0].item() == 6.0
        assert decoded[1].item() == -6.0


@requires_mps
@requires_fp4
class TestFp4Quantize:

    def test_quantize_scales_to_range(self):
        x = torch.randn(64, device="mps")
        encoded, inv_scale = torch.ops.fp8_mps.fp4_quantize(x)
        assert encoded.dtype == torch.uint8
        assert encoded.shape == (32,)
        assert inv_scale.shape == (1,)

    def test_quantize_dequantize_roundtrip(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -4.0, -6.0], device="mps")
        encoded, inv_scale = torch.ops.fp8_mps.fp4_quantize(x)
        decoded = torch.ops.fp8_mps.fp4_dequantize(encoded, inv_scale)
        for orig, dec in zip(x.cpu(), decoded.cpu().float()):
            if abs(orig) > 0.01:
                rel_err = abs(dec - orig) / abs(orig)
                assert rel_err < 0.5, f"orig={orig.item()}, decoded={dec.item()}, rel_err={rel_err:.0%}"


@requires_mps
@requires_fp4
class TestFp4DtypeConversion:

    def test_float_to_fp4_on_mps(self):
        x = torch.randn(8, device="mps")
        x_fp4 = x.to(torch.float4_e2m1fn_x2)
        assert x_fp4.dtype == torch.float4_e2m1fn_x2
        assert x_fp4.device.type == "mps"

    def test_fp4_to_float_on_mps(self):
        x = torch.randn(8, device="mps")
        x_fp4 = x.to(torch.float4_e2m1fn_x2)
        x_back = x_fp4.to(torch.float32)
        assert x_back.dtype == torch.float32
        assert x_back.device.type == "mps"

    def test_fp4_cpu_to_mps(self):
        x_cpu = torch.zeros(4, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        x_mps = x_cpu.to("mps")
        assert x_mps.dtype == torch.float4_e2m1fn_x2
        assert x_mps.device.type == "mps"
        assert torch.equal(x_mps.view(torch.uint8).cpu(), x_cpu.view(torch.uint8))

    def test_fp4_roundtrip_values(self):
        x = torch.tensor([0.5, 1.0, 2.0, 3.0, -1.0, -4.0, 0.0, 6.0], device="mps")
        x_fp4 = x.to(torch.float4_e2m1fn_x2)
        x_back = x_fp4.to(torch.float32)
        for orig, rec in zip(x.cpu(), x_back.cpu()):
            assert abs(rec - orig) < 0.5, f"orig={orig.item()}, recovered={rec.item()}"

    def test_fp4_copy(self):
        x = torch.randn(8, device="mps").to(torch.float4_e2m1fn_x2)
        dst = torch.empty(4, dtype=torch.float4_e2m1fn_x2, device="mps")
        dst.copy_(x)
        assert torch.equal(dst.view(torch.uint8).cpu(), x.view(torch.uint8).cpu())

    def test_non_fp4_ops_unaffected(self):
        x = torch.randn(8, device="mps")
        y = x.to(torch.float16)
        assert y.dtype == torch.float16
        z = x.to("cpu")
        assert z.device.type == "cpu"
