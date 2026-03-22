"""FP4 (float4_e2m1fn_x2) support on MPS."""

import pytest
import torch

from conftest import requires_mps

requires_fp4 = pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="FP4 dtype not available",
)


FP4_SPEC = {
    # nibble: expected float value (e2m1fn, bias=1)
    0b0000: 0.0, 0b0001: 0.5, 0b0010: 1.0, 0b0011: 1.5,
    0b0100: 2.0, 0b0101: 3.0, 0b0110: 4.0, 0b0111: 6.0,
    0b1000: -0.0, 0b1001: -0.5, 0b1010: -1.0, 0b1011: -1.5,
    0b1100: -2.0, 0b1101: -3.0, 0b1110: -4.0, 0b1111: -6.0,
}


@requires_mps
@requires_fp4
class TestFp4DecodeExhaustive:

    def test_all_16_values(self):
        """Decode all 16 FP4 nibble values and verify against spec."""
        scale = torch.tensor([1.0], device="mps")
        for nibble, expected in FP4_SPEC.items():
            # Pack same nibble in both halves of the byte
            byte_val = nibble | (nibble << 4)
            encoded = torch.tensor([byte_val], dtype=torch.uint8, device="mps")
            decoded = torch.ops.fp8_mps.fp4_dequantize(encoded, scale).cpu().float()
            lo, hi = decoded[0].item(), decoded[1].item()
            assert lo == expected, f"nibble 0b{nibble:04b} lo: expected {expected}, got {lo}"
            assert hi == expected, f"nibble 0b{nibble:04b} hi: expected {expected}, got {hi}"

    def test_all_256_byte_patterns(self):
        """Decode all 256 packed byte patterns and verify both nibbles."""
        all_bytes = torch.arange(256, dtype=torch.uint8, device="mps")
        scale = torch.tensor([1.0], device="mps")
        decoded = torch.ops.fp8_mps.fp4_dequantize(all_bytes, scale).cpu().float()
        for byte_val in range(256):
            lo_nibble = byte_val & 0xF
            hi_nibble = (byte_val >> 4) & 0xF
            lo_expected = FP4_SPEC[lo_nibble]
            hi_expected = FP4_SPEC[hi_nibble]
            lo_actual = decoded[byte_val * 2].item()
            hi_actual = decoded[byte_val * 2 + 1].item()
            assert lo_actual == lo_expected, (
                f"byte 0x{byte_val:02X} lo: expected {lo_expected}, got {lo_actual}"
            )
            assert hi_actual == hi_expected, (
                f"byte 0x{byte_val:02X} hi: expected {hi_expected}, got {hi_actual}"
            )


@requires_mps
@requires_fp4
class TestFp4EncodeExhaustive:

    def test_exact_values_encode_correctly(self):
        """Encoding each exact FP4 value produces the correct nibble."""
        pos_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        pos_nibbles = [0, 1, 2, 3, 4, 5, 6, 7]
        # Test positive and negative, packing pairs
        for val, nibble in zip(pos_values, pos_nibbles):
            for sign in [1.0, -1.0]:
                v = val * sign
                expected_nibble = nibble | (0b1000 if sign < 0 and val != 0.0 else 0)
                # Encode pair: [v, 0.0]
                x = torch.tensor([v, 0.0], device="mps", dtype=torch.float32)
                encoded = torch.ops.fp8_mps.fp4_encode(x).cpu()
                lo_nibble = encoded[0].item() & 0xF
                assert lo_nibble == expected_nibble, (
                    f"val={v}: expected nibble 0b{expected_nibble:04b}, got 0b{lo_nibble:04b}"
                )

    def test_midpoints_round_correctly(self):
        """Midpoints between adjacent FP4 values round to nearest."""
        pos_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        scale = torch.tensor([1.0], device="mps")
        for i in range(len(pos_values) - 1):
            mid = (pos_values[i] + pos_values[i + 1]) / 2
            # Encode and decode: should land on one of the two neighbors
            x = torch.tensor([mid, 0.0], device="mps", dtype=torch.float32)
            encoded = torch.ops.fp8_mps.fp4_encode(x)
            decoded = torch.ops.fp8_mps.fp4_dequantize(encoded, scale).cpu().float()
            result = decoded[0].item()
            assert result in (pos_values[i], pos_values[i + 1]), (
                f"midpoint {mid}: decoded to {result}, expected {pos_values[i]} or {pos_values[i+1]}"
            )

    def test_clamp_to_range(self):
        """Values outside [-6, 6] clamp to the extremes."""
        x = torch.tensor([100.0, -100.0, 1000.0, -1000.0, 0.001, -0.001, 0.0, 0.0],
                         device="mps", dtype=torch.float32)
        scale = torch.tensor([1.0], device="mps")
        encoded = torch.ops.fp8_mps.fp4_encode(x)
        decoded = torch.ops.fp8_mps.fp4_dequantize(encoded, scale).cpu().float()
        assert decoded[0].item() == 6.0
        assert decoded[1].item() == -6.0
        assert decoded[2].item() == 6.0
        assert decoded[3].item() == -6.0
        assert decoded[4].item() == 0.0
        assert decoded[5].item() == 0.0


@requires_mps
@requires_fp4
class TestFp4Encode:

    def test_encode_roundtrip(self):
        x = torch.tensor([0.0, 1.0, -1.0, 3.0, -6.0, 0.5, -0.5, 4.0], device="mps")
        encoded = torch.ops.fp8_mps.fp4_encode(x)
        decoded = torch.ops.fp8_mps.fp4_dequantize(encoded, torch.tensor([1.0], device="mps"))
        for orig, dec in zip(x.cpu(), decoded.cpu().float()):
            assert abs(dec - orig) < 0.5, f"orig={orig.item()}, decoded={dec.item()}"


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
