"""Selectable LUT/arithmetic decode backends and representative benchmarks."""

import pytest
import torch

from conftest import requires_mps


REPRESENTATIVE_ELEMENTS = 4_194_304


@requires_mps
def test_decode_backend_selection_roundtrip():
    import fp4_fp8_for_torch_mps

    fp4_fp8_for_torch_mps.set_decode_backend("arithmetic")
    assert fp4_fp8_for_torch_mps.get_decode_backend() == "arithmetic"

    fp4_fp8_for_torch_mps.set_decode_backend("lut")
    assert fp4_fp8_for_torch_mps.get_decode_backend() == "lut"

    fp4_fp8_for_torch_mps.set_decode_backend("lut4")
    assert fp4_fp8_for_torch_mps.get_decode_backend() == "lut4"

    fp4_fp8_for_torch_mps.set_decode_backend("arithmetic4")
    assert fp4_fp8_for_torch_mps.get_decode_backend() == "arithmetic4"

    fp4_fp8_for_torch_mps.set_decode_backend("auto")
    assert fp4_fp8_for_torch_mps.get_decode_backend() == "auto"

    with pytest.raises(ValueError):
        fp4_fp8_for_torch_mps.set_decode_backend("invalid")

    fp4_fp8_for_torch_mps.set_decode_backend("lut")


@requires_mps
@pytest.mark.parametrize(
    "decode_fn",
    ["fp8_dequantize_arithmetic", "fp8_dequantize_lut4", "fp8_dequantize_arithmetic4"],
)
def test_fp8_decode_backends_match_all_patterns(decode_fn):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.arange(256, dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    lut = fp8_mps_native.fp8_dequantize_lut(raw, scale)
    decoded = getattr(fp8_mps_native, decode_fn)(raw, scale)

    nan_mask = lut.isnan()
    assert decoded[nan_mask].isnan().all()
    assert torch.equal(lut[~nan_mask].cpu(), decoded[~nan_mask].cpu())


@requires_mps
@pytest.mark.parametrize(
    "decode_fn",
    ["fp4_dequantize_arithmetic", "fp4_dequantize_lut4", "fp4_dequantize_arithmetic4"],
)
def test_fp4_decode_backends_match_all_patterns(decode_fn):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.arange(256, dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    lut = fp8_mps_native.fp4_dequantize_lut(raw, scale)
    decoded = getattr(fp8_mps_native, decode_fn)(raw, scale)

    assert torch.equal(lut.cpu(), decoded.cpu())


@requires_mps
def test_default_decode_backend_is_lut():
    from fp4_fp8_for_torch_mps import fp8_mps_native

    assert fp8_mps_native.get_decode_backend() == "lut"


@requires_mps
@pytest.mark.benchmark(group="fp8_decode_backend")
def test_benchmark_fp8_lut_decode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    def run():
        result = fp8_mps_native.fp8_dequantize_lut(raw, scale)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp8_decode_backend")
def test_benchmark_fp8_arithmetic_decode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    def run():
        result = fp8_mps_native.fp8_dequantize_arithmetic(raw, scale)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp8_decode_backend")
def test_benchmark_fp8_lut4_decode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    def run():
        result = fp8_mps_native.fp8_dequantize_lut4(raw, scale)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp8_decode_backend")
def test_benchmark_fp8_arithmetic4_decode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    def run():
        result = fp8_mps_native.fp8_dequantize_arithmetic4(raw, scale)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp4_decode_backend")
def test_benchmark_fp4_lut_decode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS // 2,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    def run():
        result = fp8_mps_native.fp4_dequantize_lut(raw, scale)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp4_decode_backend")
def test_benchmark_fp4_arithmetic_decode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS // 2,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    def run():
        result = fp8_mps_native.fp4_dequantize_arithmetic(raw, scale)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp4_decode_backend")
def test_benchmark_fp4_lut4_decode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS // 2,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    def run():
        result = fp8_mps_native.fp4_dequantize_lut4(raw, scale)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp4_decode_backend")
def test_benchmark_fp4_arithmetic4_decode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS // 2,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    def run():
        result = fp8_mps_native.fp4_dequantize_arithmetic4(raw, scale)
        torch.mps.synchronize()
        return result

    benchmark(run)
