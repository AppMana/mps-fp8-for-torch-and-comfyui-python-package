"""Selectable LUT/arithmetic decode backends and representative benchmarks."""

import pytest
import torch

from conftest import requires_mps


REPRESENTATIVE_ELEMENTS = 4_194_304
FP8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)
FP8_DTYPES = [
    getattr(torch, "float8_e4m3fn", None),
    getattr(torch, "float8_e5m2", None),
    getattr(torch, "float8_e4m3fnuz", None),
    getattr(torch, "float8_e5m2fnuz", None),
    FP8_E8M0FNU,
]


def _available_dtypes(dtypes):
    return [dtype for dtype in dtypes if dtype is not None]


@requires_mps
def test_decode_backend_selection_roundtrip():
    import fp4_fp8_for_torch_mps

    fp4_fp8_for_torch_mps.set_decode_backend("arithmetic")
    assert fp4_fp8_for_torch_mps.get_decode_backend() == "arithmetic"

    fp4_fp8_for_torch_mps.set_decode_backend("lut")
    assert fp4_fp8_for_torch_mps.get_decode_backend() == "lut"

    fp4_fp8_for_torch_mps.set_decode_backend("auto")
    assert fp4_fp8_for_torch_mps.get_decode_backend() == "auto"

    with pytest.raises(ValueError):
        fp4_fp8_for_torch_mps.set_decode_backend("invalid")

    fp4_fp8_for_torch_mps.set_decode_backend("lut")


@requires_mps
@pytest.mark.parametrize(
    "decode_fn",
    ["fp8_dequantize_arithmetic"],
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
    ["fp4_dequantize_arithmetic"],
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
def test_fastest_fp8_dtype_defaults(monkeypatch):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    calls = []
    for name in (
        "fp8_encode_arithmetic_e4m3fn",
        "fp8_encode_arithmetic_e5m2",
        "fp8_encode_arithmetic_e4m3fnuz",
        "fp8_encode_lut_e5m2fnuz",
        "fp8_encode_arithmetic_e8m0fnu",
        "fp8_dequantize_lut_e4m3fn",
        "fp8_dequantize_lut_e5m2",
        "fp8_dequantize_lut_e4m3fnuz",
        "fp8_dequantize_lut_e5m2fnuz",
        "fp8_dequantize_arithmetic_e8m0fnu",
    ):
        monkeypatch.setattr(fp8_mps_native, name, lambda *args, _name=name, **kwargs: calls.append(_name) or args[0])

    x = torch.empty(1, device="mps")
    raw = torch.empty(1, dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    fp8_mps_native.fp8_encode_fastest_dtype(x, torch.float8_e4m3fn)
    fp8_mps_native.fp8_encode_fastest_dtype(x, torch.float8_e5m2)
    fp8_mps_native.fp8_encode_fastest_dtype(x, torch.float8_e4m3fnuz)
    fp8_mps_native.fp8_encode_fastest_dtype(x, torch.float8_e5m2fnuz)
    fp8_mps_native.fp8_encode_fastest_dtype(x, torch.float8_e8m0fnu)
    fp8_mps_native.fp8_dequantize_fastest_dtype(raw, scale, torch.float8_e4m3fn)
    fp8_mps_native.fp8_dequantize_fastest_dtype(raw, scale, torch.float8_e5m2)
    fp8_mps_native.fp8_dequantize_fastest_dtype(raw, scale, torch.float8_e4m3fnuz)
    fp8_mps_native.fp8_dequantize_fastest_dtype(raw, scale, torch.float8_e5m2fnuz)
    fp8_mps_native.fp8_dequantize_fastest_dtype(raw, scale, torch.float8_e8m0fnu)

    assert calls == [
        "fp8_encode_arithmetic_e4m3fn",
        "fp8_encode_arithmetic_e5m2",
        "fp8_encode_arithmetic_e4m3fnuz",
        "fp8_encode_lut_e5m2fnuz",
        "fp8_encode_arithmetic_e8m0fnu",
        "fp8_dequantize_lut_e4m3fn",
        "fp8_dequantize_lut_e5m2",
        "fp8_dequantize_lut_e4m3fnuz",
        "fp8_dequantize_lut_e5m2fnuz",
        "fp8_dequantize_arithmetic_e8m0fnu",
    ]


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
@pytest.mark.benchmark(group="fp8_e4m3fn_encode_lut_arithmetic_cpu")
def test_benchmark_fp8_e4m3fn_lut_encode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    x = torch.randn(REPRESENTATIVE_ELEMENTS, dtype=torch.float32, device="mps")

    def run():
        result = fp8_mps_native.fp8_encode_lut_e4m3fn(x)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp8_e4m3fn_encode_lut_arithmetic_cpu")
def test_benchmark_fp8_e4m3fn_arithmetic_encode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    x = torch.randn(REPRESENTATIVE_ELEMENTS, dtype=torch.float32, device="mps")

    def run():
        result = fp8_mps_native.fp8_encode(x)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp8_e4m3fn_encode_lut_arithmetic_cpu")
def test_benchmark_fp8_e4m3fn_cpu_roundtrip_encode(benchmark):
    x = torch.randn(REPRESENTATIVE_ELEMENTS, dtype=torch.float32)

    def run():
        result = x.to(torch.float8_e4m3fn).view(torch.uint8).to("mps")
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp8_e4m3fn_decode_lut_arithmetic_cpu")
def test_benchmark_fp8_e4m3fn_lut_decode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    def run():
        result = fp8_mps_native.fp8_dequantize_lut(raw, scale)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp8_e4m3fn_decode_lut_arithmetic_cpu")
def test_benchmark_fp8_e4m3fn_arithmetic_decode(benchmark):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")

    def run():
        result = fp8_mps_native.fp8_dequantize_arithmetic(raw, scale)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.benchmark(group="fp8_e4m3fn_decode_lut_arithmetic_cpu")
def test_benchmark_fp8_e4m3fn_cpu_roundtrip_decode(benchmark):
    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS,), dtype=torch.uint8).view(torch.float8_e4m3fn)

    def run():
        result = raw.to(torch.float16).to("mps")
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
@pytest.mark.benchmark(group="fp8_encode_lut_arithmetic_cpu")
def test_benchmark_fp8_lut_encode_dtype(benchmark, dtype):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    x = torch.randn(REPRESENTATIVE_ELEMENTS, dtype=torch.float32, device="mps")

    def run():
        result = fp8_mps_native.fp8_encode_lut_dtype(x, dtype)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
@pytest.mark.benchmark(group="fp8_encode_lut_arithmetic_cpu")
def test_benchmark_fp8_arithmetic_encode_dtype(benchmark, dtype):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    x = torch.randn(REPRESENTATIVE_ELEMENTS, dtype=torch.float32, device="mps")

    def run():
        result = fp8_mps_native.fp8_encode_arithmetic_dtype(x, dtype)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
@pytest.mark.benchmark(group="fp8_encode_lut_arithmetic_cpu")
def test_benchmark_fp8_cpu_roundtrip_encode_dtype(benchmark, dtype):
    x = torch.randn(REPRESENTATIVE_ELEMENTS, dtype=torch.float32)

    def run():
        result = x.to(dtype).view(torch.uint8).to("mps")
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
@pytest.mark.benchmark(group="fp8_decode_lut_arithmetic_cpu")
def test_benchmark_fp8_lut_decode_dtype(benchmark, dtype):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")
    out_dtype = torch.float32 if dtype == FP8_E8M0FNU else torch.float16

    def run():
        result = fp8_mps_native.fp8_dequantize_lut_dtype(raw, scale, dtype, out_dtype=out_dtype)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
@pytest.mark.benchmark(group="fp8_decode_lut_arithmetic_cpu")
def test_benchmark_fp8_arithmetic_decode_dtype(benchmark, dtype):
    from fp4_fp8_for_torch_mps import fp8_mps_native

    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS,), dtype=torch.uint8, device="mps")
    scale = torch.tensor([1.0], device="mps")
    out_dtype = torch.float32 if dtype == FP8_E8M0FNU else torch.float16

    def run():
        result = fp8_mps_native.fp8_dequantize_arithmetic_dtype(raw, scale, dtype, out_dtype=out_dtype)
        torch.mps.synchronize()
        return result

    benchmark(run)


@requires_mps
@pytest.mark.parametrize("dtype", _available_dtypes(FP8_DTYPES))
@pytest.mark.benchmark(group="fp8_decode_lut_arithmetic_cpu")
def test_benchmark_fp8_cpu_roundtrip_decode_dtype(benchmark, dtype):
    raw = torch.randint(0, 256, (REPRESENTATIVE_ELEMENTS,), dtype=torch.uint8).view(dtype)
    out_dtype = torch.float32 if dtype == FP8_E8M0FNU else torch.float16

    def run():
        result = raw.to(out_dtype).to("mps")
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
