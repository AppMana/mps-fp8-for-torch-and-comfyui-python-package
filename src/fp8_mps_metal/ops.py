import os
import torch
import torch.library

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

_aten_lib = torch.library.Library("aten", "IMPL")

_FP8_DTYPES = (
    getattr(torch, "float8_e4m3fn", None),
    getattr(torch, "float8_e5m2", None),
)


def _is_fp8(dtype):
    return dtype in _FP8_DTYPES


def _metal_scaled_mm(self, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False):
    from fp8_mps_metal.fp8_mps_native import fp8_scaled_mm_auto

    a = self.view(torch.uint8) if self.dtype != torch.uint8 else self
    b = mat2.view(torch.uint8) if mat2.dtype != torch.uint8 else mat2
    result = fp8_scaled_mm_auto(a, b.t().contiguous(), scale_a, scale_b)

    if bias is not None:
        result = result + bias
    if scale_result is not None:
        result = result * scale_result
    if out_dtype is not None:
        result = result.to(out_dtype)
    return result


_aten_lib.impl("_scaled_mm", _metal_scaled_mm, "MPS")


def _metal_to_copy(keyset, self, *, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, memory_format=None):
    target_dtype = dtype or self.dtype
    target_device = device or self.device
    target_is_mps = (target_device == "mps" or (isinstance(target_device, torch.device) and target_device.type == "mps"))
    source_is_fp8 = _is_fp8(self.dtype)
    target_is_fp8 = _is_fp8(target_dtype)

    if source_is_fp8 and target_is_mps and self.device.type != "mps":
        ks = keyset.remove(torch._C.DispatchKey.MPS)
        u8_mps = torch.ops.aten._to_copy.default.redispatch(
            ks, self.view(torch.uint8), dtype=torch.uint8, layout=layout,
            device=device, pin_memory=pin_memory, non_blocking=non_blocking,
            memory_format=memory_format,
        )
        result = u8_mps.view(self.dtype)
        if target_is_fp8 and target_dtype != self.dtype:
            result = result.view(torch.uint8).view(target_dtype)
        return result

    if target_is_mps and target_is_fp8 and not source_is_fp8:
        if self.device.type != "mps":
            ks = keyset.remove(torch._C.DispatchKey.MPS)
            on_mps = torch.ops.aten._to_copy.default.redispatch(
                ks, self, layout=layout, device=device, pin_memory=pin_memory,
                non_blocking=non_blocking, memory_format=memory_format,
            )
        else:
            on_mps = self
        return torch.ops.fp8_mps.encode(on_mps).view(target_dtype)

    if source_is_fp8 and self.device.type == "mps" and target_is_fp8:
        if target_dtype == self.dtype:
            return self.clone()
        return self.view(torch.uint8).view(target_dtype)

    if source_is_fp8 and self.device.type == "mps" and not target_is_fp8:
        dequantized = torch.ops.fp8_mps.dequantize(
            self.view(torch.uint8), torch.tensor([1.0], device="mps"),
        )
        if target_dtype != torch.float16:
            dequantized = dequantized.to(target_dtype)
        return dequantized

    ks = keyset.remove(torch._C.DispatchKey.MPS)
    return torch.ops.aten._to_copy.default.redispatch(
        ks, self, dtype=dtype, layout=layout, device=device,
        pin_memory=pin_memory, non_blocking=non_blocking, memory_format=memory_format,
    )


_aten_lib.impl("_to_copy", _metal_to_copy, "MPS", with_keyset=True, allow_override=True)


def _metal_copy_(keyset, self, src, non_blocking=False):
    source_is_fp8 = _is_fp8(src.dtype)
    dest_is_fp8 = _is_fp8(self.dtype)
    dest_is_mps = self.device.type == "mps"

    if source_is_fp8 and dest_is_fp8 and dest_is_mps:
        ks = keyset.remove(torch._C.DispatchKey.MPS)
        torch.ops.aten.copy_.default.redispatch(
            ks, self.view(torch.uint8), src.contiguous().view(torch.uint8), non_blocking,
        )
        return self

    if not source_is_fp8 and dest_is_fp8 and dest_is_mps:
        src_mps = src.to(device="mps") if src.device.type != "mps" else src
        encoded = torch.ops.fp8_mps.encode(src_mps)
        ks = keyset.remove(torch._C.DispatchKey.MPS)
        torch.ops.aten.copy_.default.redispatch(
            ks, self.view(torch.uint8), encoded, non_blocking,
        )
        return self

    ks = keyset.remove(torch._C.DispatchKey.MPS)
    return torch.ops.aten.copy_.default.redispatch(ks, self, src, non_blocking)


_aten_lib.impl("copy_", _metal_copy_, "MPS", with_keyset=True, allow_override=True)


@torch.library.custom_op("fp8_mps::scaled_mm", mutates_args=())
def fp8_scaled_mm(A: torch.Tensor, B: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    from fp8_mps_metal.fp8_mps_native import fp8_scaled_mm_auto
    return fp8_scaled_mm_auto(A, B, scale_a, scale_b)


@fp8_scaled_mm.register_fake
def _(A, B, scale_a, scale_b):
    return A.new_empty((A.shape[0], B.shape[0]), dtype=torch.float32)


@torch.library.custom_op("fp8_mps::encode", mutates_args=())
def fp8_encode(input: torch.Tensor) -> torch.Tensor:
    from fp8_mps_metal.fp8_mps_native import fp8_encode as _encode
    return _encode(input)


@fp8_encode.register_fake
def _(input):
    return input.new_empty(input.shape, dtype=torch.uint8)


@torch.library.custom_op("fp8_mps::quantize", mutates_args=())
def fp8_quantize(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from fp8_mps_metal.fp8_mps_native import fp8_quantize as _quantize
    return _quantize(input)


@fp8_quantize.register_fake
def _(input):
    return (input.new_empty(input.shape, dtype=torch.uint8), input.new_empty((1,), dtype=torch.float32))


@torch.library.custom_op("fp8_mps::dequantize", mutates_args=())
def fp8_dequantize(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from fp8_mps_metal.fp8_mps_native import fp8_dequantize as _dequantize
    return _dequantize(input, scale)


@fp8_dequantize.register_fake
def _(input, scale):
    return input.new_empty(input.shape, dtype=torch.float16)
