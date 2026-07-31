import logging
import os

import torch
import torch.library

logger = logging.getLogger(__name__)

#: Environment variable that disables all aten registrations when set to "1".
DISABLE_ENV_VAR = "FP8_MPS_DISABLE"

#: Attribute set on ``torch`` once registration has run, so a re-imported or
#: reloaded copy of this module never registers twice in the same process.
_SENTINEL_ATTR = "_fp4_fp8_for_torch_mps_registered"

_FP8_DTYPES = (
    getattr(torch, "float8_e4m3fn", None),
    getattr(torch, "float8_e5m2", None),
    getattr(torch, "float8_e4m3fnuz", None),
    getattr(torch, "float8_e5m2fnuz", None),
    getattr(torch, "float8_e8m0fnu", None),
)
_FP8_E4M3FN = getattr(torch, "float8_e4m3fn", None)
_FP8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)

_FP4_DTYPES = (
    getattr(torch, "float4_e2m1fn_x2", None),
)

_SUB_BYTE_DTYPES = _FP8_DTYPES + _FP4_DTYPES


def _is_fp8(dtype):
    return dtype in _FP8_DTYPES


def _is_fp4(dtype):
    return dtype in _FP4_DTYPES


def _is_sub_byte(dtype):
    return dtype in _SUB_BYTE_DTYPES


_UNIT_SCALE = None


def _unit_scale():
    """Lazily created, cached [1.0] scale tensor on MPS used by dequantize paths."""
    global _UNIT_SCALE
    if _UNIT_SCALE is None:
        _UNIT_SCALE = torch.tensor([1.0], device="mps")
    return _UNIT_SCALE


def _metal_scaled_mm(keyset, self, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False):
    # Only handle genuine FP8 inputs; anything else goes back to the original
    # kernel instead of being reinterpreted as raw bytes.
    if not (_is_fp8(self.dtype) and _is_fp8(mat2.dtype)):
        ks = keyset.remove(torch._C.DispatchKey.MPS)
        return torch.ops.aten._scaled_mm.default.redispatch(
            ks, self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum,
        )

    from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_scaled_mm_auto

    a = self.view(torch.uint8)
    b = mat2.view(torch.uint8)
    result = fp8_scaled_mm_auto(a, b.t().contiguous(), scale_a, scale_b)

    if bias is not None:
        result = result + bias
    if scale_result is not None:
        result = result * scale_result
    if out_dtype is not None:
        result = result.to(out_dtype)
    return result


def _metal_to_copy(keyset, self, *, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, memory_format=None):
    target_dtype = dtype if dtype is not None else self.dtype

    # Guarantee: this package cannot alter non-fp8/fp4 semantics. If neither
    # side of the cast is a sub-byte dtype, hand off to the native kernel
    # immediately.
    if not _is_sub_byte(self.dtype) and not _is_sub_byte(target_dtype):
        ks = keyset.remove(torch._C.DispatchKey.MPS)
        return torch.ops.aten._to_copy.default.redispatch(
            ks, self, dtype=dtype, layout=layout, device=device,
            pin_memory=pin_memory, non_blocking=non_blocking, memory_format=memory_format,
        )

    target_device = device or self.device
    target_is_mps = (target_device == "mps" or (isinstance(target_device, torch.device) and target_device.type == "mps"))
    source_sub_byte = _is_sub_byte(self.dtype)
    target_sub_byte = _is_sub_byte(target_dtype)
    source_is_fp4 = _is_fp4(self.dtype)
    target_is_fp4 = _is_fp4(target_dtype)

    # Sub-byte source -> MPS device transfer (raw bytes)
    if source_sub_byte and target_is_mps and self.device.type != "mps":
        ks = keyset.remove(torch._C.DispatchKey.MPS)
        u8_mps = torch.ops.aten._to_copy.default.redispatch(
            ks, self.view(torch.uint8), dtype=torch.uint8, layout=layout,
            device=device, pin_memory=pin_memory, non_blocking=non_blocking,
            memory_format=memory_format,
        )
        result = u8_mps.view(self.dtype)
        if target_sub_byte and target_dtype != self.dtype:
            result = result.view(torch.uint8).view(target_dtype)
        return result

    # Float -> FP4 on MPS (encode)
    if target_is_mps and target_is_fp4 and not source_sub_byte:
        if self.device.type != "mps":
            ks = keyset.remove(torch._C.DispatchKey.MPS)
            on_mps = torch.ops.aten._to_copy.default.redispatch(
                ks, self, layout=layout, device=device, pin_memory=pin_memory,
                non_blocking=non_blocking, memory_format=memory_format,
            )
        else:
            on_mps = self
        return torch.ops.fp8_mps.fp4_encode(on_mps).view(target_dtype)

    # Float -> FP8 on MPS (encode)
    if target_is_mps and _is_fp8(target_dtype) and not source_sub_byte:
        from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_encode_fastest_dtype

        if self.device.type != "mps":
            ks = keyset.remove(torch._C.DispatchKey.MPS)
            on_mps = torch.ops.aten._to_copy.default.redispatch(
                ks, self, layout=layout, device=device, pin_memory=pin_memory,
                non_blocking=non_blocking, memory_format=memory_format,
            )
        else:
            on_mps = self
        return fp8_encode_fastest_dtype(on_mps, target_dtype).view(target_dtype)

    # Sub-byte on MPS -> same type
    if source_sub_byte and self.device.type == "mps" and target_sub_byte:
        if target_dtype == self.dtype:
            return self.clone()
        return self.view(torch.uint8).view(target_dtype)

    # FP4 on MPS -> non-sub-byte (dequantize)
    if source_is_fp4 and self.device.type == "mps" and not target_sub_byte:
        dequantized = torch.ops.fp8_mps.fp4_dequantize(
            self.view(torch.uint8), _unit_scale(),
        )
        if target_dtype != torch.float16:
            dequantized = dequantized.to(target_dtype)
        return dequantized

    # FP8 on MPS -> non-sub-byte (dequantize)
    if _is_fp8(self.dtype) and self.device.type == "mps" and not target_sub_byte:
        from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_dequantize_fastest_dtype

        out_dtype = torch.float32 if target_dtype == torch.float32 or self.dtype == _FP8_E8M0FNU else torch.float16
        dequantized = fp8_dequantize_fastest_dtype(
            self.view(torch.uint8), _unit_scale(),
            self.dtype, out_dtype=out_dtype,
        )
        return dequantized if target_dtype == dequantized.dtype else dequantized.to(target_dtype)

    ks = keyset.remove(torch._C.DispatchKey.MPS)
    return torch.ops.aten._to_copy.default.redispatch(
        ks, self, dtype=dtype, layout=layout, device=device,
        pin_memory=pin_memory, non_blocking=non_blocking, memory_format=memory_format,
    )


def _metal_copy_(keyset, self, src, non_blocking=False):
    # Guarantee: this package cannot alter non-fp8/fp4 semantics. If neither
    # side of the copy is a sub-byte dtype, hand off to the native kernel
    # immediately.
    if not _is_sub_byte(src.dtype) and not _is_sub_byte(self.dtype):
        ks = keyset.remove(torch._C.DispatchKey.MPS)
        return torch.ops.aten.copy_.default.redispatch(ks, self, src, non_blocking)

    source_sub = _is_sub_byte(src.dtype)
    dest_sub = _is_sub_byte(self.dtype)
    dest_is_mps = self.device.type == "mps"

    # Sub-byte -> sub-byte on MPS (raw bytes)
    if source_sub and dest_sub and dest_is_mps:
        ks = keyset.remove(torch._C.DispatchKey.MPS)
        torch.ops.aten.copy_.default.redispatch(
            ks, self.view(torch.uint8), src.contiguous().view(torch.uint8), non_blocking,
        )
        return self

    # Float -> FP4 on MPS (encode)
    if not source_sub and _is_fp4(self.dtype) and dest_is_mps:
        src_mps = src.to(device="mps") if src.device.type != "mps" else src
        encoded = torch.ops.fp8_mps.fp4_encode(src_mps)
        ks = keyset.remove(torch._C.DispatchKey.MPS)
        torch.ops.aten.copy_.default.redispatch(
            ks, self.view(torch.uint8), encoded, non_blocking,
        )
        return self

    # Float -> FP8 on MPS (encode)
    if not source_sub and _is_fp8(self.dtype) and dest_is_mps:
        from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_encode_fastest_dtype

        src_mps = src.to(device="mps") if src.device.type != "mps" else src
        encoded = fp8_encode_fastest_dtype(src_mps, self.dtype)
        ks = keyset.remove(torch._C.DispatchKey.MPS)
        torch.ops.aten.copy_.default.redispatch(
            ks, self.view(torch.uint8), encoded, non_blocking,
        )
        return self

    ks = keyset.remove(torch._C.DispatchKey.MPS)
    return torch.ops.aten.copy_.default.redispatch(ks, self, src, non_blocking)


def _metal_embedding(keyset, weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    if _is_sub_byte(weight.dtype) and weight.device.type == "mps":
        return weight[indices]

    ks = keyset.remove(torch._C.DispatchKey.MPS)
    return torch.ops.aten.embedding.default.redispatch(
        ks, weight, indices, padding_idx, scale_grad_by_freq, sparse,
    )


_aten_lib = None


def _impl(name, fn, *, with_keyset=False, allow_override=False):
    try:
        _aten_lib.impl(name, fn, "MPS", with_keyset=with_keyset, allow_override=allow_override)
    except RuntimeError as exc:
        raise RuntimeError(
            f"fp4_fp8_for_torch_mps could not register aten::{name} for the MPS dispatch key "
            f"because another library has already registered a Python implementation for it. "
            f"Refusing to override it: two libraries interposing the same op silently corrupts "
            f"fp8/fp4 data. Disable one of them (for this package, set {DISABLE_ENV_VAR}=1). "
            f"Original error: {exc}"
        ) from exc


def _native_scaled_mm_works():
    """Probe whether torch ships a working native _scaled_mm for MPS."""
    if _FP8_E4M3FN is None:
        return False
    try:
        # 0x38 is 1.0 in float8_e4m3fn; built via view so the probe does not
        # depend on this package's cast overrides.
        a = torch.full((16, 16), 0x38, device="mps", dtype=torch.uint8).view(_FP8_E4M3FN)
        b = torch.full((16, 16), 0x38, device="mps", dtype=torch.uint8).view(_FP8_E4M3FN)
        scale = torch.ones((), device="mps", dtype=torch.float32)
        out = torch._scaled_mm(a, b.t(), scale, scale, out_dtype=torch.float16)
        return out.shape == (16, 16) and bool((out == 16.0).all().cpu())
    except Exception:
        return False


def register():
    """Register the aten MPS interposers. Idempotent; safe to call repeatedly.

    Skips registration (with a debug log) when MPS is unavailable, when
    FP8_MPS_DISABLE=1 is set, or when registration already ran in this process
    (even from a different copy of this module).
    """
    global _aten_lib

    if os.environ.get(DISABLE_ENV_VAR) == "1":
        logger.debug("fp4_fp8_for_torch_mps: %s=1, skipping aten registration", DISABLE_ENV_VAR)
        return
    if getattr(torch, _SENTINEL_ATTR, False):
        logger.debug("fp4_fp8_for_torch_mps: aten impls already registered, skipping")
        return
    if not torch.backends.mps.is_available():
        logger.debug("fp4_fp8_for_torch_mps: MPS not available, skipping aten registration")
        return

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    _aten_lib = torch.library.Library("aten", "IMPL")
    _impl("_to_copy", _metal_to_copy, with_keyset=True)
    _impl("copy_", _metal_copy_, with_keyset=True)
    _impl("embedding", _metal_embedding, with_keyset=True, allow_override=True)

    if _native_scaled_mm_works():
        logger.debug(
            "fp4_fp8_for_torch_mps: torch has a working native _scaled_mm on MPS; "
            "using it instead of registering the override"
        )
    else:
        _impl("_scaled_mm", _metal_scaled_mm, with_keyset=True)

    setattr(torch, _SENTINEL_ATTR, True)


@torch.library.custom_op("fp8_mps::scaled_mm", mutates_args=())
def fp8_scaled_mm(A: torch.Tensor, B: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_scaled_mm_auto
    return fp8_scaled_mm_auto(A, B, scale_a, scale_b)


@fp8_scaled_mm.register_fake
def _(A, B, scale_a, scale_b):
    return A.new_empty((A.shape[0], B.shape[0]), dtype=torch.float32)


@torch.library.custom_op("fp8_mps::encode", mutates_args=())
def fp8_encode(input: torch.Tensor) -> torch.Tensor:
    from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_encode as _encode
    return _encode(input)


@fp8_encode.register_fake
def _(input):
    return input.new_empty(input.shape, dtype=torch.uint8)


@torch.library.custom_op("fp8_mps::quantize", mutates_args=())
def fp8_quantize(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_quantize as _quantize
    return _quantize(input)


@fp8_quantize.register_fake
def _(input):
    return (input.new_empty(input.shape, dtype=torch.uint8), input.new_empty((1,), dtype=torch.float32))


@torch.library.custom_op("fp8_mps::dequantize", mutates_args=())
def fp8_dequantize(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from fp4_fp8_for_torch_mps.fp8_mps_native import fp8_dequantize as _dequantize
    return _dequantize(input, scale)


@fp8_dequantize.register_fake
def _(input, scale):
    return input.new_empty(input.shape, dtype=torch.float16)


@torch.library.custom_op("fp8_mps::fp4_encode", mutates_args=())
def fp4_encode(input: torch.Tensor) -> torch.Tensor:
    from fp4_fp8_for_torch_mps.fp8_mps_native import fp4_encode as _encode
    return _encode(input)


@fp4_encode.register_fake
def _(input):
    return input.new_empty(input.numel() // 2, dtype=torch.uint8)


@torch.library.custom_op("fp8_mps::fp4_quantize", mutates_args=())
def fp4_quantize(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from fp4_fp8_for_torch_mps.fp8_mps_native import fp4_quantize as _quantize
    return _quantize(input)


@fp4_quantize.register_fake
def _(input):
    return (input.new_empty(input.numel() // 2, dtype=torch.uint8), input.new_empty((1,), dtype=torch.float32))


@torch.library.custom_op("fp8_mps::fp4_dequantize", mutates_args=())
def fp4_dequantize(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from fp4_fp8_for_torch_mps.fp8_mps_native import fp4_dequantize as _dequantize
    return _dequantize(input, scale)


@fp4_dequantize.register_fake
def _(input, scale):
    return input.new_empty(input.numel() * 2, dtype=torch.float16)


register()
