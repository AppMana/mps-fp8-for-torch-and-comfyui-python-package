import importlib.resources

import torch

_lib = None
_SHADER_SOURCE = None


def _load_shader_source():
    global _SHADER_SOURCE
    if _SHADER_SOURCE is not None:
        return _SHADER_SOURCE
    ref = importlib.resources.files("fp4_fp8_for_torch_mps.shaders").joinpath("fp8_matmul.metal")
    _SHADER_SOURCE = ref.read_text(encoding="utf-8")
    return _SHADER_SOURCE


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    _lib = torch.mps.compile_shader(_load_shader_source())
    return _lib


def fp8_scaled_mm(A, B, scale_a, scale_b):
    lib = _get_lib()

    assert A.dtype == torch.uint8 and B.dtype == torch.uint8
    assert A.is_contiguous() and B.is_contiguous()

    M, K = A.shape
    N = B.shape[0]
    assert B.shape[1] == K

    if A.device.type != "mps":
        A = A.to("mps")
    if B.device.type != "mps":
        B = B.to("mps")

    scale_a = scale_a.to(device="mps", dtype=torch.float32).contiguous()
    scale_b = scale_b.to(device="mps", dtype=torch.float32).contiguous()

    scale_mode = 0 if (scale_a.numel() == 1 and scale_b.numel() == 1) else 1

    C = torch.empty(M, N, dtype=torch.float32, device="mps")

    if M == 1:
        total_threads = N * 32
        lib.fp8_scaled_vecmat_kernel(
            A, B, C, scale_a, scale_b,
            N, K, scale_mode,
            threads=(total_threads,), group_size=(256,),
        )
    else:
        lib.fp8_scaled_matmul_kernel(
            A, B, C, scale_a, scale_b,
            M, N, K, scale_mode,
            threads=(N, M), group_size=(16, 16),
        )

    return C


def fp8_dequantize(input, scale):
    lib = _get_lib()

    if input.device.type != "mps":
        input = input.to("mps")

    count = input.numel()
    output = torch.empty(input.shape, dtype=torch.float16, device="mps")

    lib.fp8_to_half_kernel(
        input.contiguous().view(-1), output.view(-1),
        count,
        threads=(count,), group_size=(256,),
    )

    scale_val = scale.to(device="mps", dtype=torch.float16)
    output = output * scale_val
    return output


def fp8_encode(input):
    lib = _get_lib()

    inp = input.to(device="mps", dtype=torch.float32).contiguous()
    count = inp.numel()
    output = torch.empty(inp.shape, dtype=torch.uint8, device="mps")

    lib.float_to_fp8_kernel(
        inp.view(-1), output.view(-1),
        count,
        threads=(count,), group_size=(256,),
    )

    return output


def fp8_quantize(input):
    lib = _get_lib()

    inp = input.to(device="mps", dtype=torch.float32).contiguous()
    count = inp.numel()

    amax = inp.abs().max().item()
    scale = 448.0 / amax if amax > 0 else 1.0

    scaled = (inp * scale).contiguous()
    output = torch.empty(inp.shape, dtype=torch.uint8, device="mps")

    lib.float_to_fp8_kernel(
        scaled.view(-1), output.view(-1),
        count,
        threads=(count,), group_size=(256,),
    )

    inv_scale = torch.tensor([1.0 / scale], dtype=torch.float32, device="mps")
    return output, inv_scale


def fp8_scaled_mm_auto(A, B, scale_a, scale_b):
    if A.shape[0] <= 16:
        return fp8_scaled_mm(A, B, scale_a, scale_b)
    return fp8_scaled_mm_fast(A, B, scale_a, scale_b)


def fp8_scaled_mm_fast(A, B, scale_a, scale_b):
    lib = _get_lib()

    if A.device.type != "mps":
        A = A.to("mps")
    if B.device.type != "mps":
        B = B.to("mps")

    M, K = A.shape
    N = B.shape[0]

    A_f16 = torch.empty(M, K, dtype=torch.float16, device="mps")
    lib.fp8_to_half_kernel(
        A.contiguous().view(-1), A_f16.view(-1),
        A.numel(),
        threads=(A.numel(),), group_size=(256,),
    )

    B_f16 = torch.empty(N, K, dtype=torch.float16, device="mps")
    lib.fp8_to_half_kernel(
        B.contiguous().view(-1), B_f16.view(-1),
        B.numel(),
        threads=(B.numel(),), group_size=(256,),
    )

    sa = scale_a.to(device="mps", dtype=torch.float16)
    sb = scale_b.to(device="mps", dtype=torch.float16)
    A_f16 = A_f16 * sa
    B_f16 = B_f16 * sb

    return (A_f16 @ B_f16.T).float()


def fp4_dequantize(input, scale):
    lib = _get_lib()

    if input.device.type != "mps":
        input = input.to("mps")

    num_bytes = input.numel()
    output = torch.empty(num_bytes * 2, dtype=torch.float16, device="mps")

    lib.fp4x2_to_half_kernel(
        input.contiguous().view(-1), output,
        num_bytes,
        threads=(num_bytes,), group_size=(256,),
    )

    scale_val = scale.to(device="mps", dtype=torch.float16)
    output = output * scale_val
    return output


def fp4_encode(input):
    lib = _get_lib()

    inp = input.to(device="mps", dtype=torch.float32).contiguous()
    num_elements = inp.numel()
    assert num_elements % 2 == 0, "FP4 x2 requires even number of elements"
    num_bytes = num_elements // 2
    output = torch.empty(num_bytes, dtype=torch.uint8, device="mps")

    lib.float_to_fp4x2_kernel(
        inp.view(-1), output,
        num_bytes,
        threads=(num_bytes,), group_size=(256,),
    )

    return output


def fp4_quantize(input):
    lib = _get_lib()

    inp = input.to(device="mps", dtype=torch.float32).contiguous()
    num_elements = inp.numel()
    assert num_elements % 2 == 0, "FP4 x2 requires even number of elements"

    amax = inp.abs().max().item()
    scale = 6.0 / amax if amax > 0 else 1.0

    scaled = (inp * scale).contiguous()
    num_bytes = num_elements // 2
    output = torch.empty(num_bytes, dtype=torch.uint8, device="mps")

    lib.float_to_fp4x2_kernel(
        scaled.view(-1), output,
        num_bytes,
        threads=(num_bytes,), group_size=(256,),
    )

    inv_scale = torch.tensor([1.0 / scale], dtype=torch.float32, device="mps")
    return output, inv_scale
