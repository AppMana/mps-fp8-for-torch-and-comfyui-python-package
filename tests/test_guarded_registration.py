"""Good-citizen behavior: guarded registration, kill switch, native-cast fidelity, coexistence."""

import os
import subprocess
import sys

import pytest
import torch

from conftest import requires_mps

COMFY_KITCHEN_PATH = os.path.expanduser("~/Documents/forks-comfy-kitchen-m1-m4")

OUR_ATEN_MPS_KEYS = (
    "aten/_to_copy/MPS",
    "aten/copy_/MPS",
    "aten/embedding/MPS",
    "aten/_scaled_mm/MPS",
)


def _run_python(code, extra_env=None):
    env = dict(os.environ)
    env.pop("FP8_MPS_DISABLE", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=300,
    )


# Deterministic inputs shared between this process and reference subprocesses.
CAST_INPUTS = """
import torch

def int8_input():
    return torch.arange(-128, 128, dtype=torch.int8).repeat(33)

def fp16_input():
    base = torch.linspace(-120.0, 120.0, 4096, dtype=torch.float32)
    frac = torch.tensor([0.0, 0.25, 0.5, -0.75, 1.5, -2.25, 100.5, -100.5], dtype=torch.float32)
    return torch.cat([base, frac]).to(torch.float16)
"""


class TestRegistrationIdempotence:

    def test_import_twice_and_register_repeatedly(self):
        import importlib

        import fp4_fp8_for_torch_mps
        importlib.import_module("fp4_fp8_for_torch_mps")
        fp4_fp8_for_torch_mps.register()
        fp4_fp8_for_torch_mps.register()

        from fp4_fp8_for_torch_mps_autoload import _autoload
        _autoload()
        _autoload()

    @requires_mps
    def test_aten_impls_registered_once(self):
        import fp4_fp8_for_torch_mps  # noqa: F401

        registered = [k for k in torch.library._impls if k in OUR_ATEN_MPS_KEYS]
        assert "aten/_to_copy/MPS" in registered
        assert "aten/copy_/MPS" in registered
        assert torch._fp4_fp8_for_torch_mps_registered is True


class TestDisableEnvVar:

    def test_disable_skips_all_aten_registration(self):
        result = _run_python(
            CAST_INPUTS + """
import torch
import torch.library

assert "fp4_fp8_for_torch_mps" not in __import__("sys").modules, "autoload should be skipped"

# Explicit import must also honor the kill switch.
import fp4_fp8_for_torch_mps

for key in (%r):
    assert key not in torch.library._impls, f"{key} registered despite FP8_MPS_DISABLE=1"
assert not getattr(torch, "_fp4_fp8_for_torch_mps_registered", False)

if torch.backends.mps.is_available():
    src = int8_input().to("mps")
    assert torch.equal(src.to(torch.float16).cpu(), int8_input().to(torch.float16))
print("OK")
""" % (OUR_ATEN_MPS_KEYS,),
            extra_env={"FP8_MPS_DISABLE": "1"},
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout


@requires_mps
class TestNativeCastFidelity:
    """int8<->fp16 casts on MPS must be bitwise-identical with and without the package."""

    REFERENCE_CODE = CAST_INPUTS + """
import torch
assert "fp4_fp8_for_torch_mps" not in __import__("sys").modules

i8 = int8_input().to("mps")
f16 = fp16_input().to("mps")
print("int8_to_fp16", i8.to(torch.float16).cpu().view(torch.uint8).numpy().tobytes().hex())
print("fp16_to_int8", f16.to(torch.int8).cpu().view(torch.uint8).numpy().tobytes().hex())
"""

    @pytest.fixture(scope="class")
    def reference(self):
        result = _run_python(self.REFERENCE_CODE, extra_env={"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
        assert result.returncode == 0, result.stderr
        out = dict(
            line.split(" ", 1)
            for line in result.stdout.strip().splitlines()
            if line.startswith(("int8_to_fp16 ", "fp16_to_int8 "))
        )
        assert set(out) == {"int8_to_fp16", "fp16_to_int8"}, result.stdout
        return out

    def _inputs(self):
        namespace = {}
        exec(compile(CAST_INPUTS, "<cast inputs>", "exec"), namespace)
        return namespace["int8_input"](), namespace["fp16_input"]()

    def test_int8_to_fp16_bitwise_identical(self, reference):
        import fp4_fp8_for_torch_mps  # noqa: F401

        i8, _ = self._inputs()
        ours = i8.to("mps").to(torch.float16).cpu().view(torch.uint8).numpy().tobytes().hex()
        assert ours == reference["int8_to_fp16"]

    def test_fp16_to_int8_bitwise_identical(self, reference):
        import fp4_fp8_for_torch_mps  # noqa: F401

        _, f16 = self._inputs()
        ours = f16.to("mps").to(torch.int8).cpu().view(torch.uint8).numpy().tobytes().hex()
        assert ours == reference["fp16_to_int8"]


@requires_mps
class TestFp8RoundTrip:

    def test_e4m3_encode_decode_matches_cpu(self):
        import fp4_fp8_for_torch_mps  # noqa: F401

        x = torch.tensor([0.0, 1.0, -1.0, 0.5, 10.0, -10.0, 100.0, 400.0, -448.0])
        cpu_bytes = x.to(torch.float8_e4m3fn).view(torch.uint8)
        mps_enc = x.to("mps").to(torch.float8_e4m3fn)
        assert torch.equal(mps_enc.view(torch.uint8).cpu(), cpu_bytes)

        decoded = mps_enc.to(torch.float32).cpu()
        cpu_decoded = x.to(torch.float8_e4m3fn).to(torch.float32)
        assert torch.equal(decoded, cpu_decoded)


class TestConflictIsLoud:

    def test_existing_python_impl_raises_clear_error(self):
        result = _run_python(
            """
import torch
import torch.library

lib = torch.library.Library("aten", "IMPL")


def other_to_copy(self, **kwargs):
    raise AssertionError("should never run")


lib.impl("_to_copy", other_to_copy, "MPS")

try:
    import fp4_fp8_for_torch_mps
except RuntimeError as exc:
    message = str(exc)
    assert "_to_copy" in message, message
    assert "fp4_fp8_for_torch_mps" in message, message
    print("CONFLICT_RAISED")
else:
    print("NO_ERROR")
""",
            extra_env={"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        )
        assert result.returncode == 0, result.stderr
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available; registration is skipped so no conflict occurs")
        assert "CONFLICT_RAISED" in result.stdout, result.stdout

    def test_autoload_survives_registration_failure(self):
        """The entry point must not break `import torch` even when registration raises."""
        result = _run_python(
            """
import torch
import torch.library

lib = torch.library.Library("aten", "IMPL")
lib.impl("_to_copy", lambda self, **kwargs: None, "MPS")

from fp4_fp8_for_torch_mps_autoload import _autoload
_autoload()
print("TORCH_IMPORT_SURVIVED")
""",
            extra_env={"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        )
        assert result.returncode == 0, result.stderr
        assert "TORCH_IMPORT_SURVIVED" in result.stdout


class TestComfyKitchenCoexistence:

    @pytest.fixture(scope="class")
    def comfy_kitchen(self):
        if not os.path.isdir(COMFY_KITCHEN_PATH):
            pytest.skip(f"{COMFY_KITCHEN_PATH} not present")
        if COMFY_KITCHEN_PATH not in sys.path:
            sys.path.insert(0, COMFY_KITCHEN_PATH)
        import fp4_fp8_for_torch_mps  # noqa: F401
        comfy_kitchen = pytest.importorskip("comfy_kitchen")
        return comfy_kitchen

    def test_eager_int8_linear_on_cpu_matches_reference(self, comfy_kitchen):
        from comfy_kitchen.backends.eager.quantization import int8_linear, quantize_int8_rowwise

        torch.manual_seed(0)
        x = torch.randn(8, 64, dtype=torch.float32)
        weight_f = torch.randn(16, 64, dtype=torch.float32)
        weight_scale = weight_f.abs().max() / 127.0
        weight = torch.clamp(torch.round(weight_f / weight_scale), -127, 127).to(torch.int8)
        bias = torch.randn(16, dtype=torch.float32)

        out = int8_linear(x, weight, weight_scale, bias=bias, out_dtype=torch.float32)

        x_8, x_scale = quantize_int8_rowwise(x)
        ref = (x_8.float() @ weight.float().T) * (x_scale.float().reshape(-1, 1) * weight_scale.float())
        ref = ref + bias

        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4), (out - ref).abs().max()

    @requires_mps
    def test_mps_int8_matmul_via_fp16_convert_unaffected(self, comfy_kitchen):
        torch.manual_seed(1)
        # Small magnitudes: |sum| <= 32 * 6 * 6 = 1152 < 2048, so every partial
        # sum is exactly representable in fp16 and the result is exact.
        a = torch.randint(-6, 7, (16, 32), dtype=torch.int8)
        b = torch.randint(-6, 7, (32, 8), dtype=torch.int8)

        a_h = a.to("mps").to(torch.float16)
        b_h = b.to("mps").to(torch.float16)

        # The int8 -> fp16 conversion must be untouched by this package.
        assert torch.equal(a_h.cpu(), a.to(torch.float16))
        assert torch.equal(b_h.cpu(), b.to(torch.float16))

        out = (a_h @ b_h).cpu().float()
        ref = a.float() @ b.float()
        assert torch.equal(out, ref), (out - ref).abs().max()
