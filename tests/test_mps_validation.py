"""MPS pattern validation tests for Apple Silicon.

Tests dtype support, basic operations, and MPS-specific quirks.
"""

import pytest
import torch

from conftest import requires_mps


@requires_mps
class TestDtypeSupport:
    """Verify that key dtypes work on MPS."""

    @pytest.mark.parametrize("dtype", [
        torch.float32, torch.float16, torch.bfloat16,
        torch.int8, torch.int16, torch.int32, torch.int64, torch.bool,
    ])
    def test_tensor_creation_and_add(self, dtype):
        """Create a tensor and perform a basic op on MPS."""
        t = torch.zeros(4, 4, dtype=dtype, device="mps")
        if dtype in (torch.float32, torch.float16, torch.bfloat16):
            r = t + 1.0
        else:
            r = t + 1
        assert r.device.type == "mps"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_matmul(self, dtype):
        """Matmul works for floating dtypes on MPS."""
        a = torch.ones(4, 4, dtype=dtype, device="mps")
        r = torch.matmul(a, a)
        assert r.shape == (4, 4)


@requires_mps
class TestMpsOperations:
    """Smoke-test operations known to have MPS quirks."""

    def test_topk(self):
        """torch.topk on MPS."""
        vals, idxs = torch.topk(torch.randn(1, 50000, device="mps"), 50)
        assert vals.shape == (1, 50)

    def test_layer_norm(self):
        """F.layer_norm on MPS."""
        import torch.nn.functional as F
        out = F.layer_norm(torch.randn(2, 128, 768, device="mps"), [768])
        assert out.shape == (2, 128, 768)

    def test_group_norm(self):
        """F.group_norm on MPS."""
        import torch.nn.functional as F
        out = F.group_norm(torch.randn(2, 32, 64, 64, device="mps"), 8)
        assert out.shape == (2, 32, 64, 64)

    def test_conv2d(self):
        """F.conv2d on MPS."""
        import torch.nn.functional as F
        out = F.conv2d(
            torch.randn(1, 3, 256, 256, device="mps"),
            torch.randn(64, 3, 3, 3, device="mps"),
        )
        assert out.shape[0] == 1

    def test_interpolate_bilinear(self):
        """F.interpolate bilinear on MPS."""
        import torch.nn.functional as F
        out = F.interpolate(
            torch.randn(1, 3, 64, 64, device="mps"),
            scale_factor=4,
            mode='bilinear',
        )
        assert out.shape == (1, 3, 256, 256)

    def test_scatter_add(self):
        """torch.scatter_add_ on MPS."""
        out = torch.zeros(10, device="mps")
        out.scatter_add_(
            0,
            torch.randint(0, 10, (100,), device="mps"),
            torch.randn(100, device="mps"),
        )
        assert out.shape == (10,)

    def test_multinomial(self):
        """torch.multinomial on MPS."""
        probs = torch.softmax(torch.randn(1, 100, device="mps"), dim=-1)
        samples = torch.multinomial(probs, 5)
        assert samples.shape == (1, 5)
