"""torch.backends autoload shim.

This lives outside the fp4_fp8_for_torch_mps package on purpose: torch imports
the entry point's module before calling it, and importing a submodule of the
package would run the package __init__ (and its aten registrations) during
``import torch`` with no way to guard it. Keeping the shim top-level and
dependency-free means loading the entry point can never fail, and everything
the package does at import time happens inside the try/except below.
"""

import logging
import os


def _autoload():
    """torch.backends entry point. Must never break ``import torch``."""
    try:
        if os.environ.get("FP8_MPS_DISABLE") == "1":
            logging.getLogger("fp4_fp8_for_torch_mps").debug(
                "FP8_MPS_DISABLE=1, skipping fp4_fp8_for_torch_mps autoload"
            )
            return
        import fp4_fp8_for_torch_mps  # noqa: F401
    except Exception:
        logging.getLogger("fp4_fp8_for_torch_mps").exception(
            "fp4_fp8_for_torch_mps failed to load; fp8/fp4 support for MPS is disabled"
        )
