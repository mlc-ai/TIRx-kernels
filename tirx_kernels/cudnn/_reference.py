# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Import cuDNN Frontend reference kernels from the pinned source install.

The reference implementations the ``cudnn`` ports compare against live in the
``cudnn-frontend`` source checkout pinned in ``reference-dependencies.json``
(v1.28.0) and installed editable by ``scripts/install_reference_dependencies.py``
into ``.reference-deps/cudnn-frontend`` — replacing any released
``nvidia-cudnn-frontend`` wheel, which predates the ``cudnn.gemm`` /
``cudnn.linear_attention`` / ``cudnn.csa`` reorganization and carries older
revisions of the kernels it does ship.

Every reference — the FROST linear-attention kernels included — runs on the
ambient CuTe-DSL, whose version the lock pins (the FROST kernels need >= 4.7.0,
the first release that ships ``cutlass.experimental``).
"""

from __future__ import annotations

import importlib
import sys
from functools import cache

_INSTALL_HINT = "run `python scripts/install_reference_dependencies.py` first"


def _rebind_submodules(package):
    """Re-attach already-loaded submodules to a freshly re-executed package.

    A failed ``import cutlass`` drops ``cutlass`` from ``sys.modules`` but leaves
    every ``cutlass.*`` submodule behind. On the retry the package body re-runs,
    and its inner ``import cutlass._mlir`` finds that submodule already loaded,
    so the import system never binds it as an attribute of the new package
    object. The upstream kernel reaches for ``cutlass._mlir.dialects.math`` in
    its amax reduction and fails with an ``AttributeError`` that has nothing to
    do with the kernel under test.
    """
    prefix = package.__name__ + "."
    for name, module in list(sys.modules.items()):
        if not name.startswith(prefix) or module is None:
            continue
        child = name[len(prefix) :]
        if "." in child:
            continue
        if getattr(package, child, None) is not module:
            setattr(package, child, module)
    return package


def import_cutlass_reference():
    """Recover from CuTeDSL's non-idempotent generated builder imports."""
    try:
        return _rebind_submodules(importlib.import_module("cutlass"))
    except RuntimeError as exc:
        message = str(exc)
        if "Attribute builder for '" not in message or "is already registered" not in message:
            raise
        mlir_ir = sys.modules.get("cutlass._mlir.ir")
        if mlir_ir is None:
            raise
        register_attribute_builder = mlir_ir.register_attribute_builder

        def register_replacing_builder(kind, replace=False):
            del replace
            return register_attribute_builder(kind, replace=True)

        mlir_ir.register_attribute_builder = register_replacing_builder
        try:
            return _rebind_submodules(importlib.import_module("cutlass"))
        finally:
            mlir_ir.register_attribute_builder = register_attribute_builder


def from_dlpack_typed(tensor, *, assumed_align=16, leading_dim=None):
    """Wrap CUDA tensors whose FP8/packed-FP4 dtypes DLPack cannot export."""
    cutlass = import_cutlass_reference()
    import torch
    from cutlass.cute.runtime import from_dlpack

    unsupported = {
        torch.float4_e2m1fn_x2: cutlass.Float4E2M1FN,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float8_e5m2: cutlass.Float8E5M2,
        torch.float8_e8m0fnu: cutlass.Float8E8M0FNU,
    }
    element_type = unsupported.get(tensor.dtype)
    storage = tensor.view(torch.uint8) if element_type is not None else tensor
    wrapped = from_dlpack(storage, assumed_align=assumed_align)
    if leading_dim is not None:
        wrapped = wrapped.mark_layout_dynamic(leading_dim=leading_dim)
    if element_type is not None:
        wrapped.element_type = element_type
    return wrapped


@cache
def load_reference_module(name: str):
    """Import one cuDNN Frontend reference module from the pinned install."""
    import_cutlass_reference()
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name and (name == exc.name or name.startswith(exc.name + ".")):
            raise RuntimeError(
                f"cannot import the cuDNN Frontend reference {name!r}; {_INSTALL_HINT}"
            ) from exc
        # A dependency of the reference is missing, not the reference itself;
        # keep the original error rather than misattributing it to the install.
        raise
