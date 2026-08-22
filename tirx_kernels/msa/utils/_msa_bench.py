# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Import and launch helpers for the MSA sparse-attention reference.

MSA ships as the ``fmha_sm100`` package, but its CuTeDSL sources import their
siblings as ``src.*`` (``from src.common import ...``), so ``python/`` and
``python/fmha_sm100/cute`` both have to be importable -- the same two entries
MSA's own benchmarks add (``benchmarks/bench_sparse_attention_ops.py``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MSA_MODULE = None

# Where the checkout may live, in priority order: an explicit override, the
# path `scripts/install_reference_dependencies.py` clones into, and the
# workspace's own kernel-libs checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANDIDATE_ROOTS = (_REPO_ROOT / ".reference-deps" / "msa", Path.home() / "kernel-libs" / "msa")


def msa_root() -> Path:
    """Return the MSA checkout root, or raise with the paths that were tried."""
    override = os.environ.get("MSA_PATH")
    candidates = (Path(override),) if override else _CANDIDATE_ROOTS
    for root in candidates:
        if (root / "python" / "fmha_sm100").is_dir():
            return root
    tried = ", ".join(str(path) for path in candidates)
    raise ImportError(f"no MSA checkout found (set MSA_PATH); tried: {tried}")


def cutedsl_paths() -> list[str]:
    """Import roots for the CuTe-DSL build MSA's sources need, or an empty list.

    MSA's CuTe-DSL sources are written against ``cutlass.cute``. CuTe-DSL 4.6
    renamed that package to ``iket``, so 4.6 releases cannot run them at all,
    and the 4.6.0 development build that still carries ``cutlass.cute`` fails
    NVVM serialization on the forward kernel's gather4 Q specializations
    (``qhead_per_kv`` 1, 2 and 4) -- the reference simply will not compile for
    a third of this kernel's dispatch domain. 4.5.3 compiles all of them, and
    ``quack-kernels`` has to match it: 0.5.1 and newer import
    ``cutlass._mlir_helpers``, which 4.5 does not have.

    Install the pair with::

        python -m pip install --no-deps --target .reference-deps/msa-cutedsl \\
          nvidia-cutlass-dsl==4.5.3 nvidia-cutlass-dsl-libs-base==4.5.3 \\
          nvidia-cutlass-dsl-libs-cu13==4.5.3 quack-kernels==0.5.0

    ``MSA_CUTEDSL_PATH`` overrides the location. When neither the override nor
    the default prefix exists, the ambient install is used unchanged.
    """
    override = os.environ.get("MSA_CUTEDSL_PATH")
    prefix = Path(override) if override else _REPO_ROOT / ".reference-deps" / "msa-cutedsl"
    packages = prefix / "nvidia_cutlass_dsl" / "python_packages"
    if not packages.is_dir():
        return []
    return [str(prefix), str(packages)]


# The import roots the MSA reference pulls in, and the only ones this recovery
# is allowed to touch. `tirx_kernels` and `tvm` are excluded above because
# dropping them mid-process would unload the module doing the dropping.
# Matched as prefixes, so a subtree can be named without its parent.
#
# Only the STRUCTURAL check below may act on these -- drop a subtree whose
# submodule is present in `sys.modules` but unreachable from its own parent.
# Purging a root wholesale on retry was tried and reverted twice: `cutlass`
# carries native nanobind extensions whose types register process-globally, so
# re-importing aborts the interpreter ("refusing to add duplicate key ERROR",
# SIGABRT), and `torch._dynamo` re-registers its PGO mega-cache artifact and
# fails every subsequent compile ("Artifact of type=pgo already registered").
# A rare missed recovery costs one workload, which the suite retries; a bad
# purge costs the whole process.
# `torch._dynamo` is listed rather than `torch`: the failure mode is the same
# (a retry finds it partially initialized, here without `utils`), but purging
# all of `torch` mid-process would be both expensive and wrong -- plenty of
# torch submodules are legitimately absent from their parent until imported,
# and torch owns live CUDA state this recovery must not disturb.
_REFERENCE_ROOTS = ("cutlass", "quack", "sympy", "src", "fmha_sm100", "torch._dynamo")

# The subset of the roots above that is safe to drop and import again inside a
# live process: pure Python, no native extension module underneath. `cutlass` is
# NOT here -- its `_mlir_libs` extension registers nanobind types on first init,
# and a second init aborts the process with "refusing to add duplicate key". Nor
# is `torch._dynamo`, which holds compiled-code caches other live objects point
# at.
_RETRY_DROP_ROOTS = ("quack", "sympy", "src", "fmha_sm100")

# Reference-builder entries so far in this process; >1 means a retry.
_ensure_calls = 0


def _drop_interrupted_imports() -> None:
    """Forget modules whose import was cut off partway through.

    The bench suite stops a worker mid-flight when it sees another process on
    the GPU, then retries the measurement inside that same worker. If the stop
    lands inside one of the reference's heavy imports -- ``quack.__init__``
    pulls in CuTe-DSL, and CuTe-DSL pulls in ``sympy`` -- the package stays in
    ``sys.modules`` with its body only partly executed. The retry then reads a
    missing attribute off it (``quack`` with no ``copy_utils``, ``sympy`` with
    no ``core``) instead of importing it again, and the workload is reported as
    a baseline error even though nothing is wrong with either implementation.

    A module still marked ``_initializing`` here is stale by construction: this
    runs from the reference builder, not from inside anyone's import.

    That flag does not catch every case -- a package can be left incomplete with
    the flag already cleared -- so the packages that have actually been seen to
    break are probed for an attribute their own ``__init__`` defines, and their
    whole subtree is dropped when the probe fails. Re-importing costs seconds,
    but only on a retry, and it happens outside the timed region.

    Neither sweep is sufficient on its own. Both need evidence left behind in
    ``sys.modules``: the flag needs the module to still be marked initializing,
    the invariant needs a submodule that made it into ``sys.modules`` without
    being bound on its parent. An import cut *before* the submodule is
    registered at all leaves neither trace, and the parent then simply lacks the
    attribute -- which is how one matrix lost two of fourteen rows to ``quack``
    with no ``copy_utils`` and ``sympy`` with no ``printing``.

    So from the second call onward -- that is, on a retry -- the pure-Python
    reference roots are dropped outright rather than inspected. Only those:
    re-importing ``cutlass`` in a live process aborts it, because its MLIR
    extension registers nanobind types that may be registered exactly once. The
    first call must not drop anything either, because it is the only chance to
    observe a ``cutlass`` that some earlier import already bound outside the
    pin, which is what the caller checks for next; the pinned build and the
    ambient one are not interchangeable.
    """
    global _ensure_calls
    _ensure_calls += 1
    for name, module in list(sys.modules.items()):
        if name.startswith(("tirx_kernels", "tvm")):
            continue
        spec = getattr(module, "__spec__", None)
        if spec is not None and getattr(spec, "_initializing", False):
            del sys.modules[name]

    # A named-attribute probe per package cannot be made complete. The import can
    # be cut at ANY statement of any `__init__`, so whichever attribute the probe
    # names, there is a stop position that leaves that one bound and a later one
    # missing -- which is exactly how this failed twice in a row on two different
    # packages (`quack` without `copy_utils`, then `cutlass.base_dsl` without
    # `runtime`). Instead of extending the list a third time, check the invariant
    # that actually matters: a package left half-executed has submodules in
    # `sys.modules` that are not reachable as attributes of their own parent.
    def _in_scope(name: str) -> bool:
        return any(name == root or name.startswith(root + ".") for root in _REFERENCE_ROOTS)

    for name in [n for n in sys.modules if _in_scope(n)]:
        parent_name, _, leaf = name.rpartition(".")
        if not parent_name:
            continue
        parent = sys.modules.get(parent_name)
        if parent is not None and not hasattr(parent, leaf):
            for stale in [
                n for n in sys.modules if n == parent_name or n.startswith(parent_name + ".")
            ]:
                del sys.modules[stale]

    if _ensure_calls > 1:
        for name in [
            n
            for n in sys.modules
            if any(n == r or n.startswith(r + ".") for r in _RETRY_DROP_ROOTS)
        ]:
            del sys.modules[name]


def ensure_msa_importable() -> None:
    """Put MSA's two import roots -- and its pinned CuTe-DSL -- on ``sys.path``."""
    _drop_interrupted_imports()
    if "cutlass" in sys.modules:
        pinned = cutedsl_paths()
        search_path = getattr(sys.modules["cutlass"], "__path__", None)
        loaded = str(next(iter(search_path))) if search_path else None
        if pinned and loaded and not loaded.startswith(pinned[0]):
            raise RuntimeError(
                "cutlass was imported before the MSA reference pinned its CuTe-DSL "
                f"build; loaded from {loaded}, expected under {pinned[0]}"
            )
    for entry in reversed(cutedsl_paths()):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    root = msa_root()
    for entry in (str(root / "python" / "fmha_sm100" / "cute"), str(root / "python")):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def prepare_scheduler_module():
    """Import ``src.sm100.prepare_scheduler`` from the MSA checkout, once."""
    global _MSA_MODULE
    if _MSA_MODULE is not None:
        return _MSA_MODULE

    ensure_msa_importable()

    import src.sm100.prepare_scheduler as prepare_scheduler

    _MSA_MODULE = prepare_scheduler
    return _MSA_MODULE


def compiled_fwd_split_atomic(case: dict):
    """Compile (or fetch from MSA's AOT cache) the forward split-slot kernel.

    Returns the compiled callable behind ``prepare_sparse_fwd_schedule_and_split``'s
    NVTX range, so the timed closure covers the kernel launch alone -- not the
    host wrapper's shape validation, its ``split_counts.zero_()``, or the
    flat-schedule kernel it runs first.
    """
    module = prepare_scheduler_module()
    return module._get_sparse_prepare_fwd_split_atomic(
        case["k2q_row_ptr"],
        case["k2q_q_indices"],
        case["scheduler_metadata"],
        case["work_count"],
        case["k2q_qsplit_indices"],
        case["split_counts"],
        case["cu_seqlens_q"],
        case["work_capacity"],
        case["max_seqlen_q"],
        case["topk"],
    )


_FWD_CACHE: dict = {}


def compiled_sparse_atten_fwd(case: dict):
    """Compile (or fetch) ``SparseAttentionForwardSm100`` and return the launchable.

    This mirrors the compile step of ``_call_sparse_forward_sm100_csr_varlen``
    (interface.py:1714-1780) instead of calling that host entry, so the timed
    closure covers the kernel launch alone: the host entry would also validate
    shapes, copy ``O_partial`` through ``reshape(-1, head_dim).contiguous()``,
    build a gather4 tensor map, and -- when handed no prebuilt schedule -- run
    the two preparation kernels first.

    Every tensor is wrapped **before** ``cute.compile``. Wrapping one afterwards
    binds this process to a host ``cuTensorMapEncodeTiled`` path that unrelated
    kernels then fail on, long after the call that caused it.
    """
    ensure_msa_importable()

    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass import Float32, Int32
    from src.common.cute_dsl_utils import to_cute_tensor
    from src.common.tma_utils import create_q_gather4_tma_desc
    from src.sm100.fwd.atten_fwd import SparseAttentionForwardSm100

    qhead_per_kv = case["qhead_per_kv"]
    q_flat = case["q_flat"]
    o_partial_flat = case["o_partial_flat"]
    lse_temperature = case.get("lse_temperature_partial")
    # Paging is `page_table is not None` at the host entry (interface.py:1670),
    # and `seqused_k` is accepted only alongside it (interface.py:251-253).
    page_table = case.get("page_table")
    seqused_k = case.get("seqused_k")
    paged = page_table is not None
    if seqused_k is not None and not paged:
        raise ValueError("seqused_k is only supported together with page_table")
    gather4_desc = (
        create_q_gather4_tma_desc(q_flat, box_x=128 if q_flat.dtype == torch.float8_e4m3fn else 64)
        if qhead_per_kv in (1, 2, 4)
        else None
    )

    key = (
        "sparse_forward_sm100_csr_varlen",
        case["head_dim"],
        case["blk_kv"],
        qhead_per_kv,
        q_flat.dtype,
        case["k"].dtype,
        case["v"].dtype,
        case["qk_dtype"],
        case["pv_dtype"],
        o_partial_flat.dtype,
        bool(case["causal"]),
        paged,
        True,  # use_prepare_scheduler
        case["blk_kv"] if paged else None,  # page_size: upstream forces == blk_kv
        seqused_k is not None,
        lse_temperature is not None,
    )
    if key not in _FWD_CACHE:
        cutlass_dtype = {
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
            torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        }
        kernel = SparseAttentionForwardSm100(
            head_dim=case["head_dim"],
            qheadperkv=qhead_per_kv,
            n_block_size=case["blk_kv"],
            paged_kv=paged,
            page_size=case["blk_kv"] if paged else None,
            has_seqused_k=seqused_k is not None,
            causal=bool(case["causal"]),
            use_prepare_scheduler=True,
            qk_dtype=cutlass_dtype[case["qk_dtype"]],
            pv_dtype=cutlass_dtype[case["pv_dtype"]],
        )
        _FWD_CACHE[key] = cute.compile(
            kernel,
            to_cute_tensor(case["k"]),
            to_cute_tensor(case["v"]),
            to_cute_tensor(case["k2q_q_indices"]),
            to_cute_tensor(case["k2q_qsplit_indices"]),
            to_cute_tensor(case["k2q_row_ptr"]),
            to_cute_tensor(case["scheduler_metadata"]),
            to_cute_tensor(case["work_count"]),
            to_cute_tensor(o_partial_flat),
            to_cute_tensor(case["lse_partial"]),
            None if lse_temperature is None else to_cute_tensor(lse_temperature),
            to_cute_tensor(q_flat),
            None if gather4_desc is None else to_cute_tensor(gather4_desc),
            None if page_table is None else to_cute_tensor(page_table),
            None if seqused_k is None else to_cute_tensor(seqused_k),
            to_cute_tensor(case["cu_seqlens_q"]),
            to_cute_tensor(case["cu_seqlens_k"]),
            Float32(case["softmax_scale"]),
            Float32(case["lse_temperature_inv_scale"]),
            Int32(case["num_kv_blocks"]),
            Int32(case["head_kv"]),
            Int32(case["max_seqlen_q"]),
            Int32(case["work_capacity"]),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compiled = _FWD_CACHE[key]

    def launch() -> None:
        compiled(
            case["k"],
            case["v"],
            case["k2q_q_indices"],
            case["k2q_qsplit_indices"],
            case["k2q_row_ptr"],
            case["scheduler_metadata"],
            case["work_count"],
            o_partial_flat,
            case["lse_partial"],
            lse_temperature,
            q_flat,
            gather4_desc,
            page_table,
            seqused_k,
            case["cu_seqlens_q"],
            case["cu_seqlens_k"],
            case["softmax_scale"],
            case["lse_temperature_inv_scale"],
            case["num_kv_blocks"],
            case["head_kv"],
            case["max_seqlen_q"],
            case["work_capacity"],
        )

    return launch


_FWD_NVFP4_CACHE: dict = {}


def compiled_sparse_atten_nvfp4_kv(case: dict):
    """Compile (or fetch) ``SparseAttentionForwardNvfp4KvSm100`` and return the launchable.

    Mirrors the compile step of ``_call_sparse_forward_sm100_csr_varlen_nvfp4_kv``
    (interface.py:1840-2011) for the same reason the BF16/FP8 sibling adapter
    mirrors its own host entry: the timed closure has to cover the kernel launch
    alone.

    Two contract details the host entry hides. ``O_partial`` reaches the kernel
    as ``reshape(-1, head_dim)``, and the caller must pass a **view** of the
    buffer it later reads -- the host entry's ``.contiguous()`` would silently
    hand the kernel a detached copy. And the V tensor scale never reaches this
    kernel at all: the interface pins ``has_v_global_scale`` false and applies
    that scale once in the combine kernel (interface.py:1862-1866), so
    ``v_global_scale`` is accepted here only to keep the contract visible.

    Every tensor is wrapped **before** ``cute.compile``; wrapping one afterwards
    poisons this process's host ``cuTensorMapEncodeTiled`` path.
    """
    ensure_msa_importable()

    import cutlass.cute as cute
    import torch
    from cutlass import Float32, Int32
    from src.common.cute_dsl_utils import to_cute_tensor
    from src.common.tma_utils import create_q_gather4_tma_desc
    from src.sm100.fwd.atten_fwd_nvfp4_kv import SparseAttentionForwardNvfp4KvSm100

    qhead_per_kv = case["qhead_per_kv"]
    q_flat = case["q_flat"]
    o_partial_flat = case["o_partial_flat"]
    lse_temperature = case.get("lse_temperature_partial")
    k_global_scale = case.get("k_global_scale")
    page_table = case.get("page_table")
    seqused_k = case.get("seqused_k")
    paged = page_table is not None
    if seqused_k is not None and not paged:
        raise ValueError("seqused_k is only supported together with page_table")
    gather4_desc = (
        create_q_gather4_tma_desc(q_flat, box_x=128 if q_flat.dtype == torch.float8_e4m3fn else 64)
        if qhead_per_kv in (1, 2, 4)
        else None
    )

    key = (
        "sparse_forward_sm100_csr_varlen_nvfp4_kv",
        case["head_dim"],
        case["blk_kv"],
        qhead_per_kv,
        q_flat.dtype,
        o_partial_flat.dtype,
        bool(case["causal"]),
        paged,
        True,  # use_prepare_scheduler
        case["blk_kv"] if paged else None,  # page_size: upstream forces == blk_kv
        seqused_k is not None,
        lse_temperature is not None,
        True,  # fp8_pair_dequant: pinned, never read from the environment
        k_global_scale is not None,
        False,  # has_v_global_scale: applied in the combine kernel instead
    )
    if key not in _FWD_NVFP4_CACHE:
        kernel = SparseAttentionForwardNvfp4KvSm100(
            head_dim=case["head_dim"],
            qheadperkv=qhead_per_kv,
            n_block_size=case["blk_kv"],
            paged_kv=paged,
            page_size=case["blk_kv"] if paged else None,
            has_seqused_k=seqused_k is not None,
            causal=bool(case["causal"]),
            use_prepare_scheduler=True,
            fp8_pair_dequant=True,
            has_k_global_scale=k_global_scale is not None,
            has_v_global_scale=False,
        )
        _FWD_NVFP4_CACHE[key] = cute.compile(
            kernel,
            to_cute_tensor(case["k"]),
            to_cute_tensor(case["v"]),
            to_cute_tensor(case["k_scale_128x4"]),
            to_cute_tensor(case["v_scale_128x4"]),
            None if k_global_scale is None else to_cute_tensor(k_global_scale),
            None,  # v_global_scale
            to_cute_tensor(case["k2q_q_indices"]),
            to_cute_tensor(case["k2q_qsplit_indices"]),
            to_cute_tensor(case["k2q_row_ptr"]),
            to_cute_tensor(case["scheduler_metadata"]),
            to_cute_tensor(case["work_count"]),
            to_cute_tensor(o_partial_flat),
            to_cute_tensor(case["lse_partial"]),
            None if lse_temperature is None else to_cute_tensor(lse_temperature),
            to_cute_tensor(q_flat),
            None if gather4_desc is None else to_cute_tensor(gather4_desc),
            None if page_table is None else to_cute_tensor(page_table),
            None if seqused_k is None else to_cute_tensor(seqused_k),
            to_cute_tensor(case["cu_seqlens_q"]),
            to_cute_tensor(case["cu_seqlens_k"]),
            Float32(case["softmax_scale"]),
            Float32(case["lse_temperature_inv_scale"]),
            Int32(case["num_kv_blocks"]),
            Int32(case["head_kv"]),
            Int32(case["max_seqlen_q"]),
            Int32(case["work_capacity"]),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    compiled = _FWD_NVFP4_CACHE[key]

    def launch() -> None:
        compiled(
            case["k"],
            case["v"],
            case["k_scale_128x4"],
            case["v_scale_128x4"],
            k_global_scale,
            None,
            case["k2q_q_indices"],
            case["k2q_qsplit_indices"],
            case["k2q_row_ptr"],
            case["scheduler_metadata"],
            case["work_count"],
            o_partial_flat,
            case["lse_partial"],
            lse_temperature,
            q_flat,
            gather4_desc,
            page_table,
            seqused_k,
            case["cu_seqlens_q"],
            case["cu_seqlens_k"],
            case["softmax_scale"],
            case["lse_temperature_inv_scale"],
            case["num_kv_blocks"],
            case["head_kv"],
            case["max_seqlen_q"],
            case["work_capacity"],
        )

    return launch


def compiled_flat_schedule(case: dict):
    """Compile (or fetch from MSA's AOT cache) the flat-schedule kernel.

    Returns the compiled callable behind ``prepare_sparse_flat_schedule``'s NVTX
    range, so the timed closure covers the kernel launch alone rather than the
    host wrapper's allocation and schedule sizing.
    """
    module = prepare_scheduler_module()
    return module._get_sparse_prepare_flat_schedule(
        case["k2q_row_ptr"],
        case["cu_seqlens_k"],
        case["scheduler_metadata"],
        case["work_count"],
        case["target_q_per_cta"],
        case["work_capacity"],
        case["head_kv"],
        case["blk_kv"],
    )
