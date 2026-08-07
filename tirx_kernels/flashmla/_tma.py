"""Shared TMA copy_async config for the sparse FlashMLA phase1 kernels."""

from __future__ import annotations

from typing import Any

from tvm.script import tirx as T


def leader_mbar(bar_ptr: Any) -> Any:
    # cta_group::2 completion routes to the CTA the mbar names; map to the pair
    # leader (rank 0) so both CTAs' issues aggregate on one barrier.
    mapped = T.alloc_local([1], "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], bar_ptr, T.uint32(0)))
    return T.reinterpret("handle", mapped[0])


def tma_config(
    *,
    dispatch: str = "tma_auto",
    mbar: Any = None,
    cta_group: int | None = None,
    cta_mask: Any = None,
    cache_hint: Any = None,
    mbarrier_addr: Any = None,
    gather4: list[Any] | None = None,
    src_selector: list[tuple[Any, Any]] | None = None,
    prefetch_tensormap: bool = True,
    tensormap_l2_promotion: str = "L2::256B",
) -> dict[str, Any]:
    """Return the FlashMLA-standard ``Tx.copy_async`` TMA config dict.

    Every TMA copy in the phase1 kernels prefetches its tensormap and asks
    for ``L2::256B`` promotion; the remaining knobs are per-site:

    - ``mbar``: completion mbarrier pointer (loads; stores omit it).
    - ``cta_group`` / ``cta_mask``: tcgen05 CTA-pair scope of the copy.
    - ``cache_hint``: ``"evict_first"`` or a ``T.uint64`` encoded L2 hint.
    - ``mbarrier_addr``: mbar addressing switch for the gather path.
    - ``gather4``: exactly four row coordinates for ``tma_explicit``.
    - ``src_selector``: first-true alternate global views for ``tma_explicit``.

    ``None`` means "leave the knob out of the config" so dispatch defaults
    apply; a misspelled keyword fails loudly instead of passing through.
    """
    cfg: dict[str, Any] = {
        "dispatch": dispatch,
        "prefetch_tensormap": prefetch_tensormap,
        "tensormap_l2_promotion": tensormap_l2_promotion,
    }
    for key, value in (
        ("mbar", mbar),
        ("cta_group", cta_group),
        ("cta_mask", cta_mask),
        ("cache_hint", cache_hint),
        ("mbarrier_addr", mbarrier_addr),
        ("gather4", gather4),
        ("src_selector", src_selector),
    ):
        if value is not None:
            cfg[key] = value
    return cfg
