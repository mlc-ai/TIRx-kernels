# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""tirx-lite overrides for the in-tree tile schedulers and CLC helpers.

The backend classes remain the source of truth for construction, state, and
ordinary Python dispatch. tirx-lite only replaces their parser-inline expansion
points, spelling those bodies directly with the public ``txl`` primitives.
"""

from __future__ import annotations

import tirx_kernels.tirx_lite as txl
from tvm.backend.cuda.lang.tile_scheduler import (
    ClusterLaunchControlScheduler as _ClusterLaunchControlScheduler,
)
from tvm.backend.cuda.lang.tile_scheduler import (
    ClusterPersistentScheduler2D as _ClusterPersistentScheduler2D,
)
from tvm.backend.cuda.lang.tile_scheduler import (
    FlashAttentionLinearScheduler as _FlashAttentionLinearScheduler,
)
from tvm.backend.cuda.lang.tile_scheduler import (
    FlashAttentionLPTScheduler as _FlashAttentionLPTScheduler,
)

from .pipeline import Pipeline, PipelineState


def query_cancel_first_ctaid_x(first_ctaid_x, handle, *, use_ld_acquire=True):
    """Decode one CLC cancellation response into ``first_ctaid_x``."""
    response = txl.local_scalar("uint128", name="clc_response")
    canceled = txl.local_scalar(txl.u32, name="clc_canceled")
    suffix = ".acquire.cta" if use_ld_acquire else ""

    txl.ptx[f"ld{suffix}.shared.b128"](response, handle)
    txl.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, response)
    txl.assign(first_ctaid_x, txl.uint32(0xFFFFFFFF))
    txl.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
        first_ctaid_x, response, pred=canceled
    )
    txl.ptx.fence.proxy.async_.shared__cta()


class ClusterPersistentScheduler2D(_ClusterPersistentScheduler2D):
    """The backend scheduler with its group-major inline points traced by tirx-lite."""

    def __init__(
        self,
        prefix: str,
        num_m_tiles,
        num_n_tiles: int,
        num_clusters: int,
        l2_group_size: int = 8,
        cluster_m: int = 1,
        cluster_n: int = 1,
        serpentine: bool = False,
    ):
        if serpentine:
            raise ValueError(
                "txl.ClusterPersistentScheduler2D does not support serpentine scheduling"
            )
        super().__init__(
            prefix,
            num_m_tiles,
            num_n_tiles,
            num_clusters,
            l2_group_size,
            cluster_m,
            cluster_n,
            serpentine=False,
        )

    def update_current_m_n_idx(self, work_idx):
        cluster_m_offset = work_idx % self._cluster_m
        t = work_idx // self._cluster_m
        cluster_n_offset = t % self._cluster_n
        tile_linear = t // self._cluster_n

        def set_tile_coords(tile_row, tile_col):
            txl.assign(self.m_idx, tile_row * self._cluster_m + cluster_m_offset)
            txl.assign(self.n_idx, tile_col * self._cluster_n + cluster_n_offset)

        self._update_group_major(tile_linear, set_tile_coords)

    def _gm_emit_zero(self, set_tile_coords):
        set_tile_coords(0, 0)

    def _gm_emit_full_only(self, tile_linear, set_tile_coords):
        full_groups = self._FULL_GROUPS
        group_size = self._l2_group_size
        group_span = self._l2_group_size * self._N_TILE_COLS
        with txl.If((full_groups > 0) & (tile_linear < full_groups * group_span)):
            with txl.Then():
                group_id = tile_linear // group_span
                within_group = tile_linear % group_span
                tile_row = group_id * group_size + within_group % group_size
                tile_col = within_group // group_size
                set_tile_coords(tile_row, tile_col)
            with txl.Else():
                set_tile_coords(0, 0)

    def _gm_emit_tail_only(self, tile_linear, set_tile_coords):
        full_groups = self._FULL_GROUPS
        tail_rows = self._TAIL_ROWS
        group_size = self._l2_group_size
        group_span = self._l2_group_size * self._N_TILE_COLS
        with txl.If(tail_rows > 0):
            with txl.Then():
                rem = tile_linear - full_groups * group_span
                tile_row = full_groups * group_size + rem % tail_rows
                tile_col = rem // tail_rows
                set_tile_coords(tile_row, tile_col)
            with txl.Else():
                set_tile_coords(0, 0)

    def _gm_emit_full_and_tail(self, tile_linear, set_tile_coords):
        full_groups = self._FULL_GROUPS
        tail_rows = self._TAIL_ROWS
        group_size = self._l2_group_size
        group_span = self._l2_group_size * self._N_TILE_COLS
        with txl.If((full_groups > 0) & (tile_linear < full_groups * group_span)):
            with txl.Then():
                group_id = tile_linear // group_span
                within_group = tile_linear % group_span
                tile_row = group_id * group_size + within_group % group_size
                tile_col = within_group // group_size
                set_tile_coords(tile_row, tile_col)
            with txl.Else():
                with txl.If(tail_rows > 0):
                    with txl.Then():
                        rem = tile_linear - full_groups * group_span
                        tile_row = full_groups * group_size + rem % tail_rows
                        tile_col = rem // tail_rows
                        set_tile_coords(tile_row, tile_col)
                    with txl.Else():
                        set_tile_coords(0, 0)

    def init(self, cluster_id):
        txl.assign(self.linear_idx, cluster_id)
        txl.assign(self.tile_count, txl.int32(0))
        self.update_current_m_n_idx(cluster_id)

    def next_tile(self):
        txl.assign(self.linear_idx, self.linear_idx + self._num_clusters)
        txl.assign(self.tile_count, self.tile_count + txl.int32(1))
        self.update_current_m_n_idx(self.linear_idx)

    def next_tile_stride(self, stride: int):
        txl.assign(self.linear_idx, self.linear_idx + stride)
        txl.assign(self.tile_count, self.tile_count + txl.int32(1))
        self.update_current_m_n_idx(self.linear_idx)


class FlashAttentionLinearScheduler(_FlashAttentionLinearScheduler):
    """The backend linear scheduler with its three inline methods overridden."""

    def update_current_m_n_idx(self, linear_idx):
        head_m_product = self._num_heads * self._num_m_blocks
        txl.assign(self.batch_idx, linear_idx // head_m_product)
        txl.assign(self.head_idx, linear_idx % head_m_product // self._num_m_blocks)
        txl.assign(self.m_block_idx, linear_idx % self._num_m_blocks)

    def init(self, cta_id):
        txl.assign(self.linear_idx, cta_id)
        self.update_current_m_n_idx(cta_id)

    def next_tile(self):
        txl.assign(self.linear_idx, self.linear_idx + self._num_ctas)
        self.update_current_m_n_idx(self.linear_idx)


class FlashAttentionLPTScheduler(_FlashAttentionLPTScheduler):
    """The backend LPT scheduler with its three inline methods overridden."""

    def update_current_m_n_idx(self, linear_idx):
        bidhb = linear_idx // self._l2_major
        l2_mod = linear_idx % self._l2_major
        num_hb_remainder = txl.max(self._num_hb % self._l2_swizzle, 1)
        in_full_group = bidhb < self._num_hb_quotient
        m_block_raw = txl.Select(
            in_full_group, l2_mod // self._l2_swizzle, l2_mod // num_hb_remainder
        )
        bidhb_residual = txl.Select(
            in_full_group, l2_mod % self._l2_swizzle, l2_mod % num_hb_remainder
        )
        bidhb_actual = bidhb * self._l2_swizzle + bidhb_residual
        txl.assign(self.batch_idx, bidhb_actual // self._num_heads)
        txl.assign(self.head_idx, bidhb_actual % self._num_heads)
        txl.assign(self.m_block_idx, self._num_m_blocks - 1 - m_block_raw)

    def init(self, cta_id):
        txl.assign(self.linear_idx, cta_id)
        self.update_current_m_n_idx(cta_id)

    def next_tile(self):
        if self._num_ctas is None:
            txl.assign(self.linear_idx, self._total_tasks)
        else:
            txl.assign(self.linear_idx, self.linear_idx + self._num_ctas)
            self.update_current_m_n_idx(self.linear_idx)


class _CLCWorker(ClusterPersistentScheduler2D):
    def __init__(self, clc, prefix):
        super().__init__(
            prefix,
            num_m_tiles=clc._num_m_tiles,
            num_n_tiles=clc._num_n_tiles,
            num_clusters=clc._num_m_tiles * clc._num_n_tiles,
            l2_group_size=clc._l2_group_size,
        )
        self._clc = clc
        self._sa = PipelineState(1, 0)
        self._done = txl.local_scalar(txl.i32, name=f"{prefix}_done")
        self._nxt = txl.local_scalar(txl.u32, name=f"{prefix}_next")

    def reset(self):
        txl.assign(self._done, txl.int32(0))

    def init(self, cluster_id):
        super().init(cluster_id)
        txl.assign(self._done, txl.int32(0))

    def valid(self):
        return self._done == txl.int32(0)

    def consume(self):
        self._clc.sched_arr.full.wait(0, self._sa.phase)
        self._sa.advance()
        query_cancel_first_ctaid_x(self._nxt, txl.address_of(self._clc.clc_handle[0]))
        self._clc.sched_fin.empty.arrive(0, remote=0, pred=True)

    def consume_wg(self, wg_id, warp_id, lane_id):
        self._clc.sched_arr.full.wait(0, self._sa.phase)
        self._sa.advance()
        query_cancel_first_ctaid_x(self._nxt, txl.address_of(self._clc.clc_handle[0]))
        txl.cuda.warpgroup_sync(wg_id + 1)
        with txl.If((warp_id == 0) & (lane_id == 0)), txl.Then():
            self._clc.sched_fin.empty.arrive(0, remote=0, pred=True)

    def advance_coords(self):
        with txl.If(self._nxt != txl.uint32(0xFFFFFFFF)), txl.Then():
            self.update_current_m_n_idx(self._nxt // self._clc._cta_group)

    def mark_done_if_drained(self):
        with txl.If(self._nxt == txl.uint32(0xFFFFFFFF)), txl.Then():
            txl.assign(self._done, txl.int32(1))


class ClusterLaunchControlScheduler(_ClusterLaunchControlScheduler):
    """The backend CLC object routed through tirx-lite-native pipeline expansion."""

    def __init__(self, pool, num_m_tiles, num_n_tiles, l2_group_size, cta_group, finish_arrivals):
        self._num_m_tiles = num_m_tiles
        self._num_n_tiles = num_n_tiles
        self._l2_group_size = l2_group_size
        self._cta_group = cta_group
        self.sched_arr = Pipeline(pool, 1, full="tma", empty="mbar", init_empty=1)
        self.sched_fin = Pipeline(pool, 1, full="mbar", empty="mbar", init_empty=finish_arrivals)
        self.clc_handle = pool.alloc((4,), txl.u32, align=16)
        self._s_done = txl.local_scalar(txl.i32, name="clc_scheduler_done")
        self._s_nxt = txl.local_scalar(txl.u32, name="clc_scheduler_next")

    def worker(self, prefix):
        return _CLCWorker(self, prefix)

    def run_scheduler(self, cbx):
        with txl.If(txl.cuda.elect_sync()), txl.Then():
            sa = PipelineState(1, 0)
            sf = PipelineState(1, 1)
            txl.assign(self._s_done, txl.int32(0))
            with txl.While(self._s_done == txl.int32(0)):
                with txl.If(cbx == 0), txl.Then():
                    self.sched_fin.empty.wait(0, sf.phase)
                    sf.advance()
                    txl.ptx[
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                    ](txl.address_of(self.clc_handle[0]), txl.address_of(self.sched_arr.full.buf[0]))
                self.sched_arr.full.arrive(0, 16)
                self.sched_arr.full.wait(0, sa.phase)
                sa.advance()
                query_cancel_first_ctaid_x(self._s_nxt, txl.address_of(self.clc_handle[0]))
                self.sched_fin.empty.arrive(0, remote=0, pred=True)
                with txl.If(self._s_nxt == txl.uint32(0xFFFFFFFF)), txl.Then():
                    txl.assign(self._s_done, txl.int32(1))


__all__ = [
    "ClusterLaunchControlScheduler",
    "ClusterPersistentScheduler2D",
    "FlashAttentionLPTScheduler",
    "FlashAttentionLinearScheduler",
    "query_cancel_first_ctaid_x",
]
