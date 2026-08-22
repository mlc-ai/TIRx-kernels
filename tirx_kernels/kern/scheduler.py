# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""Kern overrides for the in-tree tile schedulers and CLC helpers.

The backend classes remain the source of truth for construction, state, and
ordinary Python dispatch. Kern only replaces their parser-inline expansion
points, spelling those bodies directly with the public ``K`` primitives.
"""

from __future__ import annotations

import tirx_kernels.kern as K
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
    response = K.local_scalar("uint128", name="clc_response")
    canceled = K.local_scalar(K.u32, name="clc_canceled")
    suffix = ".acquire.cta" if use_ld_acquire else ""

    K.ptx[f"ld{suffix}.shared.b128"](response, handle)
    K.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, response)
    K.assign(first_ctaid_x, K.uint32(0xFFFFFFFF))
    K.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
        first_ctaid_x, response, pred=canceled
    )
    K.ptx.fence.proxy.async_.shared__cta()


class ClusterPersistentScheduler2D(_ClusterPersistentScheduler2D):
    """The backend scheduler with its group-major inline points traced by Kern."""

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
                "K.ClusterPersistentScheduler2D does not support serpentine scheduling"
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
            K.assign(self.m_idx, tile_row * self._cluster_m + cluster_m_offset)
            K.assign(self.n_idx, tile_col * self._cluster_n + cluster_n_offset)

        self._update_group_major(tile_linear, set_tile_coords)

    def _gm_emit_zero(self, set_tile_coords):
        set_tile_coords(0, 0)

    def _gm_emit_full_only(self, tile_linear, set_tile_coords):
        full_groups = self._FULL_GROUPS
        group_size = self._l2_group_size
        group_span = self._l2_group_size * self._N_TILE_COLS
        with K.If((full_groups > 0) & (tile_linear < full_groups * group_span)):
            with K.Then():
                group_id = tile_linear // group_span
                within_group = tile_linear % group_span
                tile_row = group_id * group_size + within_group % group_size
                tile_col = within_group // group_size
                set_tile_coords(tile_row, tile_col)
            with K.Else():
                set_tile_coords(0, 0)

    def _gm_emit_tail_only(self, tile_linear, set_tile_coords):
        full_groups = self._FULL_GROUPS
        tail_rows = self._TAIL_ROWS
        group_size = self._l2_group_size
        group_span = self._l2_group_size * self._N_TILE_COLS
        with K.If(tail_rows > 0):
            with K.Then():
                rem = tile_linear - full_groups * group_span
                tile_row = full_groups * group_size + rem % tail_rows
                tile_col = rem // tail_rows
                set_tile_coords(tile_row, tile_col)
            with K.Else():
                set_tile_coords(0, 0)

    def _gm_emit_full_and_tail(self, tile_linear, set_tile_coords):
        full_groups = self._FULL_GROUPS
        tail_rows = self._TAIL_ROWS
        group_size = self._l2_group_size
        group_span = self._l2_group_size * self._N_TILE_COLS
        with K.If((full_groups > 0) & (tile_linear < full_groups * group_span)):
            with K.Then():
                group_id = tile_linear // group_span
                within_group = tile_linear % group_span
                tile_row = group_id * group_size + within_group % group_size
                tile_col = within_group // group_size
                set_tile_coords(tile_row, tile_col)
            with K.Else():
                with K.If(tail_rows > 0):
                    with K.Then():
                        rem = tile_linear - full_groups * group_span
                        tile_row = full_groups * group_size + rem % tail_rows
                        tile_col = rem // tail_rows
                        set_tile_coords(tile_row, tile_col)
                    with K.Else():
                        set_tile_coords(0, 0)

    def init(self, cluster_id):
        K.assign(self.linear_idx, cluster_id)
        K.assign(self.tile_count, K.int32(0))
        self.update_current_m_n_idx(cluster_id)

    def next_tile(self):
        K.assign(self.linear_idx, self.linear_idx + self._num_clusters)
        K.assign(self.tile_count, self.tile_count + K.int32(1))
        self.update_current_m_n_idx(self.linear_idx)

    def next_tile_stride(self, stride: int):
        K.assign(self.linear_idx, self.linear_idx + stride)
        K.assign(self.tile_count, self.tile_count + K.int32(1))
        self.update_current_m_n_idx(self.linear_idx)


class FlashAttentionLinearScheduler(_FlashAttentionLinearScheduler):
    """The backend linear scheduler with its three inline methods overridden."""

    def update_current_m_n_idx(self, linear_idx):
        head_m_product = self._num_heads * self._num_m_blocks
        K.assign(self.batch_idx, linear_idx // head_m_product)
        K.assign(self.head_idx, linear_idx % head_m_product // self._num_m_blocks)
        K.assign(self.m_block_idx, linear_idx % self._num_m_blocks)

    def init(self, cta_id):
        K.assign(self.linear_idx, cta_id)
        self.update_current_m_n_idx(cta_id)

    def next_tile(self):
        K.assign(self.linear_idx, self.linear_idx + self._num_ctas)
        self.update_current_m_n_idx(self.linear_idx)


class FlashAttentionLPTScheduler(_FlashAttentionLPTScheduler):
    """The backend LPT scheduler with its three inline methods overridden."""

    def update_current_m_n_idx(self, linear_idx):
        bidhb = linear_idx // self._l2_major
        l2_mod = linear_idx % self._l2_major
        num_hb_remainder = K.max(self._num_hb % self._l2_swizzle, 1)
        in_full_group = bidhb < self._num_hb_quotient
        m_block_raw = K.Select(
            in_full_group, l2_mod // self._l2_swizzle, l2_mod // num_hb_remainder
        )
        bidhb_residual = K.Select(
            in_full_group, l2_mod % self._l2_swizzle, l2_mod % num_hb_remainder
        )
        bidhb_actual = bidhb * self._l2_swizzle + bidhb_residual
        K.assign(self.batch_idx, bidhb_actual // self._num_heads)
        K.assign(self.head_idx, bidhb_actual % self._num_heads)
        K.assign(self.m_block_idx, self._num_m_blocks - 1 - m_block_raw)

    def init(self, cta_id):
        K.assign(self.linear_idx, cta_id)
        self.update_current_m_n_idx(cta_id)

    def next_tile(self):
        if self._num_ctas is None:
            K.assign(self.linear_idx, self._total_tasks)
        else:
            K.assign(self.linear_idx, self.linear_idx + self._num_ctas)
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
        self._done = K.local_scalar(K.i32, name=f"{prefix}_done")
        self._nxt = K.local_scalar(K.u32, name=f"{prefix}_next")

    def reset(self):
        K.assign(self._done, K.int32(0))

    def init(self, cluster_id):
        super().init(cluster_id)
        K.assign(self._done, K.int32(0))

    def valid(self):
        return self._done == K.int32(0)

    def consume(self):
        self._clc.sched_arr.full.wait(0, self._sa.phase)
        self._sa.advance()
        query_cancel_first_ctaid_x(self._nxt, K.address_of(self._clc.clc_handle[0]))
        self._clc.sched_fin.empty.arrive(0, remote=0, pred=True)

    def consume_wg(self, wg_id, warp_id, lane_id):
        self._clc.sched_arr.full.wait(0, self._sa.phase)
        self._sa.advance()
        query_cancel_first_ctaid_x(self._nxt, K.address_of(self._clc.clc_handle[0]))
        K.cuda.warpgroup_sync(wg_id + 1)
        with K.If((warp_id == 0) & (lane_id == 0)), K.Then():
            self._clc.sched_fin.empty.arrive(0, remote=0, pred=True)

    def advance_coords(self):
        with K.If(self._nxt != K.uint32(0xFFFFFFFF)), K.Then():
            self.update_current_m_n_idx(self._nxt // self._clc._cta_group)

    def mark_done_if_drained(self):
        with K.If(self._nxt == K.uint32(0xFFFFFFFF)), K.Then():
            K.assign(self._done, K.int32(1))


class ClusterLaunchControlScheduler(_ClusterLaunchControlScheduler):
    """The backend CLC object routed through Kern-native pipeline expansion."""

    def __init__(self, pool, num_m_tiles, num_n_tiles, l2_group_size, cta_group, finish_arrivals):
        self._num_m_tiles = num_m_tiles
        self._num_n_tiles = num_n_tiles
        self._l2_group_size = l2_group_size
        self._cta_group = cta_group
        self.sched_arr = Pipeline(pool, 1, full="tma", empty="mbar", init_empty=1)
        self.sched_fin = Pipeline(pool, 1, full="mbar", empty="mbar", init_empty=finish_arrivals)
        self.clc_handle = pool.alloc((4,), K.u32, align=16)
        self._s_done = K.local_scalar(K.i32, name="clc_scheduler_done")
        self._s_nxt = K.local_scalar(K.u32, name="clc_scheduler_next")

    def worker(self, prefix):
        return _CLCWorker(self, prefix)

    def run_scheduler(self, cbx):
        with K.If(K.cuda.elect_sync()), K.Then():
            sa = PipelineState(1, 0)
            sf = PipelineState(1, 1)
            K.assign(self._s_done, K.int32(0))
            with K.While(self._s_done == K.int32(0)):
                with K.If(cbx == 0), K.Then():
                    self.sched_fin.empty.wait(0, sf.phase)
                    sf.advance()
                    K.ptx[
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                    ](K.address_of(self.clc_handle[0]), K.address_of(self.sched_arr.full.buf[0]))
                self.sched_arr.full.arrive(0, 16)
                self.sched_arr.full.wait(0, sa.phase)
                sa.advance()
                query_cancel_first_ctaid_x(self._s_nxt, K.address_of(self.clc_handle[0]))
                self.sched_fin.empty.arrive(0, remote=0, pred=True)
                with K.If(self._s_nxt == K.uint32(0xFFFFFFFF)), K.Then():
                    K.assign(self._s_done, K.int32(1))


__all__ = [
    "ClusterLaunchControlScheduler",
    "ClusterPersistentScheduler2D",
    "FlashAttentionLPTScheduler",
    "FlashAttentionLinearScheduler",
    "query_cancel_first_ctaid_x",
]
