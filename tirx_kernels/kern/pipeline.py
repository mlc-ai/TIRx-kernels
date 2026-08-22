# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""Parser-free overrides for the in-tree pipeline helper classes."""

from __future__ import annotations

import tirx_kernels.kern as K
from tvm.backend.cuda.lang.pipeline import MBarrier as _MBarrier
from tvm.backend.cuda.lang.pipeline import Pipeline as _Pipeline
from tvm.backend.cuda.lang.pipeline import PipelineState as _PipelineState
from tvm.backend.cuda.lang.pipeline import TMABar as _TMABar
from tvm.backend.cuda.lang.pipeline import _tcgen05_commit_is_unicast


class PipelineState(_PipelineState):
    """The in-tree state model with its two parser macros traced natively."""

    def init(self, phase):
        K.assign(self.stage, K.int32(0))
        K.assign(self.phase, phase)

    def advance(self):
        if self.depth > 1:
            K.assign(self.stage, self.stage + K.int32(1))
            with K.If(self.stage == self.depth), K.Then():
                K.assign(self.stage, K.int32(0))
                K.assign(self.phase, self.phase ^ K.int32(1))
        else:
            K.assign(self.phase, self.phase ^ K.int32(1))


class MBarrier(_MBarrier):
    """The in-tree mbarrier wrapper with native instruction-emitting methods."""

    def _init(self, count):
        with K.If(self.leader), K.Then():
            with K.unroll(0, self.depth) as i:
                K.ptx.mbarrier.init.shared.b64(self.buf.ptr_to([i]), K.uint32(count))

    def _wait(self, stage, phase):
        K.cuda.mbarrier_wait(self.buf.ptr_to([stage]), phase ^ self.phase_offset)

    def _arrive(self, bar):
        K.ptx.mbarrier.arrive.shared.b64(bar, K.uint32(1))


class TMABar(MBarrier):
    """The in-tree TMA barrier with its local-arrive macro traced natively."""

    arrive = _TMABar.arrive

    def _arrive_tma_local(self, bar, tx_count=None):
        if tx_count is None:
            K.ptx.mbarrier.arrive.shared.b64(bar, K.uint32(1))
        else:
            K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar, K.uint32(tx_count))


class TCGen05Bar(MBarrier):
    """The in-tree tcgen05 barrier with its arrive macro traced natively."""

    def arrive(self, stage, cta_group=1, cta_mask=None, pred=None):
        if _tcgen05_commit_is_unicast(cta_mask):
            K.ptx[
                f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.b64"
            ](self.buf.ptr_to([stage]), pred=pred)
        else:
            K.ptx[
                f"tcgen05.commit.cta_group::{cta_group}"
                ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
            ](self.buf.ptr_to([stage]), K.Cast("uint16", cta_mask), pred=pred)


_BAR_KINDS = {"tma": TMABar, "tcgen05": TCGen05Bar, "mbar": MBarrier}


class Pipeline(_Pipeline):
    """The in-tree pipeline constructor routed through the native barriers."""

    def __init__(
        self,
        pool,
        stages,
        *,
        full,
        empty,
        init_full=1,
        init_empty=1,
        empty_phase_offset=0,
        leader=None,
    ):
        self.stages = stages
        self.full = _BAR_KINDS[full](pool, stages, leader=leader)
        self.full.init(init_full)
        self.empty = _BAR_KINDS[empty](pool, stages, phase_offset=empty_phase_offset, leader=leader)
        self.empty.init(init_empty)


__all__ = ["MBarrier", "Pipeline", "PipelineState", "TCGen05Bar", "TMABar"]
