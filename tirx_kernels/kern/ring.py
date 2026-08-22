# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Unsigned subtract-wrap state for phase-tracked software rings."""

import tirx_kernels.kern as K


class RingState:
    """Own a ``(stage, phase)`` cursor with a fixed unsigned stride.

    Unlike ``PipelineState``, this preserves the subtract-wrap lowering used
    by rings whose stage is topology-partitioned or whose hot path must avoid
    signed/unit-stride canonicalization.
    """

    def __init__(self, depth: int, phase=0, *, stage=0, stride: int = 1):
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"ring depth must be a positive integer, got {depth!r}")
        if not isinstance(stride, int) or not 1 <= stride <= depth:
            raise ValueError(f"ring stride must be in [1, {depth}], got {stride!r}")
        self.depth = depth
        self.stride = stride
        self.stage = K.local_scalar(K.u32, name="stage")
        self.phase = K.local_scalar(K.u32, name="phase")
        self.init(stage, phase)

    def init(self, stage=0, phase=0):
        K.assign(self.stage, K.Cast(K.u32, stage))
        K.assign(self.phase, K.Cast(K.u32, phase))

    def advance(self):
        K.assign(self.stage, self.stage + K.uint32(self.stride))
        with K.If(self.stage >= K.uint32(self.depth)), K.Then():
            K.assign(self.stage, self.stage - K.uint32(self.depth))
            K.assign(self.phase, self.phase ^ K.uint32(1))


__all__ = ["RingState"]
