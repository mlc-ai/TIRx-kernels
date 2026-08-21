# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Unsigned subtract-wrap state for phase-tracked software rings."""

from tvm.script import tirx as T


@T.meta_class
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
        self.stage = T.local_scalar("uint32")
        self.phase = T.local_scalar("uint32")
        self.init(stage, phase)

    @T.inline
    def init(self, stage=0, phase=0):
        self.stage = T.cast(stage, "uint32")
        self.phase = T.cast(phase, "uint32")

    @T.inline
    def advance(self):
        self.stage = self.stage + T.uint32(self.stride)
        if self.stage >= T.uint32(self.depth):
            self.stage = self.stage - T.uint32(self.depth)
            self.phase = self.phase ^ T.uint32(1)


__all__ = ["RingState"]
