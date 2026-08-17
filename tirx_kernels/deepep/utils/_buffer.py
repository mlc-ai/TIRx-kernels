# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""NCCL symmetric-window allocation for the TIRx DeepEP kernels.

The source kernels translate peer pointers through the NCCL device API
(`NCCLGin`, common/handle.cuh). The TIRx port replaces that with
host-precomputed peer base-pointer tables (sanctioned substitution 1 in
`.agents/sketch/deepep_dispatch.md`), obtained here with the same host API
the source backend uses (`ncclGetLsaDevicePointer`,
csrc/kernels/backend/nccl.cu:141-152).
"""

from __future__ import annotations

import ctypes
import os

import torch
import torch.distributed as dist

# DeepEP `layout::WorkspaceLayout::get_num_bytes()` (common/layout.cuh:43-80),
# recomputed in Python; must stay in sync with the source.
WORKSPACE_NUM_BYTES = (
    16  # barrier counter + signals
    + (1024 + 2048) * 8  # notify reduction workspace
    + 1024 * 8 * 2  # scaleup rank send/recv counts
    + 2048 * 8 * 2  # scaleup expert send/recv counts
    + 1024 * 4  # scaleup atomic sender counters
    + 1024 * 4 * 2  # scaleout rank send/recv counts
    + 2048 * 4 * 2  # scaleout expert send/recv counts
    + 1024 * 1024 * 8  # scaleout channel signaled tails
    + 1024 * 1024 * 4  # channel scaleup tails
    + 2 * 2 * 8  # PP send/recv counts
    + (32 + 1) * 1024 * 4  # AGRS signals
)
assert WORKSPACE_NUM_BYTES == 12_820_528

SYMMETRIC_ALIGNMENT = 2 * 1024 * 1024  # 2 MB (backend/symmetric.hpp:16)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


WORKSPACE_ALIGNED_BYTES = align_up(WORKSPACE_NUM_BYTES, SYMMETRIC_ALIGNMENT)

_NCCL_WIN_STRICT_ORDERING = 0x02


def _load_nccl() -> ctypes.CDLL:
    """Resolve the libnccl instance already loaded by deep_ep / torch."""

    candidates = [
        None,  # already loaded symbols (deep_ep's _C links libnccl)
        "/host-libs/libnccl.so.2",
        "libnccl.so.2",
    ]
    for candidate in candidates:
        try:
            if candidate is None:
                library = ctypes.CDLL(None)
            else:
                library = ctypes.CDLL(candidate, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
            if hasattr(library, "ncclGetLsaDevicePointer"):
                return library
        except OSError:
            continue
    raise RuntimeError("no loaded libnccl exports ncclGetLsaDevicePointer")


def _check(result: int, call: str) -> None:
    if result != 0:
        raise RuntimeError(f"{call} failed with ncclResult_t={result}")


class SymmetricWindow:
    """One NCCL-registered symmetric window: [workspace | recv buffer]."""

    def __init__(self, group: dist.ProcessGroup, num_buffer_bytes: int) -> None:
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        from tirx_kernels.runner import external_references_enabled

        if external_references_enabled():
            # Keep the original dedicated communicator when DeepEP is present,
            # so the TIRx window does not share reference communication state.
            from deep_ep.utils.comm import get_nccl_comm_handle

            self._comm_handle = get_nccl_comm_handle(group, force_new_comm=True)
            comm = ctypes.c_void_p(self._comm_handle.get())
        else:
            device = torch.device("cuda", self.rank)
            backend = group._get_backend(device)
            comm = ctypes.c_void_p(backend._comm_ptr())
        self._nccl = _load_nccl()

        self.workspace_bytes = WORKSPACE_ALIGNED_BYTES
        self.buffer_bytes = align_up(num_buffer_bytes, SYMMETRIC_ALIGNMENT)
        self.total_bytes = self.workspace_bytes + self.buffer_bytes

        device = torch.device("cuda", self.rank)

        base = ctypes.c_void_p()
        _check(
            self._nccl.ncclMemAlloc(ctypes.byref(base), ctypes.c_size_t(self.total_bytes)),
            "ncclMemAlloc",
        )
        self.base_ptr = int(base.value)
        self.buffer_ptr = self.base_ptr + self.workspace_bytes

        window = ctypes.c_void_p()
        _check(
            self._nccl.ncclCommWindowRegister(
                comm,
                base,
                ctypes.c_size_t(self.total_bytes),
                ctypes.byref(window),
                ctypes.c_int(_NCCL_WIN_STRICT_ORDERING),
            ),
            "ncclCommWindowRegister",
        )
        self._window = window
        self._comm = comm

        # Workspace must start zeroed (barrier signals, counters, port scratch).
        # Use the libcudart instance already loaded by torch (runtime API keeps
        # the rank-local primary context current on this thread).
        cudart = None
        maps = open(f"/proc/{os.getpid()}/maps").read()
        for line in maps.splitlines():
            if "libcudart.so" in line:
                cudart = ctypes.CDLL(line.split()[-1], mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
                break
        if cudart is None:
            raise RuntimeError("no loaded libcudart found")
        result = cudart.cudaMemset(
            ctypes.c_void_p(self.base_ptr), 0, ctypes.c_size_t(self.workspace_bytes)
        )
        if result != 0:
            raise RuntimeError(f"cudaMemset failed with cudaError_t={result}")

        peer_ws = []
        for peer in range(self.world_size):
            ptr = ctypes.c_void_p()
            _check(
                self._nccl.ncclGetLsaDevicePointer(
                    window, ctypes.c_int64(0), ctypes.c_int(peer), ctypes.byref(ptr)
                ),
                "ncclGetLsaDevicePointer",
            )
            peer_ws.append(int(ptr.value))
        self.peer_ws_ptrs = torch.tensor(peer_ws, dtype=torch.int64, device=device)
        self.peer_buf_ptrs = torch.tensor(
            [p + self.workspace_bytes for p in peer_ws], dtype=torch.int64, device=device
        )

    def destroy(self) -> None:
        if self._window is not None:
            _check(
                self._nccl.ncclCommWindowDeregister(self._comm, self._window),
                "ncclCommWindowDeregister",
            )
            self._window = None
        if self.base_ptr:
            _check(self._nccl.ncclMemFree(ctypes.c_void_p(self.base_ptr)), "ncclMemFree")
            self.base_ptr = 0


def get_theoretical_num_sms(
    world_size: int, num_experts: int, num_topk: int, prefer_overlap_with_compute: bool = True
) -> int:
    """Return the launch width for the ported e256/k6 single-node path.

    The retained specialization always uses the source model's non-overlap
    floor of 64 SMs.  Keeping that invariant here avoids importing a runtime
    package merely to rediscover a constant already fixed by the kernel port.
    """

    if (world_size, num_experts, num_topk) != (8, 256, 6):
        raise ValueError("DeepEP ports are specialized to world=8, experts=256, topk=6")
    from tirx_kernels.runner import PREPARE_NUM_SMS_ENV

    prepared_sms = os.environ.get(PREPARE_NUM_SMS_ENV)
    device_sms = (
        int(prepared_sms)
        if prepared_sms
        else torch.cuda.get_device_properties(0).multi_processor_count
    )
    return min(64, device_sms)


__all__ = [
    "SYMMETRIC_ALIGNMENT",
    "WORKSPACE_ALIGNED_BYTES",
    "WORKSPACE_NUM_BYTES",
    "SymmetricWindow",
    "align_up",
    "get_theoretical_num_sms",
]
