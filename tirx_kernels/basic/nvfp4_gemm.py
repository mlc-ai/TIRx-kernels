# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from enum import IntEnum
from pathlib import Path

import tvm
from tirx_kernels.runner import PREPARE_CUDA_ARCH_ENV, bench
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_smem_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.tirx.lang.pipeline import MBarrier, Pipeline, PipelineState, TMABar
from tvm.tirx.lang.smem_desc import SmemDescriptor
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D


class WarpRole(IntEnum):
    MMA = 0
    TMA = 2
    EPILOGUE = 4


_TMA_G2S_2D = (
    "cp.async.bulk.tensor.2d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
)
_TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
)
_TMA_S2G_EVICT_FIRST = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
_TCGEN05_CP_2SM = "tcgen05.cp.cta_group::2.32x128b.warpx4"
_MMA_NVFP4_2SM = "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
_TMEM_LD_X2 = "tcgen05.ld.sync.aligned.16x256b.x2.b32"
_TMEM_LD_X4 = "tcgen05.ld.sync.aligned.16x256b.x4.b32"
_TMEM_LD_X8 = "tcgen05.ld.sync.aligned.16x256b.x8.b32"
_EVICT_NORMAL_L2_POLICY = 0x1000000000000000
_EVICT_FIRST_L2_POLICY = 0x12F0000000000000


def prepare_data(M: int, N: int, K: int, *, return_origin: bool = False):
    import torch
    from flashinfer import SfLayout, nvfp4_quantize

    torch.manual_seed(0)
    A_origin = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B_origin = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A_global_sf = 448 * 6 / A_origin.float().abs().nan_to_num().max()
    B_global_sf = 448 * 6 / B_origin.float().abs().nan_to_num().max()
    A_fp4, A_sf = nvfp4_quantize(
        A_origin, A_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    B_fp4, B_sf = nvfp4_quantize(
        B_origin, B_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    alpha = 1.0 / (A_global_sf * B_global_sf)
    C_ref = torch.mm(A_origin, B_origin.T)
    if return_origin:
        return (A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref, A_origin, B_origin)
    return (A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref)


_CUBLASLT_EXT = None


def _load_cublaslt_nvfp4_ext():
    """Build and load the shape-independent cuBLASLt reference before READY."""
    global _CUBLASLT_EXT
    if _CUBLASLT_EXT is not None:
        return _CUBLASLT_EXT

    from torch.utils import cpp_extension

    source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#define CHECK_CUBLAS_THROW(call)                                             \
  do {                                                                       \
    cublasStatus_t status = call;                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
      throw std::runtime_error("cuBLASLt error status=" +                    \
                               std::to_string(static_cast<int>(status)));     \
    }                                                                        \
  } while (0)

#define CHECK_CUDA_THROW(call)                                               \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                \
                               cudaGetErrorString(err));                     \
    }                                                                        \
  } while (0)

struct Nvfp4Plan {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  cublasLtMatrixLayout_t layout_d = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  void* workspace = nullptr;
  size_t workspace_size = 128 * 1024 * 1024;

  Nvfp4Plan(int M, int N, int K) {
    CHECK_CUBLAS_THROW(cublasLtCreate(&handle));
    CHECK_CUBLAS_THROW(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));

    // TN layout mapping for the cuBLASLt NVFP4 matmul:
    // cuBLAS "A" is logical B, cuBLAS "B" is logical A, TN writes row-major D
    // through a column-major NxM view.
    CHECK_CUBLAS_THROW(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_4F_E2M1, K, N, K));
    CHECK_CUBLAS_THROW(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_4F_E2M1, K, M, K));
    CHECK_CUBLAS_THROW(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16BF, N, M, N));
    CHECK_CUBLAS_THROW(cublasLtMatrixLayoutCreate(&layout_d, CUDA_R_16BF, N, M, N));

    CHECK_CUDA_THROW(cudaMalloc(&workspace, workspace_size));
    CHECK_CUBLAS_THROW(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS_THROW(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    void* dummy_scale = workspace;
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dummy_scale, sizeof(dummy_scale)));
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dummy_scale, sizeof(dummy_scale)));

    int returned = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        handle, desc, layout_a, layout_b, layout_c, layout_d, preference,
        1, &heuristic, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
      throw std::runtime_error("cuBLASLt NVFP4 heuristic returned no algorithm");
    }
  }

  ~Nvfp4Plan() {
    if (workspace) cudaFree(workspace);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (layout_a) cublasLtMatrixLayoutDestroy(layout_a);
    if (layout_b) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_c) cublasLtMatrixLayoutDestroy(layout_c);
    if (layout_d) cublasLtMatrixLayoutDestroy(layout_d);
    if (desc) cublasLtMatmulDescDestroy(desc);
    if (handle) cublasLtDestroy(handle);
  }
};

static std::mutex g_mu;
static std::unordered_map<std::string, std::unique_ptr<Nvfp4Plan>> g_plans;

static Nvfp4Plan* get_plan(int M, int N, int K) {
  std::lock_guard<std::mutex> lock(g_mu);
  std::string key = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
  auto it = g_plans.find(key);
  if (it == g_plans.end()) {
    it = g_plans.emplace(key, std::make_unique<Nvfp4Plan>(M, N, K)).first;
  }
  return it->second.get();
}

void nvfp4_cublaslt(torch::Tensor A, torch::Tensor B, torch::Tensor A_scale,
                    torch::Tensor B_scale, double alpha, torch::Tensor D,
                    int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A_scale.is_cuda() && B_scale.is_cuda() && D.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kByte && B.scalar_type() == at::kByte,
              "A and B must be uint8 packed FP4 tensors");
  TORCH_CHECK(A_scale.scalar_type() == at::kByte && B_scale.scalar_type() == at::kByte,
              "scale tensors must be uint8 FP8 payloads");
  TORCH_CHECK(D.scalar_type() == at::kBFloat16, "D must be bf16");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && A_scale.is_contiguous() &&
              B_scale.is_contiguous() && D.is_contiguous(), "all tensors must be contiguous");

  Nvfp4Plan* plan = get_plan(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

  auto* A_ptr = reinterpret_cast<const __nv_fp4x2_e2m1*>(A.data_ptr<uint8_t>());
  auto* B_ptr = reinterpret_cast<const __nv_fp4x2_e2m1*>(B.data_ptr<uint8_t>());
  auto* A_scale_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(A_scale.data_ptr<uint8_t>());
  auto* B_scale_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(B_scale.data_ptr<uint8_t>());
  auto* D_ptr = reinterpret_cast<__nv_bfloat16*>(D.data_ptr<at::BFloat16>());

  const void* cublas_a_scale = B_scale_ptr;
  const void* cublas_b_scale = A_scale_ptr;
  CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
      plan->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &cublas_a_scale, sizeof(cublas_a_scale)));
  CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
      plan->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &cublas_b_scale, sizeof(cublas_b_scale)));

  float alpha_f = static_cast<float>(alpha);
  float beta = 0.0f;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  CHECK_CUBLAS_THROW(cublasLtMatmul(
      plan->handle, plan->desc, &alpha_f,
      B_ptr, plan->layout_a,
      A_ptr, plan->layout_b,
      &beta,
      D_ptr, plan->layout_c,
      D_ptr, plan->layout_d,
      &plan->heuristic.algo,
      plan->workspace, plan->workspace_size,
      stream));
}
"""
    extra_include_paths = []
    extra_ldflags = ["-lcublas", "-lcublasLt"]
    if cpp_extension.CUDA_HOME:
        extra_include_paths.append(f"{cpp_extension.CUDA_HOME}/include")
        extra_ldflags.insert(0, f"-L{cpp_extension.CUDA_HOME}/lib64")

    name = "nvfp4_cublaslt_baseline_ext"
    build_directory = Path(cpp_extension._get_build_directory(name, verbose=False))
    lock_fd = os.open(build_directory / "lock.flock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # PyTorch's FileBaton is not process-death-safe. The outer flock proves
        # no suite process is building here, so a remaining baton is stale.
        (build_directory / "lock").unlink(missing_ok=True)
        extra_cuda_cflags = []
        if prepare_arch := os.environ.get(PREPARE_CUDA_ARCH_ENV):
            arch = prepare_arch.removeprefix("sm_")
            extra_cuda_cflags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
        _CUBLASLT_EXT = cpp_extension.load_inline(
            name=name,
            cpp_sources=[source],
            functions=["nvfp4_cublaslt"],
            with_cuda=True,
            extra_include_paths=extra_include_paths,
            extra_cflags=["-O3"],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            build_directory=str(build_directory),
            verbose=False,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return _CUBLASLT_EXT


def _mapa_u64(ptr, rank):
    """`mapa.u64` into a declared register, returned as an ordinary value.

    PTX has no defining form, so mapa writes a register the caller declares;
    a one-element local buffer gives both a writable lvalue and an Expr.
    """
    mapped = T.alloc_local([1], "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    return mapped[0]


@T.inline
def _mul_f32x2_inplace(values, index, multiplier):
    """Keep the packed multiply's register lifetime explicit in inline PTX."""
    packed: T.uint64
    rhs: T.uint64
    T.ptx.mov.b64(packed, values[index], values[index + 1])
    T.ptx.mov.b64(rhs, multiplier, multiplier)
    T.ptx.mul.rz.ftz.f32x2(packed, packed, rhs)
    T.ptx.mov.b64(values[index], values[index + 1], packed)


@T.jit
def _kernel(
    A_packed: T.Buffer((M, K // 2), "uint8"),
    B_packed: T.Buffer((N, K // 2), "uint8"),
    SFA_in: T.Buffer((M, K // 16), "uint8", layout=sf_smem_layout(M, K // 16, sf_per_mma=4)),
    SFB_in: T.Buffer((N, K // 16), "uint8", layout=sf_smem_layout(N, K // 16, sf_per_mma=4)),
    alpha: T.Buffer((1,), "float32"),
    D: T.Buffer((M, N), "bfloat16"),
    *,
    M: T.constexpr,
    N: T.constexpr,
    K: T.constexpr,
    # Fixed hardware + tile/cluster/pipeline choices (tir_ws_kernel never
    # overrides these). Derived quantities are computed from them below.
    SM_COUNT: T.constexpr = 148,
    CTA_GROUP: T.constexpr = 2,
    CLUSTER_M: T.constexpr = 2,
    CLUSTER_N: T.constexpr = 1,
    CTA_M: T.constexpr = 128,
    CTA_N: T.constexpr = 128,
    CTA_K: T.constexpr = 256,
    MMA_K: T.constexpr = 64,
    EPI_TILE: T.constexpr = 64,
    TMEM_LD_SIZE: T.constexpr = 64,
    WB_PIPE_DEPTH: T.constexpr = 2,
    PIPE_DEPTH: T.constexpr = 5,
    TMEM_PIPE_DEPTH: T.constexpr = 1,
    L2_GROUP_SIZE: T.constexpr = 8,
    NUM_WARPS: T.constexpr = 8,
    OVERLAP_EPI: T.constexpr = True,
):
    # Derived shapes (formulas, so they track the params above).
    CLUSTER_SIZE = T.meta_var(CLUSTER_M * CLUSTER_N)
    MMA_N = T.meta_var(CTA_N * CTA_GROUP)
    SFB_N = T.meta_var(MMA_N)
    MMA_K_BLOCKS = T.meta_var(CTA_K // MMA_K)
    SF_CTA_K = T.meta_var(CTA_K // 16)
    NUM_CLUSTERS = T.meta_var(SM_COUNT // CLUSTER_SIZE)
    D_SWIZZLE_MODE = T.meta_var(
        SwizzleMode.SWIZZLE_32B_ATOM
        if EPI_TILE == 16
        else SwizzleMode.SWIZZLE_64B_ATOM
        if EPI_TILE == 32
        else SwizzleMode.SWIZZLE_128B_ATOM
    )
    A_BYTES = T.meta_var(CTA_M * (CTA_K // 2) * CTA_GROUP)
    B_BYTES = T.meta_var(CTA_N * (CTA_K // 2) * CTA_GROUP)
    SFA_BYTES = T.meta_var(CTA_M * SF_CTA_K * CTA_GROUP)
    SFB_BYTES = T.meta_var(SFB_N * SF_CTA_K * CTA_GROUP)
    K_TILES = T.meta_var(K // CTA_K)
    CLUSTER_M_TILES = T.meta_var(M // CTA_M // CLUSTER_M)
    CLUSTER_N_TILES = T.meta_var(N // MMA_N // CLUSTER_N)
    TMEM_LD = T.meta_var(
        _TMEM_LD_X2 if EPI_TILE == 16 else _TMEM_LD_X4 if EPI_TILE == 32 else _TMEM_LD_X8
    )

    # The packed FP4 operands are ordinary 2-D row-major byte tensors.  The
    # scale factors retain FlashInfer's layout_128x4 packing: represent that
    # physical layout as a 3-D uint16 tensor map rather than reinterpreting or
    # reshuffling any E4M3 payloads in the kernel.
    A_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    B_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    SFA_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    SFB_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    D_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        A_tensor_map,
        "uint8",
        2,
        A_packed.data,
        K // 2,
        M,
        K // 2,
        CTA_K // 2,
        CTA_M,
        1,
        1,
        0,
        SwizzleMode.SWIZZLE_128B_ATOM.value,
        2,
        0,
    )
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        B_tensor_map,
        "uint8",
        2,
        B_packed.data,
        K // 2,
        N,
        K // 2,
        CTA_K // 2,
        CTA_N,
        1,
        1,
        0,
        SwizzleMode.SWIZZLE_128B_ATOM.value,
        2,
        0,
    )
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        SFA_tensor_map,
        "uint16",
        3,
        SFA_in.data,
        256,
        K // 64,
        M // 128,
        512,
        K * 8,
        256,
        4,
        1,
        1,
        1,
        1,
        0,
        0,
        2,
        0,
    )
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        SFB_tensor_map,
        "uint16",
        3,
        SFB_in.data,
        256,
        K // 64,
        N // 128,
        512,
        K * 8,
        256,
        4,
        1,
        1,
        1,
        1,
        0,
        0,
        2,
        0,
    )
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        D_tensor_map,
        "bfloat16",
        2,
        D.data,
        N,
        M,
        N * 2,
        EPI_TILE,
        CTA_M,
        1,
        1,
        0,
        D_SWIZZLE_MODE.value,
        2,
        0,
    )
    T.device_entry()
    cluster_rank = T.cta_id_in_cluster([CLUSTER_SIZE], preferred=[CLUSTER_SIZE])
    cta_idx = T.cta_id([SM_COUNT])
    tid_in_cta = T.thread_id([NUM_WARPS * 32])
    lane_id = T.lane_id([32])
    tid_in_wg = T.thread_id_in_wg([128])
    wg_id = T.warpgroup_id([NUM_WARPS // 4])
    warp_id = T.warp_id([NUM_WARPS])
    if warp_id == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(A_tensor_map)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(B_tensor_map)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(SFA_tensor_map)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(SFB_tensor_map)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(D_tensor_map)))
    cb_m: T.let = cluster_rank % CLUSTER_M
    cb_n: T.let = cluster_rank // CLUSTER_M
    pair_id: T.let = cluster_rank // CTA_GROUP
    id_in_pair: T.let = cluster_rank % CTA_GROUP
    pair_leader_rank: T.let = pair_id * CTA_GROUP
    tile_scheduler = ClusterPersistentScheduler2D(
        "tile_scheduler",
        num_m_tiles=CLUSTER_M_TILES,
        num_n_tiles=CLUSTER_N_TILES,
        num_clusters=NUM_CLUSTERS,
        l2_group_size=L2_GROUP_SIZE,
    )
    tile_scheduler.init(cta_idx // CLUSTER_SIZE)
    m_idx = T.meta_var(tile_scheduler.m_idx)
    n_idx = T.meta_var(tile_scheduler.n_idx)
    cta_m = T.meta_var(m_idx * CLUSTER_M + cb_m)
    cta_n = T.meta_var(n_idx * CLUSTER_N + cb_n)
    a_m = T.meta_var(cta_m * CTA_M)
    d_m = T.meta_var(cta_m * CTA_M)
    b_n = T.meta_var(cta_n * MMA_N + id_in_pair * CTA_N)
    d_n = T.meta_var(cta_n * MMA_N)
    pool = T.SMEMPool()
    A_smem_packed = pool.alloc_tcgen05_mma_AB((PIPE_DEPTH, CTA_M, CTA_K // 2), "uint8")
    B_smem_packed = pool.alloc_tcgen05_mma_AB((PIPE_DEPTH, CTA_N, CTA_K // 2), "uint8")
    SFA_smem = pool.alloc(
        (PIPE_DEPTH, CTA_M, SF_CTA_K),
        "uint8",
        layout=sf_smem_layout(128, 16, sf_per_mma=4, pipe_depth=PIPE_DEPTH),
        align=1024,
    )
    SFB_smem = pool.alloc(
        (PIPE_DEPTH, SFB_N, SF_CTA_K),
        "uint8",
        layout=sf_smem_layout(SFB_N, 16, sf_per_mma=4, pipe_depth=PIPE_DEPTH),
        align=1024,
    )
    output_smem = pool.alloc_tcgen05_mma_AB(
        (WB_PIPE_DEPTH, CTA_M, EPI_TILE), "bfloat16", swizzle_mode=D_SWIZZLE_MODE
    )
    tmem_addr = pool.alloc([1], "uint32", align=4)
    mbar_leader = tid_in_cta == 32
    smem_pipe = Pipeline(pool, PIPE_DEPTH, full="tma", empty="tcgen05", leader=mbar_leader)
    tile_full_bar = TMABar(pool, PIPE_DEPTH, leader=mbar_leader)
    tile_full_bar.init(1)
    scale_full_bar = TMABar(pool, PIPE_DEPTH, leader=mbar_leader)
    scale_full_bar.init(1)
    tmem_pipe = Pipeline(
        pool,
        TMEM_PIPE_DEPTH,
        full="tcgen05",
        empty="mbar",
        init_empty=CTA_GROUP,
        leader=mbar_leader,
    )
    tmem_finished = MBarrier(pool, 1, leader=mbar_leader)
    tmem_finished.init(1)
    pool.commit()
    if mbar_leader:
        T.ptx.fence.mbarrier_init.release.cluster()
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
    tmem = tmem_pool.alloc((CTA_M, 512), "float32")
    A_smem = A_smem_packed.view("float4_e2m1fn")
    B_smem = B_smem_packed.view("float4_e2m1fn")
    sf_mma_k = T.meta_var(4)
    SFB_n_chunks = T.meta_var(SFB_N // 128)
    tmem_pool.move_base_to(448)
    SFA_tmem = tmem_pool.alloc_sf(
        (128, sf_mma_k * MMA_K_BLOCKS), "float8_e4m3fn", sf_per_mma=sf_mma_k
    )
    tmem_pool.move_base_to(464)
    SFB_tmem = tmem_pool.alloc_sf(
        (128 * SFB_n_chunks, sf_mma_k * MMA_K_BLOCKS), "float8_e4m3fn", sf_per_mma=sf_mma_k
    )
    sf_desc = T.meta_var(SmemDescriptor())
    desc_a = T.meta_var(SmemDescriptor())
    desc_b = T.meta_var(SmemDescriptor())
    sf_desc.init(T.reinterpret("handle", T.uint64(0)), ldo=0, sdo=8, swizzle=0)
    desc_a.init(A_smem.ptr_to([0, 0, 0]), ldo=0, sdo=64, swizzle=3)
    desc_b.init(B_smem.ptr_to([0, 0, 0]), ldo=0, sdo=64, swizzle=3)
    # Publish the weak shared-memory store performed by tcgen05.alloc through
    # the existing cluster release/acquire before any other warp consumes the
    # TMEM base address.
    tmem_pool.commit()
    T.ptx.barrier.cluster.arrive.release.aligned()
    T.ptx.barrier.cluster.wait.acquire()
    if tid_in_cta < 32:
        T.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{CTA_GROUP}.sync.aligned"]()
    pair_mask: T.int32
    pair_mask = 0
    pair_mask = pair_mask | 1 << pair_leader_rank
    pair_mask = pair_mask | 1 << pair_leader_rank + 1
    tma_cur = PipelineState(PIPE_DEPTH, 1)
    mma_smem = PipelineState(PIPE_DEPTH, 0)
    mma_tmem = PipelineState(TMEM_PIPE_DEPTH, 1)
    accum: T.int32
    accum = 0
    epi_cur = PipelineState(TMEM_PIPE_DEPTH, 0)
    epi_wb_state = PipelineState(WB_PIPE_DEPTH, 1)
    alpha_local: T.float32
    T.ptx.ld.global_.nc.f32(alpha_local, alpha.ptr_to([0]))
    if warp_id == int(WarpRole.TMA):

        @T.inline
        def issue_tma_load(k_tile: T.int32):
            stage = tma_cur.stage
            k = T.meta_var(k_tile * CTA_K // 2)
            smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
            if id_in_pair == 0:
                tile_bytes = T.meta_var(A_BYTES + B_BYTES)
                _rem1 = T.alloc_local([1], "uint64")
                T.ptx.mapa.shared__cluster.u64(
                    _rem1[0], tile_full_bar.ptr_to([stage]), T.uint32(pair_leader_rank)
                )
                T.ptx.mbarrier.arrive.expect_tx.b64(
                    _rem1[0], T.uint32(tile_bytes), pred=T.bool(True)
                )
            single_cta_mask: T.int32 = 1 << id_in_pair
            mapped_tile_bar = _mapa_u64(tile_full_bar.ptr_to([stage]), 0)
            T.evaluate(
                T.ptx[_TMA_G2S_2D](
                    A_smem_packed.ptr_to([stage, 0, 0]),
                    T.address_of(A_tensor_map),
                    T.cast(k, "int32"),
                    T.cast(a_m, "int32"),
                    T.cuda.cvta_generic_to_shared(T.reinterpret("handle", mapped_tile_bar)),
                    T.cast(single_cta_mask, "uint16"),
                    T.uint64(_EVICT_NORMAL_L2_POLICY),
                )
            )
            T.evaluate(
                T.ptx[_TMA_G2S_2D](
                    B_smem_packed.ptr_to([stage, 0, 0]),
                    T.address_of(B_tensor_map),
                    T.cast(k, "int32"),
                    T.cast(b_n, "int32"),
                    T.cuda.cvta_generic_to_shared(T.reinterpret("handle", mapped_tile_bar)),
                    T.cast(single_cta_mask, "uint16"),
                    T.uint64(_EVICT_NORMAL_L2_POLICY),
                )
            )

        if T.cuda.elect_sync():
            while tile_scheduler.valid():
                for k_tile in T.serial(K_TILES):
                    issue_tma_load(k_tile)
                    tma_cur.advance()
                tile_scheduler.next_tile()
    elif warp_id == int(WarpRole.TMA) + 1:

        @T.inline
        def issue_scale_tma_load(k_tile: T.int32):
            stage = tma_cur.stage
            sf_k = T.meta_var(k_tile * SF_CTA_K)
            sf_m = T.meta_var((a_m // 128) * 128)
            sf_n = T.meta_var((d_n // 128) * 128)
            smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
            if id_in_pair == 0:
                scale_bytes = T.meta_var(SFA_BYTES + SFB_BYTES)
                _rem2 = T.alloc_local([1], "uint64")
                T.ptx.mapa.shared__cluster.u64(
                    _rem2[0], scale_full_bar.ptr_to([stage]), T.uint32(pair_leader_rank)
                )
                T.ptx.mbarrier.arrive.expect_tx.b64(
                    _rem2[0], T.uint32(scale_bytes), pred=T.bool(True)
                )
            single_cta_mask: T.int32 = 1 << id_in_pair
            # SFA: each CTA loads its half (single_cta_mask). SFB: multicast to
            # both CTAs (pair_mask).
            mapped_sfa_bar = _mapa_u64(scale_full_bar.ptr_to([stage]), 0)
            T.evaluate(
                T.ptx[_TMA_G2S_3D](
                    SFA_smem.ptr_to([stage, 0, 0]),
                    T.address_of(SFA_tensor_map),
                    T.int32(0),
                    T.cast(sf_k // 4, "int32"),
                    T.cast(sf_m // 128, "int32"),
                    T.cuda.cvta_generic_to_shared(T.reinterpret("handle", mapped_sfa_bar)),
                    T.cast(single_cta_mask, "uint16"),
                    T.uint64(_EVICT_NORMAL_L2_POLICY),
                )
            )
            mapped_sfb_bar = _mapa_u64(scale_full_bar.ptr_to([stage]), 0)
            if SFB_N == 128:
                if id_in_pair == 0:
                    T.evaluate(
                        T.ptx[_TMA_G2S_3D](
                            SFB_smem.ptr_to([stage, 0, 0]),
                            T.address_of(SFB_tensor_map),
                            T.int32(0),
                            T.cast(sf_k // 4, "int32"),
                            T.cast(sf_n // 128, "int32"),
                            T.cuda.cvta_generic_to_shared(T.reinterpret("handle", mapped_sfb_bar)),
                            T.cast(pair_mask, "uint16"),
                            T.uint64(_EVICT_NORMAL_L2_POLICY),
                        )
                    )
            else:
                T.evaluate(
                    T.ptx[_TMA_G2S_3D](
                        SFB_smem.ptr_to([stage, cb_m * 128, 0]),
                        T.address_of(SFB_tensor_map),
                        T.int32(0),
                        T.cast(sf_k // 4, "int32"),
                        T.cast(sf_n // 128 + cb_m, "int32"),
                        T.cuda.cvta_generic_to_shared(T.reinterpret("handle", mapped_sfb_bar)),
                        T.cast(pair_mask, "uint16"),
                        T.uint64(_EVICT_NORMAL_L2_POLICY),
                    )
                )

        if T.cuda.elect_sync():
            while tile_scheduler.valid():
                for k_tile in T.serial(K_TILES):
                    issue_scale_tma_load(k_tile)
                    tma_cur.advance()
                tile_scheduler.next_tile()
    elif (warp_id == int(WarpRole.MMA)) & (id_in_pair == 0):

        @T.inline
        def execute_mma():
            stage = mma_smem.stage
            scale_full_bar.wait(mma_smem.stage, mma_smem.phase)
            tile_full_bar.wait(mma_smem.stage, mma_smem.phase)
            for flat in T.unroll(CTA_M // 32):
                sfa_row = T.meta_var(flat % 4 * 32)
                sfa_shared_addr: T.uint32 = T.cuda.cvta_generic_to_shared(
                    T.ptr_byte_offset(
                        SFA_smem.ptr_to([0, 0, 0]), (stage * CTA_M + sfa_row) * SF_CTA_K, "uint8"
                    )
                )
                sfa_cp_desc: T.uint64 = T.bitwise_or(
                    T.bitwise_and(sf_desc.desc, T.bitwise_not(T.uint64(0x3FFF))),
                    T.cast(
                        T.bitwise_and(
                            T.shift_right(sfa_shared_addr, T.uint32(4)), T.uint32(0x3FFF)
                        ),
                        "uint64",
                    ),
                )
                T.ptx[_TCGEN05_CP_2SM](
                    T.cast(SFA_tmem.allocated_addr[0] + flat % 4 * 4, "uint32"), sfa_cp_desc
                )
            for flat in T.unroll(SFB_N // 32):
                sfb_row = T.meta_var(flat % 4 * 32 + flat // 4 * 128)
                sfb_shared_addr: T.uint32 = T.cuda.cvta_generic_to_shared(
                    T.ptr_byte_offset(
                        SFB_smem.ptr_to([0, 0, 0]), (stage * SFB_N + sfb_row) * SF_CTA_K, "uint8"
                    )
                )
                sfb_cp_desc: T.uint64 = T.bitwise_or(
                    T.bitwise_and(sf_desc.desc, T.bitwise_not(T.uint64(0x3FFF))),
                    T.cast(
                        T.bitwise_and(
                            T.shift_right(sfb_shared_addr, T.uint32(4)), T.uint32(0x3FFF)
                        ),
                        "uint64",
                    ),
                )
                T.ptx[_TCGEN05_CP_2SM](
                    T.cast(
                        SFB_tmem.allocated_addr[0] + flat % 4 * SFB_n_chunks * 4 + flat // 4 * 4,
                        "uint32",
                    ),
                    sfb_cp_desc,
                )
            desc_i: T.uint32
            T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
                T.address_of(desc_i),
                d_dtype="float32",
                a_dtype="float4_e2m1fn",
                b_dtype="float4_e2m1fn",
                sfa_dtype="float8_e4m3fn",
                sfb_dtype="float8_e4m3fn",
                sfa_tmem_addr=SFA_tmem.allocated_addr[0],
                sfb_tmem_addr=SFB_tmem.allocated_addr[0],
                M=CTA_M * CTA_GROUP,
                N=MMA_N,
                K=MMA_K,
                trans_a=False,
                trans_b=False,
                n_cta_groups=CTA_GROUP,
            )
            for ki in T.unroll(MMA_K_BLOCKS):
                desc_a_ki = T.meta_var(
                    desc_a.add_16B_offset((stage * CTA_M * CTA_K + ki * MMA_K) // 32)
                )
                desc_b_ki = T.meta_var(
                    desc_b.add_16B_offset((stage * CTA_N * CTA_K + ki * MMA_K) // 32)
                )
                sf_linear = T.meta_var(ki * sf_mma_k)
                T.ptx[_MMA_NVFP4_2SM](
                    T.cast(tmem.allocated_addr[0], "uint32"),
                    desc_a_ki,
                    desc_b_ki,
                    desc_i,
                    T.cuda.get_tmem_addr(
                        SFA_tmem.allocated_addr[0],
                        sf_linear % 512 // 16,
                        sf_linear % 16 // sf_mma_k * sf_mma_k + sf_linear // 512,
                    ),
                    T.cuda.get_tmem_addr(
                        SFB_tmem.allocated_addr[0],
                        sf_linear % 512 // 16,
                        sf_linear % 16 // sf_mma_k * sf_mma_k * SFB_n_chunks + sf_linear // 512,
                    ),
                    tvm.tirx.any(ki != 0, T.cast(accum, "bool")),
                )
            accum = 1
            smem_pipe.empty.arrive(mma_smem.stage, cta_group=CTA_GROUP, cta_mask=pair_mask)

        if T.cuda.elect_sync():
            while tile_scheduler.valid():
                tmem_pipe.empty.wait(mma_tmem.stage, mma_tmem.phase)
                accum = 0
                for k_tile in T.serial(K_TILES):
                    execute_mma()
                    mma_smem.advance()
                tmem_pipe.full.arrive(mma_tmem.stage, cta_group=CTA_GROUP, cta_mask=pair_mask)
                mma_tmem.advance()
                tile_scheduler.next_tile()
    elif warp_id >= int(WarpRole.EPILOGUE):

        @T.inline
        def regs_to_smem(reg_bf16_words, chunk_index: T.constexpr, fragment_cols: T.constexpr):
            # Each epilogue warp owns 32 rows.  stmatrix.x4 publishes two
            # 16-row halves and one 16-column band at a time.  The former
            # ldstmatrix dispatcher addressed the physical row-major tile
            # first and then applied the hardware atom swizzle.  Reproduce
            # that ordering explicitly instead of applying the buffer's
            # logical tiled layout to the stmatrix seed coordinate.
            swizzle_mask = T.meta_var(0x40 if EPI_TILE == 16 else 0xC0 if EPI_TILE == 32 else 0x1C0)
            for cj in T.unroll(EPI_TILE // 16):
                for mm in T.unroll(2):
                    linear = T.meta_var(
                        epi_wb_state.stage * CTA_M * EPI_TILE
                        + warp_id % 4 * 32 * EPI_TILE
                        + (lane_id % 16 + mm * 16) * EPI_TILE
                        + lane_id // 16 * 8
                        + cj * 16
                    )
                    swizzled = T.meta_var(
                        T.bitwise_xor(
                            linear, T.shift_right(T.bitwise_and(linear, swizzle_mask), T.int32(3))
                        )
                    )
                    word_base = T.meta_var(
                        mm * (fragment_cols // 4) + chunk_index * (EPI_TILE // 4) + cj * 4
                    )
                    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        T.ptr_byte_offset(output_smem.ptr_to([0, 0, 0]), swizzled * 2, "bfloat16"),
                        reg_bf16_words[word_base],
                        reg_bf16_words[word_base + 1],
                        reg_bf16_words[word_base + 2],
                        reg_bf16_words[word_base + 3],
                    )

        @T.inline
        def epilogue():
            tmem_pipe.full.wait(epi_cur.stage, epi_cur.phase)

            # Per-chunk store: R->S (stmatrix) then S->G (TMA). Shared by both schedules.
            @T.inline
            def store_epi_chunk(
                reg_bf16_words,
                chunk_index: T.constexpr,
                linear_n: T.constexpr,
                fragment_cols: T.constexpr,
            ):
                T.ptx.cp.async_.bulk.wait_group.read(WB_PIPE_DEPTH - 1)
                T.cuda.warpgroup_sync(1)
                regs_to_smem(reg_bf16_words, chunk_index, fragment_cols)
                T.cuda.warpgroup_sync(1)
                d_n_out: T.int32
                d_n_out = d_n + linear_n
                if tid_in_wg == 0:
                    T.ptx.fence.proxy.async_.shared__cta()
                    T.evaluate(
                        T.ptx[_TMA_S2G_EVICT_FIRST](
                            T.address_of(D_tensor_map),
                            T.cast(d_n_out, "int32"),
                            T.cast(d_m, "int32"),
                            output_smem.ptr_to([epi_wb_state.stage, 0, 0]),
                            T.uint64(_EVICT_FIRST_L2_POLICY),
                        )
                    )
                    T.ptx.cp.async_.bulk.commit_group()
                epi_wb_state.advance()

            # Fusion vs fission of {load; scale+cast; store}: overlap fuses and reuses
            # a small (128, EPI_TILE) frag; non-overlap splits the loops, needing a big
            # (128, MMA_N) frag (all chunks live between load and store).
            if OVERLAP_EPI:
                reg_f32 = T.alloc_local((EPI_TILE,), "float32")
                reg_bf16_words = T.alloc_local((EPI_TILE // 2,), "uint32", align=16)
                for no in T.unroll(MMA_N // EPI_TILE):
                    linear_n = T.meta_var(no * EPI_TILE)
                    for slab in T.unroll(2):
                        reg_base = T.meta_var(slab * (EPI_TILE // 2))
                        T.ptx[TMEM_LD](
                            *[reg_f32[reg_base + j] for j in range(EPI_TILE // 2)],
                            T.cuda.get_tmem_addr(tmem.allocated_addr[0], slab * 16, linear_n),
                        )
                    if no == MMA_N // EPI_TILE - 1:
                        T.ptx.tcgen05.wait__ld.sync.aligned()
                        if tid_in_wg == 0:
                            tmem_pipe.empty.arrive(
                                epi_cur.stage, remote=pair_leader_rank, pred=True, count=1
                            )
                    for pair in T.unroll(EPI_TILE // 2):
                        _mul_f32x2_inplace(reg_f32, pair * 2, alpha_local)
                    for pair in T.unroll(EPI_TILE // 2):
                        # PTX's packed cvt names the high-half source first.
                        T.ptx.cvt.rn.bf16x2.f32(
                            reg_bf16_words[pair], reg_f32[pair * 2 + 1], reg_f32[pair * 2]
                        )
                    store_epi_chunk(reg_bf16_words, 0, linear_n, EPI_TILE)
            else:
                # Preserve the dispatcher's two-slab register ordering: the
                # first/second 64 TMEM rows occupy the two local halves.
                reg_all_f32 = T.alloc_local((MMA_N,), "float32")
                reg_all_pairs = reg_all_f32.view("uint64")
                reg_all_bf16_words = T.alloc_local((MMA_N // 2,), "uint32", align=16)
                for no in T.unroll(MMA_N // EPI_TILE):
                    linear_n = T.meta_var(no * EPI_TILE)
                    for slab in T.unroll(2):
                        reg_base = T.meta_var(no * (EPI_TILE // 2) + slab * (MMA_N // 2))
                        T.ptx[TMEM_LD](
                            *[reg_all_f32[reg_base + j] for j in range(EPI_TILE // 2)],
                            T.cuda.get_tmem_addr(tmem.allocated_addr[0], slab * 16, linear_n),
                        )
                T.ptx.tcgen05.wait__ld.sync.aligned()
                for pair in T.unroll(MMA_N // 2):
                    T.ptx.mul.rz.ftz.f32x2(
                        reg_all_pairs[pair],
                        T.cuda.make_float2(reg_all_f32[pair * 2], reg_all_f32[pair * 2 + 1]),
                        T.cuda.make_float2(alpha_local, alpha_local),
                    )
                for pair in T.unroll(MMA_N // 2):
                    T.ptx.cvt.rn.bf16x2.f32(
                        reg_all_bf16_words[pair], reg_all_f32[pair * 2 + 1], reg_all_f32[pair * 2]
                    )
                if tid_in_wg == 0:
                    tmem_pipe.empty.arrive(
                        epi_cur.stage, remote=pair_leader_rank, pred=True, count=1
                    )
                T.cuda.warpgroup_sync(1)
                for no in T.unroll(MMA_N // EPI_TILE):
                    linear_n = T.meta_var(no * EPI_TILE)
                    store_epi_chunk(reg_all_bf16_words, no, linear_n, MMA_N)

        while tile_scheduler.valid():
            epilogue()
            epi_cur.advance()
            tile_scheduler.next_tile()
        if tid_in_wg == 0:
            T.ptx.cp.async_.bulk.wait_group.read(0)
        T.cuda.warpgroup_sync(1)
    if warp_id == int(WarpRole.EPILOGUE):
        if T.cuda.elect_sync():
            _rem3 = T.alloc_local([1], "uint64")
            T.ptx.mapa.shared__cluster.u64(
                _rem3[0], tmem_finished.ptr_to([0]), T.uint32(pair_leader_rank + 1 - id_in_pair)
            )
            T.ptx.mbarrier.arrive.b64(_rem3[0], T.uint32(1), pred=T.bool(True))
        T.cuda.mbarrier_wait_acquire_cluster(tmem_finished.ptr_to([0]), 0)
        tmem_dealloc_addr: T.uint32
        T.ptx.ld.shared.u32(tmem_dealloc_addr, tmem_addr.ptr_to([0]))
        T.ptx[f"tcgen05.dealloc.cta_group::{CTA_GROUP}.sync.aligned.b32"](
            tmem_dealloc_addr, T.uint32(512)
        )


def tir_ws_kernel(M: int, N: int, K: int):
    assert M % 128 == 0 and N % 256 == 0 and K % 256 == 0
    assert (M // 128) % 2 == 0
    assert (K // 16) % 4 == 0
    config = dict(TIRX_CONFIGS.get((M, N, K), {}))
    return _kernel.specialize(M=M, N=N, K=K, **config)


TIRX_CONFIGS = {
    # Per-shape launch/pipeline tuning. The cluster N tile spans CTA_GROUP CTAs,
    # so CTA_N = (cluster N tile) / CTA_GROUP.
    (1024, 1024, 1024): {
        "SM_COUNT": 64,
        "CTA_N": 64,
        "EPI_TILE": 32,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 12,
        "OVERLAP_EPI": True,
    },
    (2048, 2048, 2048): {
        "SM_COUNT": 128,
        "CTA_N": 128,
        "EPI_TILE": 32,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 4,
        "OVERLAP_EPI": True,
    },
    (4096, 4096, 4096): {
        "SM_COUNT": 148,
        "CTA_N": 128,
        "EPI_TILE": 32,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 4,
        "OVERLAP_EPI": False,
    },
    (8192, 8192, 8192): {
        "SM_COUNT": 148,
        "CTA_N": 128,
        "EPI_TILE": 16,
        "PIPE_DEPTH": 4,
        "L2_GROUP_SIZE": 1,
        "OVERLAP_EPI": False,
    },
    (16384, 16384, 16384): {
        "SM_COUNT": 148,
        "CTA_N": 128,
        "EPI_TILE": 16,
        "PIPE_DEPTH": 4,
        "L2_GROUP_SIZE": 12,
        "OVERLAP_EPI": False,
    },
}


KERNEL_META = {"name": "nvfp4_gemm", "category": "basic", "compute_capability": 10}
CONFIGS = [
    {"M": s, "N": s, "K": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]


def get_kernel(M, N, K):
    return tir_ws_kernel(M, N, K)


def _compile_executable(M: int, N: int, K: int):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(get_kernel(M, N, K))


def run_test(M=1024, N=1024, K=1024):
    """Compile, run, and verify kernel."""
    import torch
    import torch.nn.functional as F

    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = prepare_data(M, N, K)
    alpha_tensor = torch.tensor([alpha], device="cuda", dtype=torch.float)
    out = torch.empty_like(C_ref).to("cuda").to(torch.bfloat16)
    ex = _compile_executable(M, N, K)
    ex.mod(A_fp4, B_fp4, A_sf, B_sf, alpha_tensor, out)
    cosine_sim = F.cosine_similarity(
        out.reshape(-1).float(), C_ref.to("cuda").reshape(-1).float(), dim=0
    )
    assert cosine_sim > 0.97, f"nvfp4_gemm cosine_sim {cosine_sim:.6f} <= 0.97"


def _flashinfer_autotune_cache_path(
    M: int, N: int, K: int, *, backend: str = "auto"
) -> Path | None:
    """Return an environment-specific, per-shape FlashInfer cache path."""
    cache_root = os.environ.get("TIRX_BENCH_CACHE_DIR")
    if not cache_root:
        return None

    # FlashInfer rejects cache metadata from a different software/GPU stack.
    # Put each stack in its own directory as well, so an obsolete file cannot
    # prevent the current process from saving its newly tuned result.
    import flashinfer
    from flashinfer.autotuner import _collect_metadata

    environment = _collect_metadata()
    digest = hashlib.sha256(
        json.dumps(environment, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    version = str(getattr(flashinfer, "__version__", "unknown")).replace("/", "_")
    backend_suffix = "" if backend == "auto" else f"_{backend}"
    return (
        Path(cache_root)
        / "flashinfer"
        / f"{version}-{digest}"
        / f"nvfp4_gemm{backend_suffix}_{M}x{N}x{K}.json"
    )


def _flashinfer_tuned_choice(
    M: int, N: int, K: int, cache_path: Path | None, *, expected_runner: str | None = None
) -> tuple[str, object]:
    """Read the exact-shape runner/tactic selected by FlashInfer's autotuner."""
    choices: list[tuple[str, object]] = []
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        packed_shape_prefix = f"(({M}, {K // 2}), ({K // 2}, {N})"
        choices.extend(
            (value[0], value[1])
            for key, value in payload.items()
            if key.startswith("('fp4_gemm', ") and packed_shape_prefix in key
        )
    else:
        from flashinfer.autotuner import AutoTuner

        for key, (_, tactic, _) in AutoTuner.get().profiling_cache.items():
            if (
                key.custom_op == "fp4_gemm"
                and len(key.nearest_profile) >= 2
                and key.nearest_profile[0] == (M, K // 2)
                and key.nearest_profile[1] == (K // 2, N)
            ):
                choices.append((key.runner_class_name, tactic))

    if expected_runner is not None:
        choices = [choice for choice in choices if choice[0] == expected_runner]

    unique = list(
        dict.fromkeys((runner, json.dumps(tactic, sort_keys=True)) for runner, tactic in choices)
    )
    if len(unique) != 1:
        raise RuntimeError(
            "FlashInfer autotune did not produce exactly one fp4_gemm choice "
            f"for M={M}, N={N}, K={K}, expected_runner={expected_runner}: {choices}"
        )
    runner, tactic_json = unique[0]
    tactic = json.loads(tactic_json)
    if tactic == -1:
        raise RuntimeError(
            f"FlashInfer autotune fell back to {runner} tactic=-1 for M={M}; "
            "refusing to benchmark an untuned fallback"
        )
    return runner, tactic


# timer=None inherits the global default (proton). Proton matters here: the
# flashinfer/cublaslt references carry heavy per-call host dispatch (Python + internal
# cudaDeviceSynchronize), and since the nvfp4 kernel (~28µs) is faster than that dispatch,
# event wall-clock is host-starved and over-credits us ~4x. Proton measures pure GPU
# kernel time -> honest ~parity (verified 0.996 vs event 4.11).
def prepare_bench(M=1024, N=1024, K=1024, **kwargs):
    """Compile TIRx and the device-independent cuBLASLt extension before READY."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    state = {
        "config": {"M": M, "N": N, "K": K, **kwargs},
        "executable": _compile_executable(M, N, K),
        "cublaslt_extension": _load_cublaslt_nvfp4_ext(),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, **kwargs):
    """Benchmark."""
    import flashinfer
    import torch

    config_kwargs = {**prepared["config"], **kwargs}
    M = config_kwargs.pop("M")
    N = config_kwargs.pop("N")
    K = config_kwargs.pop("K")
    metadata = {}
    ex = prepared["executable"]

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = prepare_data(M, N, K)
    alpha_value = float(alpha.item())
    alpha_tensor = torch.tensor([alpha_value], device="cuda", dtype=torch.float)
    out_tir = torch.empty_like(C_ref).to("cuda").to(torch.bfloat16)

    funcs = {"tir": lambda: ex.mod(A_fp4, B_fp4, A_sf, B_sf, alpha_tensor, out_tir)}
    flashinfer_backend = os.environ.get("TIRX_NVFP4_FLASHINFER_BACKEND", "auto")
    if flashinfer_backend not in {"auto", "cutlass"}:
        raise ValueError(
            f"TIRX_NVFP4_FLASHINFER_BACKEND must be 'auto' or 'cutlass', got {flashinfer_backend!r}"
        )
    expected_flashinfer_runner = "CutlassFp4GemmRunner" if flashinfer_backend == "cutlass" else None
    flashinfer_cache_path = _flashinfer_autotune_cache_path(M, N, K, backend=flashinfer_backend)
    flashinfer_context_kwargs = {"tuning_buckets": (M,), "round_up": False}
    if flashinfer_cache_path is not None:
        flashinfer_context_kwargs["cache"] = str(flashinfer_cache_path)

    def _flashinfer():
        out_fi = torch.empty_like(out_tir)
        cache_hit_before_tune = False
        if flashinfer_cache_path is not None and flashinfer_cache_path.exists():
            try:
                _flashinfer_tuned_choice(
                    M, N, K, flashinfer_cache_path, expected_runner=expected_flashinfer_runner
                )
                cache_hit_before_tune = True
            except (json.JSONDecodeError, RuntimeError):
                pass

        def run():
            return flashinfer.mm_fp4(
                A_fp4,
                B_fp4.T,
                A_sf,
                B_sf.T,
                alpha,
                out=out_fi,
                block_size=16,
                backend=flashinfer_backend,
                use_nvfp4=True,
            )

        # Tune/load exactly this benchmark shape and persist the selection in
        # the suite cache. Both profiling and all cache I/O happen before the
        # launch closure is handed to bench().
        with flashinfer.autotune(True, **flashinfer_context_kwargs):
            run()
        torch.cuda.synchronize()

        # Exercise the normal non-tuning lookup once before timing and reject
        # a silent heuristic fallback. Keep the exact same bucket override that
        # was used while tuning: cuDNN runner cache keys include its mapper.
        with flashinfer.autotune(False, **flashinfer_context_kwargs):
            run()
        torch.cuda.synchronize()
        runner, tactic = _flashinfer_tuned_choice(
            M, N, K, flashinfer_cache_path, expected_runner=expected_flashinfer_runner
        )
        sample_rows = min(M, 256)
        sample_cols = min(N, 256)
        cosine_similarity = torch.nn.functional.cosine_similarity(
            out_fi[:sample_rows, :sample_cols].reshape(-1).float(),
            C_ref[:sample_rows, :sample_cols].reshape(-1).float(),
            dim=0,
        ).item()
        if cosine_similarity <= 0.97:
            raise RuntimeError(
                "FlashInfer tuned NVFP4 output failed validation: "
                f"cosine_similarity={cosine_similarity:.6f}"
            )
        metadata.update(
            {
                "flashinfer_autotune_cache": (
                    "hit"
                    if cache_hit_before_tune
                    else "miss"
                    if flashinfer_cache_path is not None
                    else "memory"
                ),
                "flashinfer_tuning_bucket": M,
                "flashinfer_requested_backend": flashinfer_backend,
                "flashinfer_runner": runner,
                "flashinfer_tactic": tactic,
                "flashinfer_cosine_similarity": cosine_similarity,
            }
        )
        return run

    def _cublaslt():
        ext = prepared["cublaslt_extension"]
        out_cublaslt = torch.empty_like(out_tir)
        return lambda: ext.nvfp4_cublaslt(
            A_fp4, B_fp4, A_sf, B_sf, alpha_value, out_cublaslt, M, N, K
        )

    # FlashInfer is a required reference for this benchmark. Prepare and
    # validate its tuned launch before entering bench() so a bad/missing cache
    # fails the workload instead of being downgraded to an optional baseline
    # construction error.
    flashinfer_run = _flashinfer()
    # Load the file and install the exact-M mapper once, outside all timer
    # calls. Timed FlashInfer launches then perform only an in-memory cache
    # lookup and the selected kernel launch.
    with flashinfer.autotune(False, **flashinfer_context_kwargs):
        result = bench(
            funcs,
            warmup=warmup,
            repeat=repeat,
            timer=timer,
            references={
                "flashinfer": lambda: flashinfer_run,
                "cublaslt_nvfp4": _cublaslt,
            },
            **config_kwargs,
        )
    result["metadata"] = {**result.get("metadata", {}), **metadata}
    return result


def run_bench(M=1024, N=1024, K=1024, *, warmup=None, repeat=None, timer=None, **kwargs):
    protocol = {name: kwargs.pop(name) for name in ("rounds", "cooldown_s") if name in kwargs}
    return prepare_bench(M=M, N=N, K=K, **kwargs).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, **protocol
    )
