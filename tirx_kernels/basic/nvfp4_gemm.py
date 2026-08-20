# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import ctypes
import fcntl
import functools
import hashlib
import json
import os
import subprocess
import sys
from enum import IntEnum
from pathlib import Path

import tirx_kernels.kern as K
import tvm
from tirx_kernels.runner import PREPARE_CUDA_ARCH_ENV, bench
from tvm.backend.cuda.lang import MBarrier, Pipeline, PipelineState, TMABar
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.script.builder import ir as I


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
    """Build if needed, then load the shape-independent cuBLASLt reference."""
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
        prepare_arch = os.environ.get(PREPARE_CUDA_ARCH_ENV, "sm_100a")
        arch = prepare_arch.removeprefix("sm_")
        extra_cuda_cflags = [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
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


def _build_cublaslt_nvfp4_ext() -> None:
    """Populate the extension cache without initializing CUDA in this process."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tirx_kernels.basic.nvfp4_gemm import "
            "_load_cublaslt_nvfp4_ext; _load_cublaslt_nvfp4_ext()",
        ],
        check=True,
        env=env,
    )


_W_MMA = int(WarpRole.MMA)
_W_TMA = int(WarpRole.TMA)
_W_EPILOGUE = int(WarpRole.EPILOGUE)
_DEFAULTS = {
    "SM_COUNT": 148,
    "CTA_GROUP": 2,
    "CLUSTER_M": 2,
    "CLUSTER_N": 1,
    "CTA_M": 128,
    "CTA_N": 128,
    "CTA_K": 256,
    "MMA_K": 64,
    "EPI_TILE": 64,
    "TMEM_LD_SIZE": 64,
    "WB_PIPE_DEPTH": 2,
    "PIPE_DEPTH": 5,
    "TMEM_PIPE_DEPTH": 1,
    "L2_GROUP_SIZE": 8,
    "NUM_WARPS": 8,
    "OVERLAP_EPI": True,
}


def _mapa_u64(ptr, rank):
    """The original's defining ``mapa.u64`` spelling."""
    mapped = K.local_scalar("uint64")
    K.ptx.mapa.u64(mapped, ptr, K.uint32(rank))
    return mapped


def _mul_f32x2_inplace(values, index, multiplier):
    """The original inline helper, preserving both explicit registers."""
    packed = K.local_scalar("uint64")
    rhs = K.local_scalar("uint64")
    K.ptx.mov.b64(packed, values[index], values[index + 1])
    K.ptx.mov.b64(rhs, multiplier, multiplier)
    K.ptx.mul.rz.ftz.f32x2(packed, packed, rhs)
    K.ptx.mov.b64(values[index], values[index + 1], packed)


def make_kernel(M, N, KDIM):
    """Trace one registry shape with every original config value baked in."""
    if M % 128 != 0 or N % 256 != 0 or KDIM % 256 != 0:
        raise ValueError("M, N, K must be divisible by 128, 256, 256 respectively")
    if (M // 128) % 2 != 0:
        raise ValueError("M/128 must be divisible by the two-CTA cluster")
    if (KDIM // 16) % 4 != 0:
        raise ValueError("K/16 must be divisible by four")

    cfg = {**_DEFAULTS, **TIRX_CONFIGS.get((M, N, KDIM), {})}
    SM_COUNT = cfg["SM_COUNT"]
    CTA_GROUP = cfg["CTA_GROUP"]
    CLUSTER_M = cfg["CLUSTER_M"]
    CLUSTER_N = cfg["CLUSTER_N"]
    CTA_M = cfg["CTA_M"]
    CTA_N = cfg["CTA_N"]
    CTA_K = cfg["CTA_K"]
    MMA_K = cfg["MMA_K"]
    EPI_TILE = cfg["EPI_TILE"]
    WB_PIPE_DEPTH = cfg["WB_PIPE_DEPTH"]
    PIPE_DEPTH = cfg["PIPE_DEPTH"]
    TMEM_PIPE_DEPTH = cfg["TMEM_PIPE_DEPTH"]
    L2_GROUP_SIZE = cfg["L2_GROUP_SIZE"]
    NUM_WARPS = cfg["NUM_WARPS"]
    OVERLAP_EPI = cfg["OVERLAP_EPI"]

    CLUSTER_SIZE = CLUSTER_M * CLUSTER_N
    MMA_N = CTA_N * CTA_GROUP
    SFB_N = MMA_N
    MMA_K_BLOCKS = CTA_K // MMA_K
    SF_CTA_K = CTA_K // 16
    NUM_CLUSTERS = SM_COUNT // CLUSTER_SIZE
    D_SWIZZLE_MODE = K.SW32B if EPI_TILE == 16 else K.SW64B if EPI_TILE == 32 else K.SW128B
    A_BYTES = CTA_M * (CTA_K // 2) * CTA_GROUP
    B_BYTES = CTA_N * (CTA_K // 2) * CTA_GROUP
    SFA_BYTES = CTA_M * SF_CTA_K * CTA_GROUP
    SFB_BYTES = SFB_N * SF_CTA_K * CTA_GROUP
    K_TILES = KDIM // CTA_K
    CLUSTER_M_TILES = M // CTA_M // CLUSTER_M
    CLUSTER_N_TILES = N // MMA_N // CLUSTER_N
    TMEM_LD = _TMEM_LD_X2 if EPI_TILE == 16 else _TMEM_LD_X4 if EPI_TILE == 32 else _TMEM_LD_X8

    @K.kernel(warps=NUM_WARPS, arch="sm_100a", grid=SM_COUNT)
    def nvfp4_gemm_kern(
        A_tensor_map: K.TensorMap,
        B_tensor_map: K.TensorMap,
        SFA_tensor_map: K.TensorMap,
        SFB_tensor_map: K.TensorMap,
        D_tensor_map: K.TensorMap,
        alpha: K.gptr[K.f32],
    ):
        # The cluster-local rank remains a low-level scope; K owns the global
        # persistent-CTA grid.
        cluster_rank = I.cta_id_in_cluster([CLUSTER_SIZE], preferred=[CLUSTER_SIZE])
        cta_idx = K.cta_id()
        tid_in_cta = K.thread_id()
        lane_id = K.lane_id()
        warp_id = K.warp_id()

        sp = K.specialize()
        r_mma = sp.role("mma", warps=[_W_MMA])
        r_idle = sp.role("idle", warps=[_W_MMA + 1])
        r_tma = sp.role("tma", warps=[_W_TMA])
        r_scale = sp.role("scale", warps=[_W_TMA + 1])
        r_epi = sp.role("epilogue", warps=list(range(_W_EPILOGUE, NUM_WARPS)))

        with r_mma:
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                for tmap in (
                    A_tensor_map,
                    B_tensor_map,
                    SFA_tensor_map,
                    SFB_tensor_map,
                    D_tensor_map,
                ):
                    K.ptx.prefetch.tensormap(K.address_of(tmap))

        cb_m = cluster_rank % CLUSTER_M
        cb_n = cluster_rank // CLUSTER_M
        pair_id = cluster_rank // CTA_GROUP
        id_in_pair = cluster_rank % CTA_GROUP
        pair_leader_rank = pair_id * CTA_GROUP
        tile_scheduler = ClusterPersistentScheduler2D(
            "tile_scheduler",
            num_m_tiles=CLUSTER_M_TILES,
            num_n_tiles=CLUSTER_N_TILES,
            num_clusters=NUM_CLUSTERS,
            l2_group_size=L2_GROUP_SIZE,
        )
        tile_scheduler.init(cta_idx // CLUSTER_SIZE)
        m_idx = tile_scheduler.m_idx
        n_idx = tile_scheduler.n_idx
        cta_m = m_idx * CLUSTER_M + cb_m
        cta_n = n_idx * CLUSTER_N + cb_n
        a_m = cta_m * CTA_M
        d_m = cta_m * CTA_M
        b_n = cta_n * MMA_N + id_in_pair * CTA_N
        d_n = cta_n * MMA_N

        # One underlying in-tree pool.  K only gives A/B/output their swizzled
        # tensor views; all protocol allocations and explicit rebases use the
        # original pool directly.
        smem = K.smem_pool()
        pool = smem.pool
        A_smem_packed = smem.alloc((PIPE_DEPTH, CTA_M, CTA_K // 2), K.u8, swizzle=K.SW128B)
        B_smem_packed = smem.alloc((PIPE_DEPTH, CTA_N, CTA_K // 2), K.u8, swizzle=K.SW128B)
        SFA_smem = smem.alloc((PIPE_DEPTH, CTA_M, SF_CTA_K), "uint8", align=1024)
        SFB_smem = smem.alloc((PIPE_DEPTH, SFB_N, SF_CTA_K), "uint8", align=1024)
        output_smem = smem.alloc((WB_PIPE_DEPTH, CTA_M, EPI_TILE), K.bf16, swizzle=D_SWIZZLE_MODE)
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
        smem.commit()
        with K.If(mbar_leader), K.Then():
            K.ptx.fence.mbarrier_init.release.cluster()

        tmem = K.decl_buffer((CTA_M, 512), "float32", scope="tmem", allocated_addr=0)
        A_smem = A_smem_packed.buf.view("float4_e2m1fn")
        B_smem = B_smem_packed.buf.view("float4_e2m1fn")
        sf_mma_k = 4
        SFB_n_chunks = SFB_N // 128
        SFA_tmem = K.decl_buffer(
            (128, sf_mma_k * MMA_K_BLOCKS), "float8_e4m3fn", scope="tmem", allocated_addr=448
        )
        SFB_tmem = K.decl_buffer(
            (128 * SFB_n_chunks, sf_mma_k * MMA_K_BLOCKS),
            "float8_e4m3fn",
            scope="tmem",
            allocated_addr=464,
        )
        sf_desc = K.SmemDescriptor()
        desc_a = K.SmemDescriptor()
        desc_b = K.SmemDescriptor()
        sf_desc.init(K.reinterpret("handle", K.uint64(0)), ldo=0, sdo=8, swizzle=0)
        desc_a.init(A_smem.ptr_to([0, 0, 0]), ldo=0, sdo=64, swizzle=3)
        desc_b.init(B_smem.ptr_to([0, 0, 0]), ldo=0, sdo=64, swizzle=3)
        with K.If(warp_id == 0), K.Then():
            K.ptx[f"tcgen05.alloc.cta_group::{CTA_GROUP}.sync.aligned.shared::cta.b32"](
                K.address_of(tmem_addr[0]), K.uint32(512)
            )
            K.cuda.warp_sync()
        K.ptx.barrier.cluster.arrive.release.aligned()
        K.ptx.barrier.cluster.wait.acquire()
        with K.If(tid_in_cta < 32), K.Then():
            K.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{CTA_GROUP}.sync.aligned"]()

        pair_mask = K.local_scalar("int32")
        K.assign(pair_mask, K.int32(0))
        K.assign(pair_mask, pair_mask | (K.int32(1) << pair_leader_rank))
        K.assign(pair_mask, pair_mask | (K.int32(1) << (pair_leader_rank + 1)))
        tma_cur = PipelineState(PIPE_DEPTH, 1)
        mma_smem = PipelineState(PIPE_DEPTH, 0)
        mma_tmem = PipelineState(TMEM_PIPE_DEPTH, 1)
        accum = K.local_scalar("int32")
        K.assign(accum, 0)
        epi_cur = PipelineState(TMEM_PIPE_DEPTH, 0)
        epi_wb_state = PipelineState(WB_PIPE_DEPTH, 1)
        alpha_local = K.local_scalar("float32")
        K.ptx.ld.global_.nc.f32(alpha_local, alpha.ptr_to([0]))

        # These five role blocks are intentionally adjacent: K folds their
        # guards into the original if/elif dispatch chain.
        with r_tma:

            def issue_tma_load(k_tile):
                stage = tma_cur.stage
                k = k_tile * CTA_K // 2
                smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
                with K.If(id_in_pair == 0), K.Then():
                    rem = K.local_scalar("uint64")
                    K.ptx.mapa.shared__cluster.u64(
                        rem, tile_full_bar.ptr_to([stage]), K.uint32(pair_leader_rank)
                    )
                    K.ptx.mbarrier.arrive.expect_tx.b64(
                        rem, K.uint32(A_BYTES + B_BYTES), pred=K.bool(True)
                    )
                single_cta_mask = K.int32(1) << id_in_pair
                mapped_tile_bar = _mapa_u64(tile_full_bar.ptr_to([stage]), 0)
                K.ptx[_TMA_G2S_2D](
                    A_smem_packed[stage].ptr_to(0, 0),
                    K.address_of(A_tensor_map),
                    K.Cast("int32", k),
                    K.Cast("int32", a_m),
                    K.cuda.cvta_generic_to_shared(K.reinterpret("handle", mapped_tile_bar)),
                    K.Cast("uint16", single_cta_mask),
                    K.uint64(_EVICT_NORMAL_L2_POLICY),
                )
                K.ptx[_TMA_G2S_2D](
                    B_smem_packed[stage].ptr_to(0, 0),
                    K.address_of(B_tensor_map),
                    K.Cast("int32", k),
                    K.Cast("int32", b_n),
                    K.cuda.cvta_generic_to_shared(K.reinterpret("handle", mapped_tile_bar)),
                    K.Cast("uint16", single_cta_mask),
                    K.uint64(_EVICT_NORMAL_L2_POLICY),
                )

            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                with K.While(tile_scheduler.valid()):
                    with K.serial(K_TILES) as k_tile:
                        issue_tma_load(k_tile)
                        tma_cur.advance()
                    tile_scheduler.next_tile()

        with r_scale:

            def issue_scale_tma_load(k_tile):
                stage = tma_cur.stage
                sf_k = k_tile * SF_CTA_K
                sf_m = (a_m // 128) * 128
                sf_n = (d_n // 128) * 128
                smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
                with K.If(id_in_pair == 0), K.Then():
                    rem = K.local_scalar("uint64")
                    K.ptx.mapa.shared__cluster.u64(
                        rem, scale_full_bar.ptr_to([stage]), K.uint32(pair_leader_rank)
                    )
                    K.ptx.mbarrier.arrive.expect_tx.b64(
                        rem, K.uint32(SFA_BYTES + SFB_BYTES), pred=K.bool(True)
                    )
                single_cta_mask = K.int32(1) << id_in_pair
                mapped_sfa_bar = _mapa_u64(scale_full_bar.ptr_to([stage]), 0)
                K.ptx[_TMA_G2S_3D](
                    SFA_smem.ptr_to([stage, 0, 0]),
                    K.address_of(SFA_tensor_map),
                    K.int32(0),
                    K.Cast("int32", sf_k // 4),
                    K.Cast("int32", sf_m // 128),
                    K.cuda.cvta_generic_to_shared(K.reinterpret("handle", mapped_sfa_bar)),
                    K.Cast("uint16", single_cta_mask),
                    K.uint64(_EVICT_NORMAL_L2_POLICY),
                )
                mapped_sfb_bar = _mapa_u64(scale_full_bar.ptr_to([stage]), 0)
                if SFB_N == 128:
                    with K.If(id_in_pair == 0), K.Then():
                        K.ptx[_TMA_G2S_3D](
                            SFB_smem.ptr_to([stage, 0, 0]),
                            K.address_of(SFB_tensor_map),
                            K.int32(0),
                            K.Cast("int32", sf_k // 4),
                            K.Cast("int32", sf_n // 128),
                            K.cuda.cvta_generic_to_shared(K.reinterpret("handle", mapped_sfb_bar)),
                            K.Cast("uint16", pair_mask),
                            K.uint64(_EVICT_NORMAL_L2_POLICY),
                        )
                else:
                    K.ptx[_TMA_G2S_3D](
                        SFB_smem.ptr_to([stage, cb_m * 128, 0]),
                        K.address_of(SFB_tensor_map),
                        K.int32(0),
                        K.Cast("int32", sf_k // 4),
                        K.Cast("int32", sf_n // 128 + cb_m),
                        K.cuda.cvta_generic_to_shared(K.reinterpret("handle", mapped_sfb_bar)),
                        K.Cast("uint16", pair_mask),
                        K.uint64(_EVICT_NORMAL_L2_POLICY),
                    )

            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                with K.While(tile_scheduler.valid()):
                    with K.serial(K_TILES) as k_tile:
                        issue_scale_tma_load(k_tile)
                        tma_cur.advance()
                    tile_scheduler.next_tile()

        with r_mma:
            with K.If(id_in_pair == 0), K.Then():

                def execute_mma():
                    stage = mma_smem.stage
                    scale_full_bar.wait(mma_smem.stage, mma_smem.phase)
                    tile_full_bar.wait(mma_smem.stage, mma_smem.phase)
                    for flat in range(CTA_M // 32):
                        sfa_row = flat % 4 * 32
                        sfa_shared_addr = K.local_scalar("uint32")
                        K.assign(
                            sfa_shared_addr,
                            K.cuda.cvta_generic_to_shared(
                                K.ptr_byte_offset(
                                    SFA_smem.ptr_to([0, 0, 0]),
                                    (stage * CTA_M + sfa_row) * SF_CTA_K,
                                    "uint8",
                                )
                            ),
                        )
                        sfa_cp_desc = K.local_scalar("uint64")
                        K.assign(
                            sfa_cp_desc,
                            K.bitwise_or(
                                K.bitwise_and(sf_desc.desc, K.bitwise_not(K.uint64(0x3FFF))),
                                K.Cast(
                                    "uint64",
                                    K.bitwise_and(
                                        K.shift_right(sfa_shared_addr, K.uint32(4)),
                                        K.uint32(0x3FFF),
                                    ),
                                ),
                            ),
                        )
                        K.ptx[_TCGEN05_CP_2SM](
                            K.Cast("uint32", SFA_tmem.allocated_addr[0] + flat % 4 * 4), sfa_cp_desc
                        )
                    for flat in range(SFB_N // 32):
                        sfb_row = flat % 4 * 32 + flat // 4 * 128
                        sfb_shared_addr = K.local_scalar("uint32")
                        K.assign(
                            sfb_shared_addr,
                            K.cuda.cvta_generic_to_shared(
                                K.ptr_byte_offset(
                                    SFB_smem.ptr_to([0, 0, 0]),
                                    (stage * SFB_N + sfb_row) * SF_CTA_K,
                                    "uint8",
                                )
                            ),
                        )
                        sfb_cp_desc = K.local_scalar("uint64")
                        K.assign(
                            sfb_cp_desc,
                            K.bitwise_or(
                                K.bitwise_and(sf_desc.desc, K.bitwise_not(K.uint64(0x3FFF))),
                                K.Cast(
                                    "uint64",
                                    K.bitwise_and(
                                        K.shift_right(sfb_shared_addr, K.uint32(4)),
                                        K.uint32(0x3FFF),
                                    ),
                                ),
                            ),
                        )
                        K.ptx[_TCGEN05_CP_2SM](
                            K.Cast(
                                "uint32",
                                SFB_tmem.allocated_addr[0]
                                + flat % 4 * SFB_n_chunks * 4
                                + flat // 4 * 4,
                            ),
                            sfb_cp_desc,
                        )
                    desc_i = K.local_scalar("uint32")
                    K.cuda.tcgen05.encode_instr_descriptor_block_scaled(
                        K.address_of(desc_i),
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
                    for ki in range(MMA_K_BLOCKS):
                        desc_a_ki = desc_a.add_16B_offset(
                            (stage * CTA_M * CTA_K + ki * MMA_K) // 32
                        )
                        desc_b_ki = desc_b.add_16B_offset(
                            (stage * CTA_N * CTA_K + ki * MMA_K) // 32
                        )
                        sf_linear = ki * sf_mma_k
                        K.ptx[_MMA_NVFP4_2SM](
                            K.Cast("uint32", tmem.allocated_addr[0]),
                            desc_a_ki,
                            desc_b_ki,
                            desc_i,
                            K.cuda.get_tmem_addr(
                                SFA_tmem.allocated_addr[0],
                                sf_linear % 512 // 16,
                                sf_linear % 16 // sf_mma_k * sf_mma_k + sf_linear // 512,
                            ),
                            K.cuda.get_tmem_addr(
                                SFB_tmem.allocated_addr[0],
                                sf_linear % 512 // 16,
                                sf_linear % 16 // sf_mma_k * sf_mma_k * SFB_n_chunks
                                + sf_linear // 512,
                            ),
                            K.bool(True) if ki else K.Cast("bool", accum),
                        )
                    K.assign(accum, 1)
                    smem_pipe.empty.arrive(mma_smem.stage, cta_group=CTA_GROUP, cta_mask=pair_mask)

                with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                    with K.While(tile_scheduler.valid()):
                        tmem_pipe.empty.wait(mma_tmem.stage, mma_tmem.phase)
                        K.assign(accum, 0)
                        with K.serial(K_TILES):
                            execute_mma()
                            mma_smem.advance()
                        tmem_pipe.full.arrive(
                            mma_tmem.stage, cta_group=CTA_GROUP, cta_mask=pair_mask
                        )
                        mma_tmem.advance()
                        tile_scheduler.next_tile()

        with r_idle:
            pass

        with r_epi:
            tid_in_wg = K.tid_in_role()

            def regs_to_smem(reg_bf16_words, chunk_index, fragment_cols):
                for cj in range(EPI_TILE // 16):
                    for mm in range(2):
                        row = warp_id % 4 * 32 + lane_id % 16 + mm * 16
                        col = lane_id // 16 * 8 + cj * 16
                        word_base = (
                            mm * (fragment_cols // 4) + chunk_index * (EPI_TILE // 4) + cj * 4
                        )
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                            output_smem[epi_wb_state.stage].ptr_to(row, col),
                            reg_bf16_words[word_base],
                            reg_bf16_words[word_base + 1],
                            reg_bf16_words[word_base + 2],
                            reg_bf16_words[word_base + 3],
                        )

            def store_epi_chunk(reg_bf16_words, chunk_index, linear_n, fragment_cols):
                K.ptx.cp.async_.bulk.wait_group.read(WB_PIPE_DEPTH - 1)
                K.cuda.warpgroup_sync(1)
                regs_to_smem(reg_bf16_words, chunk_index, fragment_cols)
                K.cuda.warpgroup_sync(1)
                d_n_out = d_n + linear_n
                with K.If(tid_in_wg == 0), K.Then():
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.ptx[_TMA_S2G_EVICT_FIRST](
                        K.address_of(D_tensor_map),
                        K.Cast("int32", d_n_out),
                        K.Cast("int32", d_m),
                        output_smem[epi_wb_state.stage].ptr_to(0, 0),
                        K.uint64(_EVICT_FIRST_L2_POLICY),
                    )
                    K.ptx.cp.async_.bulk.commit_group()
                epi_wb_state.advance()

            def epilogue():
                tmem_pipe.full.wait(epi_cur.stage, epi_cur.phase)
                if OVERLAP_EPI:
                    reg_f32 = K.alloc_local((EPI_TILE,), "float32")
                    reg_bf16_words = K.alloc_local((EPI_TILE // 2,), "uint32", align=16)
                    for no in range(MMA_N // EPI_TILE):
                        linear_n = no * EPI_TILE
                        for slab in range(2):
                            reg_base = slab * (EPI_TILE // 2)
                            K.ptx[TMEM_LD](
                                *[reg_f32[reg_base + j] for j in range(EPI_TILE // 2)],
                                K.cuda.get_tmem_addr(tmem.allocated_addr[0], slab * 16, linear_n),
                            )
                        if no == MMA_N // EPI_TILE - 1:
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            with K.If(tid_in_wg == 0), K.Then():
                                tmem_pipe.empty.arrive(
                                    epi_cur.stage, remote=pair_leader_rank, pred=True, count=1
                                )
                        for pair in range(EPI_TILE // 2):
                            _mul_f32x2_inplace(reg_f32, pair * 2, alpha_local)
                        for pair in range(EPI_TILE // 2):
                            K.ptx.cvt.rn.bf16x2.f32(
                                reg_bf16_words[pair], reg_f32[pair * 2 + 1], reg_f32[pair * 2]
                            )
                        store_epi_chunk(reg_bf16_words, 0, linear_n, EPI_TILE)
                else:
                    reg_all_f32 = K.alloc_local((MMA_N,), "float32")
                    reg_all_pairs = reg_all_f32.view("uint64")
                    reg_all_bf16_words = K.alloc_local((MMA_N // 2,), "uint32", align=16)
                    for no in range(MMA_N // EPI_TILE):
                        linear_n = no * EPI_TILE
                        for slab in range(2):
                            reg_base = no * (EPI_TILE // 2) + slab * (MMA_N // 2)
                            K.ptx[TMEM_LD](
                                *[reg_all_f32[reg_base + j] for j in range(EPI_TILE // 2)],
                                K.cuda.get_tmem_addr(tmem.allocated_addr[0], slab * 16, linear_n),
                            )
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    for pair in range(MMA_N // 2):
                        K.ptx.mul.rz.ftz.f32x2(
                            reg_all_pairs[pair],
                            K.cuda.make_float2(reg_all_f32[pair * 2], reg_all_f32[pair * 2 + 1]),
                            K.cuda.make_float2(alpha_local, alpha_local),
                        )
                    for pair in range(MMA_N // 2):
                        K.ptx.cvt.rn.bf16x2.f32(
                            reg_all_bf16_words[pair],
                            reg_all_f32[pair * 2 + 1],
                            reg_all_f32[pair * 2],
                        )
                    with K.If(tid_in_wg == 0), K.Then():
                        tmem_pipe.empty.arrive(
                            epi_cur.stage, remote=pair_leader_rank, pred=True, count=1
                        )
                    K.cuda.warpgroup_sync(1)
                    for no in range(MMA_N // EPI_TILE):
                        store_epi_chunk(reg_all_bf16_words, no, no * EPI_TILE, MMA_N)

            with K.While(tile_scheduler.valid()):
                epilogue()
                epi_cur.advance()
                tile_scheduler.next_tile()
            with K.If(tid_in_wg == 0), K.Then():
                K.ptx.cp.async_.bulk.wait_group.read(0)
            K.cuda.warpgroup_sync(1)

        with K.If(warp_id == _W_EPILOGUE), K.Then():
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                rem = K.local_scalar("uint64")
                K.ptx.mapa.shared__cluster.u64(
                    rem, tmem_finished.ptr_to([0]), K.uint32(pair_leader_rank + 1 - id_in_pair)
                )
                K.ptx.mbarrier.arrive.b64(rem, K.uint32(1), pred=K.bool(True))
            K.cuda.mbarrier_wait_acquire_cluster(tmem_finished.ptr_to([0]), 0)
            tmem_dealloc_addr = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tmem_dealloc_addr, tmem_addr.ptr_to([0]))
            K.ptx[f"tcgen05.dealloc.cta_group::{CTA_GROUP}.sync.aligned.b32"](
                tmem_dealloc_addr, K.uint32(512)
            )

    return nvfp4_gemm_kern


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


SWIZZLE_NONE, SWIZZLE_32B, SWIZZLE_64B, SWIZZLE_128B = 0, 1, 2, 3


class _AlignedTensorMap:
    __slots__ = ("_storage", "ptr")

    def __init__(self):
        self._storage = ctypes.create_string_buffer(192)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tiled(dtype, tensor, *, dims, strides_bytes, box, swizzle):
    descriptor = _AlignedTensorMap()
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        descriptor.ptr,
        dtype,
        len(dims),
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides_bytes,
        *box,
        *((1,) * len(dims)),
        0,
        swizzle,
        2,
        0,
    )
    return descriptor


def _shape_config(M, N, K):
    return {**_DEFAULTS, **TIRX_CONFIGS.get((M, N, K), {})}


def _build_tensor_maps(M, N, K, A, B, SFA, SFB, D):
    cfg = _shape_config(M, N, K)
    cta_m, cta_n, cta_k, epi = (cfg["CTA_M"], cfg["CTA_N"], cfg["CTA_K"], cfg["EPI_TILE"])
    d_swizzle = SWIZZLE_32B if epi == 16 else SWIZZLE_64B if epi == 32 else SWIZZLE_128B
    enc = functools.partial(_encode_tiled)
    return [
        enc(
            "uint8",
            A,
            dims=(K // 2, M),
            strides_bytes=(K // 2,),
            box=(cta_k // 2, cta_m),
            swizzle=SWIZZLE_128B,
        ),
        enc(
            "uint8",
            B,
            dims=(K // 2, N),
            strides_bytes=(K // 2,),
            box=(cta_k // 2, cta_n),
            swizzle=SWIZZLE_128B,
        ),
        enc(
            "uint16",
            SFA,
            dims=(256, K // 64, M // 128),
            strides_bytes=(512, K * 8),
            box=(256, 4, 1),
            swizzle=SWIZZLE_NONE,
        ),
        enc(
            "uint16",
            SFB,
            dims=(256, K // 64, N // 128),
            strides_bytes=(512, K * 8),
            box=(256, 4, 1),
            swizzle=SWIZZLE_NONE,
        ),
        enc(
            "bfloat16", D, dims=(N, M), strides_bytes=(N * 2,), box=(epi, cta_m), swizzle=d_swizzle
        ),
    ]


class _Runner:
    def __init__(self, M, N, K):
        self.shape = M, N, K
        self.lib = make_kernel(M, N, K).compile()
        self._maps = None
        self._map_key = None

    def __call__(self, A, B, SFA, SFB, alpha, D):
        key = tuple(int(tensor.data_ptr()) for tensor in (A, B, SFA, SFB, D))
        if key != self._map_key:
            self._maps = _build_tensor_maps(*self.shape, A, B, SFA, SFB, D)
            self._map_key = key
        self.lib(*[descriptor.ptr for descriptor in self._maps], alpha.view(-1))


KERNEL_META = {"name": "nvfp4_gemm", "category": "basic", "compute_capability": 10}
CONFIGS = [
    {"M": s, "N": s, "K": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]


def get_kernel(M, N, K):
    return make_kernel(M, N, K).func


def _compile_executable(M: int, N: int, K: int):
    return _Runner(M, N, K)


def run_test(M=1024, N=1024, K=1024):
    """Compile, run, and verify kernel."""
    import torch
    import torch.nn.functional as F

    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = prepare_data(M, N, K)
    alpha_tensor = torch.tensor([alpha], device="cuda", dtype=torch.float)
    out = torch.empty_like(C_ref).to("cuda").to(torch.bfloat16)
    ex = _compile_executable(M, N, K)
    ex(A_fp4, B_fp4, A_sf, B_sf, alpha_tensor, out)
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

        for key, (tactic, _) in AutoTuner.get().profiling_cache.items():
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
    """Compile TIRx and populate the cuBLASLt extension cache before READY."""
    from tirx_kernels.runner import external_references_enabled, prepared_gpu_benchmark

    if external_references_enabled():
        _build_cublaslt_nvfp4_ext()
    state = {
        "config": {"M": M, "N": N, "K": K, **kwargs},
        "executable": _compile_executable(M, N, K),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, **kwargs):
    """Benchmark."""
    import flashinfer
    import torch

    from tirx_kernels.runner import external_references_enabled

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
    funcs = {"tir": lambda: ex(A_fp4, B_fp4, A_sf, B_sf, alpha_tensor, out_tir)}
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
        ext = _load_cublaslt_nvfp4_ext()
        out_cublaslt = torch.empty_like(out_tir)
        return lambda: ext.nvfp4_cublaslt(
            A_fp4, B_fp4, A_sf, B_sf, alpha_value, out_cublaslt, M, N, K
        )

    if external_references_enabled():
        # Prepare and validate the tuned reference launch before entering bench().
        flashinfer_run = _flashinfer()
        # Keep the exact-M mapper installed while timing FlashInfer.
        with flashinfer.autotune(False, **flashinfer_context_kwargs):
            result = bench(
                funcs,
                warmup=warmup,
                repeat=repeat,
                timer=timer,
                references={"flashinfer": lambda: flashinfer_run, "cublaslt_nvfp4": _cublaslt},
                **config_kwargs,
            )
    else:
        result = bench(funcs, warmup=warmup, repeat=repeat, timer=timer, **config_kwargs)
    result["metadata"] = {**result.get("metadata", {}), **metadata}
    return result


def run_bench(M=1024, N=1024, K=1024, *, warmup=None, repeat=None, timer=None, **kwargs):
    protocol = {name: kwargs.pop(name) for name in ("rounds", "cooldown_s") if name in kwargs}
    return prepare_bench(M=M, N=N, K=K, **kwargs).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, **protocol
    )
