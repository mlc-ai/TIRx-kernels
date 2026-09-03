// This file is a TIRx port of code from fast.cu
// (https://github.com/pranjalssh/fast.cu @
// 2dfe5e26aecfd9e5f27bf9d5837deea01acda24b), Copyright (c) 2024 Pranjal
// Shankhdhar SPDX-License-Identifier: Apache-2.0 AND MIT
// SPDX-FileCopyrightText: Copyright TIRx authors

#include "gemm9.cuh"

#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

constexpr int kTableMode = 1;

struct SourceHandle {
  CUtensorMap a;
  CUtensorMap b;
  CUtensorMap sfa;
  CUtensorMap sfb;
  __half *c;
  int m;
  int n;
  int k;
  int clusters;
};

struct DevicePlan {
  int sms = 0;
  int clusters = 0;
  bool ready = false;
  l2side::RuntimeMap map;
  sched::Census census;
};

DevicePlan &device_plan() {
  static DevicePlan plan;
  return plan;
}

void configure_source(DevicePlan &plan) {
  if (plan.ready)
    return;
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  plan.sms = prop.multiProcessorCount;
  plan.clusters = plan.sms / nvfp4::CTA_GROUP;
  if (plan.sms != 152 || plan.clusters != 76)
    std::abort();
  CUDA_CHECK(cudaFuncSetAttribute(nvfp4::nvfp4_gemm_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  int(sizeof(nvfp4::SmemCD))));
  CUDA_CHECK(cudaFuncSetAttribute(nvfp4::l2a_cluster_probe,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  int(sizeof(nvfp4::SmemCD))));

  constexpr uint64_t kProbeBytes = uint64_t(2) << 20;
  void *raw = nullptr;
  CUDA_CHECK(cudaMalloc(&raw, 2 * kProbeBytes));
  char *arena = reinterpret_cast<char *>(
      (reinterpret_cast<uintptr_t>(raw) + kProbeBytes - 1) &
      ~(kProbeBytes - 1));
  plan.map = l2side::probe_stable(arena, kProbeBytes, 3);
  CUDA_CHECK(cudaFree(raw));
  if (plan.map.nsm != plan.sms || plan.map.hash != l2side::kExpectedHash ||
      plan.map.model_mismatches != 0) {
    std::abort();
  }

  unsigned *d_smids = nullptr;
  const size_t count = size_t(plan.clusters) * nvfp4::CTA_GROUP;
  CUDA_CHECK(cudaMalloc(&d_smids, count * sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(d_smids, 0xff, count * sizeof(unsigned)));
  nvfp4::l2a_cluster_probe<<<nvfp4::launch_grid(plan.clusters), nvfp4::TB_SIZE,
                             sizeof(nvfp4::SmemCD)>>>(d_smids);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<unsigned> smids(count);
  CUDA_CHECK(cudaMemcpy(smids.data(), d_smids, count * sizeof(unsigned),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_smids));
  plan.census = sched::classify_census(smids, plan.map, plan.sms);
  plan.ready = true;
}

size_t operand_read_bytes(int m, int n, int k) {
  const size_t row_stride = size_t((k / 2 + 15) & ~15);
  const size_t sf_inner = size_t((k + 63) / 64) * 4;
  const size_t sfa = size_t((m + 127) / 128) * 128 * sf_inner;
  const size_t sfb = size_t((n + 127) / 128) * 128 * sf_inner;
  return size_t(m + n) * row_stride + sfa + sfb;
}

std::vector<int> make_schedule(DevicePlan &plan, int m, int n, int k) {
  const int mblocks = (m + nvfp4::CLUSTER_M - 1) / nvfp4::CLUSTER_M;
  const int nblocks = (n + nvfp4::BLOCK_N - 1) / nvfp4::BLOCK_N;
  const int total = mblocks * nblocks;
  if (total > nvfp4::L2A_ROUTE_WORK_CAP)
    std::abort();
  const sched::ScheduleMode mode =
      sched::pick_schedule(operand_read_bytes(m, n, k), mblocks, nblocks);
  std::vector<int> table =
      sched::build_schedule(plan.census, mblocks, nblocks, mode);
  if (table.empty()) {
    table.resize(total);
    std::iota(table.begin(), table.end(), 0);
  }
  return table;
}

} // namespace

extern "C" int fastcu_nvfp4_prepare_schedule(int m, int n, int k, int *route,
                                             int *sm_side, int *cluster_side,
                                             cudaStream_t stream) {
  DevicePlan &plan = device_plan();
  configure_source(plan);
  std::vector<int> table = make_schedule(plan, m, n, k);
  sched::upload_side_constants(plan.map, plan.census);
  sched::upload_table(table, kTableMode);
  std::array<int, nvfp4::L2A_ROUTE_WORK_CAP> route_full{};
  std::copy(table.begin(), table.end(), route_full.begin());
  std::array<int, nvfp4::L2A_CLUSTER_CAP> cluster_full{};
  std::copy(plan.census.cluster_side.begin(), plan.census.cluster_side.end(),
            cluster_full.begin());
  CUDA_CHECK(cudaMemcpyAsync(route, route_full.data(), sizeof(route_full),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(sm_side, plan.map.sm_side.data(),
                             sizeof(plan.map.sm_side), cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(cluster_side, cluster_full.data(),
                             sizeof(cluster_full), cudaMemcpyHostToDevice,
                             stream));
  return 0;
}

extern "C" void *fastcu_nvfp4_create(void *a, void *b, void *sfa, void *sfb,
                                     void *c, int m, int n, int k) {
  DevicePlan &plan = device_plan();
  configure_source(plan);
  auto *handle = new SourceHandle{};
  const int row_stride = (k / 2 + 15) & ~15;
  handle->a = host::make_ab_tmap(a, m, k, row_stride);
  handle->b = host::make_ab_tmap(b, n, k, row_stride);
  handle->sfa = host::make_sf_tmap(static_cast<uint8_t *>(sfa), m, k);
  handle->sfb = host::make_sf_tmap(static_cast<uint8_t *>(sfb), n, k);
  handle->c = static_cast<__half *>(c);
  handle->m = m;
  handle->n = n;
  handle->k = k;
  handle->clusters = plan.clusters;
  return handle;
}

extern "C" int fastcu_nvfp4_launch(void *opaque, cudaStream_t stream) {
  auto *handle = static_cast<SourceHandle *>(opaque);
  nvfp4::nvfp4_gemm_kernel<<<nvfp4::launch_grid(handle->clusters),
                             nvfp4::TB_SIZE, sizeof(nvfp4::SmemCD), stream>>>(
      handle->a, handle->b, handle->sfa, handle->sfb, handle->c, handle->m,
      handle->n, handle->k, kTableMode * nvfp4::L2A_ROUTE_WORK_CAP);
  return int(cudaPeekAtLastError());
}

extern "C" int fastcu_nvfp4_source_placement_errors(unsigned *value) {
  return int(cudaMemcpyFromSymbol(value, nvfp4::l2a_placement_errors,
                                  sizeof(*value), 0, cudaMemcpyDeviceToHost));
}

extern "C" void fastcu_nvfp4_destroy(void *opaque) {
  delete static_cast<SourceHandle *>(opaque);
}

extern "C" int fastcu_nvfp4_ptx_version() { return 93; }
