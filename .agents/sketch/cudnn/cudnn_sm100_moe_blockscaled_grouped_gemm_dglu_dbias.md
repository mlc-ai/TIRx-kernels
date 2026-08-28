<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116),
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 MoE block-scaled grouped GEMM + dGLU backward: coarse WASP pipeline sketch

This is a non-executable execution sketch. It freezes the storage, eight-warp
role split, MoE persistent tile scheduling, asynchronous protocols, tile
dataflow, dGLU backward epilogue, dbias/dprob/amax reductions, and FP8 output
scale-factor path for the single parameterized TIRx module
[`tirx_kernels/cudnn/dglu/moe_blockscaled_grouped_gemm_dglu_dbias.py`](../../../tirx_kernels/cudnn/dglu/moe_blockscaled_grouped_gemm_dglu_dbias.py)
and its device code in
[`_moe_blockscaled_grouped_gemm_dglu_dbias/kernel.py`](../../../tirx_kernels/cudnn/dglu/_moe_blockscaled_grouped_gemm_dglu_dbias/kernel.py).
After the reviewer gate that module is the executable source of truth; this
sketch remains frozen.

The source is
`python/cudnn/gemm/cutedsl/grouped/dglu/moe_blockscaled_grouped_gemm_dglu_dbias.py`
at commit `7b5327b32907b9dd21d85a393d62f9573d7f0116`, SHA256
`c5c4279b3ad578971d64f361897fbe539e1eaa7a94eff035bc0caabcae6a3070`.
Unqualified source-line citations are to that file; `sched N`, `utils N`, and
`ext N` cite `moe_persistent_scheduler.py`, `moe_utils.py`, and
`moe_sched_extension.py`.

The PTX `.file` numbering is **not** uniform across the exports, so resolve a
`.loc` against the export it came from:

| export | .file 2 | .file 3 | .file 4 | .file 5 |
| --- | --- | --- | --- | --- |
| `anchor`, `fp4`, `situglu` | `moe_persistent_scheduler.py` | `moe_utils.py` | `moe_sched_extension.py` | -- |
| `scalar_geglu` | `moe_persistent_scheduler.py` | `moe_utils.py` | `moe_sched_extension.py` | `moe_kernel_helpers.py` |
| `discrete_dynamic` | `moe_utils.py` | `moe_persistent_scheduler.py` | `moe_sched_extension.py` | -- |

Files 2 and 3 are **swapped** in `discrete_dynamic`, which is the export the
dynamic-scheduler annotations cite.

The primary evidence is the writer's line-info export under
`.porting/moe_blockscaled_grouped_gemm_dglu_dbias/writer_source_export/`.
`PTX n` means the numbered line in the 6,657-line `anchor/` artifact, SHA256
`6392fa5214c6b2871166ed600007344820e420b7aa841a83c7f02e4f4ab6fd24`. Four
further exports cover the branches the anchor does not reach:

| export | SHA256 (first 16) | lines | reaches |
| --- | --- | --- | --- |
| `anchor/` | `6392fa5214c6b287` | 6657 | the anchor below |
| `fp4/` | `e5227dcb087af5c7` | 4205 | FP4 E2M1 A/B, E8M0 vec 16, BF16 D, amax |
| `discrete_dynamic/` | `da46c2e0bfdcc44a` | 6977 | the helper kernel, discrete B/SFB descriptors, dynamic scheduler |
| `scalar_geglu/` | `ee14b34c4b28809e` | 7427 | dGeGLU and the non-vectorized epilogue |
| `situglu/` | `c57033e404bb0fb8` | 7483 | dSiTU-GLU on the `situ_beta1 == 4.0` closed form, which replaces the sigmoid with a tanh identity |

The instruction-annotated body is the fixed primary specialization:

| axis | anchor value |
| --- | --- |
| problem | 4 experts x 256 padded tokens, `N=512`, `K=512` |
| weights / schedule | dense, static |
| activation | dSwiGLU, packed `f32x2` |
| A/B | FP8 E4M3, K-major/K-major |
| scale | E8M0, vector size 32 |
| C / D | BF16 / E4M3 with row and column scale factors |
| MMA tile / cluster | `(256, 256)` two-CTA / `(2, 1)` |
| optional outputs | dbias on, prob and dprob on, amax off |
| fixed | FP32 accumulator, CTA tile M 128, `FIX_PAD_SIZE` 256, overlap margin 0 |

The same sketch owns every accepted compile-time branch listed in **Static
specialization boundary**. Those branches alter shapes, descriptor constants,
conversion families, pipeline depths, role presence, and predicates; they do not
add a role or a workflow. `d_dtype = E5M2` is out of scope because the source's
own `cvt_f32x4_to_f8x4_pack_i32` (`1231-1247`) emits no instruction for it. The
`N = 192` and `N = 64` SFB pointer shifts (`908-925`, `2766-2783`) are
unreachable because `is_valid_mma_tiler_and_cluster_shape` pins MMA tile N to
256. `store_d_directly` and `epilogue_prefetch_more` are hardcoded false
(`489`, `570`) and their branches are not ported.

First-class layouts are forbidden throughout both this sketch and the device
kernel. No layout object is constructed, passed, returned, stored on a buffer,
or manipulated by layout algebra. Source layouts are resolved before device
tracing into ordinary integer extents, byte strides, byte offsets, swizzle enum
fields, TensorMap bytes, and raw MMA descriptor bits. Every device-side SMEM
buffer is linear, every TMEM region is named by integer row/column addresses,
and every address helper below returns scalar integers or raw pointers only.
Abstract `tile` wording means a logical rectangular region of the algorithm; it
never denotes a TIRx tile primitive or a layout-bearing value.

## Pipeline at a glance

| warp / role | tile program | publication / reuse edges |
| --- | --- | --- |
| warp 7, scheduler | walks the MoE persistent work list from `padded_offsets`, publishes each `(expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt)` record into a two-stage SMEM ring, and terminates the other seven warps with `expert_idx = -1` | tile-info-empty -> record store -> `fence.proxy.async.shared::cta` -> named barrier 4 -> tile-info-full; producer tail drains both stages |
| warp 5, TMA producer | prefetches every TensorMap, then per work tile rebinds the per-expert A/SFA row window and B/SFB expert slice and issues four TMA loads per K tile | tile-info-full -> per-tile setup; AB-empty (speculative probe then conditional blocking wait) -> four TMA loads -> AB-full transaction bytes; producer tail waits until all AB stages are reusable |
| warp 4, MMA | waits for the TMEM pointer on named barrier 3, then per work tile acquires one accumulator stage, and per K tile copies SFA and SFB SMEM -> TMEM and issues four block-scaled MMA instructions | AB-full -> scale copy and MMA; `tcgen05.commit` publishes AB-empty per K tile and accumulator-full per work tile; accumulator-empty gates reuse; only the leader CTA of the pair runs the loop |
| warps 0-3, epilogue | warp 0 allocates all 512 TMEM columns; all four wait on named barrier 3, then per work tile load one accumulator subtile TMEM -> registers, release the accumulator early, consume two C stages, evaluate the dGLU derivative, accumulate dprob, reduce dbias through SMEM, accumulate amax, quantize row and column scale factors, stage D through a two-slot SMEM ring, and TMA-store it | accumulator-full -> TMEM load; early elected release publishes accumulator-empty after `iter_acc_early_release` subtiles; C-full -> register load -> C-empty; named barrier 2 brackets each SMEM D stage and the dbias reduction's three stages; TMA bulk-group wait protects D-stage reuse; per-tile tails emit the amax and dprob global atomics |
| warp 6, epilogue C producer | mirrors the epilogue's subtile order, including the reverse walk on alternating tiles, and issues two TMA loads for each of the eight subtiles -- sixteen 32-column blocks, the whole 2N tile | tile-info-full -> per-tile setup; C-empty -> TMA load -> C-full; producer tail drains the C ring |
| whole CTA/cluster prologue | initialize and publish the AB, accumulator, tile-info, and C mbarrier groups in declaration order, arrive on the cluster, build scalar addresses and masks, then wait; an all-empty problem exits before any role runs | two `fence.mbarrier_init.release.cluster` epochs -- one after the four pipelines' barriers, one after the TMEM-deallocation barrier; `barrier.cluster.arrive.relaxed` then a delayed `barrier.cluster.wait`; named barrier 3 publishes the TMEM pointer to the MMA and epilogue warps |
| helper kernel (discrete weights or dynamic schedule) | one thread per expert writes that expert's B and SFB TensorMap image into the workspace; block 0 additionally resets the dynamic-scheduler counter | a separate launch on the same stream orders the descriptor writes against the main kernel's tensormap proxy reads |

The four consumer roles each keep their own tile-info cursor over the identical
record stream; they communicate work only through that ring. Source `2454-2489`,
`2491-2671`, `2673-2872`, `2874-3488`, `3490-3570`; anchor PTX `320-817`,
`819-1181`, `1183-1558`, `1560-6331`, `6333-6648`.

## Primitive vocabulary

Structural forms describe placement and views without moving data:

```python
specialize(...)                 # compile-time dtype/major/tile/cluster/mode branch
launch(...)                     # grid, eight warps, cluster, dynamic SMEM, min occupancy
gmem_region(name, shape, dtype, major)
smem_bytes(name, offset, byte_count, alignment)
smem_scalar(dtype)                                  # one word inside the protocol header
tmem_region(name, start_column, row_count, column_count, dtype)
rmem_words(name, count, dtype)
byte_ptr(base, scalar_byte_offset)
raw_mma_descriptor(base_ptr, ldo, sdo, swizzle_enum)
mbarrier_array(name, stages, arrivals)
pipeline_state(name, stages, phase, index, count)
moe_scheduler(padded_offsets, expert_shape, cta_tile, cluster_shape, grid)
expert_window(tensor, padded_offsets, expert_idx)   # scalar row/slice rebase
expert_weights(tensor, expert_idx, P)               # dense slice or discrete ptr
logical_region(tensor, scalar_coordinates, scalar_extents)
```

Directional movement is explicit:

```python
copy_g2s(src_gmem, dst_smem, tensor_map, mbarrier, multicast_mask, desc_ptr=None)
copy_s2t(raw_smem_descriptor_u64, dst_tmem)
copy_t2r(src_tmem, dst_rmem)
copy_s2r(src_smem, dst_rmem)
copy_r2s(src_rmem, dst_smem)
copy_s2g(src_smem, dst_gmem, tensor_map)
load_gmem(src_gmem, scalar_index); store_gmem(src_rmem, dst_gmem)
load_smem(src, dst); store_smem(src, dst)
load_smem_vector(src, word_count)                   # contiguous multi-word read
store_tensormap_image(src_descriptor_bytes, dst_gmem)
```

Basic computation remains decomposed:

```python
gemm(acc_tmem, a_smem, sfa_tmem, b_smem, sfb_tmem, accumulate)
mul(a, b); add(a, b); sub(a, b); neg(a); abs(a); min(a, b); max(a, b)
select(cond, a, b); and_(a, b)                # predicate forms, lower to selp.f32
exp2(x); rcp(x); tanh(x); sigmoid(x)          # fastmath forms
cast(src, dst_dtype); pack_f8x4(src_f32x4); pack_bf16x2(a, b)
round_to_sf(x)                      # E8M0 round-up then upcast back to f32
reduce_max(src, dst, nan_semantics); reduce_add(src, dst)
warp_reduce_max(value); warp_reduce_add(value)      # 32-lane redux forms
atomic_max_bits(nonnegative_f32, dst_f32)
atomic_add_f32(value, dst_f32); atomic_add_bf16x2(packed, dst_bf16x2)
```

Schedule operations stay visible:

```python
thread_idx(); warp_id(); lane_id(); warp_uniform(value)   # role dispatch
block_idx_x(); block_idx_in_cluster(); grid_dim()
union(*bit_masks)                                   # multicast mask accumulation
init(barrier, arrivals); prefetch(tensor_map)
elect_one(); wait(barrier, phase); arrive_expect_tx(barrier, bytes)
try_wait_acquire(barrier, phase); wait_plain_if_not_ready(barrier, phase, token)
arrive(barrier); advance(state.index, state.phase, state.count)
producer_tail(state, stages)                        # drain loop at every tail
commit(barrier, cta_mask); release(barrier, cta_mask)
fence_mbarrier_init_cluster(); cluster_arrive(); cluster_wait(); cta_sync()
fence_async_shared_cta(); fence_view_async_tmem_load(); named_barrier(id, threads)
tma_commit_group(); tma_wait_group(pending)
alloc_tmem(columns); relinquish_tmem_alloc_permit(); dealloc_tmem(columns)
exit_cta()
```

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# source 249-378, 380-586, 701-760; PTX 16-60
# ===========================================================================
P = specialize(
    group_m_list, N, K, L,
    weight_mode, sched, act,
    ab_dtype, sf_dtype, sf_vec_size, c_dtype, d_dtype, b_major,
    mma_tiler_mn, cluster_shape_mn, vectorized_f32,
    with_dbias, with_prob, with_amax, discrete_col_sfd,
    linear_offset, geglu_alpha, glu_clamp_max, glu_clamp_min,
    situ_beta1, situ_beta2,
)
FIX_PAD = 256                            # every expert range is a multiple
TOKENS = sum(align_up(m, FIX_PAD) for m in group_m_list)   # anchor 1024
N_OUT = 2 * N                            # interleaved gate/up columns
ATOM_THR = 2 if P.mma_tiler_mn[0] == 256 else 1
CTA_TILE_M = P.mma_tiler_mn[0] // ATOM_THR               # always 128
CTA_TILE_N = 256                                          # pinned by the source
K_TILE = 4 * mma_instruction_k(P)        # anchor 4 * 32 == 128; FP4 4 * 64 == 256
EPI_TILE = (128, 32)
EPI_SUBTILES = CTA_TILE_N // 32          # 8 accumulator subtiles per tile
WARPS = 8
EPI_WARPS, MMA_WARP, TMA_WARP, C_WARP, SCHED_WARP = (0, 1, 2, 3), 4, 5, 6, 7
AB_STAGES = source_ab_stages(P)          # anchor 4
ACC_STAGES = 1                           # CTA_TILE_N == 256 forces one stage
C_STAGES = 4 if ab_bits(P) == 8 else 2   # anchor 4
D_STAGES = 2
TILE_STAGES = 2
OVERLAPPING_ACCUM = True                 # ACC_STAGES == 1 and CTA_TILE_N == 256
ACC_EARLY_RELEASE = ceil_div(48, 32) - 1 # subtile index that frees the accumulator

# TensorMaps describe A/SFA/C/D (and D_col when scale factors are generated) and,
# for dense weights, B/SFB. They are already-encoded descriptors, not layouts.
# Discrete weights instead read a per-expert descriptor image out of the
# workspace, so B/SFB arrive as gmem addresses rather than grid constants.
ABI = (
    A_ptr, B_ptr_or_pointer_array, SFA_ptr, SFB_ptr_or_pointer_array,
    C_ptr, D_row_ptr, [D_col_ptr, SFD_row_ptr, SFD_col_ptr, norm_const_ptr],
    [amax_ptr], padded_offsets_ptr, alpha_ptr, beta_ptr,
    [prob_ptr, dprob_ptr], [dbias_ptr], [workspace_ptr],
    map_A, map_SFA, map_C, map_D, [map_D_col], [map_B, map_SFB],
)

launch(
    grid=(cluster_m, cluster_n, max_active_clusters(cluster_m * cluster_n)),
    block=(256, 1, 1),
    cluster=(cluster_m, cluster_n, 1),
    arch="sm_100a",
    min_blocks_per_sm=1,
    dynamic_smem=source_shared_storage_size(P),
)
# instruction_selection: `.maxntid 256, 1, 1` (from `max_number_threads`),
#   `.minnctapersm 1`, and `.extern .shared .align 1024 .b8`; extent: one launch
#   specialization (source 1119-1132; PTX 47, 48, 14)

# ===========================================================================
# Storage and synchronization objects
# source 980-1048, 2272-2396, sched 419-447; PTX 111-307
# ===========================================================================
AB_FULL  = mbarrier_array(stages=AB_STAGES, arrivals=1)
AB_EMPTY = mbarrier_array(stages=AB_STAGES,
                          arrivals=num_mcast_ctas_a(P) + num_mcast_ctas_b(P) - 1)
ACC_FULL  = mbarrier_array(stages=ACC_STAGES, arrivals=1)
ACC_EMPTY = mbarrier_array(stages=ACC_STAGES, arrivals=4 * ATOM_THR)
TILE_FULL  = mbarrier_array(stages=TILE_STAGES, arrivals=32)
TILE_EMPTY = mbarrier_array(stages=TILE_STAGES, arrivals=224)
# Dynamic scheduling only: a one-stage cluster pipeline plus the 16-byte slot the
# elected CTA broadcasts the next work index into. Both sit in the scheduler's
# storage straight after sInfo, which is why every object below them shifts by 32
# bytes on that branch.
SCHED_CLUSTER_FULL  = mbarrier_array(stages=1, arrivals=1) if P.sched == "dynamic" else None
SCHED_CLUSTER_EMPTY = mbarrier_array(stages=1, arrivals=32 * cluster_size) if P.sched == "dynamic" else None
sSchedBroadcast     = smem_bytes("sched_broadcast", source_sched_broadcast_offset(P),
                                 16 if P.sched == "dynamic" else 0, 16)
C_FULL  = mbarrier_array(stages=C_STAGES, arrivals=1)
C_EMPTY = mbarrier_array(stages=C_STAGES, arrivals=4)
TMEM_DEALLOC = smem_scalar(i64)
TMEM_PTR = smem_scalar(u32)

# Anchor dynamic-SMEM byte map, read back from the export's mbarrier and TMA
# operands. Non-anchor offsets follow the same declaration order and alignments;
# no two live objects alias.
#   AB_FULL 0..31; AB_EMPTY 32..63; ACC_FULL 64..71; ACC_EMPTY 72..79
#   TILE_FULL 80..95; TILE_EMPTY 96..111; sInfo 112..143
#   C_FULL 144..175; C_EMPTY 176..207; TMEM_DEALLOC 208..215; TMEM_PTR 216..219
#   sC 1024..33791; sD 33792..41983; sD_col 41984..50175
#   sA 50176..115711; sB 115712..181247; sSFA 181248..183295
#   sSFB 183296..187391; sAmax 187392..187407; sDbias 187520..220287
sC     = smem_bytes("sC", source_sC_offset(P), C_STAGES * epi_bytes(P, c_dtype), 1024)
sD     = smem_bytes("sD", source_sD_offset(P), D_STAGES * epi_bytes(P, d_dtype), 1024)
sD_col = smem_bytes("sD_col", source_sD_col_offset(P),
                    D_STAGES * epi_bytes(P, d_dtype) if P.generate_sfd else 0, 1024)
sA     = smem_bytes("sA", source_sA_offset(P), AB_STAGES * a_stage_bytes(P), 1024)
sB     = smem_bytes("sB", source_sB_offset(P), AB_STAGES * b_stage_bytes(P), 1024)
sSFA   = smem_bytes("sSFA", source_sSFA_offset(P), AB_STAGES * sfa_stage_bytes(P), 1024)
sSFB   = smem_bytes("sSFB", source_sSFB_offset(P), AB_STAGES * sfb_stage_bytes(P), 1024)
sAmax  = smem_bytes("sAmax", source_sAmax_offset(P), 4 * 4, 4)
sDbias = smem_bytes("sDbias", source_sDbias_offset(P),
                    128 * 64 * 4 if P.with_dbias else 4, 128)
sInfo  = smem_bytes("sInfo", source_sInfo_offset(P), 16 * TILE_STAGES, 16)

# Pure compile-time/scalar-integer address formulas. They neither create nor
# consume a layout value; swizzle XOR and stage strides fold into integers.
sA_addr, sB_addr, sSFA_addr, sSFB_addr = (scalar_address_formula(P, r)
                                          for r in ("A", "B", "SFA", "SFB"))
sC_addr, sD_addr, sD_col_addr = (scalar_address_formula(P, r)
                                 for r in ("C", "D", "D_col"))
sAmax_addr  = lambda warp: 4 * warp
# The dbias buffer is a per-warp transpose scratch: 64 columns of 32 lanes, one
# 8192-byte pane per epilogue warp. The store side is a plain (column, lane)
# index; the read side gives each lane a column pair and XORs the row group so
# the eight vector reads of a column land in different banks. That XOR only
# permutes the order of the eight groups, so the column sum is unchanged.
sDbias_addr = lambda warp, column, lane: 4 * (warp * 64 * 32 + column * 32 + lane)
dbias_col_a = lambda lane: 2 * lane if lane < 16 else 32 + 2 * (lane - 16)
dbias_swizzle = lambda column: ((column >> 1) & 0x7) << 2
sDbias_read_addr = lambda warp, column, group: 4 * (
    warp * 64 * 32 + column * 32 + ((group * 4) ^ dbias_swizzle(column))
)
# The cross-warp stage reuses the first 256 bytes of the same buffer.
sDbias_partial_addr = lambda warp, lane: 4 * (warp * 64 + lane * 2)
# Output column a lane's packed pair lands on, and the guard extent.
dbias_column = lambda tile_n, subtile, lane: (
    tile_n * (CTA_TILE_N * 2) + (2 * subtile + (0 if lane < 16 else 1)) * 32
    + 2 * (lane if lane < 16 else lane - 16)
)
dbias_addr = lambda expert, tile_n, subtile, lane: 2 * (
    expert * N_OUT + dbias_column(tile_n, subtile, lane)
)
# Scale-factor store columns and the extents their guards compare against.
sfd_row_column = lambda tile_n, subtile: (
    tile_n * (EPI_SUBTILES // 2) + subtile // 2
) * 32 * 4
sfd_col_column = sfd_row_column
sfd_row_extent = scale_factor_extent(P, "row")
sfd_col_extent = scale_factor_extent(P, "col")
sInfo_addr  = lambda field, stage: 4 * (field + 4 * stage)

a_desc   = raw_mma_descriptor(sA, A_LDO(P), A_SDO(P), A_SWIZZLE_ENUM(P))
b_desc   = raw_mma_descriptor(sB, B_LDO(P), B_SDO(P), B_SWIZZLE_ENUM(P))
sfa_desc = raw_mma_descriptor(sSFA, SF_LDO(P), SF_SDO(P), 0)
sfb_desc = raw_mma_descriptor(sSFB, SF_LDO(P), SF_SDO(P), 0)

# All 512 columns are allocated. Under OVERLAPPING_ACCUM the two accumulator
# stages are strided by (256 - 48) columns so the pair spans 464 columns and the
# 48 scale-factor columns sit above them.
tAcc = tmem_region("acc", start_column=0, row_count=128,
                   column_count=464, dtype=f32, stage_stride=256 - 48)
tSFA = tmem_region("SFA", start_column=464, row_count=128,
                   column_count=16, dtype=sf_dtype)
tSFB = tmem_region("SFB", start_column=480, row_count=128,
                   column_count=32, dtype=sf_dtype)

rAcc  = rmem_words("acc", 32, f32)          # one tcgen05.ld.32x32b.x32 payload
rC1   = rmem_words("C_gate", 32, c_dtype)
rC2   = rmem_words("C_up", 32, c_dtype)
rD1   = rmem_words("D_gate", 32, d_dtype)
rD2   = rmem_words("D_up", 32, d_dtype)
rD1c  = rmem_words("D_gate_col", 32, d_dtype)   # generate_sfd only
rD2c  = rmem_words("D_up_col", 32, d_dtype)     # generate_sfd only
# Each SFD axis carries two fragments, as the source does (3028-3029 row,
# 3053-3054 col): an FP32 scale fragment the quantize loops write, and the
# packed sf_dtype fragment `pack_f8x4` produces for the store.
rSFDr    = rmem_words("SFD_row_pvscale", 4, f32)   # four subtile scales per store
rSFDc    = rmem_words("SFD_col_pvscale", 4, f32)
rSFDr_p  = rmem_words("SFD_row_packed", 4, sf_dtype)
rSFDc_p  = rmem_words("SFD_col_packed", 4, sf_dtype)
rescaled = rmem_words("sfd_col_rescaled", 32, f32)  # column pass, before packing
# The column scale fragment does not stay in registers: the anchor gives it a
# 32-byte local depot (`.local .align 32 .b8 __local_depot0[32]`, PTX 50),
# spills it with `st.local.b32` at PTX 4702 and 5889 -- both `.loc 1 1526` --
# and reloads it with `ld.local.v4.b32` at 6042 to feed the column pack. The row
# fragment stays register-resident (`mov.b64` 6026-6027 into the pack).
rInfo = rmem_words("tile_info", 4, i32)
# Which record fields a role actually loads. The MMA warp needs only the expert
# index; nobody loads field 3.
EXPERT_TILE_FIELDS = (0, 1, 2)
EXPERT_ONLY_FIELD = (0,)

# ===========================================================================
# Optional pre-kernel: per-expert descriptor images and counter reset
# source 612-699, utils 59-104; discrete_dynamic PTX 16-142
# ===========================================================================
if P.weight_mode == "discrete" or P.sched == "dynamic":
    # grid=(L,1,1) for discrete weights, (1,1,1) for dense weights with the
    # dynamic scheduler; one thread per block.
    expert = block_idx_x()
    if P.weight_mode == "discrete":
        b_base   = load_gmem(B_pointer_array, expert)
        sfb_base = load_gmem(SFB_pointer_array, expert)
        # instruction_selection: `ld.global.b64`; extent: two scalar loads
        #   (source 660, 678; discrete_dynamic PTX 45, 80)
        store_tensormap_image(tensormap_bytes_B(P, b_base),
                              byte_ptr(workspace, 256 * expert))
        # instruction_selection: two `st.global.v4.b64`, 32 bytes each -- the
        #   backend elides the descriptor's all-zero tail; extent: one B
        #   TensorMap image (source 676, utils 59-104; discrete_dynamic
        #   PTX 76-77)
        store_tensormap_image(tensormap_bytes_SFB(P, sfb_base),
                              byte_ptr(workspace, 256 * expert + 128))
        # instruction_selection: two `st.global.v4.b64`; extent: one SFB
        #   TensorMap image (source 691; discrete_dynamic PTX 130-131)
    if P.sched == "dynamic" and expert == 0:
        store_gmem(0, byte_ptr(workspace, sched_counter_offset(P)))
        # instruction_selection: `st.global.b32`; extent: one counter reset
        #   (source 692-699; discrete_dynamic PTX 137)
    # The launch boundary between this kernel and the main kernel is what orders
    # these writes against the main kernel's tensormap-proxy reads; the source
    # emits no tensormap fence.

# ===========================================================================
# Coordinates, descriptor prefetch, and barrier initialization
# source 2233-2270, 2272-2346, 2398-2452; PTX 63-307
# ===========================================================================
warp = warp_uniform(warp_id())
lane = lane_id()
tid  = thread_idx()
cta_rank = block_idx_in_cluster()
cluster_rank = block_idx_in_cluster()
cluster_v = cluster_rank % ATOM_THR            # position inside the CTA pair
cluster_m_coord = (cluster_rank // ATOM_THR) % max(1, cluster_m // ATOM_THR)
cluster_n_coord = cluster_rank // cluster_m
is_leader_cta = (block_idx_x() % ATOM_THR) == 0
total_tokens = load_gmem(padded_offsets, L - 1)
# instruction_selection: `ld.global.b32`; extent: one scalar, the last padded
#   offset (source 2241; PTX 104)

if warp == TMA_WARP:
    for descriptor in (map_A, map_SFA, *dense_only(map_B, map_SFB),
                       map_C, map_D, *sfd_only(map_D_col)):
        prefetch(descriptor)
        # instruction_selection: `prefetch.tensormap`; extent: seven anchor
        #   descriptor prefetches (source 2245-2256; PTX 111-129)

# Barrier initialization follows the declaration order of the four pipelines.
# Each group has its own elected issuer and its own publication epoch.
if warp == 0 and elect_one():
    for stage in range(AB_STAGES):
        init(AB_FULL[stage], arrivals=1)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: four
        #   anchor barriers at offsets 0..24 (source 2283; PTX 163-166)
    for stage in range(AB_STAGES):
        init(AB_EMPTY[stage], arrivals=num_mcast_ctas_a(P) + num_mcast_ctas_b(P) - 1)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: four
        #   barriers at offsets 32..56, anchor arrival count 1 (PTX 176-179)
    init(ACC_FULL[0], arrivals=1)
    init(ACC_EMPTY[0], arrivals=4 * ATOM_THR)
    # instruction_selection: `mbarrier.init.shared.b64`; extent: two barriers
    #   at offsets 64, 72 (source 2297; PTX 193, 206)
    # Declaration order follows the source's construction order: the C pipeline
    # is built before the tile-info pipeline.
    for stage in range(C_STAGES):
        init(C_FULL[stage], arrivals=1)
        init(C_EMPTY[stage], arrivals=4)
    # instruction_selection: `mbarrier.init.shared.b64`; extent: eight barriers
    #   at offsets 144..200 (source 2313; PTX 218-236)
    for stage in range(TILE_STAGES):
        init(TILE_FULL[stage], arrivals=32)
        init(TILE_EMPTY[stage], arrivals=224)
    # instruction_selection: `mbarrier.init.shared.b64`; extent: four barriers
    #   at offsets 80..104 (source 2331; PTX 250-265)

fence_mbarrier_init_cluster(); cta_sync()
# instruction_selection: `fence.mbarrier_init.release.cluster` then `bar.sync 0`;
#   extent: the tile-info pipeline's own non-cluster publication
#   (source 2331-2340; PTX 272-273)
# The tile-info pipeline is built without `defer_sync`, so it emits the first
# epoch itself; everything initialized after that fence belongs to the second.
if warp == 0 and elect_one():
    if P.sched == "dynamic":
        init(SCHED_CLUSTER_FULL[0], arrivals=1)
        init(SCHED_CLUSTER_EMPTY[0], arrivals=32 * cluster_size)
        # instruction_selection: `mbarrier.init.shared.b64`; extent: the two
        #   cluster-broadcast barriers. `internal_init` builds that pipeline with
        #   `defer_sync=True` after the first fence, so they are published by the
        #   second epoch alongside the TMEM barrier, not by the first
        #   (sched 457-476, called from source 2346; discrete_dynamic PTX 407,
        #   422)
    init(TMEM_DEALLOC, arrivals=32)
    # instruction_selection: `mbarrier.init.shared.b64` with an arrival count of
    #   32, one warp; extent: the two-CTA TMEM deallocation barrier at offset 208
    #   (source 2357-2364; PTX 283)
fence_mbarrier_init_cluster()
# instruction_selection: `fence.mbarrier_init.release.cluster`; extent: TMEM
#   barrier publication (PTX 286)
if cluster_m * cluster_n > 1:
    cluster_arrive()
    # instruction_selection: `barrier.cluster.arrive.relaxed`; extent: the
    #   arrive half of the delayed cluster handshake (source 2367-2368; PTX 288)

perform_scalar_address_and_descriptor_setup()
# A and SFA multicast over the cluster's N extent. B multicasts over the *vmnk*
# M extent, which the two-CTA MMA has already halved -- for the anchor's (2,1)
# cluster that extent is 1, so B is a plain single-CTA load. SFB rides its own
# cta_group::1 cluster layout, whose M extent is not halved, so it is the one
# operand the anchor does multicast.
# Every mask is the image of the vmnk cluster layout at this CTA's coordinate,
# varying one mode. That layout is `tiled_divide((cluster_m, cluster_n, 1),
# (ATOM_THR,))`, so a coordinate `(v, m, n, k)` has flat rank
# `v + ATOM_THR * m + cluster_m * n` -- the `v` term is what distinguishes the
# two CTAs of a pair and must appear in every mask.
cta_bit = lambda v, m, n: 1 << (v + ATOM_THR * m + cluster_m * n)
a_mcast_mask = sfa_mcast_mask = union(cta_bit(cluster_v, cluster_m_coord, n)
                                      for n in range(cluster_n))
b_mcast_mask = union(cta_bit(cluster_v, m, cluster_n_coord)
                     for m in range(cluster_m // ATOM_THR))
sfb_mcast_mask = union(1 << (m + cluster_m * cluster_n_coord) for m in range(cluster_m))
# instruction_selection: with a static cluster shape each mask folds to an
#   immediate. The anchor's A, B and SFA masks are single-bit -- `1 << cta_rank`,
#   so 1 on the leader and 2 on its peer -- which is why those three lower to the
#   non-multicast copy forms; SFB rides its own cta_group::1 layout and folds to
#   `mov.b16 %rs84, 3`; extent: four scalar masks (source 2398-2409; PTX 1043
#   and 1045 for SFB, 1007/1016/1030 for the non-multicast A/B/SFA copies)

# The MMA warp's two commit ops carry different masks, built inside the pipeline
# constructors rather than here.
#
# AB-empty release: the union of the A and B images *and* the same two images
# recomputed at the peer coordinate, `v ^ 1`. Without the peer terms the other
# CTA of the pair would never see its AB-empty barrier arrive.
peer_v = cluster_v ^ 1
ab_consumer_mask = union(
    a_mcast_mask, b_mcast_mask,
    union(cta_bit(peer_v, cluster_m_coord, n) for n in range(cluster_n)),
    union(cta_bit(peer_v, m, cluster_n_coord) for m in range(cluster_m // ATOM_THR)),
)
# instruction_selection: `xor.b32` on the v coordinate, two `shl.b32`, `or.b32`
#   then `cvt.u16.u32`, computed at runtime and equal to 3 for the anchor --
#   unlike the accumulator mask below, which really is an immediate; extent: one
#   mask, consumed by the `tcgen05.commit` (source 2283; PTX 1272-1278, 1481)

# Accumulator-full publication: the image over the v mode, i.e. every CTA of the
# pair, because both CTAs' epilogue warps wait on it (consumer group 4 * 2).
acc_producer_mask = union(cta_bit(v, cluster_m_coord, cluster_n_coord)
                          for v in range(ATOM_THR))
# instruction_selection: folds to `mov.b16 %rs85, 3` for the anchor; extent: one
#   mask, a separate site from the AB mask (source 2297; PTX 1497, consumed by
#   the `tcgen05.commit` at PTX 1497-1498)
# Multicast ownership is structural: A and SFA multicast across equal M
# coordinates (the cluster N dimension), B across equal N coordinates (the
# cluster M dimension after the two-CTA MMA halves it), and SFB across the same
# N coordinates under its own cta_group::1 cluster layout -- which is why the
# anchor multicasts SFB but not B.

if cluster_m * cluster_n > 1:
    cluster_wait()
    # instruction_selection: `barrier.cluster.wait`; extent: delayed matching
    #   wait before any staged-tensor use (source 2444-2447; PTX 302)
else:
    named_barrier(id=1, threads=256)
    # instruction_selection: `bar.sync 1,256`; extent: the singleton fallback

if total_tokens <= 0:
    exit_cta()
    # instruction_selection: `exit`; extent: the empty-problem early out
    #   (source 2450-2451; PTX 303-307)
k_tile_cnt = ceil_div(K, K_TILE)

# ===========================================================================
# Role 1: warp 7, MoE persistent tile scheduler producer
# source 2454-2489, sched 758-1113; PTX 320-817
# ===========================================================================
if warp == SCHED_WARP:
    sched = moe_scheduler(padded_offsets, (L, N, K),
                          (CTA_TILE_M, CTA_TILE_N), (cluster_m, cluster_n),
                          grid_dim(), dynamic=P.sched == "dynamic")
    info_prod = pipeline_state(stages=TILE_STAGES, phase=1, index=0, count=0)

    work = sched.initial_work_tile()
    # The static walk linearizes the short side first for L2 reuse and advances
    # experts through an O(1) cached search over padded_offsets; the dynamic walk
    # replaces the linear index with a global atomic counter that lane 0 fetches
    # and broadcasts to the cluster's shared memory.
    # instruction_selection: static form `ld.global.b32` over padded_offsets
    #   plus integer divmod; dynamic form: `atom.global.add.u32` on the gmem
    #   counter by an elected lane, `shfl.sync.idx.b32` to spread it across the
    #   warp, then per peer two `mapa.shared::cluster.u32` and one
    #   `st.async.shared::cluster.mbarrier::complete_tx::bytes.u32` into the
    #   broadcast slot, a further `mapa` plus
    #   `mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remote_mbar], 4`, and
    #   two `mbarrier.try_wait.parity.shared.b64` sites on the consuming side;
    #   extent: one work-tile lookup per iteration (sched 927-1113 static,
    #   53-138 and 840-896 dynamic; discrete_dynamic PTX 486, 499, 502, 513, 515,
    #   517, 526, 528, 540)

    while work.is_valid:
        wait(TILE_EMPTY[info_prod.index], info_prod.phase)
        # instruction_selection: retry loop containing
        #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: one
        #   tile-info-empty acquire (source 2464; PTX 527, 537)
        if elect_one():
            store_smem((work.expert_idx, work.tile_m_idx,
                        work.tile_n_idx, work.k_tile_cnt),
                       byte_ptr(sInfo, sInfo_addr(0, info_prod.index)))
            # instruction_selection: `st.shared.v4.b32`; extent: one 16-byte
            #   record (source 2466-2469; PTX 555)
        fence_async_shared_cta()
        # instruction_selection: `fence.proxy.async.shared::cta`; extent: record
        #   visibility before the barrier (source 2470; PTX 558)
        named_barrier(id=4, threads=32)
        # instruction_selection: `bar.sync 4,32`; extent: scheduler-warp-local
        #   rendezvous before commit (source 2472; PTX 560)
        arrive(TILE_FULL[info_prod.index])
        # instruction_selection: `mbarrier.arrive.shared.b64`; extent: one
        #   commit (source 2473; PTX 563)
        advance(info_prod.index, info_prod.phase, info_prod.count, TILE_STAGES)
        work = sched.advance_to_next_work()

    # Termination record: expert_idx = -1 stops all seven consumer roles.
    wait(TILE_EMPTY[info_prod.index], info_prod.phase)
    # instruction_selection: retry loop containing
    #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: the sentinel's
    #   tile-info-empty acquire (source 2479; PTX 743)
    if elect_one():
        store_smem((-1, 0, 0, 0), byte_ptr(sInfo, sInfo_addr(0, info_prod.index)))
        # instruction_selection: `st.shared.v4.b32` of the immediate quad
        #   `{-1, 0, 0, 0}`; extent: the sentinel record (source 2481-2484;
        #   PTX 761)
    fence_async_shared_cta()
    # instruction_selection: `fence.proxy.async.shared::cta`; extent: sentinel
    #   visibility (source 2485; PTX 764)
    named_barrier(id=4, threads=32)
    # instruction_selection: `bar.sync 4,32`; extent: the sentinel rendezvous
    #   (source 2486; PTX 766)
    arrive(TILE_FULL[info_prod.index])
    # instruction_selection: `mbarrier.arrive.shared.b64`; extent: the sentinel
    #   commit (source 2487; PTX 769)
    advance(info_prod.index, info_prod.phase, info_prod.count, TILE_STAGES)
    for live in producer_tail(info_prod, TILE_STAGES):
        wait(TILE_EMPTY[live.index], live.phase)
        # instruction_selection: two retry loops containing
        #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: the
        #   two-stage tail (source 2489; PTX 786, 809)

# ===========================================================================
# Consumer preamble shared by roles 2-5: read one tile-info record
# source 2501-2508, 2735-2743, 2949-2956, 3494-3503
# ===========================================================================
def take_tile_info(info_cons, fields):
    wait(TILE_FULL[info_cons.index], info_cons.phase)
    # instruction_selection: retry loop containing
    #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: one
    #   tile-info-full acquire (PTX 830 in the TMA warp, 1198 in the MMA warp,
    #   1585 in the epilogue, 6346 in the C warp)
    for field in fields:
        load_smem(byte_ptr(sInfo, sInfo_addr(field, info_cons.index)), rInfo[field])
    # instruction_selection: `ld.shared.b32`; extent: three loads -- expert,
    #   tile_m, tile_n -- in the TMA, epilogue and C warps, and one load of
    #   expert alone in the MMA warp. The record's fourth field, `k_tile_cnt`,
    #   is never read by any consumer: they all use the `k_tile_cnt` computed
    #   once at source 2452 (PTX 840-842, 1208, 1595-1597, 6356-6358)
    fence_async_shared_cta()
    # instruction_selection: `fence.proxy.async.shared::cta`; extent: record
    #   acquire ordering (PTX 844)
    arrive(TILE_EMPTY[info_cons.index])
    # instruction_selection: `mbarrier.arrive.shared.b64`; extent: one release
    #   (PTX 847)
    advance(info_cons.index, info_cons.phase, info_cons.count, TILE_STAGES)
    return rInfo[0] >= 0

# ===========================================================================
# Role 2: warp 5, persistent TMA producer with elected issue sites
# source 2491-2671, ext 99-330; PTX 819-1181
# ===========================================================================
if warp == TMA_WARP:
    ab_prod  = pipeline_state(stages=AB_STAGES, phase=1, index=0, count=0)
    info_cons = pipeline_state(stages=TILE_STAGES, phase=0, index=0, count=0)
    valid = take_tile_info(info_cons, EXPERT_TILE_FIELDS)

    while valid:
        expert, tile_m, tile_n = rInfo[0], rInfo[1], rInfo[2]
        # The extension rebases A and SFA onto this expert's token window and,
        # for dense weights, selects B/SFB along L; for discrete weights it
        # instead yields the workspace address of that expert's descriptor.
        a_window   = expert_window(A, padded_offsets, expert)
        sfa_window = expert_window(SFA, padded_offsets, expert)
        b_source, b_desc_ptr     = expert_weights(B, expert, P)
        sfb_source, sfb_desc_ptr = expert_weights(SFB, expert, P)
        # instruction_selection: the dense form folds the expert offset into
        #   the TMA coordinate operands and emits only integer arithmetic
        #   (ext 237-346; PTX 870-926). The discrete form emits the same
        #   `shr.s32`/`add.s32`/`and.b64` token-offset arithmetic and adds a
        #   `cvta.global.u64` on the per-expert descriptor address
        #   `workspace + 256 * expert (+128)`; extent: scalar setup either way
        #   (ext 199 and 228; discrete_dynamic PTX 1197-1200, 1240-1243)
        mma_tile_m = tile_m // ATOM_THR

        ab_empty_ready = True
        if k_tile_cnt > 0:
            ab_empty_ready = try_wait_acquire(AB_EMPTY[ab_prod.index], ab_prod.phase)
            # instruction_selection:
            #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`; extent:
            #   initial speculative AB-empty probe (source 2595-2596; PTX 881, 909)

        while ab_prod.count < k_tile_cnt:      # anchor body is nounroll
            k_tile = ab_prod.count
            wait_plain_if_not_ready(AB_EMPTY[ab_prod.index], ab_prod.phase, ab_empty_ready)
            # instruction_selection: only when the speculative token is false, a
            #   retry loop containing `mbarrier.try_wait.parity.shared.b64
            #   ...,10000000`; extent: conditional blocking retry per K tile
            #   (source 2614; PTX 950, 958)
            if is_leader_cta and elect_one():
                arrive_expect_tx(AB_FULL[ab_prod.index], source_stage_bytes(P))
                # instruction_selection: `mov.b32` of the byte immediate then
                #   `mbarrier.arrive.expect_tx.shared.b64`, under the leader-CTA
                #   predicate and an elected lane; extent: one arrival of 68,608
                #   anchor bytes -- `num_tma_load_bytes` counts the whole CTA
                #   pair, so it is `atom_thr` times the 34,304-byte per-CTA stage
                #   (source 931, 2614; PTX 968-977)
            if ab_prod.count + 1 < k_tile_cnt:
                ab_empty_ready = try_wait_acquire(AB_EMPTY[ab_prod.next_index],
                                                 ab_prod.next_phase)
                # instruction_selection:
                #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                #   extent: the next stage's speculative probe, issued before the
                #   four copies so its latency overlaps them (source 2615-2618;
                #   PTX 992)

            if elect_one():
                copy_g2s(logical_region(a_window, (mma_tile_m, k_tile), A_box(P)),
                         byte_ptr(sA, sA_addr(ab_prod.index)),
                         map_A, AB_FULL[ab_prod.index], a_mcast_mask)
            # instruction_selection: anchor
            #   `cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2`;
            #   extent: one 128 x K_TILE A tile (source 2621; PTX 1007)
            if elect_one():
                copy_g2s(logical_region(b_source, (tile_n, k_tile), B_box(P)),
                         byte_ptr(sB, sB_addr(ab_prod.index)),
                         map_B, AB_FULL[ab_prod.index], b_mcast_mask,
                         desc_ptr=b_desc_ptr)
            # instruction_selection: anchor
            #   `cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2`;
            #   the discrete export adds a workspace descriptor operand and the
            #   multicast form appears when the cluster M extent exceeds one;
            #   extent: one 128 x K_TILE B tile (source 2629; PTX 1016)
            if elect_one():
                copy_g2s(logical_region(sfa_window, (mma_tile_m, k_tile), SFA_box(P)),
                         byte_ptr(sSFA, sSFA_addr(ab_prod.index)),
                         map_SFA, AB_FULL[ab_prod.index], sfa_mcast_mask)
            # instruction_selection: anchor
            #   `cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2`;
            #   extent: one E8M0 SFA tile (source 2638; PTX 1030)
            if elect_one():
                copy_g2s(logical_region(sfb_source, (tile_n, k_tile), SFB_box(P)),
                         byte_ptr(sSFB, sSFB_addr(ab_prod.index)),
                         map_SFB, AB_FULL[ab_prod.index], sfb_mcast_mask,
                         desc_ptr=sfb_desc_ptr)
            # instruction_selection: anchor
            #   `cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint.cta_group::2`,
            #   mask 0b11; extent: one E8M0 SFB tile (source 2646; PTX 1045)

            advance(ab_prod.index, ab_prod.phase, ab_prod.count, AB_STAGES)
        valid = take_tile_info(info_cons, EXPERT_TILE_FIELDS)

    for live in producer_tail(ab_prod, AB_STAGES):
        wait(AB_EMPTY[live.index], live.phase)
        # instruction_selection: four retry loops containing
        #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: the
        #   four-stage producer tail (source 2671; PTX 1104, 1127, 1150, 1173)

# ===========================================================================
# Role 3: warp 4, persistent block-scaled MMA consumer/producer
# source 2673-2872; PTX 1183-1558
# ===========================================================================
if warp == MMA_WARP:
    named_barrier(id=3, threads=160)
    # instruction_selection: `bar.sync 3,160`; extent: MMA plus four epilogue
    #   warps waiting for the TMEM allocation (source 2680; PTX 1188)
    load_smem(TMEM_PTR, acc_tmem_base)
    # instruction_selection: `ld.shared.b32`; extent: one pointer word
    #   (source 2685; PTX 1190)
    sfa_tmem_base = acc_tmem_base + 464
    sfb_tmem_base = acc_tmem_base + 464 + 16
    # The two SFB chunks are four TMEM columns apart, not sixteen: one
    # `tcgen05.cp` writes a 32x128b block, which is four columns wide.
    SFB_CHUNK_COLUMNS = 4

    ab_cons  = pipeline_state(stages=AB_STAGES, phase=0, index=0, count=0)
    acc_prod = pipeline_state(stages=ACC_STAGES, phase=1, index=0, count=0)
    info_cons = pipeline_state(stages=TILE_STAGES, phase=0, index=0, count=0)
    valid = take_tile_info(info_cons, EXPERT_ONLY_FIELD)

    while valid:
        ab_full_ready = True
        if k_tile_cnt > 0 and is_leader_cta:
            ab_full_ready = try_wait_acquire(AB_FULL[ab_cons.index], ab_cons.phase)
            # instruction_selection:
            #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`; extent:
            #   speculative AB-full probe (source 2749-2751; PTX 1297)
        # Under OVERLAPPING_ACCUM the stage index is the producer phase's
        # complement, so successive tiles alternate between the two strided
        # accumulator regions without a second mbarrier stage.
        acc_stage = acc_prod.phase ^ 1 if OVERLAPPING_ACCUM else acc_prod.index

        if is_leader_cta:
            wait(ACC_EMPTY[acc_prod.index], acc_prod.phase)
            # instruction_selection: retry loop containing
            #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: one
            #   accumulator acquire per work tile (source 2788; PTX 1309, 1316)
            accumulate = False
            for k_tile in range(k_tile_cnt):        # anchor body is nounroll
                wait_plain_if_not_ready(AB_FULL[ab_cons.index], ab_cons.phase, ab_full_ready)
                # instruction_selection: conditional retry loop containing
                #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent:
                #   one AB-full acquire per K tile (source 2792; PTX 1333, 1343)
                if k_tile < k_tile_cnt - 1:
                    ab_full_ready = try_wait_acquire(AB_FULL[ab_cons.next_index],
                                                     ab_cons.next_phase)
                    # instruction_selection:
                    #   `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                    #   extent: the next K tile's speculative probe
                    #   (source 2796-2798; PTX 1366)

                copy_s2t(sfa_desc_for(ab_cons.index), tSFA)
                # instruction_selection: `tcgen05.cp.cta_group::2.32x128b.warpx4`;
                #   extent: one SFA chunk (source 2808; PTX 1381)
                for chunk in range(sfb_chunks(P)):
                    copy_s2t(sfb_desc_for(ab_cons.index, chunk),
                             tSFB + SFB_CHUNK_COLUMNS * chunk)
                # instruction_selection: `tcgen05.cp.cta_group::2.32x128b.warpx4`;
                #   extent: two anchor SFB chunks whose TMEM destinations are
                #   columns 480 and 484 while their SMEM descriptors are 512
                #   bytes apart, half of `sfb_stage_bytes` (source 2813;
                #   PTX 1389, 1397)

                for kblock in range(4):
                    gemm(tAcc[acc_stage], a_desc_for(ab_cons.index, kblock), tSFA,
                         b_desc_for(ab_cons.index, kblock), tSFB, accumulate)
                    accumulate = True
                # instruction_selection:
                #   `tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32`
                #   (FP4 exports `kind::mxf4nvf4` with `block16`/`block32`
                #   selected by the scale vector size); extent: four issues per
                #   K tile, the first with the accumulate field clear
                #   (source 2830-2853; PTX 1420, 1439, 1455, 1474)

                commit(AB_EMPTY[ab_cons.index], ab_consumer_mask)
                # instruction_selection:
                #   `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`;
                #   extent: one asynchronous AB release per K tile
                #   (source 2852; PTX 1481)
                advance(ab_cons.index, ab_cons.phase, ab_cons.count, AB_STAGES)
            commit(ACC_FULL[acc_prod.index], acc_producer_mask)
            # instruction_selection:
            #   `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`;
            #   extent: one accumulator publication per work tile
            #   (source 2854; PTX 1498)
        advance(acc_prod.index, acc_prod.phase, acc_prod.count, ACC_STAGES)
        valid = take_tile_info(info_cons, EXPERT_ONLY_FIELD)

    for live in producer_tail(acc_prod, ACC_STAGES):
        wait(ACC_EMPTY[live.index], live.phase)
        # instruction_selection: one retry loop containing
        #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: the
        #   single-stage accumulator tail (source 2872; PTX 1550)


# ===========================================================================
# Role 4: warps 0-3, dGLU backward epilogue
# source 2874-3488; PTX 1560-6331
# ===========================================================================
if warp in EPI_WARPS:
    if warp == 0:
        alloc_tmem(512)
        # instruction_selection:
        #   `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32`; extent:
        #   one 512-column allocation. It is `.sync.aligned`, so all 32 lanes of
        #   warp 0 execute it and the only guard is the warp predicate -- there
        #   is no elected lane here (source 2881; PTX 1565-1570)
    named_barrier(id=3, threads=160)
    # instruction_selection: `bar.sync 3,160`; extent: TMEM pointer publication
    #   (source 2886; PTX 1573)
    load_smem(TMEM_PTR, tmem_base)
    # instruction_selection: `ld.shared.b32`; extent: one pointer word
    #   (source 2891; PTX 1575)
    if P.generate_sfd:
        norm_const = load_gmem(norm_const_ptr, 0)
        rcp_limit = dtype_rcp_limit(P.d_dtype)      # 1/448 for E4M3
        # instruction_selection: `ld.global.b32`; extent: one loop-invariant
        #   scalar, hoisted above the persistent loop head (source 2922;
        #   PTX 1577, before the loop head at 1626)

    acc_cons  = pipeline_state(stages=ACC_STAGES, phase=0, index=0, count=0)
    c_cons    = pipeline_state(stages=C_STAGES, phase=0, index=0, count=0)
    d_prod    = pipeline_state(stages=D_STAGES // 2, phase=1, index=0, count=0)
    info_cons = pipeline_state(stages=TILE_STAGES, phase=0, index=0, count=0)
    valid = take_tile_info(info_cons, EXPERT_TILE_FIELDS)
    prev_subtiles = 0

    while valid:
        expert, tile_m, tile_n = rInfo[0], rInfo[1], rInfo[2]
        alpha_val = load_gmem(alpha, expert)
        beta_val  = load_gmem(beta, expert)
        # instruction_selection: two `ld.global.b32`; extent: per-expert scalars
        #   (source 2979-2980; PTX 1627-1633)
        square_alpha = mul(alpha_val, alpha_val)
        d_window = expert_window(D_row, padded_offsets, expert)
        if P.generate_sfd:
            d_col_window   = expert_window(D_col, padded_offsets, expert)
            sfd_row_window = expert_window(SFD_row, padded_offsets, expert)
            # With discrete_col_sfd the source rebinds SFD_col to the row-shaped
            # mapping and repartitions it, so the column scales land in the
            # caller's buffer under the row index formula.
            sfd_col_window = (sfd_row_window if P.discrete_col_sfd
                              else expert_window(SFD_col, padded_offsets, expert))

        # The accumulator stage alternates with the consumer phase, and the
        # subtile walk reverses on the tiles that land in stage 0.
        acc_stage = acc_cons.phase if OVERLAPPING_ACCUM else acc_cons.index
        reverse_subtile = OVERLAPPING_ACCUM and acc_stage == 0

        row = (tile_m // ATOM_THR) * 256 + (block_idx_x() % ATOM_THR) * 128 + tid
        prob_val = 1.0
        if P.with_prob:
            prob_val = load_gmem(expert_window(prob, padded_offsets, expert), row)
            # instruction_selection: `ld.global.b32`; extent: one scalar per
            #   thread, indexed by the thread's own row (source 3070-3071; PTX 1718-1719)
        dprob_acc = 0.0
        amax_gate = amax_up = 0.0

        wait(ACC_FULL[acc_cons.index], acc_cons.phase)
        # instruction_selection: retry loop containing
        #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: one
        #   accumulator acquire per work tile (source 3078; PTX 1722, 1729)

        for subtile in range(EPI_SUBTILES):       # anchor body is nounroll
            real_subtile = (EPI_SUBTILES - 1 - subtile) if reverse_subtile else subtile
            copy_t2r(tAcc[acc_stage, real_subtile], rAcc)
            # instruction_selection: `tcgen05.ld.sync.aligned.32x32b.x32.b32`
            #   alone -- what follows it is `mov.b64` register unpacking, and the
            #   kernel's only `tcgen05.wait::ld` belongs to the early-release
            #   branch below; extent: one 32-word accumulator subtile per thread
            #   (source 3121-3122; PTX 1770)

            if OVERLAPPING_ACCUM and subtile == ACC_EARLY_RELEASE:
                fence_view_async_tmem_load()
                # instruction_selection: `tcgen05.wait::ld.sync.aligned`; extent:
                #   the kernel's single TMEM read fence, guarding the early
                #   release (source 3135; PTX 1792)
                if elect_one():
                    arrive(ACC_EMPTY[acc_cons.index])
                    # instruction_selection: `mapa.shared::cluster.u32` then
                    #   `mbarrier.arrive.shared::cluster.b64`; extent: one elected
                    #   early release, freeing the accumulator once the
                    #   scale-factor columns are no longer overlapped
                    #   (source 3137; PTX 1617, 1800)
                advance(acc_cons.index, acc_cons.phase, acc_cons.count, ACC_STAGES)

            wait(C_FULL[c_cons.index], c_cons.phase)
            # instruction_selection: retry loop containing
            #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: the
            #   gate stage's C-full acquire (source 3141; PTX 1841)
            copy_s2r(byte_ptr(sC, sC_addr(c_cons.index)), rC1)
            # instruction_selection: `ld.shared.v4.b32` sequence; extent: one
            #   128 x 32 gate subtile per thread (source 3142-3146; PTX 1853-1871)
            fence_async_shared_cta()
            # instruction_selection: `fence.proxy.async.shared::cta`; extent: one
            #   fence before the release (source 3147; PTX 1877)
            if lane == 0:
                arrive(C_EMPTY[c_cons.index])
                # instruction_selection: `mbarrier.arrive.shared.b64` under a
                #   `tid & 31 == 0` predicate; extent: one arrival per epilogue
                #   warp, matching the arrival count of four (source 3148;
                #   PTX 1882)
            advance(c_cons.index, c_cons.phase, c_cons.count, C_STAGES)
            wait(C_FULL[c_cons.index], c_cons.phase)
            # instruction_selection: retry loop containing
            #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: the
            #   up stage's C-full acquire (source 3150; PTX 1902)
            copy_s2r(byte_ptr(sC, sC_addr(c_cons.index)), rC2)
            # instruction_selection: `ld.shared.v4.b32` sequence; extent: one
            #   128 x 32 up subtile per thread (source 3151-3155; PTX 1914-1932)
            fence_async_shared_cta()
            # instruction_selection: `fence.proxy.async.shared::cta`; extent: one
            #   fence before the release (source 3156; PTX 1938)
            if lane == 0:
                arrive(C_EMPTY[c_cons.index])
                # instruction_selection: `mbarrier.arrive.shared.b64` under the
                #   same lane predicate; extent: one arrival per epilogue warp
                #   (source 3157; PTX 1943)
            advance(c_cons.index, c_cons.phase, c_cons.count, C_STAGES)

            # ---- dGLU backward -------------------------------------------
            # gate = rC1, up = rC2. dSwiGLU and dSiTU-GLU scale both by beta;
            # dGeGLU does not, and clamps instead.
            acc = mul(rAcc, square_alpha)
            # instruction_selection: `mul.rn.f32x2` under vectorized_f32, plain
            #   `mul.f32` otherwise; extent: one 32-word register tile, 16 packed
            #   issues (source 1721-1729; PTX 1954)
            # `prob_val` is folded in only when the specialization has prob; with
            # `with_prob` false the source emits no multiply at all rather than
            # multiplying by a constant one. `square_alpha` is written per work
            # tile here while the source recomputes it per subtile (3171); the
            # backend hoists it to PTX 1739, so the emitted code matches this
            # placement rather than the source text's.
            gate = mul(cast(rC1, f32), beta_val) if P.act != "dgeglu" else cast(rC1, f32)
            up   = mul(cast(rC2, f32), beta_val) if P.act != "dgeglu" else cast(rC2, f32)
            # instruction_selection: `cvt.f32.bf16` then `mul.rn.f32x2` under
            #   vectorized_f32, plain `mul.f32` otherwise; extent: one 32-word
            #   register tile (source 1730-1747, 2031-2060; PTX 1956-1967)
            if P.act == "dswiglu":
                sig   = sigmoid(gate)
                # instruction_selection: `mul.rn.f32x2` by the immediate
                #   `0fBFB8AA3B` (= -log2 e), two `ex2.approx.ftz.f32`, an
                #   `add.rn.f32x2` with 1.0, then two `rcp.approx.ftz.f32`. The
                #   negation is folded into the multiplier immediate, so the
                #   sigmoid itself emits no `neg.f32`; extent: one 32-word
                #   fastmath sigmoid (source 1748-1766; PTX 1969-1987)
                swish = mul(gate, sig)
                # instruction_selection: `mul.rn.f32x2`; extent: one 32-word
                #   register tile (source 1767-1772; PTX 1989)
                d_prob = mul(mul(up, swish), acc)
                # instruction_selection: `mul.rn.f32x2` x2; extent: one 32-word
                #   register tile. `up * swish` is formed first and the
                #   pre-prob accumulator multiplies last (source 1778-1788;
                #   PTX 1991, 1993)
                acc_prob = mul(acc, prob_val)
                # instruction_selection: `mul.rn.f32x2`; extent: one 32-word
                #   register tile, shared by both gradients below and computed
                #   once (source 1792-1795; PTX 1996)
                d_up   = mul(acc_prob, swish)
                # instruction_selection: `mul.rn.f32x2`; extent: one 32-word
                #   register tile (source 1797-1805; PTX 1998)
                d_gate = mul(mul(mul(acc_prob, up), sig),
                             add(1.0, mul(gate, sub(1.0, sig))))
                # instruction_selection: `neg.f32` then `add.rn.f32x2` for the
                #   `1 - sigmoid` term, and `mul.rn.f32x2` for the products.
                #   The chain multiplies left to right -- `acc_prob * up`, then
                #   `* sig`, then `* (1 + gate*(1-sig))` last -- and never forms
                #   `up * sig`; extent: 32 words (source 1807-1851; PTX 2001,
                #   2003, 2005-2016)
            elif P.act == "dgeglu":
                y_gate = min(gate, glu_clamp_max)
                y_up   = max(min(up, glu_clamp_max), glu_clamp_min)
                # instruction_selection: `setp.ge.f32` and `setp.le.f32` feeding
                #   `selp.f32` -- the backend lowers these clamps to predicated
                #   selects, not to a `min`/`max` instruction; extent: 64
                #   `setp.ge.f32` (the gate clamp at source 2158 and the up
                #   clamp at 2159) and 32 `setp.le.f32` (source 2160)
                #   (scalar_geglu PTX 2019-2026). The source also has a packed
                #   BF16 clamp path at 2067-2069 (`fmin_bf16x2` /
                #   `fmax_bf16x2`); no exported specialization reaches it, so it
                #   carries no PTX evidence here.
                sig    = sigmoid(mul(y_gate, geglu_alpha))
                # instruction_selection: the fastmath sigmoid expands inline to
                #   `mul.f32` x64, `ex2.approx.ftz.f32` x32, `add.f32` x32 and
                #   `rcp.approx.ftz.f32` x32 -- 1/(1+exp(-x)) with no library
                #   call and no `tanh`; extent: 32 elements per subtile
                #   (source 2162; scalar_geglu PTX 2028-2036, repeated at 3726)
                d_gate = mul(mul(mul(mul(acc, prob_val), sig),
                                 add(1.0, mul(mul(geglu_alpha, y_gate), sub(1.0, sig)))),
                             add(y_up, linear_offset))
                d_up   = mul(mul(mul(acc, prob_val), y_gate), sig)
                # instruction_selection: `d_gate` lowers to `mul.f32` x96,
                #   `fma.rn.f32` x32, `add.f32` x32 and `sub.f32` x32 -- the
                #   backend contracts `1 + alpha*y1*(1-sig)` into one FMA;
                #   `d_up` is `mul.f32` x64. The two `mul.f32` at source 2154
                #   and 2157 (x32 each) produce `acc*alpha^2` and the prob
                #   scaling that both expressions share; extent: 32 elements
                #   per subtile (source 2154, 2157, 2164-2165; scalar_geglu PTX
                #   2015, 2017, 2037-2050, 2051-2052)
                d_gate = mul(d_gate, select(gate <= glu_clamp_max, 1.0, 0.0))
                d_up   = mul(d_up, select(and_(up >= glu_clamp_min,
                                               up <= glu_clamp_max), 1.0, 0.0))
                # instruction_selection: `setp.le.f32` then `selp.f32` and
                #   `mul.f32`. The two-sided bound folds to one `abs.f32` plus a
                #   single `setp.le.f32` because the clamp limits are symmetric,
                #   so no `setp.ge.f32` is emitted here; extent: 32 of each per
                #   mask, 64 `setp.le.f32` across the two (source 2167-2171;
                #   scalar_geglu PTX 2054-2065, one of several identical
                #   unrolled copies -- 2599-2610 is the same block). The
                #   vectorized
                #   branch at source 2120-2130 computes the same masks with
                #   packed multiplies and is not reached by any export.
                d_prob = mul(mul(mul(y_gate, sig), add(y_up, linear_offset)), acc)
                # instruction_selection: `mul.f32` x96 -- three chained scalar
                #   multiplies, reusing the already-clamped `y_up +
                #   linear_offset` computed for `d_gate` rather than recomputing
                #   the add; extent: 32 elements per subtile (source 2174;
                #   scalar_geglu PTX 2067-2069)
            elif P.act == "dsituglu" and situ_beta1 == 4.0:
                # A distinct closed form, not a vectorization of the branch
                # below: with t = tanh(gate/4), the identity
                # sigmoid(g) = 0.5 + t / (1 + t^2) removes the exponential
                # entirely, which is why this export contains zero
                # `ex2.approx.ftz.f32`. It always uses packed f32x2 arithmetic,
                # independently of `vectorized_f32`.
                gate_tanh = tanh(mul(gate, 1.0 / situ_beta1))
                up_tanh   = tanh(mul(up, 1.0 / situ_beta2))
                # instruction_selection: `tanh.approx.f32`, two scalar
                #   evaluations per packed pair -- the packing applies to the
                #   surrounding `mul`/`add`, not to the tanh itself; extent: 64
                #   evaluations over the subtile -- 16 per source line across the
                #   four scalar calls, 64 in the export, against 0
                #   `tanh.approx.f32x2` (source 1945-1950; situglu PTX 1981,
                #   1987, 1993, 1999)
                gate_tanh_sq   = mul(gate_tanh, gate_tanh)
                denom_rcp      = rcp(add(1.0, gate_tanh_sq))
                # instruction_selection: `mul.rn.f32x2` x16 for the square,
                #   `add.rn.f32x2` x16 for `1 + t^2`, then `rcp.approx.ftz.f32`
                #   x16 per half of the pair (32 total) -- the reciprocal is the
                #   one step with no packed form; extent: 32 elements per
                #   subtile (source 1952-1956; situglu PTX 2004, 2009, 2012,
                #   2014)
                sig            = add(0.5, mul(gate_tanh, denom_rcp))
                gate_value     = mul(mul(situ_beta1, gate_tanh), sig)
                up_value       = mul(situ_beta2, up_tanh)
                gate_grad      = mul(sub(1.0, gate_tanh_sq),
                                     add(0.5, mul(2.0, mul(gate_tanh,
                                                           mul(denom_rcp, denom_rcp)))))
                up_grad        = sub(1.0, mul(up_tanh, up_tanh))
                # instruction_selection: `mul.rn.f32x2` and `add.rn.f32x2`
                #   throughout, x16 per source line (x32 where a line holds two
                #   packed products, as `gate_value` at 1962 does). Both
                #   `1 - t^2` forms are spelled as an add of a negation, not a
                #   subtract: `neg.f32` x32 then `add.rn.f32x2` x16 each, since
                #   there is no packed `sub.rn.f32x2`; extent over source
                #   1958-1981, 144 `mul.rn.f32x2`, 64 `add.rn.f32x2` and 64
                #   `neg.f32` -- the remaining 112 packed multiplies belong to
                #   the `d_gate`/`d_up`/`d_prob` annotation below, and the
                #   export's file totals of 480 and 96 also count the
                #   pre-activation scaling at 1927-1943 and the quantize path,
                #   so neither is this block's extent (source 1958-1981;
                #   situglu PTX 2017-2061)
                act_grad = mul(acc, prob_val)
                d_gate = mul(mul(act_grad, up_value), gate_grad)
                d_up   = mul(mul(act_grad, gate_value), up_grad)
                d_prob = mul(mul(acc, gate_value), up_value)
                # instruction_selection: `mul.rn.f32x2` throughout -- x16 for
                #   the shared `acc * prob` at source 1984, then x32 each for
                #   `d_gate` (1985), `d_up` (1986) and `d_prob` (1990). `d_prob`
                #   multiplies the pre-prob accumulator, so it reads `grad`
                #   rather than the prob-scaled value the other two share;
                #   extent: 32 elements per subtile (source 1984-1990; situglu
                #   PTX 2064, 2066-2069, 2071-2074, 2076-2078)
            else:                                  # dsituglu, general beta1
                sig       = sigmoid(gate)
                gate_tanh = tanh(mul(gate, 1.0 / situ_beta1))
                up_tanh   = tanh(mul(up, 1.0 / situ_beta2))
                # instruction_selection: `tanh.approx.f32`; extent: two scalar
                #   evaluations per element (source 1997-2019). No exported
                #   specialization reaches this branch, so it carries no PTX
                #   evidence here.
                gate_value = mul(mul(situ_beta1, gate_tanh), sig)
                up_value   = mul(situ_beta2, up_tanh)
                gate_grad  = add(mul(sub(1.0, mul(gate_tanh, gate_tanh)), sig),
                                 mul(mul(mul(situ_beta1, gate_tanh), sig), sub(1.0, sig)))
                up_grad    = sub(1.0, mul(up_tanh, up_tanh))
                act_grad = mul(acc, prob_val)
                d_gate = mul(mul(act_grad, up_value), gate_grad)
                d_up   = mul(mul(act_grad, gate_value), up_grad)
                d_prob = mul(mul(acc, gate_value), up_value)

            if P.with_prob:
                reduce_add(d_prob, dprob_acc)
                # instruction_selection: `add.rn.f32x2` tree under
                #   vectorized_f32, otherwise a scalar `add.f32` reduction;
                #   extent: one 32-word reduction per subtile, 16
                #   `add.rn.f32x2` -- the anchor's other 48 belong to the
                #   activation itself (source 3200-3221)

            if P.with_dbias:
                # Three stages, each separated by named barrier 2: transpose the
                # register tile into this warp's pane, reduce each column pair
                # inside the warp, then exchange the four warps' partials through
                # the head of the same buffer so warp 0 alone issues the atomic.
                for n in range(32):
                    store_smem(d_gate[n], byte_ptr(sDbias, sDbias_addr(warp, n, lane)))
                    store_smem(d_up[n],   byte_ptr(sDbias, sDbias_addr(warp, 32 + n, lane)))
                # instruction_selection: `st.shared.b32`; extent: 64 stores per
                #   thread (source 1642-1643; PTX 2932-3058, address setup at
                #   2928-2931)
                named_barrier(id=2, threads=128)
                # instruction_selection: `bar.sync 2,128`; extent: the transpose
                #   rendezvous (source 1645; PTX 3060)

                # Lane l < 16 owns gate columns (2l, 2l+1); lane l >= 16 owns up
                # columns (2(l-16), 2(l-16)+1) of the same subtile pair.
                col_a = dbias_col_a(lane)
                col_b = col_a + 1
                sum_a = sum_b = 0.0
                for group in range(8):
                    reduce_add(load_smem_vector(byte_ptr(sDbias,
                                    sDbias_read_addr(warp, col_a, group)), 4),
                               sum_a)
                    reduce_add(load_smem_vector(byte_ptr(sDbias,
                                    sDbias_read_addr(warp, col_b, group)), 4),
                               sum_b)
                # instruction_selection: `shl.b32`/`and.b32`/`or.b32` for the XOR
                #   offset, then `ld.shared.b32` and an `add.f32` tree; extent: 64
                #   loads per lane, eight four-element groups per column
                #   (source 1663-1672; PTX 3064-3069, 3071-3090, at
                #   `.loc 1 1663` and `.loc 1 1667`)

                named_barrier(id=2, threads=128)
                # instruction_selection: `bar.sync 2,128`; extent: the barrier
                #   that frees the pane for the partial exchange (source 1679;
                #   PTX 3461)
                store_smem((sum_a, sum_b),
                           byte_ptr(sDbias, sDbias_partial_addr(warp, lane)))
                # instruction_selection: two `st.shared.b32`; extent: this warp's
                #   column-pair partial (source 1684; PTX 3470-3472)
                named_barrier(id=2, threads=128)
                # instruction_selection: `bar.sync 2,128`; extent: the exchange
                #   rendezvous (source 1685; PTX 3474)
                if warp == 0:
                    cta_sum_a = cta_sum_b = 0.0
                    for peer in range(4):
                        reduce_add(load_smem_vector(byte_ptr(sDbias,
                                        sDbias_partial_addr(peer, lane)), 2),
                                   (cta_sum_a, cta_sum_b))
                    # instruction_selection: eight `ld.shared.b32` and an
                    #   `add.f32` tree, under the warp-0 predicate; extent: four
                    #   warps' partials (source 1687-1694; PTX 3476, 3496-3511,
                    #   3546)
                    if dbias_column(tile_n, real_subtile, lane) < N_OUT:
                        atomic_add_bf16x2(pack_bf16x2(cta_sum_a, cta_sum_b),
                                          byte_ptr(dbias,
                                                   dbias_addr(expert, tile_n, real_subtile, lane)))
                        # instruction_selection: `cvt.rn.bf16x2.f32` then
                        #   `red.global.add.noftz.bf16x2`, guarded by
                        #   `n_offset < dbias_n_total`; extent: one packed atomic
                        #   per lane, covering two adjacent output columns
                        #   (source 1696-1698; the else-branch at 1699-1702 is dead because `dbias_cross_warp_reduce` is bound to `generate_dbias` at source 770; PTX 3549)

            if P.with_amax:
                amax_gate = max(amax_gate, reduce_max(abs(d_gate)))
                amax_up   = max(amax_up, reduce_max(abs(d_up)))
                # instruction_selection: `abs.f32`, then a `max.NaN.f32` tree
                #   for the in-register vector reduce (including its
                #   `max.NaN.f32 ..., 0f00000000` init), then a plain `max.f32`
                #   for the running accumulation across subtiles --
                #   `cute.arch.fmax` defaults to `nan=False`; extent: two 32-word
                #   maxima (source 1198-1199; fp4 PTX 3532-3583, 3585-3636)

            if P.generate_sfd:
                # Row quantization takes the absolute maximum over each
                # sf_vec_size run inside the register tile; column quantization
                # takes it across the warp's 32 lanes instead.
                for half, (d_value, d_reg, d_reg_col) in enumerate(
                        ((d_gate, rD1, rD1c), (d_up, rD2, rD2c))):
                    slot = (real_subtile * 2 + half) % 4
                    row_scale = mul(mul(reduce_max(abs(d_value)), rcp_limit), norm_const)
                    # instruction_selection: `abs.f32`, then a `max.NaN.f32`
                    #   tree, then two `mul.f32`; extent: one scale per
                    #   sf_vec_size run (source 1334-1341; PTX 3631-3662 and
                    #   4829-4860 for the absolute values, 3664-3680 and
                    #   4862-4878 for the trees)
                    rSFDr[slot] = row_scale        # register write, not SMEM
                    quantized = mul(d_value,
                                    mul(norm_const, rcp(round_to_sf(row_scale))))
                    # instruction_selection:
                    #   `cvt.rp.satfinite.ue8m0x2.f32` then
                    #   `cvt.rn.bf16x2.ue8m0x2` to round the scale toward
                    #   positive infinity, then `rcp.approx.ftz.f32`,
                    #   `min.NaN.f32` against FLT_MAX -- the NaN-propagating form
                    #   is what `nan=True` selects, and the anchor has 66
                    #   `min.NaN.f32` and no plain `min.f32` -- and
                    #   `mul.rn.f32x2`; extent: one 32-word rescale, 16 packed
                    #   multiplies per call and 32 across the two
                    #   (source 1357-1373; PTX 3695 and 3697 for the round-trip,
                    #   3704 `rcp`, 3706 `mul.f32`, 3711 `min.NaN.f32`, and the
                    #   rescale loop 3718-3782 at `.loc 1 1368`)
                    d_reg[:] = pack_f8x4(quantized)    # register write
                    # instruction_selection: `cvt.rn.satfinite.e4m3x2.f32` pairs
                    #   packed with `mov.b32`, emitted inside `// begin inline
                    #   asm` blocks; extent: eight packed words
                    #   (source 1379-1390; PTX 3787, 3789, 3802)
                    # One loop over the eight four-element groups. Each group's
                    # four warp reductions serve both purposes: the lane-select
                    # that decides which one this lane stores as the scale
                    # factor, and the rescale of that group's own elements.
                    # There is no second reduction pass.
                    for group in range(8):
                        column_max = [warp_reduce_max(abs(d_value[4 * group + j]))
                                      for j in range(4)]
                        # instruction_selection: `redux.sync.max.NaN.f32`;
                        #   extent: four 32-lane reductions per group, 32 per
                        #   call and 64 per subtile across the two halves
                        #   (source 1415-1446)
                        combined = mul(rcp_limit, norm_const)
                        # instruction_selection: one scalar `mul.f32`, hoisted
                        #   out of the group loop; extent: one combined scale
                        #   reused by both packed multiplies below and by every
                        #   group (source 1449; PTX 3923)
                        column_scale = [mul(m, combined) for m in column_max]
                        # instruction_selection: `mul.rn.f32x2` x2; extent: the
                        #   four scales of one group, 16 packed multiplies at
                        #   each of the two sites over the subtile
                        #   (source 1450-1461; PTX 3926, 3930)
                        for j in range(4):
                            if lane == 4 * group + j:
                                rSFDc[slot] = column_scale[j]   # register write
                        # instruction_selection: `setp.eq.b32` then `selp.f32`,
                        #   so lane l keeps column l's scale; extent: 64
                        #   `selp.f32` in the anchor (source 1463-1470, 1526)
                        rescaled[4 * group:4 * group + 4] = [
                            mul(d_value[4 * group + j],
                                mul(norm_const, rcp(round_to_sf(column_scale[j]))))
                            for j in range(4)
                        ]
                        # instruction_selection: `cvt.rp.satfinite.ue8m0x2.f32`
                        #   and `cvt.rn.bf16x2.ue8m0x2` to round the four scales,
                        #   `rcp.approx.ftz.f32` x4, `min.NaN.f32` x4 against
                        #   FLT_MAX, then `mul.rn.f32x2` x2 for the scales and x2
                        #   for the rescale; extent: one group
                        #   (source 1472-1524)
                    d_reg_col[:] = pack_f8x4(rescaled)   # register write
                    # instruction_selection: `cvt.rn.satfinite.e4m3x2.f32` pairs
                    #   packed with `mov.b32`, in a separate loop after the
                    #   rescale; extent: eight packed words, the other 32 of the
                    #   anchor's 64 E4M3 converts (source 1528-1540)
                if subtile % 2 == 1:
                    # Both packs are unconditional and precede both guards, as
                    # the source has them at 3327-3328 and the export at
                    # 6033-6053 -- only the stores are predicated.
                    rSFDr_p[:] = pack_f8x4(rSFDr)
                    # instruction_selection: `cvt.rp.satfinite.ue8m0x2.f32`
                    #   x2 then `mov.b32` to join the halves -- E8M0
                    #   round-up, not the E4M3 family used for D; extent:
                    #   four row scales per store. These are two of the four
                    #   `cvt.rp` in the anchor that have no matching
                    #   `cvt.rn.bf16x2.ue8m0x2` upcast partner, the other 34
                    #   being the dequantize reads (source 3327; PTX 6033,
                    #   6035, 6037)
                    rSFDc_p[:] = pack_f8x4(rSFDc)
                    # instruction_selection: same E8M0 pack, reading the
                    #   column fragment back from its local depot first
                    #   with `ld.local.v4.b32`; extent: four column scales
                    #   per store, the other two upcast-less `cvt.rp`
                    #   (source 3328; PTX 6042, 6049, 6051, 6053)
                    # Two independent guards over two different extents.
                    if sfd_row_column(tile_n, real_subtile) < sfd_row_extent:
                        store_gmem(rSFDr_p,
                                   sfd_row_addr(sfd_row_window, tile_m, tile_n, real_subtile))
                        # instruction_selection: `shl.b32` then `setp.le.s32`
                        #   guarding one `st.global.b32` that carries the four
                        #   packed E8M0 row scales contiguously; extent: one
                        #   store every second subtile (source 3329-3330;
                        #   PTX 6058-6059, branch at 6061, store at 6077)
                    if sfd_col_column(tile_n, real_subtile) < sfd_col_extent:
                        store_gmem(rSFDc_p,
                                   sfd_col_addr(sfd_col_window, tile_m, tile_n, real_subtile))
                        # instruction_selection: a separate `shl.b32` plus
                        #   `setp.le.s32` over a different extent, guarding four
                        #   `st.global.b8` at stride 4 -- the column scales are
                        #   not contiguous the way the row scales are; extent:
                        #   four stores every second subtile (source 3331-3332;
                        #   PTX 6088, 6090, 6092, 6110, 6114, 6118, 6122)
            else:
                rD1[:] = cast(d_gate, P.d_dtype)   # register write, not SMEM
                rD2[:] = cast(d_up,   P.d_dtype)   # register write, not SMEM
                # instruction_selection: `cvt.rn.bf16x2.f32` for BF16 output, or
                #   `cvt.rn.f16x2.f32`, or a plain `mov.b32` for FP32; extent:
                #   two 32-word converts (source 3335-3338; fp4 PTX 3643-3675)

            # ---- staged D store ------------------------------------------
            if warp == 0:
                # The D store pipeline is a bulk-group counter, not an mbarrier:
                # there is no D barrier in the storage map.
                tma_wait_group(pending=0)
                # instruction_selection: `cp.async.bulk.wait_group.read 0`;
                #   extent: one bulk-group wait guarding D-slot reuse
                #   (source 3373-3375; PTX 6129)
            named_barrier(id=2, threads=128)
            # instruction_selection: `bar.sync 2,128`; extent: the four-warp
            #   rendezvous before writing the SMEM slot (source 3376; PTX 6162)
            slot1 = prev_subtiles % D_STAGES; prev_subtiles += 1
            slot2 = prev_subtiles % D_STAGES; prev_subtiles += 1
            # Source order interleaves D and D_col per slot.
            copy_r2s(rD1, byte_ptr(sD, sD_addr(slot1)))
            # instruction_selection: `st.shared.v4.b32`; extent: the gate D
            #   subtile (source 3379; PTX 6174, 6176)
            if P.generate_sfd:
                copy_r2s(rD1c, byte_ptr(sD_col, sD_col_addr(slot1)))
                # instruction_selection: `st.shared.v4.b32`; extent: the gate
                #   D_col subtile (source 3385; PTX 6179, 6181)
            copy_r2s(rD2, byte_ptr(sD, sD_addr(slot2)))
            # instruction_selection: `st.shared.v4.b32`; extent: the up D subtile
            #   (source 3392; PTX 6193, 6195)
            if P.generate_sfd:
                copy_r2s(rD2c, byte_ptr(sD_col, sD_col_addr(slot2)))
                # instruction_selection: `st.shared.v4.b32`; extent: the up D_col
                #   subtile (source 3398; PTX 6198, 6200)
            fence_async_shared_cta()
            # instruction_selection: `fence.proxy.async.shared::cta`; extent:
            #   SMEM visibility to the TMA proxy (source 3404; PTX 6201-6202)
            named_barrier(id=2, threads=128)
            # instruction_selection: `bar.sync 2,128`; extent: the store
            #   rendezvous (source 3405; PTX 6203-6204)
            if warp == 0:
                # The two D subtiles of one accumulator subtile land in the
                # adjacent halves of the 2N region, which is why the column index
                # is 2 * real_subtile + {0,1} against a tile that only nominally
                # holds EPI_SUBTILES columns.
                copy_s2g(byte_ptr(sD, sD_addr(slot1)),
                         logical_region(d_window, (tile_m, tile_n * 2,
                                                   2 * real_subtile + 0), EPI_TILE), map_D)
                copy_s2g(byte_ptr(sD, sD_addr(slot2)),
                         logical_region(d_window, (tile_m, tile_n * 2,
                                                   2 * real_subtile + 1), EPI_TILE), map_D)
                # instruction_selection:
                #   `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint`;
                #   extent: two 128 x 32 D subtiles (source 3410-3418; PTX 6224, 6230)
                if P.generate_sfd:
                    copy_s2g(byte_ptr(sD_col, sD_col_addr(slot1)),
                             logical_region(d_col_window, (tile_m, tile_n * 2,
                                                           2 * real_subtile + 0), EPI_TILE),
                             map_D_col)
                    copy_s2g(byte_ptr(sD_col, sD_col_addr(slot2)),
                             logical_region(d_col_window, (tile_m, tile_n * 2,
                                                           2 * real_subtile + 1), EPI_TILE),
                             map_D_col)
                    # instruction_selection: same bulk-tensor store form; extent:
                    #   two 128 x 32 D_col subtiles (source 3420-3430; PTX 6233, 6235)
                tma_commit_group()
                # instruction_selection: `cp.async.bulk.commit_group`; extent:
                #   one group per subtile pair (source 3432; PTX 6237)
                advance(d_prod.index, d_prod.phase, d_prod.count, D_STAGES // 2)
            named_barrier(id=2, threads=128)
            # instruction_selection: `bar.sync 2,128`; extent: end-of-subtile
            #   rendezvous (source 3433; PTX 6240)
            # The subtile loop is `.pragma "nounroll"` with a trip count of 8
            #   (source 3093; PTX 1751, loop bound at 6243).

        if not OVERLAPPING_ACCUM:
            if elect_one():
                arrive(ACC_EMPTY[acc_cons.index])
                # instruction_selection: `mbarrier.arrive.shared::cluster.b64`;
                #   extent: the end-of-tile accumulator release taken only when
                #   the overlapping layout is off (source 3439-3440). The anchor
                #   has OVERLAPPING_ACCUM on, so this branch is absent from it.
            advance(acc_cons.index, acc_cons.phase, acc_cons.count, ACC_STAGES)

        valid = take_tile_info(info_cons, EXPERT_TILE_FIELDS)

        if P.with_amax:
            for slot, value in ((0, amax_gate), (1, amax_up)):
                warp_max = warp_reduce_max(value)
                # instruction_selection: `redux.sync.max.NaN.f32`; extent: one
                #   32-lane reduction (source 1204-1209; fp4 PTX 3815, 3862)
                if lane == 0:
                    store_smem(warp_max, byte_ptr(sAmax, sAmax_addr(warp)))
                    # instruction_selection: `st.shared.b32` under a lane-0
                    #   predicate; extent: one word per epilogue warp
                    #   (source 1212-1214; fp4 PTX 3823)
                named_barrier(id=2, threads=128)
                # instruction_selection: `bar.sync 2,128`; extent: the barrier
                #   that publishes the four warps' maxima (source 1216;
                #   fp4 PTX 3829)
                if warp == 0 and lane == 0:
                    atomic_max_bits(reduce_max(load_smem_vector(sAmax, 4)),
                                    byte_ptr(amax, 8 * expert + 4 * slot))
                    # instruction_selection: plain `max.f32` over the four
                    #   slots -- `cute.arch.fmax` defaults to `nan=False`, as
                    #   it does for the per-thread running maximum; only the
                    #   in-register vector reduces and the per-column
                    #   `redux.sync.max.NaN.f32` propagate NaN -- then
                    #   `atom.global.max.s32` on the non-negative bit pattern;
                    #   extent: one atomic per expert half (source 1219-1226;
                    #   fp4 PTX 3835, 3839, 3843, 3847, 3853, 3901)
                named_barrier(id=2, threads=128)
                # instruction_selection: `bar.sync 2,128`; extent: the trailing
                #   barrier, without which the second call's stores would race
                #   the peers still reading this call's slots (source 1229;
                #   fp4 PTX 3858)
        if P.with_prob:
            atomic_add_f32(dprob_acc,
                           byte_ptr(expert_window(dprob, padded_offsets, expert), 4 * row))
            # instruction_selection: `atom.global.add.f32`; extent: one atomic
            #   per thread per work tile (source 3471-3476; PTX 6284)
        # The next-tile record fetch for this role is at PTX 6255-6274.

    if warp == 0:
        relinquish_tmem_alloc_permit()
        # instruction_selection:
        #   `tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned`, a
        #   warp-wide instruction guarded by the warp predicate alone; extent:
        #   one release (source 3481; PTX 6295-6296)
    named_barrier(id=2, threads=128)
    # instruction_selection: `bar.sync 2,128`; extent: pre-free rendezvous
    #   (source 3482; PTX 6300-6301)
    if warp == 0:
        dealloc_tmem(512)
        # instruction_selection: on the two-CTA path an
        #   `mbarrier.arrive.shared::cluster.b64` and its matching wait, then
        #   `tcgen05.dealloc.cta_group::2.sync.aligned.b32`; extent: one
        #   deallocation, again warp-wide (source 3483; PTX 6309-6311, 6318,
        #   6328). Its own `.loc` names 3489, a comment line, so the source
        #   citation here is the `tmem.free` call that emits it.
    tma_wait_group(pending=0)
    # instruction_selection: `cp.async.bulk.wait_group.read 0`; extent: the D
    #   store tail (source 3488; PTX 6331)

# ===========================================================================
# Role 5: warp 6, epilogue C producer mirroring the epilogue's subtile order
# source 3490-3570; PTX 6333-6648
# ===========================================================================
if warp == C_WARP:
    c_prod = pipeline_state(stages=C_STAGES, phase=1, index=0, count=0)
    info_cons = pipeline_state(stages=TILE_STAGES, phase=0, index=0, count=0)
    valid = take_tile_info(info_cons, EXPERT_TILE_FIELDS)
    is_reverse = True

    while valid:
        # This warp keeps its own copy of the epilogue's alternating direction so
        # the two roles agree on which C subtile belongs to which accumulator
        # subtile without any extra handshake.
        reverse_subtile = is_reverse
        is_reverse = not is_reverse
        expert, tile_m, tile_n = rInfo[0], rInfo[1], rInfo[2]
        c_window = expert_window(C, padded_offsets, expert)
        d_tile_n = tile_n * 2                      # the 2N interleaved column tile

        # The C warp walks all eight accumulator subtiles, two C loads each, so
        # it covers all sixteen 32-column blocks of the 2N tile.
        for subtile in range(EPI_SUBTILES):
            real_subtile = (EPI_SUBTILES - 1 - subtile) if reverse_subtile else subtile
            wait(C_EMPTY[c_prod.index], c_prod.phase)
            # instruction_selection: retry loop containing
            #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: one
            #   C-empty acquire (source 3539; PTX 6423)
            if elect_one():
                arrive_expect_tx(C_FULL[c_prod.index], source_c_stage_bytes(P))
            # instruction_selection: `mbarrier.arrive.expect_tx.shared.b64`;
            #   extent: one elected arrival, anchor 8,192 bytes (PTX 6438)
            if elect_one():
                copy_g2s(logical_region(c_window,
                                        (tile_m, d_tile_n, 2 * real_subtile + 0),
                                        EPI_TILE),
                         byte_ptr(sC, sC_addr(c_prod.index)), map_C,
                         C_FULL[c_prod.index], mask=None)
            # instruction_selection:
            #   `cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
            #   extent: one 128 x 32 gate subtile, its column operand
            #   `base + (real_subtile << 6)` (source 3540; PTX 6446, 6458)
            advance(c_prod.index, c_prod.phase, c_prod.count, C_STAGES)
            wait(C_EMPTY[c_prod.index], c_prod.phase)
            # instruction_selection: retry loop containing
            #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: the
            #   second C-empty acquire (source 3547; PTX 6476)
            if elect_one():
                arrive_expect_tx(C_FULL[c_prod.index], source_c_stage_bytes(P))
                # instruction_selection: `mbarrier.arrive.expect_tx.shared.b64`;
                #   extent: the second elected arrival (PTX 6491)
            if elect_one():
                copy_g2s(logical_region(c_window,
                                        (tile_m, d_tile_n, 2 * real_subtile + 1),
                                        EPI_TILE),
                         byte_ptr(sC, sC_addr(c_prod.index)), map_C,
                         C_FULL[c_prod.index], mask=None)
            # instruction_selection:
            #   `cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
            #   extent: one 128 x 32 up subtile, its column operand the gate
            #   operand plus 32 (source 3548; PTX 6501, 6505)
            advance(c_prod.index, c_prod.phase, c_prod.count, C_STAGES)
        valid = take_tile_info(info_cons, EXPERT_TILE_FIELDS)

    for live in producer_tail(c_prod, C_STAGES):
        wait(C_EMPTY[live.index], live.phase)
        # instruction_selection: four retry loops containing
        #   `mbarrier.try_wait.parity.shared.b64 ...,10000000`; extent: the
        #   four-stage C tail (source 3570; PTX 6571, 6594, 6617, 6640)
```

## Pipeline inventory

| pipeline | stages | producer | consumer | publication | reuse |
| --- | --- | --- | --- | --- | --- |
| tile info | 2 | warp 7 | warps 0-6 (224 threads) | `mbarrier.arrive` after `fence.proxy.async.shared::cta` and named barrier 4 | each consumer arrives on tile-info-empty after copying the three record words it reads -- the MMA warp reads only the expert index, and `k_tile_cnt` is read by nobody |
| A/B/SFA/SFB | `num_ab_stage` (anchor 4) | warp 5, TMA transaction bytes | warp 4 | TMA `mbarrier.arrive.expect_tx` completion | `tcgen05.commit` multicast release per K tile |
| accumulator | 1, doubled in TMEM by the overlapping layout | warp 4 | warps 0-3 | `tcgen05.commit` after the last K tile | elected `mbarrier.arrive` after `ACC_EARLY_RELEASE` subtiles |
| C | 4 for FP8 A/B, 2 for FP4 | warp 6 | warps 0-3 | TMA `mbarrier.arrive.expect_tx` completion | per-subtile arrive after the register load |
| D store | `num_d_stage // 2` bulk groups over a 2-slot SMEM ring | warps 0-3, issued by warp 0 | TMA store unit | `cp.async.bulk.commit_group` | `cp.async.bulk.wait_group.read` before reusing a slot |

## TensorMap fields and tail behaviour

| map | rank | box | multicast | notes |
| --- | --- | --- | --- | --- |
| A | 3 | `(128, K_TILE, 1)` | across the cluster N extent | rebased per expert by a scalar row offset |
| B | 3 | `(CTA_TILE_N / ATOM_THR, K_TILE, 1)` | across the cluster M extent after the two-CTA split | dense: grid constant with an L coordinate; discrete: a per-expert image read from the workspace |
| SFA | 4 | one 32x4x4 atom group per 128 rows | as A | interleaved atom mapping, not a layout object |
| SFB | 4 | as SFA over N | across the cluster M extent of its own `cta_group::1` layout | the anchor multicasts SFB while B is not multicast |
| C | 3 | `(128, 32, 1)` | none | two loads per accumulator subtile: gate then up |
| D, D_col | 3 | `(128, 32, 1)` | none | column index `2 * real_subtile + {0,1}` walks the adjacent half of the 2N region |

Tail behaviour: expert ranges are padded to 256 rows, so no partial M tile
exists; the scheduler bounds the sweep by `padded_offsets[L-1]` and never emits
a tile past it. A zero-token expert contributes no tiles. `total_tokens <= 0`
exits every CTA before any role starts. The scale-factor stores are predicated
on the column staying inside the scale-factor extent, which is what covers an
`N_OUT` that is not a multiple of `32 * 4`.

## Static specialization boundary

| axis | accepted values | emitted-code consequence |
| --- | --- | --- |
| weight_mode | dense, discrete | discrete adds the pre-kernel, a workspace descriptor address on the B/SFB TMA copies, and an expert-indexed pointer load |
| sched | static, dynamic | dynamic adds the counter reset in the pre-kernel and replaces the linear index with `atom.global.add.u32` plus a cluster-shared broadcast |
| act | dswiglu, dgeglu, dsituglu | selects the epilogue expression set; only dgeglu skips the beta scaling and adds clamp masks; `situ_beta1 == 4.0` selects a distinct closed form that replaces the sigmoid with a tanh identity and so emits no `ex2.approx.ftz.f32` at all, and that branch always uses packed `f32x2` arithmetic regardless of `vectorized_f32` |
| vectorized_f32 | false, true | true replaces scalar `mul.f32`/`add.f32` with `mul.rn.f32x2`/`add.rn.f32x2`; FP8 C excludes it. The dSiTU-GLU `beta1 == 4.0` branch ignores this knob and is always packed |
| ab_dtype | FP4 E2M1, FP8 E4M3/E5M2 | selects `kind::mxf4nvf4` versus `kind::mxf8f6f4`, `K_TILE` 256 versus 128, and the C stage count |
| sf_dtype / sf_vec_size | E8M0 x {16, 32}, E4M3 x 16 | selects `block16` versus `block32` on the MMA and the scale-factor rounding family |
| c_dtype | FP32, FP16, BF16, E4M3, E5M2 | selects the C stage bytes and the register convert family |
| d_dtype | FP16, BF16, FP32 for FP4 A/B; E4M3 for FP8 A/B | FP8 output turns on the scale-factor path, D_col, and the packed `cvt.rn.satfinite.e4m3x2.f32` store |
| b_major | k, n (FP8 only) | changes the B TMA box and descriptor leading-dimension offset |
| mma_tiler_mn | `(128, 256)`, `(256, 256)` | `(256, 256)` selects `cta_group::2` on every tcgen05 instruction and halves the per-CTA B extent |
| cluster_shape_mn | cluster tiler M in {128, 256}, each extent a power of two at most 4 | selects the multicast masks and the `max_active_clusters` grid |
| with_dbias, with_prob, with_amax | false, true | each gates a reduction, its SMEM scratch, and its global atomic |
| discrete_col_sfd | false, true | true rebinds SFD_col to the row-shaped index mapping |
| activation knobs | folded FP32 constants | appear as immediates in the epilogue expressions |

Out of scope: `d_dtype = E5M2` (the source's FP8 packing helper emits nothing
for it), MMA tile N other than 256, the `N = 192`/`N = 64` SFB pointer shifts
that only such tiles could reach, `store_d_directly`, `epilogue_prefetch_more`,
`use_single_group_runtime_offsets`, and the Rubin SM107 fork.

`use_single_group_runtime_offsets` (source 263, 306-307, 2237-2240) defaults
false and the constructor rejects it unless `expert_cnt == 1`. When set it
replaces the `padded_offsets` array with a one-element register tensor holding
`mA`'s row extent, so the single expert's bound comes from the A tensor instead
of from gmem. Every specialization exported here takes the false path, where the
branch emits nothing and `total_tokens` is the `ld.global.b32` at source 2241
(PTX 104) that the sketch already carries.

## TIRx module and benchmark contract

- registry name `cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias`, category
  `cudnn`, compute capability 10.
- the device definition imports only `tirx_kernels.kern as K`; every copy,
  compute, and synchronization operation above is spelled through `K.ptx`,
  `K.specialize`, `K.smem_pool`, `K.Pipeline`, and `K.RingState`.
- exported symbols: `KERNEL_META`, `CONFIGS`, `BENCH_CONFIGS`, `get_kernel`,
  `prepare_data`, `run_test`, `prepare_bench`, `run_gpu`, `run_bench`.
- `get_kernel` returns `[helper, main]` when the specialization needs the
  pre-kernel and `[main]` otherwise, so both launches are timed the way the
  source times them.
- correctness compares TIRx against both an FP32 oracle and the upstream kernel
  compiled from `CUDNN_FRONTEND_PATH`, at the upstream tolerances: 1e-1/1e-2
  base, 0.125 for E4M3 output, and a dbias tolerance scaled by
  `0.008 * sqrt(tiles per expert)` to absorb BF16 atomic ordering.
- `prepare_bench` compiles only TIRx; the reference import and JIT happen inside
  `run_gpu`, which validates before timing.
- the performance gate is one complete `python -m tirx_kernels.bench_suite`
  run with references over the 49-row matrix, requiring
  `mean(cudnn_frontend_us) / mean(tirx_us) > 0.99` on every row.

## Instruction selection is a lowering consequence

| placement or schedule decision | anchor consequence |
| --- | --- |
| two-CTA MMA tile `(256, 256)` | every tcgen05 instruction carries `cta_group::2`, and the per-CTA B extent is 128 |
| FP8 A/B with E8M0 vec 32 | `tcgen05.mma...kind::mxf8f6f4.block_scale.block32`, four issues per K tile |
| scale factors staged in SMEM then TMEM | `tcgen05.cp.cta_group::2.32x128b.warpx4`, one SFA plus two SFB chunks per stage |
| accumulator read as one 32-word subtile | `tcgen05.ld.sync.aligned.32x32b.x32.b32` plus `tcgen05.wait::ld.sync.aligned` |
| overlapping accumulator with early release | the release arrives after subtile `ACC_EARLY_RELEASE`, not at the end of the tile |
| packed-f32x2 epilogue | 336 `mul.rn.f32x2` and 64 `add.rn.f32x2` in the anchor |
| fast sigmoid | 98 `rcp.approx.ftz.f32` and 32 `ex2.approx.ftz.f32` |
| E8M0 scale rounding | 38 `cvt.rp.satfinite.ue8m0x2.f32` with 34 `cvt.rn.bf16x2.ue8m0x2` upcasts, and 66 `min.NaN.f32` clamping the reciprocals |
| E4M3 output packing | 64 `cvt.rn.satfinite.e4m3x2.f32` |
| column scale over the warp | 64 `redux.sync.max.NaN.f32` with 64 `selp.f32` selecting each lane's own column |
| scale-factor stores | one `st.global.b32` for the contiguous row scales, four `st.global.b8` at stride 4 for the column scales |
| dbias through BF16 atomics | one `cvt.rn.bf16x2.f32` feeding `red.global.add.noftz.bf16x2`, issued by warp 0 only |
| dprob per thread per tile | one `atom.global.add.f32` |
| D staged through SMEM then TMA | four `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint` per subtile pair with one `cp.async.bulk.commit_group` |

Counting convention: instruction counts are instruction lines minus predicated
lines in the anchor export.

Extent denominators: an `extent:` figure counts the occurrences an annotation's
own operation contributes, not the export's total for that mnemonic. Where one
mnemonic is emitted by several annotated operations the per-annotation figures
sum to the export total rather than each matching it -- in `scalar_geglu`, for
instance, the clamp's 32 `setp.le.f32` and the two output masks' 64 together
account for that export's 96. A recount that compares a single annotation
against a whole-export histogram will therefore read low, and that is not a
mismatch. Figures explicitly labelled as file or export totals (the dSiTU-GLU
note at 1142 is the only one) are the exception and say so.

Line-info caveat: `.loc 1 701 0` is NVVM's no-line-info fallback -- source 701 is
the `@cute.jit` decorator on `__call__` -- and the FP8 converts carry `.loc`
values pointing at parameter lines 703-705. The 64 `redux.sync.max.NaN.f32`, 64
`abs.f32`, 64 `selp.f32`, 66 `min.NaN.f32`, the dbias
`red.global.add.noftz.bf16x2` and the dprob `atom.global.add.f32` all fall in
those regions, so their PTX citations above are pinned structurally, between
neighbouring instructions that do carry usable line info, rather than by `.loc`
alone.

Two further `.loc` values point at non-statement lines and are cited here by the
statement that emits them, not by what `.loc` says: `tcgen05.dealloc` carries
`.loc 1 3489`, a comment, and comes from `tmem.free` at source 3483; and an
`mbarrier.try_wait` carries `.loc 1 3573`, a function signature, and belongs to
the C pipeline's `producer_tail` at source 3570.
