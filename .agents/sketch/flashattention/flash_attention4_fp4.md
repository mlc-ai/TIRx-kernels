<!--
This sketch is a design artifact, not source. It is written once, reviewed
once, and then frozen: the implementation follows it, and where the two
disagree the TIRx module linked below is what runs.

Documents a TIRx port of hao-ai-lab/flash-attention-fp4
(https://github.com/hao-ai-lab/flash-attention-fp4 @ 5aa37a9680f7b76a11799b5f4846100ed5a3e6d8),
flash_attn/cute/flash_fwd_sm100_fp4.py. SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# flash_attention4_fp4: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, warp roles,
pipelines, control flow, and PTX-level operations of
[`tirx_kernels/flashattention/flash_attention4_fp4.py`](../../../tirx_kernels/flashattention/flash_attention4_fp4.py).
That TIRx module is the authoritative implementation.

The port covers **one kernel**: `FlashAttentionForwardSm100.kernel`
(`flash_fwd_sm100_fp4.py:1330-1964`, cited below as `:NNNN`) with its trace-time
launcher `__call__` (`:442-1327`), `__init__`/`_setup_attributes` (`:119-440`) and
every role helper it dispatches to: `load` (`:1967-2235`), `mma` (`:2238-2722`),
`softmax_loop`/`softmax_step` (`:2727-3063`, `:3559-3883`),
`_fused_log2_group_quant` (`:3171-3295`), `_pack_fp8` (`:3332-3346`),
`correction_loop`/`correction_rescale`/`correction_epilogue` (`:3886-4360`),
`epilogue_s2g` (`:4405-4510`), `load_Q`/`load_KV` (`:4512-4589`) and
`mainloop_s2t_copy_and_partition` (`:4591-4631`). The MMA issue sequences live in
`blackwell_helpers.py` (`bh:`): `gemm_ptx_partial_fp4` (`bh:1224-1554`),
`gemm_ptx_partial` (`bh:402-665`), `gemm_blockscaled_generic` (`bh:1191-1220`),
`tmem_ld_red_max` (`bh:1777-1829`), the packers `packed_float_to_e2m1` (`bh:1624`),
`packed_float_to_ue4m3` (`bh:1596`), `packed_float_to_ue8m0` (`bh:1849`) and
`ceil_f32` (`bh:1833`). Softmax arithmetic is `SoftmaxSm100` (`softmax.py`, `sm:`),
the packed fp32 idioms are `utils.py` (`ut:`), masking is `mask.py` (`mk:`), block
ranges are `block_info.py` (`bi:`), tile order is `tile_scheduler.py` (`ts:`).

**Every tensor core here sees block-scaled or narrow operands directly.** Q and K
arrive as packed NVFP4 (E2M1 + E4M3 scale per 16) or MXFP8 (E4M3 + E8M0 scale
per 32); V is BF16, plain E4M3, packed NVFP4 or MXFP8; when V is quantized the
softmax warps quantize P on the fly and write P's scale factors to shared
memory for the MMA warp to stage into TMEM. Scale factors travel GMEM -> SMEM
(TMA) -> TMEM (`tcgen05.cp`) and are consumed by `tcgen05.mma...block_scale`.

## Scope and instantiations

Fixed for every specialization in this port:

| axis | value | why it is fixed |
| --- | --- | --- |
| `m_block_size`, `n_block_size` | 128, 128 | `_flash_attn_fwd` defaults (`interface.py:218-219`) |
| `q_stage` | 2 | `not is_split_kv` (`:166`); the CTA owns 256 Q rows |
| `epi_stage`, `acc_stage` | 2, 1 | `:369-370` |
| `cta_group` | 1 | `use_2cta_instrs` hard-asserted False (`:624-625`) |
| `cluster` | (1,1,1) | `:174`, `:676` |
| `pack_gqa` | **False** | the interface default is True for GQA (`interface.py:432`), but the fork's FP4 kernel is numerically wrong on that path (measured cos_sim 0.94/0.96 vs fp64 where the unpacked path gives 1.0000); both sides run unpacked, `kv_head = head // qhead_per_kvhead` (`:2014-2016`) |
| `use_tma_KV`, `use_tma_O` | True | no paged KV, no varlen (`:153`, `:607`) |
| `s0_s1_barrier` | False | `:198`; the 8 `s0_s1_sequence` mbarriers are allocated and never initialized |
| `quant_qk` | True | Q/K scales are always supplied |
| SM103 knobs | `use_ldred_rowmax=True`, `force_e2e=False`, `fp4_pv_log2_quant=True`, `fp8_pv_use_explicit_pack=True`, no store pipelining, `fp8_pv_zero_fill_regs=True`, `sfqk_tmem_slot="s"` | the device probe (`:72-95`) and env defaults (`:282-335`); pinned by the harness |
| P handoff split knobs | fp8 3/4, fp4 1/2, bf16 3/4 | `:1105-1125` defaults |
| P store repetition | 16 bf16, 8 fp8/fp4/mxfp8 at every head_dim | `:2790-2807`; the harness pins `FA4_FP8_PV_TMEM_STORE_REP=8`, which the source honours before its d=64 default of 4 (`:2798-2801`) |
| `mLSE`, `learnable_sink`, `score_mod`, `mask_mod`, block sparsity, `v_descale`, split-KV, varlen, paged, local | absent / off | not in the README GB300 dispatch domain |

In scope, i.e. the specializations this module compiles:

| axis | values | what changes in device code |
| --- | --- | --- |
| `qk_format` | `nvfp4` (E2M1, E4M3 scale per 16) / `mxfp8` (E4M3, E8M0 scale per 32) | Q/K tile bytes (8 KB vs 16 KB), TMA swizzle (64B vs 128B), the QK MMA (`kind::mxf4nvf4.block_scale.scale_vec::4X`, K=64, 2 k-tiles at d=128 vs `kind::mxf8f6f4.block_scale.block32`, K=32, 4 k-tiles with per-k `sf_id`), SF bytes per tile (1024 vs 512), `sfqk` TMEM columns (8+8 vs 16+16) |
| `pv_format` | `bf16` / `fp8` / `nvfp4` / `mxfp8` | V tile bytes and majorness, the PV MMA kind (`f16` K=16 / `f8f6f4` K=32 / `mxf4nvf4` 4X K=64 / `mxf8f6f4` block32 K=32), the P program (bf16x2 pack / e4m3x2 pack / log-domain group quant to E2M1 + E4M3 SF / to E4M3 + E8M0 SF), P columns in TMEM (64 / 32 / 16 / 32), P store shape and split, whether SFP/SFV exist, `rescale_threshold` (8.0 vs 0.0) and `max_offset` (0 / 0 / log2 6 / log2 448), whether the hw tile maxes feed the group maxes (mxfp8 PV only), `kv_stage` |
| `head_dim` | 128; 64 only for `nvfp4+bf16` and `nvfp4+fp8` | QK k-tiles 2 -> 1, Q/K swizzle 64B -> 32B, `kv_stage` (4 -> 10, 4 -> 20), register split 192/80/48 -> 200/64/48, PV N 128 -> 64 (f16 idesc `0x08210490` -> `0x08110490`, f8f6f4 `0x08210010` -> `0x08110010`), O tile bytes, one TMA store per stage; the P store shape does not change (rep 8 pinned) |
| `is_causal` | False / True | scheduler (persistent static vs single-tile LPT), `n_block_max` clipping, non-persistent grid, and the softmax body: three inlined `softmax_step` copies (first, diagonal loop, remaining loop), ALL masked with the software row max; the hw `ld.red` max and (mxfp8 PV) hw group maxes are consumed only by the non-causal steady copy |
| `qhead_per_kvhead` | 1 / >1 | only the KV head index and the tile count; the device program is otherwise identical (unpacked GQA) |

Out of scope, with the predicate that excludes each:

- **`pack_gqa=True`** -- numerically wrong upstream (see the fixed-axes table).
- **varlen / paged / split-KV / local / LSE / sink / score_mod / mask_mod / block-sparse** -- all gated
  off at the host; the compiled-out arms are not transcribed.
- **e2e exp2 emulation** (`force_e2e`), the store-pipelined P paths, the fused fp8
  pack, the range-unrolled fp8 path -- env-gated off; the export contains none of
  their instructions.
- **2-CTA** -- asserted off.
- Tile (`Tx`) primitives are out of scope everywhere.

## The line-info exports this sketch is annotated from

Every `instruction_selection` annotation below is read out of a line-info PTX
export, not out of the source text. Sixteen exports are preserved under
`.porting/flash_attention4_fp4/ptx_lineinfo/<name>/`, produced by
`.porting/flash_attention4_fp4/export_driver.py` (real quantized data, s=512 so
both `softmax_step` copies exist, `h=32`, `kv=32` or `kv=8`), with
`CUTE_DSL_NO_CACHE=1 CUTE_DSL_KEEP=ptx CUTE_DSL_LINEINFO=1`. All are
`.version 9.3 .target sm_103a`.

| name | mode | causal / GQA | lines | `.loc` |
| --- | --- | --- | ---: | ---: |
| `nvfp4_bf16_d128` | 1 | no / MHA | 7617 | 2135 |
| `nvfp4_fp8_d128` | 2 | no / MHA | 8148 | 2285 |
| `nvfp4_nvfp4_d128` | 3 | no / MHA | 9247 | 2738 |
| `nvfp4_mxfp8_d128` | 4 | no / MHA | 9068 | 2562 |
| `mxfp8_bf16_d128` | 5 | no / MHA | 7430 | 2033 |
| `mxfp8_fp8_d128` | 6 | no / MHA | 7969 | 2188 |
| `<mode>_d128_causal_gqa` (six) | 1-6 | yes / h32kv8 | e.g. 11056 | e.g. 3428 |
| `nvfp4_bf16_d64` (+ `_causal_gqa`) | 1 | both | 6854 | 2017 |
| `nvfp4_fp8_d64` (+ `_causal_gqa`) | 2 | both | 7429 | 2171 |

The `.file` table sits at the tail of each export and its numbering is assigned PER
EXPORT: 1 `flash_fwd_sm100_fp4.py`, 2 `tile_scheduler.py`, 3 `copy_utils.py`, 4
`block_info.py` are stable, but 5/6 are `mma_sm100_desc.py`/`blackwell_helpers.py` in the
NVFP4-QK exports and swapped in the MXFP8-QK ones (the generic QK path emits
`.loc 5 1214` there), and 7/8/9 are `mask/utils/softmax` in the non-causal NVFP4+BF16
export, `softmax/mask/utils` in its causal build, and `mask/softmax/utils` in the causal
mode-4 build. Resolve the table before attributing any `.loc N`. The annotations below
therefore cite helper sites by FILE NAME and line (e.g. `blackwell_helpers.py:1828`,
`mask.py:488`, `utils.py:466-473`, `softmax.py:238`), never by a bare file number. The inline-asm blocks of `gemm_ptx_partial*`
and `tmem_ld_red_max` carry `.loc 1 442` (the `__call__` decorator line), so
their identity is settled by opcode and operand shape, not by line.

Facts that settle otherwise-ambiguous source text (unqualified counts are
`nvfp4_bf16_d128`; counting convention: instruction lines minus predicated lines):

1. **Every packed fp32 op is RN without ftz**: `fma.rn.f32x2` 128, `add.rn.f32x2`
   127, `mul.rn.f32x2` 256; zero `.rz` anywhere. The scalar sites `mul.f32` (4),
   `sub.f32` (3), `add.f32` (2) carry no rounding modifier. The only contracted
   scalar FMAs are in the fp4/mxfp8 group-quant row sum (`fma.rn.f32` 17 in mode 3).
2. **`tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max`** is issued 4x per
   softmax step in every mode (8 per non-causal export, 12 per causal one), but its
   max is *consumed* only in the non-causal steady copy (`softmax.py:238` `max.f32 hw, row_max`
   appears exactly once, in the non-causal exports). Every masked copy -- the first
   step, and BOTH steady copies of a causal kernel (`unmasked = const_expr(not
   is_causal ...)` at `:3001-3006` is False there) -- applies the r2p mask and
   recomputes the row max with the software tree: 66 `max.f32` per copy
   (`utils.py:418-432`), 198 in the causal export, and the hw-max instruction is absent
   from it. For mxfp8 PV the same rule governs the hw tile maxes feeding the group
   maxes (`:3213-3221`, non-causal steady copy only). In mode 3 the 16-element group
   maxes add 10 `max.f32` per group.
3. **One softmax body, two inlined `softmax_step` copies** (first + steady):
   `ex2.approx.ftz.f32` = 257 = 2x128 + 1 (`acc_scale`); the causal export has a
   third copy (386 = 3x128 + 2).
4. **Descriptor high words**: Q/K NVFP4 d128 `0x80004020` (SWIZZLE_64B, SBO 512 B,
   version 1), d64 `0xC0004010` (SWIZZLE_32B, SBO 256 B), e4m3 K-major `0x40004040`
   (SWIZZLE_128B, SBO 1024 B) -- the same `0x40004040` is the MN-major bf16 V word and
   the MN-major e4m3 V word at d128; at d64 the e4m3 V word is `0x80004020` (SW64B,
   a 64-column e4m3 row is one 64 B swizzle atom). K-tile steps on the low word: `+0x2` (32 B) for FP4 K=64 and e4m3 K=32
   K-major tiles, `+0x80` (2048 B = 16 rows) for MN-major V per K=16.
5. **Instruction descriptors** (`mov.b32 idesc, ...` immediates): QK
   `mxf4nvf4` `0x08201680`; PV `f16` N=128 `0x08210490`, N=64 `0x08110490`; PV
   `f8f6f4` N=128 `0x08210010`, N=64 `0x08110010`; PV `mxf4nvf4` `0x08201680`; every `mxf8f6f4` use
   `0x08A00000 | k<<29 | k<<4` for k = 0..3 (`a_sf_id`/`b_sf_id`). Two spellings carry
   it: the generic QK path of modes 5/6 (`kind::mxf8f6f4.block_scale.block32`) ALSO
   writes k into the top two bits of both SF TMEM address operands (`0x80`,
   `0x40000080`, `0x80000080`, `0xC0000080` for SFQ column 128), whereas the
   inline-asm PV path of mode 4 (`kind::mxf8f6f4.block_scale.scale_vec::1X`) keeps
   static `[tmem_scale + 0x0]` operands and carries k only in the idesc immediate.
6. **The MXFP8 QK path is the generic `cute.gemm` one**: it spells
   `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32` with a
   `mov.pred` accumulate flag per instruction and precomputed per-k A/B descriptors
   (four A descriptors and four B descriptors `+2k`), not the `scale_vec::1X` inline-asm chain.
7. **TMA shapes**: Q/K/V/O are 4-D maps (coordinates `{col, row, head, batch}`), the
   scale factors 5-D: SFQ/SFK issue `{0, 0, block, head, batch}` when a 128-row tile
   spans two 512 B chunks (SF_TILE_K == 2: nvfp4 d128) and `{0, block, 0, head, batch}`
   when it spans one (SF_TILE_K == 1: mxfp8 d128, nvfp4 d64 -- the block index moves to
   dim 1, e.g. mxfp8_fp8_d128 `{%r=0, 2*m_block, %r=0, head, batch}`); mxfp8 SFV (one
   chunk) `{0, 0, n_block, kv_head, batch}`; nvfp4 SFV `{0, 2*n_block, 0, kv_head, batch}`;
   all with
   `.L2::cache_hint` and a zero policy. BF16 V and O at d=128 need two 64-column
   issues per tile (`+0x80` column coordinate 64); e4m3 V and FP4 V one.
8. **SMEM byte map** (mode 1, from the `+imm` operands): mbarriers 0..327,
   `tmem_holding_buf` 328, `sScale` 1024, `sO` 3072, `sQ` 68608, `sV` 84992 (K
   aliases V), `sSFQ` 216064, `sSFK` 218112; each field starts on a 1024 B boundary.
9. **mbarrier counts** (`mov.b32` before `mbarrier.init`): q_full/q_empty 1/1,
   kv_full/kv_empty 1/1, P_full_O_rescaled 256, S_full 1, O_full 1,
   softmax_corr full/empty 128/128, corr_epi full/empty 128/32, tmem_dealloc 384,
   P_full_2 128, sfqk_load 128; 31 inits at kv_stage 4 (23 + 2 * kv_stage).
10. **setmaxnreg**: `dec 48` x3 (warps 14, 12, 13), `dec 24` (warp 15), `inc 192`
    (softmax), `dec 80` (correction); d=64 exports show `inc 200` / `dec 64`.

Reproduce any export with:

```bash
mkdir -p .porting/flash_attention4_fp4/ptx_lineinfo/<name>
CUTE_DSL_NO_CACHE=1 CUTE_DSL_KEEP=ptx CUTE_DSL_LINEINFO=1 \
CUTE_DSL_DUMP_DIR=$PWD/.porting/flash_attention4_fp4/ptx_lineinfo/<name> \
python .porting/flash_attention4_fp4/export_driver.py <qk> <pv> <head_dim> <causal 0|1> <h> <kv> [seq]
```

## Pipeline at a glance

16 warps, 512 threads, one CTA per work tile (persistent when non-causal). Role
selection is a flat sequence of `if warp_idx ...` blocks in source order
(`:1737-1963`); each role runs its own copy of the tile scheduler loop.

| Warps | Role-local tile program | Main publication / reuse edges |
| --- | --- | --- |
| 0..3 | softmax warpgroup 0, Q stage 0: per KV block load S from TMEM (hw row max), mask, online max, scale-subtract, exp2 + P conversion (+ P scales), store P to TMEM, row sum | waits `S_full[0]`; publishes `sfqk_load[1]`, `softmax_corr_full[0]` + `sScale`, `P_full_O_rescaled[0]`, `P_full_2[0]` (+ `sSFP[0]`); waits `softmax_corr_empty[0]` |
| 4..7 | softmax warpgroup 1, same body with runtime `stage = 1` | the `[1]` slots; publishes `sfqk_load[0]` (and pre-arrives it once) |
| 8..11 | correction warpgroup: rescale O in TMEM when a row max moved, then normalize O, convert to bf16 and stage it in `sO` | waits `softmax_corr_full[s]`; publishes `softmax_corr_empty[1-s]`, `P_full_O_rescaled[s]` (pre-arrives both once), `corr_epi_full[s]`; waits `O_full[s]`, `corr_epi_empty[s]` |
| 12 | the single MMA-issue warp; TMEM allocator; stages scale factors SMEM -> TMEM | waits `q_full`, `kv_full`, `sfqk_load[s]`, `P_full_O_rescaled[s]`, `P_full_2[s]`; commits `S_full[s]`, `kv_empty`, `q_empty`, `O_full[s]`; waits `tmem_dealloc` |
| 13 | epilogue: TMA-stores `sO[s]` | waits `corr_epi_full[s]`; publishes `corr_epi_empty[s]` |
| 14 | load: TMA Q(+SFQ) x2, then K(+SFK), V(+SFV) per KV block, newest block first | waits `q_empty`, `kv_empty`; arrives `q_full`, `kv_full` with tx bytes |
| 15 | idle; `setmaxnreg.dec 24` and falls out (`:1737-1739`) | none |

**KV is a ring of `kv_stage` slots shared by K and V**: the loader alternates
K_i, V_i into successive slots (`:2195-2214`); when K is narrower than V (FP4 K,
BF16/FP8 V) K lives inside V's slot with V's stage stride (`:1526-1536`); when
they have the same width V aliases K (`:1523-1525`).

**Blocks are visited newest-first**: `n_block = n_block_max-1 .. n_block_min`
(`:2187-2214`); the first block is the one that gets the seqlen mask.

**Two Q tiles per CTA, interleaved in the MMA warp**: for every KV block the MMA
warp issues PV(stage 0), PV(stage 1) and then QK(stage 0), QK(stage 1)
(`:2494-2637`); softmax warpgroup `s` sees only stage `s`.

## Primitive vocabulary

Structural operations do not compute values:

```python
tile(space, dtype, shape, align)   # a 1-D linear allocation; no layout attached
offset(name, ...)                   # a named scalar byte/column offset function
reg_tile(shape, dtype)              # role-local registers
tensormap(...)                      # a TMA descriptor: dims, strides, box, swizzle
tmem_cols(base, count)              # a TMEM column interval (lanes are implicit: 32x32b)
```

Copies always state their storage direction:

```python
copy_g2s(map, coords, dst_smem, bar)      # TMA load, completes on `bar`
copy_s2g(map, coords, src_smem)           # TMA store, bulk group
copy_s2t(desc, dst_cols)                  # tcgen05.cp: 512 B SF chunk -> 4 TMEM columns
copy_t2r(cols, regs)                      # tcgen05.ld
copy_t2r_max(cols, regs, red)             # tcgen05.ld.red ... .max: values + their max
copy_r2t(cols, regs)                      # tcgen05.st
store_shared(smem, byte_off, value)       # st.shared
load_shared(smem, byte_off) -> reg        # ld.shared
```

The computational vocabulary:

```python
gemm(dst_cols, a, b, idesc, sfa=None, sfb=None, accumulate)  # one tcgen05.mma
fill(dst, value)
fma(dst, a, b, c, lanes=1)          # lanes=2 is one packed two-lane op
mul(dst, a, b, lanes=1)
add(dst, a, b, lanes=1)
sub(dst, a, b)
max(dst, a, b) / max3(dst, a, b, c) # 2- and 3-source max.f32
exp2(dst, src)                      # ex2.approx.ftz
rcp(dst, src)                       # rcp.approx.ftz
ceil(dst, src)                      # cvt.rpi.f32.f32
select(dst, pred, a, b)
cast_bf16x2(dst, hi, lo)            # cvt.rn.bf16x2.f32
cast_e4m3x2(dst, hi, lo)            # cvt.rn.satfinite.e4m3x2.f32
cast_e2m1x2(dst, hi, lo)            # cvt.rn.satfinite.e2m1x2.f32 (one byte)
cast_ue8m0x2(dst, hi, lo)           # cvt.rz.satfinite.ue8m0x2.f32
shift_right_u32(dst, src, n)        # shr.u32 (saturating)
```

Small index and mask decodes are written as expressions so no primitive hides a
computation. There is deliberately no `tree_max`, `tree_sum`, `quantize`,
`softmax`, `attention`, `mask`, `TMA`, `TCGEN05`, `descriptor` or `epilogue`
primitive: every reduction tree, every group loop and every conversion pass is
written out where it occurs.

Schedule operations: `pipe`, `init`, `wait`, `arrive`, `arrive_expect_tx`,
`commit` (the matrix engine's arrive: `tcgen05.commit`), `release`, `fence`, `elect`,
`set_register_budget`, `allocate_tmem`, `relinquish_tmem_alloc_permit`,
`free_tmem`, `wait_tmem_ld`, `wait_tmem_st`, `bulk_commit_group`,
`bulk_wait_group_read`, and cursor (`slot`, `phase`) updates.

Two of them have a fixed lowering everywhere they appear, stated here once and
referenced inline as `# release: elect+commit` / `# arrive: mbarrier.arrive`:

- `release(kv_load.empty[s])` (`pipeline_kv.consumer_release`, `:2485/2579/2632/2715`)
  = `elect.sync` + `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64
  [kv_empty + 8s]`; extent: scalar (the MMA warp's slot release IS a tcgen05 commit).
- `arrive(x)` = `mbarrier.arrive.shared.b64 [mbar + off], 1` executed by every thread of
  the calling role (128 for a softmax or correction warpgroup, 32 for the epilogue warp);
  extent: scalar per thread. 28 static sites in the non-causal export, 32 in the causal one.

**THE_WAIT.** Every mbarrier wait in this kernel is the same inline-asm retry
loop, never one instruction:

```text
LAB_WAIT: mbarrier.try_wait.parity.shared.b64 P1, [addr], phase, 10000000;
          @P1 bra.uni DONE;  bra.uni LAB_WAIT;  DONE:
```

38 sites are `.shared` in a non-causal export (40 in a causal one: the third
`softmax_step` copy adds an `S_full` and a `softmax_corr.empty` wait); the 4 embedded
inside the PV MMA chains are `.shared::cta` with plain `bra`. Extent below is "one
retry loop".

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
BLK_M = BLK_N = 128;  Q_STAGE = 2;  EPI_STAGE = 2;  TMEM_COLS = 512     # :129-130,:166,:369-370,:219
qk_format  in {nvfp4, mxfp8}      # q_width 4 | 8, sf_vec 16 | 32, SF dtype E4M3 | E8M0
pv_format  in {bf16, fp8, nvfp4, mxfp8}   # v_width 16 | 8 | 4 | 8, quant_pv = pv in {nvfp4, mxfp8}
D  = head_dim in {128, 64};  DV = D                                        # :157-161
QK_K   = 64 if q_width == 4 else 32;  QK_KT = D // QK_K                  # 256-bit operand tile :344-352
PV_K   = {16: 16, 8: 32, 4: 64}[v_width];  PV_KT = BLK_N // PV_K
SF_TILE_K    = D // (4 * sf_vec)            # 512 B SF chunks per Q/K tile: 2 (nvfp4 d128), 1 (mxfp8 d128, nvfp4 d64)
SF_TILE_K_PV = BLK_N // (4 * sf_vec_pv)     # 2 (nvfp4 PV), 1 (mxfp8 PV)
regs_softmax, regs_correction, regs_other = (192, 80, 48) if D >= 96 else (200, 64, 48)   # :260-274
kv_stage   # _setup_attributes :369-424: (227 KiB - fixed) // per-stage; fp8-V cap 4 -> {4,4,13,7,3,4}@d128, {10,20}@d64
P_COLS     = BLK_N * v_width // 32          # 64 | 32 | 16 | 32
P_REP      = 16 if v_width == 16 else 8       # :2790-2807 with FA4_FP8_PV_TMEM_STORE_REP=8 pinned (also at d=64)
P_CHUNKS   = P_COLS // P_REP;  P_SPLIT = mbar_p_split(P_CHUNKS)                  # 3/4 | 3/4 | 1/2 | 3/4 (fp8 d64: 3/4 too)
PV_SPLIT   = mbar_p_split(PV_KT)            # 6/8 | 3/4 | 1/2 | 3/4                                    :1105-1125
max_offset = {nvfp4: log2(6) = 0x40257007, mxfp8: log2(448) = 0x410CEAED}.get(pv_format, 0.0)      # :2862-2878
rescale_threshold = 0.0 if quant_pv else 8.0
is_causal;  qhead_per_kvhead = H // HKV;  pack_gqa = False;  is_persistent = not is_causal
# instruction_selection: none; extent: compile-time constants only.

# Runtime ABI (the port's entry; the source passes cute tensors + TMA atoms + layouts :1275-1320)
mQ, mK        # packed bytes (b, s, h, D/2) e2m1x2 | (b, s, h, D) e4m3
mV            # (b, s_k, hkv, DV) bf16 | e4m3 (MN-major) ; (b, hkv, DV, s_k/2) e2m1x2 | (b, hkv, DV, s_k) e4m3 (K-major)
mO            # (b, s_q, h, DV) bf16
mSFQ, mSFK    # bytes [b][h][s/128][D/(4 sf_vec)][32][4][4]  (the 7-D (32,4,rest_m,4,rest_k,h,b) view :892,:913)
mSFV          # quant_pv only: [b][hkv][DV/128][s_k/(4 sf_vec_pv)][32][4][4]                       :964
softmax_scale_log2      # f32 = softmax_scale * log2(e), host-computed                        :1244-1246
tmaps: Q, K, V, O, SFQ, SFK, [SFV]   # CUtensorMaps encoded on the host (see the TensorMap table)
# instruction_selection: .param .align 64 .b8 x7 (128-byte tensormaps) and .param .f32; extent: ABI only.
#   The export's entry takes 4 tensor params, 7 tensormap params (`_param_4..7,15,17` plus
#   the SF maps), one f32 and the scheduler/shape structs; the port passes the same maps by value.

launch(grid = min(num_SMs, tiles) if is_persistent else tiles,       # ts:338-348 / ts:522-530
       block = 512, cluster = (1,1,1), smem = sizeof(SharedStorage), min_blocks_per_mp = 1)
tiles = B * H * ceil(S_Q / 256)
# instruction_selection: none; extent: launch geometry (:1320-1327).

# ===========================================================================
# Storage -- one 1-D linear allocation per SharedStorage field, in declaration order
# (:1139-1192), each 1024-aligned. Byte map for nvfp4+bf16 d128 confirmed from the export's
# `+imm` operands (fact 8).
# ===========================================================================
mbar    = tile("shared", "u64", [33 + 2*kv_stage])      # 41 slots / 328 B at kv_stage 4: q_full[2] @0, q_empty[2] @16, kv_full[kv] @32,
                                                         # kv_empty[kv] @32+8kv, P_full_O_rescaled[2], S_full[2],
                                                         # O_full[2], sc_full[2], sc_empty[2], ce_full[2], ce_empty[2],
                                                         # s0_s1[8] (never initialized), tmem_dealloc[1], P_full_2[2],
                                                         # sfqk_load[2], sfpv_load[2] (never used)          :1080-1097
tmem_holding_buf = tile("shared", "u32", [1])            # @328
sScale  = tile("shared", "f32", [2 * 128 * 2], align=1024)     # @1024; [stage*128 + row] used, +256.. unused (no LSE)
sO      = tile("shared", "bf16", [2 * 128 * DV], align=1024)   # @3072; 128-B swizzled 64-column halves (see sO_off)
sQ      = tile("shared", "u8", [2 * 128 * D * q_width // 8], align=1024)     # @68608; K-major, 64B/32B/128B swizzle
sK      = tile("shared", "u8", [1 if k_aliases_v else kv_stage * 128 * D * k_width // 8], align=1 or 1024)
sV      = tile("shared", "u8", [1 if v_aliases_k else kv_stage * 128 * DV * v_width // 8], align=1 or 1024)  # @84992
sSFQ    = tile("shared", "u8", [2 * 512 * SF_TILE_K], align=1024)            # @216064
sSFK    = tile("shared", "u8", [kv_stage * 512 * SF_TILE_K], align=1024)     # @218112
sSFP    = tile("shared", "u8", [2 * 512 * SF_TILE_K_PV if quant_pv else 1], align=1024)
sSFV    = tile("shared", "u8", [kv_stage * 512 * SF_TILE_K_PV if quant_pv else 1], align=1024)
# K/V aliasing (:1523-1539): FP4 K inside BF16/FP8 V -> K stage s starts at sV + s * v_stage_bytes;
# same width -> V stage s starts at sK + s * k_stage_bytes.
def kv_slot_off(s):   return s * max(k_stage_bytes, v_stage_bytes)
def sf_chunk_off(stage, c): return stage * 512 * SF_TILE_K + 512 * c        # chunk = 32 rows x 4 groups x 4
def sO_off(stage, r, c):    # r row 0..127, c column 0..DV-1, bf16
    return stage * 128 * DV * 2 + (c // 64) * 16384 + (r // 8) * 1024 + (r % 8) * 128 \
           + (((c % 64) // 8) ^ (r % 8)) * 16 + (c % 8) * 2
def sSFP_off(stage, lane, warp, k_outer): return stage * 512 * SF_TILE_K_PV + lane * 16 + (warp % 4) * 4 + 512 * k_outer   # :3757-3773

# TMEM: 512 columns, all names absolute (the allocation base is asserted to be 0).
S    = [tmem_cols(0, 128),   tmem_cols(128, 128)]           # :243
O    = [tmem_cols(256, DV),  tmem_cols(256 + DV, DV)]       # :244-247
P    = [tmem_cols(64, P_COLS), tmem_cols(192, P_COLS)]      # aliases the upper half of S       :251-256
SFQ  = [tmem_cols(128, 4*QK_KT), tmem_cols(0, 4*QK_KT)]     # parked on the OPPOSITE S stage      :1618-1644
SFK  = [tmem_cols(128 + 4*QK_KT, 4*QK_KT), tmem_cols(4*QK_KT, 4*QK_KT)]                          # :1650-1664
SFP  = [tmem_cols(0, 4*SF_TILE_K_PV),   tmem_cols(128, 4*SF_TILE_K_PV)]        # quant_pv; on S's own stage :1671-1682
SFV  = [tmem_cols(0 + sfp_cols, ...),   tmem_cols(128 + sfp_cols, ...)]        # sfp_cols = 4*SF_TILE_K_PV = 8 (fp4) | 4 (mxfp8) :1689-1700
# instruction_selection: none; extent: column arithmetic. Export: SFQ/SFK chunks land at columns
#   128,132 / 136,140 for stage 0 and 0,4 / 8,12 for stage 1 (nvfp4 d128); 128 / 144 (mxfp8 d128). SFP/SFV
#   chunks land at 0,4 / 8,12 (mode 3, stage 0; +128 for stage 1) and at 0 / 4 (mode 4, stage 0; 128 / 132 stage 1).

# mbarriers (counts: fact 9)
q_load        = pipe(2,        full=1 (+tx),  empty=1)       # empty arrive = tcgen05.commit          :1446-1451
kv_load       = pipe(kv_stage, full=1 (+tx),  empty=1)       # PipelineTmaUmma                        :4644-4658
P_full_O_rescaled = mbar(2, 256)      # 128 softmax + 128 correction arrivals                          :1479-1482
S_full        = mbar(2, 1)            # tcgen05.commit                                                :1484
O_full        = mbar(2, 1)            # tcgen05.commit, tail PV only                                   :1487
softmax_corr  = pipe(2, full=128, empty=128)                                                          # :1455-1459
corr_epi      = pipe(2, full=128, empty=32)                                                           # :1469-1475
tmem_dealloc  = mbar(1, 384)          # softmax 256 + correction 128                                   :1496-1506
P_full_2      = mbar(2, 128)                                                                          # :1492-1494
sfqk_load     = mbar(2, 128)                                                                          # :1510-1512
# instruction_selection: mbarrier.init.shared.b64 x(23 + 2*kv_stage) with the count in a register;
#   extent: scalar each. Initialized by warps 1,2,4,5,6,7,8 (:1443-1512) and the kv pipeline ctor
#   (:1514), then fence.mbarrier_init.release.cluster + bar.sync 0 (one CTA-wide pair).

# ===========================================================================
# Prologue (:1392-1514)
# ===========================================================================
warp = warp_id()                          # make_warp_uniform
if warp == 0:
    prefetch_descriptor(tmap) for tmap in (Q, K, V, O, SFQ, SFK, [SFV])
    # instruction_selection: prefetch.tensormap x6/7 (:1394-1407); extent: scalar each.
# barrier inits as above; then
fence("mbarrier_init.release.cluster"); barrier()   # cta-wide
# instruction_selection: fence.mbarrier_init.release.cluster; bar.sync 0; extent: scalar.

# ===========================================================================
# Role dispatch -- flat `if` blocks in source order (:1737-1963)
# ===========================================================================
if warp == 15: set_register_budget(dec=24)                              # :1737-1739 (port: 48, K warpgroup-uniform)
if warp == 14: set_register_budget(dec=regs_other); load_warp()          # :1750-1782
if warp == 12: set_register_budget(dec=regs_other); mma_warp()           # :1787-1843
if warp == 13: set_register_budget(dec=regs_other); epilogue_warp()      # :1848-1861
if warp <  8:  set_register_budget(inc=regs_softmax); softmax_warpgroup(stage = warp >> 2)   # :1866-1918
if 8 <= warp < 12: set_register_budget(dec=regs_correction); correction_warpgroup()          # :1939-1962
# instruction_selection: setmaxnreg.dec.sync.aligned.u32 48 x3, 24 x1, 80 x1; setmaxnreg.inc 192 x1
#   (fact 10); extent: scalar each. One softmax body with runtime `stage`, not two copies (fact 3).
```

```python
# ===========================================================================
# Load warp 14 (:1967-2235, :4515-4592)
# ===========================================================================
def load_warp():
    q_phase = 1;  kv = cursor(kv_stage, phase=1)        # producer state       :2001-2004
    sched = scheduler(); tile = sched.initial()
    while tile.valid:                                      # :2007
        m_block, head, batch = tile;  kv_head = head // qhead_per_kvhead          # :2014-2016
        n_min, n_max = block_range(m_block)                # bi:24-55: n_max = ceil(S_K/128), causal min(., ceil(((m_block+1)*256 + S_K - S_Q)/128))
        # ---- Q0 + SFQ0 -----------------------------------------------------------------
        load_Q(block = 2*m_block + 0, stage = 0)
        load_K(n_max - 1); kv.advance()
        load_Q(block = 2*m_block + 1, stage = 1);  q_phase ^= 1
        load_V(n_max - 1); kv.advance()
        for i in range(n_max - 1 - n_min):                 # rolled, unroll=1    :2202
            n = n_max - 2 - i
            load_K(n); kv.advance();  load_V(n); kv.advance()
        sched.advance(); tile = sched.current()

    def load_Q(block, stage):                              # :4512-4529
        wait(q_load.empty[stage], q_phase)
        # instruction_selection: THE_WAIT; extent: one retry loop.
        with elect():
            arrive_expect_tx(q_load.full[stage], tma_bytes_q)     # 9216 (nvfp4 d128: 8192 tile + 1024 SF) | 16896 (mxfp8) | 4608 (nvfp4 d64)
            # instruction_selection: elect.sync; mbarrier.arrive.expect_tx.shared.b64 with `mov.b32 9216`; extent: scalar.
        with elect():
            copy_g2s(Q, {0, block*128, head, batch}, sQ + stage*q_stage_bytes, q_load.full[stage])
            # instruction_selection: cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes
            #   .L2::cache_hint (policy 0), coords {0, block<<7, head, batch} with block = 2*m_block + stage the 128-row
            #   block index (`%r16 = m_block<<1`, `%r20 = %r16 + 1`, row = block<<7); extent: one (D*q_width/8 B, 128 rows) box.
        with elect():
            copy_g2s(SFQ, coords_sf(block, head, batch), sSFQ + stage*512*SF_TILE_K, q_load.full[stage])
            # instruction_selection: cp.async.bulk.tensor.5d...complete_tx::bytes.L2::cache_hint, coords
            #   {0, 0, block, head, batch} when SF_TILE_K == 2 (nvfp4 d128) and {0, block, 0, head, batch} when
            #   SF_TILE_K == 1 (mxfp8 d128, nvfp4 d64: mxfp8_fp8_d128 export `{%r=0, 2*m_block, %r=0, ...}`), with the
            #   same 128-row block index (2*m_block + stage); extent: one (256 x SF_TILE_K) u16 box = 512*SF_TILE_K bytes.

    def load_K(n) / load_V(n):                             # :4531-4589
        slot, phase = kv.slot, kv.phase
        wait(kv_load.empty[slot], phase)                   # instruction_selection: THE_WAIT; extent: one retry loop.
        with elect():
            arrive_expect_tx(kv_load.full[slot], tma_bytes_k | tma_bytes_v)    # K: 9216 nvfp4 d128 | 16896 mxfp8 | 4608 nvfp4 d64 ; V: 32768 bf16 d128 | 16384 bf16 d64 | 16384 e4m3 d128 | 8192 e4m3 d64 | 9216 fp4 | 16896 mxfp8
            # instruction_selection: elect.sync; mbarrier.arrive.expect_tx.shared.b64 (`.loc 1 4565`); extent: scalar.
        with elect():                                                       # :4574 (cute.copy lowers to one elect per copy)
            copy_g2s(K, {0, n*128, kv_head, batch}, sK_slot(slot), kv_load.full[slot])
            # instruction_selection: elect.sync + cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx
            #   ::bytes.L2::cache_hint (`.loc 1 4574`); extent: one K tile. V: bf16 d128 issues TWO such elected 4d
            #   copies (column coords 0 and 64, dst +0 and +16384) -- 8 4d loads per export vs 6 for e4m3/FP4 V (fact 7).
            #   MN-major V (bf16/e4m3) uses the K form {0|64, n*128, kv_head, batch}; K-major V puts the key block on
            #   dim 0: nvfp4 V {n*64 B, 0, kv_head, batch} (E3c line 702 `{%r=n<<7, 0, kv_head, batch}`, in 4-bit
            #   elements), mxfp8 V {n*128, 0, kv_head, batch} (mode-4 export `.loc 1 4574`).
        with elect():                                                       # :4589
            copy_g2s(SFK, coords_sf(n, kv_head, batch), sSFK slot, kv_load.full[slot])                     # load_K
            # instruction_selection: elect.sync + cp.async.bulk.tensor.5d...complete_tx::bytes.L2::cache_hint (`.loc 1
            #   4589`), coords {0, 0, n, kv_head, batch} when SF_TILE_K == 2 (E3c line 610) and {0, n, 0, kv_head, batch}
            #   when SF_TILE_K == 1 (mxfp8_fp8_d128 / nvfp4_fp8_d64 exports); extent: one 512*SF_TILE_K B box.
            [copy_g2s(SFV, coords_sfv(n), sSFV slot, kv_load.full[slot])]                                   # load_V, quant_pv
            # instruction_selection: same 5d copy (`.loc 1 4589`); coords differ per pv_format: nvfp4 PV (sf_vec 16,
            #   two 512 B chunks per 128 keys) {0, 2n, 0, kv_head, batch} (E3c lines 714/847 `{0, n<<1, 0, ...}`,
            #   box (256, 2, 1, 1, 1)); mxfp8 PV (sf_vec 32, one chunk) {0, 0, n, kv_head, batch} (mode-4 export,
            #   box (256, 1, 1, 1, 1)); extent: one 512*SF_TILE_K_PV B box.
```

```python
# ===========================================================================
# MMA warp 12 (:1787-1843, :2238-2722)
# ===========================================================================
def mma_warp():
    allocate_tmem(512, tmem_holding_buf); warp_sync()                       # :1791-1794
    # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [buf], 512; bar.warp.sync; extent: warp.
    q_phase = 0; kv = cursor(kv_stage, phase=0); p_phase = 0; sfqk_phase = 0     # :2352-2356, :2403
    sched = scheduler(); tile = sched.initial()
    while tile.valid:
        n_min, n_max = block_range(m_block);  n_blocks = n_max - n_min
        # ---- QK0, QK1 on the newest block -------------------------------------------- :2423-2489
        for stage in (0, 1):
            wait(q_load.full[stage], q_phase)                                   # THE_WAIT
            if stage == 0: wait(kv_load.full[kv.slot], kv.phase)                # THE_WAIT
            fence("tcgen05.after_thread_sync")                                  # :2441
            # instruction_selection: tcgen05.fence::after_thread_sync; extent: scalar (4 static).
            wait(sfqk_load[stage], sfqk_phase)                                  # :2442  THE_WAIT
            for c in range(SF_TILE_K):                                              # :2446-2450 (one cute.copy = all SFQ chunks)
                with elect(): copy_s2t(sf_desc(sSFQ, sf_chunk_off(stage, c)), SFQ[stage] + 4c)
            for c in range(SF_TILE_K):                                              # :2455-2459 (then all SFK chunks)
                with elect(): copy_s2t(sf_desc(sSFK, sf_chunk_off(kv.slot, c)), SFK[stage] + 4c)
                # instruction_selection: tcgen05.cp.cta_group::1.32x128b.warpx4 [col], desc, one per 512 B chunk, each under
                #   its own elect.sync, issue order SFQ c0, SFQ c1, SFK c0, SFK c1 (columns 128,132 then 136,140 for stage 0);
                #   the chunk descriptor is the SF base descriptor +32 (16 B units) per chunk and + stage_bytes/16 per slot;
                #   extent: 4 columns per instruction (16 per export at nvfp4 d128).
            gemm_qk(stage, kv.slot)
            with elect(): commit(S_full[stage])                                  # :2477-2478
            # instruction_selection: elect.sync; tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: scalar.
        q_phase ^= 1; sfqk_phase ^= 1
        release(kv_load.empty[kv.slot]); kv.advance()                           # release: elect+commit  :2485-2486
        # ---- steady loop over the remaining blocks ---------------------------------- :2494-2637
        accumulate = False
        for i in range(n_blocks - 1):                                            # rolled, unroll=1
            wait(kv_load.full[kv.slot], kv.phase)                                # V_i           THE_WAIT
            v_slot = kv.slot
            for stage in (0, 1):
                wait(P_full_O_rescaled[stage], p_phase)                          # :2512          THE_WAIT
                if quant_pv:
                    for c in range(SF_TILE_K_PV):                                       # :2527-2531 (all SFP chunks first)
                        with elect(): copy_s2t(sf_desc(sSFP, sf_chunk_off(stage, c)), SFP[stage] + 4c)
                    for c in range(SF_TILE_K_PV):                                       # :2535-2539 (then all SFV chunks)
                        with elect(): copy_s2t(sf_desc(sSFV, sf_chunk_off(v_slot, c)), SFV[stage] + 4c)
                        # instruction_selection: tcgen05.cp...32x128b.warpx4 x SF_TILE_K_PV each, issue order SFP c0..,
                        #   then SFV c0.. (mode 3 columns 0,4 then 8,12; mode 4: 0 then 4); extent: 4 columns each
                        #   (32 cp per export in mode 3, 24 in mode 4).
                gemm_pv(stage, v_slot, accumulate, wait_bar=(P_full_2[stage], p_phase))
                if stage == 1: release(kv_load.empty[v_slot])                    # release: elect+commit  :2578-2580
                if stage == 0: kv.advance(); wait(kv_load.full[kv.slot], kv.phase)   # K_i           :2585-2587
                fence("tcgen05.after_thread_sync"); wait(sfqk_load[stage], sfqk_phase)   # :2598-2599
                all SFQ[stage] chunks, then all SFK[stage] chunks, as above      # :2600-2613
                gemm_qk(stage, kv.slot)
                with elect(): commit(S_full[stage])                              # :2626-2627
            release(kv_load.empty[kv.slot]); kv.advance(); p_phase ^= 1; sfqk_phase ^= 1; accumulate = True   # release: elect+commit :2632
        with elect():
            for stage in (0, 1): commit(q_load.empty[stage])                     # :2641-2643
        # ---- tail PV on the last V ---------------------------------------------------- :2645-2716
        wait(kv_load.full[kv.slot], kv.phase)
        for stage in (0, 1):
            wait(P_full_O_rescaled[stage], p_phase)
            [all SFP[stage] chunks, then all SFV[stage] chunks]                  # :2663-2679
            gemm_pv(stage, kv.slot, accumulate, wait_bar=(P_full_2[stage], p_phase))
            with elect(): commit(O_full[stage])                                  # :2710-2711
        p_phase ^= 1; release(kv_load.empty[kv.slot]); kv.advance()             # release: elect+commit  :2715
        sched.advance(); tile = sched.current()
    relinquish_tmem_alloc_permit(); wait(tmem_dealloc, 0); free_tmem(load_shared(tmem_holding_buf), 512)   # :1834-1843
    # instruction_selection: tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned; THE_WAIT; ld.shared.b32;
    #   tcgen05.dealloc.cta_group::1.sync.aligned.b32; extent: warp.

def gemm_qk(stage, slot):                     # bh:1326-1408 (nvfp4) / bh:1191-1220 (mxfp8)
    a_lo = ((sQ + stage*q_stage_bytes) >> 4) | LBO_bits;  b_lo = ((sK_slot(slot)) >> 4) | LBO_bits
    for k in range(QK_KT):
        gemm(S[stage], desc(a_lo + 2k, HI_QK), desc(b_lo + 2k, HI_QK), idesc_qk(k),
             sfa = SFQ[stage] + sf_col(k), sfb = SFK[stage] + sf_col(k), accumulate = (k != 0))
    # instruction_selection, nvfp4: `@leader tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
    #   [tmem_acc], smem_desc_a, smem_desc_b, idesc, [tmem_scale_a], [tmem_scale_b], 0|1` x QK_KT inside ONE inline-asm
    #   block: `elect.sync _|leader_thread`, descriptor halves assembled with `mov.b64 {lo,hi}`, hi = 0x80004020
    #   (d128) / 0xC0004010 (d64), lo stepped `add.u32 ..., 0x2` per k-tile, idesc `mov.b32 0x08201680` then
    #   `and 0xC0000000 / shr 1 / or` and `and / shr 26 / or` of the two SF address top bits (zero here), SF column +4
    #   per k-tile (scale_vec::4X), first instruction's accumulate operand is the immediate 0 (zero_init), the rest 1;
    #   extent: QK_KT instructions per issue, 8 per export (d128), 4 (d64).
    # instruction_selection, mxfp8: `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d], a_desc, b_desc,
    #   idesc, [sfa], [sfb], p` x4, one elect.sync region each, `mov.pred p, 0|-1`, precomputed per-k descriptors
    #   (`%rd7,12,13,14`; b `+2k`), idesc 0x08A00000 | k<<29 | k<<4 and the SF address operands with the same k in
    #   their top two bits (0x80, 0x40000080, 0x80000080, 0xC0000080 for column 128); the SF column does not move
    #   (four K=32 phases share one 128-K scale column); extent: 4 instructions per issue (fact 5, 6).

def gemm_pv(stage, slot, accumulate, wait_bar):       # bh:1409-1554 (TS form) / bh:402-665 (f16, f8f6f4)
    b_lo = (sV_slot(slot) >> 4) | LBO_bits
    for k in range(PV_KT):
        if k == PV_SPLIT: wait(*wait_bar)                              # embedded second-half P wait
        gemm(O[stage], P[stage] + 8k, desc(b_lo + k*B_STEP, HI_PV), idesc_pv(k),
             [sfa = SFP[stage] + sf_col_pv(k), sfb = SFV[stage] + sf_col_pv(k)], accumulate = accumulate or k != 0)
    # instruction_selection, bf16 V: `@leader tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a + 0x8k], smem_desc_b,
    #   idesc, p|1` x8 in one asm block, idesc 0x08210490 (0x08110490 at DV=64), hi 0x40004040, B lo stepped
    #   +0x80 per K=16 (2048 B = 16 MN-major rows), `setp.ne.b32 p, accumulate, 0` on k=0 only, THE_WAIT
    #   (`.shared::cta`) between k=5 and k=6 (PV_SPLIT 6/8); extent: 8 per issue, 32 per export.
    # instruction_selection, fp8 V: kind::f8f6f4, idesc 0x08210010 (DV=128) / 0x08110010 (DV=64), K=32, 4 per issue,
    #   split 3/4 (THE_WAIT `.shared::cta` before k=3); the MN-major e4m3 V descriptor is hi 0x40004040 (SW128B) with
    #   B step +0x100 (32 rows x 128 B) at DV=128, but hi 0x80004020 (SW64B: a 64-column e4m3 row is 64 B) with B step
    #   +0x80 (32 rows x 64 B) at DV=64.
    # instruction_selection, nvfp4 V: kind::mxf4nvf4.block_scale.scale_vec::4X [acc], [tmem_a + 0x8k], desc_b, idesc
    #   0x08201680, [sfp + 4k], [sfv + 4k], p|1; K=64, 2 per issue, split 1/2; V K-major hi 0x80004020, B lo +0x2.
    # instruction_selection, mxfp8 V (mode 4, inline asm bh:1409-1554): `@leader_thread tcgen05.mma.cta_group::1
    #   .kind::mxf8f6f4.block_scale.scale_vec::1X [acc], [tmem_a + 0x8k], desc_b, idesc, [sfp + 0x0], [sfv + 0x0], p|1`
    #   with `mov.b32 idesc, 0x08A00000 | k<<29 | k<<4` then `or.b32 idesc, idesc, sf_dyn` (the always-zero top bits of
    #   the SF base), K=32, 4 per issue, THE_WAIT (`.shared::cta`) before k=3 (split 3/4); V K-major e4m3 hi 0x40004040,
    #   B lo +0x2. Unlike the generic QK path the SF address operands are static `+0x0`.
```

```python
# ===========================================================================
# Softmax warpgroups 0..7 (:2727-3063, :3559-3883); thread = one S row (`tidx & 127`)
# ===========================================================================
def softmax_warpgroup(stage):
    si_phase = 0; corr_phase = 1                                          # :2810-2811
    if stage == 1: arrive(sfqk_load[0])                                   # :2815-2817 (one-time pre-arrive)
    # instruction_selection: mbarrier.arrive.shared.b64 [mbar+296]; extent: scalar.
    sched = scheduler(); tile = sched.initial()
    while tile.valid:
        n_min, n_max = block_range(m_block)
        row_max = -inf; row_sum = 0                                        # softmax.reset :2887
        wait(softmax_corr.empty[stage], corr_phase); corr_phase ^= 1       # :2924-2927  THE_WAIT
        step(n_max - 1, first=True, mask_seqlen=True, mask_causal=is_causal)   # :2964-2974 (mask_fn(mask_seqlen=True))
        if is_causal:                                                          # :2976-2993, bi:105-121
            for n in range(n_max - 2, n_min_causal - 1, -1): step(n, mask_causal=True)
            # the `:3007` loop below runs with the DEFAULT mask_fn (mask_causal=True, mask_seqlen=False) and
            # `mask_is_noop=False`: identical body, only the bounds differ -- the export has three masked copies.
            for n in range(min(n_max - 1, n_min_causal) - 1, n_min - 1, -1): step(n, mask_causal=True)
        else:
            for n in range(n_max - 2, n_min - 1, -1): step(n, unmasked=True)   # :3007-3014, mask_is_noop=True
        store_shared(sScale, (tidx + stage*128) * 4, row_sum)              # :3034
        # instruction_selection: st.shared.b32 [sScale + stage*512 + tidx*4]; extent: scalar.
        arrive(softmax_corr.full[stage])                                   # :3039  arrive: mbarrier.arrive [mbar+144+8s]
        sched.advance(); tile = sched.current()
    arrive(tmem_dealloc)                                                   # :1918  arrive: mbarrier.arrive [mbar+272]

def step(n, first=False, mask_seqlen=False, mask_causal=False, unmasked=False):
    wait(S_full[stage], si_phase)                                          # :3623  THE_WAIT
    s = reg_tile([128], "f32"); tile_max = reg_tile([4], "f32")
    for j in range(4): copy_t2r_max(S[stage] + 32j, s[32j:32j+32], tile_max[j])      # bh:1806-1829
    # instruction_selection: tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max {32 regs}, max, [addr] x4 with addr
    #   = base | 32j (`or.b32 ..., 32/64/96`); then tcgen05.wait::ld.sync.aligned; extent: 32 values + 1 max each.
    hw_max = max(max(max(tile_max[0], tile_max[1]), tile_max[2]), tile_max[3])       # bh:1828
    # instruction_selection: max.f32 x3 (2-source, blackwell_helpers.py:1828) emitted ONLY in the unmasked copy (the
    #   non-causal steady step); in the first copy and in every causal copy the fold is dead and eliminated -- the 33rd
    #   ld.red register is simply unused there (3 static in a non-causal export, 0 in a causal one); extent: scalar.
    wait_tmem_ld(); arrive(sfqk_load[1 - stage])                            # :3646-3648
    # instruction_selection: mbarrier.arrive.shared.b64 [mbar + 296 + (stage^1)*8]; extent: scalar.
    if mask_seqlen or mask_causal:                                          # mk:373-500 (r2p)
        seqlen_limit = S_K - max(n, 0) * 128                                    # mk:401-403
        causal_limit = (tidx + (2*m_block + stage)*128) + (S_K - n*128 - S_Q) + 1   # mk:481-486 (row + causal_row_offset + 1)
        limit = seqlen_limit            if mask_seqlen and not mask_causal  else \
                causal_limit            if mask_causal and not mask_seqlen  else \
                min(causal_limit, seqlen_limit)                                 # both: first causal step only (mk:487-488)
        # instruction_selection: `min.s32` exists only in the first causal copy (`mask.py:488`, x1 in the causal export);
        #   the diagonal/remaining causal copies compute `row + causal_row_offset + 1` with no min (`mask.py:486`).
        for c in range(4):
            bits = shift_right_u32(0xFFFFFFFF, max((c+1)*32 - limit, 0))
            for i in range(32): s[32c+i] = select(bits & (1<<i), s[32c+i], -inf)
        # instruction_selection: max.s32/shl/sub/max.s32 for the limit, `shr.u32` (inline asm) per chunk, then per
        #   element `and.b32 / setp.(ne|eq).b32 / selp.f32 ..., 0fFF800000`; extent: 128 selects per masked step.
    if unmasked:  new_max = max(hw_max, row_max)                             # sm:229-239
    else:                                                                    # sm:242-259, ut:410-432
        l0 = max(s[0], s[1]) if first else max3(row_max, s[0], s[1])         # ut:415-419: the previous row max folds
        l = [l0, max(s[2],s[3]), max(s[4],s[5]), max(s[6],s[7])]              #   into the FIRST partial, never at the tail
        for i in range(8, 128, 8): l[j] = max3(l[j], s[i+2j], s[i+2j+1])  for j in 0..3     # ut:426-430
        new_max = max3(max(l[0], l[1]), l[2], l[3])                           # ut:431-432
        # instruction_selection: max.f32 x66 per copy -- first copy 5 two-source (utils.py:418, 422-424, 431) + 61
        #   three-source (427-430, 432); non-first masked copies 4 two-source (422-424, 431) + 62 three-source
        #   (416, 427-430, 432); extent: 128 values.
    safe = select(new_max != -inf, new_max, 0.0)                            # sm:217/245
    # instruction_selection: setp.neu.f32 ..., 0fFF800000; selp.f32; extent: scalar.
    if not first:
        acc_scale_ = softmax_scale_log2 * (row_max - safe);  acc_scale = exp2(acc_scale_)      # sm:218-219
        # instruction_selection: sub.f32; mul.f32 (no .rn); ex2.approx.ftz.f32; extent: scalar.
        if rescale_threshold == 8.0 and acc_scale_ >= -8.0: new_max, safe, acc_scale = row_max, row_max, 1.0   # sm:220-224
        # instruction_selection: setp.ge.f32 ..., 0fC1000000; selp.f32 x3; extent: scalar (absent in quant_pv modes).
        store_shared(sScale, (tidx + stage*128)*4, acc_scale)                # :3683
        # instruction_selection: st.shared.b32 [sScale + stage*512 + tidx*4] (`.loc 1 3683`); extent: scalar,
        #   non-first copies only (1 per export, 2 in causal builds).
    row_max = new_max
    arrive(softmax_corr.full[stage])                                        # :3685  arrive: mbarrier.arrive [mbar+144+8s]
    bias = max_offset - safe * softmax_scale_log2                            # sm:271-285
    for i in range(0, 128, 2): (s[i], s[i+1]) = fma((s[i], s[i+1]), scale_log2, bias, lanes=2)
    # instruction_selection: mul.f32; sub.f32 (from `mov.b32` of the max_offset constant); fma.rn.f32x2 x64; extent: 128 values.
    p_words = reg_tile([P_COLS], "u32"); fill(p_words, 0)                   # :3699-3700 (fp8_pv_zero_fill_regs)
    # instruction_selection: none -- the zero fill is folded into the full overwrite below (no `mov` survives).
    # ---- P per mode ----------------------------------------------------------------------------------
    if pv_format == bf16:                                                    # sm:303-346 apply_exp2_convert
        for frag in range(4):
            for i in range(32): s[32frag+i] = exp2(s[32frag+i])
            for i in range(16): p_words[16frag+i] = cast_bf16x2(s[32frag+2i+1], s[32frag+2i])
        # instruction_selection: ex2.approx.ftz.f32 x128; cvt.rn.bf16x2.f32 x64 (hi = odd element); extent: 128 values.
    elif pv_format == fp8:                                                   # :3809-3821, bh:1596-1620
        for i in range(128): s[i] = exp2(s[i])
        for i in range(32): p_words[i] = {cast_e4m3x2(s[4i+1], s[4i]) | cast_e4m3x2(s[4i+3], s[4i+2]) << 16}
        # instruction_selection: ex2.approx.ftz.f32 x128; cvt.rn.satfinite.e4m3x2.f32 x64 + mov.b32 {lo,hi} x32; extent: 128 values.
    elif pv_format == nvfp4:                                                 # :3171-3295 _fused_log2_group_quant
        acc = 0
        for g in range(8):                                                   # 16-element groups, unrolled
            m = fmax-tree over s[16g:16g+16]                                  # max.f32 x5 + max3 x5 (10 per group)
            b = max(m, -100.0) + (-log2 6);  nb = -b                          # :3224
            for i in range(0, 16, 2): (s[16g+i], s[16g+i+1]) = fma((..), (1.0, 1.0), (nb, nb), lanes=2)   # :3231-3236
            for i in range(16): s[16g+i] = exp2(s[16g+i]);  sf[g] = exp2(b)  # :3237-3242
            gs = packed add-tree over the 16 exps;  acc = fma(sf[g], gs, acc)  # :3245 (first: fma(sf0, gs0, +0.0))
            p_words[2g], p_words[2g+1] = 8 x cast_e2m1x2 bytes                 # bh:1624-1666
        row_sum = acc if first else fma(row_sum, acc_scale, acc)              # :3272-3275
        sf_words = [cast_e4m3x2(sf[1], sf[0]) | cast_e4m3x2(sf[3], sf[2]) << 16, ... sf[4..7]]    # bh:1596
        store_shared(sSFP, sSFP_off(stage, lane, warp, 0), sf_words[0]); store_shared(sSFP, sSFP_off(stage, lane, warp, 1), sf_words[1])   # :3757-3773
        # instruction_selection, PER INLINED softmax_step COPY (two copies per non-causal export): group maxes
        #   max.f32 x80 (10 per group, 2/3-source) + `max.f32 m, m, 0fC2C80000` x8 + `add.f32 b, m, 0fC0257007` x8
        #   (`.loc 1 3224`), neg.f32 x8 (`.loc 1 3235`), fma.rn.f32x2 x64, ex2.approx.ftz.f32 x136 (128 + 8 sf),
        #   group sums add.rn.f32x2 x56 + add.f32 x8 (`utils.py:466-473`), fma.rn.f32 x8 (`.loc 1 3245`) + x1 for row_sum
        #   (`.loc 1 3275`, non-first copy), cvt.rn.satfinite.e2m1x2.f32 x64 (`.loc 1 447-450`), cvt.rn.satfinite.e4m3x2
        #   .f32 x4, st.shared.b32 x2 at [sSFP + stage*1024 + lane*16 + (warp%4)*4 (+512)] (`.loc 1 3773`);
        #   extent: 128 values / 8 groups. Export totals: ex2 273 = 2x136 + 1, neg.f32 16, fma.rn.f32 17.
    elif pv_format == mxfp8:                                                 # same helper with 32-element groups
        for g in range(4):
            m = (tile_max[g] * scale_log2 + hw_bias) if unmasked else fmax-tree over s[32g:32g+32]   # :3213-3223
            b = ceil(max(m, -100.0) + (-log2 448))                            # :3224-3228, bh:1833
            fma / exp2 as above over 32 values; sf[g] = exp2(b); gs; acc = fma(sf[g], gs, acc)
            p_words[8g:8g+8] = cast_e4m3x2 pairs                              # :3263-3269
        row_sum as above;  sf_word = cast_ue8m0x2(sf[1], sf[0]) | cast_ue8m0x2(sf[3], sf[2]) << 16   # bh:1849-1873
        store_shared(sSFP, sSFP_off(stage, lane, warp, 0), sf_word)
        # instruction_selection, PER COPY: st.shared.b32 x1 at [sSFP + stage*512 + lane*16 + (warp%4)*4] (`.loc 1
        #   3773`; 2 per mode-4 export = 1 per inlined copy), cvt.rpi.f32.f32 x4 (`.loc 1 442`), cvt.rn.satfinite.e4m3x2.f32 x64,
        #   cvt.rz.satfinite.ue8m0x2.f32 x2, ex2 x132 (128 + 4 sf), fma.rn.f32 x4 (`.loc 1 3245` group chain) + x1
        #   row_sum (non-first) + x4 (`.loc 1 3221`, hw group maxes -- NON-causal steady copy only; the first copy and
        #   every causal copy use the 32-element software fmax tree instead); extent: 128 values / 4 groups.
        #   Export totals (mode 4, non-causal): fma.rn.f32 13 = 8 + 4 + 1, cvt.rpi 8, ue8m0x2 4.
    # ---- P to TMEM with the split handoff ---------------------------------------------------- :3861-3870
    for c in range(P_SPLIT):     copy_r2t(P[stage] + c*P_REP, p_words[c*P_REP : (c+1)*P_REP])
    wait_tmem_st(); arrive(P_full_O_rescaled[stage])
    for c in range(P_SPLIT, P_CHUNKS): copy_r2t(P[stage] + c*P_REP, p_words[...])
    wait_tmem_st(); arrive(P_full_2[stage])
    # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32 x4 (bf16: 3 + 1) | .x8 x4 (fp8/mxfp8, d128 and
    #   d64 alike: 3 + 1) | .x8 x2 (nvfp4: 1 + 1); each group followed by tcgen05.wait::st.sync.aligned and
    #   mbarrier.arrive.shared.b64 (all 128 threads) on [mbar+96+8*stage] then [mbar+280+8*stage]; extent: P_REP
    #   columns per instruction. The fp8 d64 export has zero `.x4`: `.loc 1 3862` .x8 x3, `.loc 1 3867` .x8 x1 per copy.
    wait(softmax_corr.empty[stage], corr_phase)                              # :3874  THE_WAIT
    if not quant_pv:                                                         # sm:261-266, ut:455-473
        init = row_sum * acc_scale (not first)
        l = [(s0,s1)+(init,0) , (s2,s3), (s4,s5), (s6,s7)]; l[j] += (s[i+2j], s[i+2j+1]) for i in 8..120 step 8
        row_sum = (l0+l1+l2+l3).x + .y
        # instruction_selection: mul.f32; add.rn.f32x2 x63 (+1); add.f32; extent: 128 values.
    si_phase ^= 1; corr_phase ^= 1
```

```python
# ===========================================================================
# Correction warpgroup 8..11 (:3885-4360); thread = one O row
# ===========================================================================
def correction_warpgroup():
    arrive(P_full_O_rescaled[0]); arrive(P_full_O_rescaled[1])               # :3926-3927 arrive: mbarrier.arrive [mbar+96],[+104]
    sc_phase = 0; o_phase = 0; ce_phase = 1                                   # :3929-3931
    sched = scheduler(); tile = sched.initial()
    while tile.valid:
        n_min, n_max = block_range(m_block)
        wait(softmax_corr.full[0], sc_phase); arrive(softmax_corr.empty[0])   # :3969-3972 THE_WAIT; arrive: mbarrier.arrive [mbar+160]
        wait(softmax_corr.full[1], sc_phase); sc_phase ^= 1
        for i in range(n_max - n_min - 1):                                    # rolled, unroll=1   :3979
            for stage in (0, 1):
                wait(softmax_corr.full[stage], sc_phase)                      # THE_WAIT
                scale = load_shared(sScale, (tidx + stage*128)*4)             # :3989
                # instruction_selection: ld.shared.b32 [sScale + stage*512 + tidx*4]; extent: scalar.
                if vote_any(scale < 1.0):                                     # :3990
                    # instruction_selection: setp.lt.f32 ..., 0f3F800000; vote.sync.ballot.b32; setp.eq.b32; branch; extent: warp.
                    for t in range(DV // 16):                                 # :4196-4225
                        o = reg_tile([16], "f32"); copy_t2r(O[stage] + 16t, o)
                        for j in range(0, 16, 2): (o[j], o[j+1]) = mul((o[j], o[j+1]), (scale, scale), lanes=2)
                        copy_r2t(O[stage] + 16t, o)
                    wait_tmem_st()
                    # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32 x8, mul.rn.f32x2 x8 each,
                    #   tcgen05.st.sync.aligned.32x32b.x16.b32 x8, then one tcgen05.wait::st.sync.aligned; extent: DV columns.
                arrive(P_full_O_rescaled[stage]); arrive(softmax_corr.empty[1 - stage])     # :3998-4001 arrive: mbarrier.arrive
            sc_phase ^= 1
        arrive(softmax_corr.empty[1])                                          # :4004  arrive: mbarrier.arrive [mbar+168]
        for stage in (0, 1):                                                   # :4021-4076
            wait(softmax_corr.full[stage], sc_phase)
            row_sum = load_shared(sScale, (tidx + stage*128)*4)               # :4029
            # instruction_selection: ld.shared.b32 [sScale + stage*512 + tidx*4] (`.loc 1 4029`); extent: scalar.
            arrive(softmax_corr.empty[stage])                                  # :4034  arrive: mbarrier.arrive
            d = select(row_sum != 0.0, row_sum, 1.0);  scale = rcp(d)         # :4047-4051 (v_descale = 1.0 folded)
            # instruction_selection: setp.ne.f32 (one compare covers the `== 0 or NaN` source test); selp.f32;
            #   rcp.approx.ftz.f32; extent: scalar.
            wait(O_full[stage], o_phase); wait(corr_epi.empty[stage], ce_phase)          # :4052-4058
            for t in range(DV // 16):                                          # :4293-4305
                o = reg_tile([16], "f32"); copy_t2r(O[stage] + 16t, o)
                for j in range(0, 16, 2): (o[j], o[j+1]) = mul((o[j], o[j+1]), (scale, scale), lanes=2)
                h = [cast_bf16x2(o[2j+1], o[2j]) for j in range(8)]
                store_shared(sO, sO_off(stage, tidx, 16t),     h[0:4])
                store_shared(sO, sO_off(stage, tidx, 16t + 8), h[4:8])
            # instruction_selection: tcgen05.ld...x16 x8; mul.rn.f32x2 x8 each; cvt.rn.bf16x2.f32 x8 each;
            #   st.shared.v4.b32 x2 each (32 per export at d128, 16 at d64); extent: 16 columns per iteration.
            fence("proxy.async.shared::cta")                                    # :4307-4310
            # instruction_selection: fence.proxy.async.shared::cta; extent: scalar.
            arrive(corr_epi.full[stage]); arrive(P_full_O_rescaled[stage])      # :4073-4076 arrive: mbarrier.arrive [mbar+176+8s], [mbar+96+8s]
        o_phase ^= 1; sc_phase ^= 1; ce_phase ^= 1
        sched.advance(); tile = sched.current()
    arrive(tmem_dealloc)                                                       # :1962  arrive: mbarrier.arrive [mbar+272]

# ===========================================================================
# Epilogue warp 13 (:4405-4510)
# ===========================================================================
def epilogue_warp():
    phase = 0; sched = scheduler(); tile = sched.initial()
    while tile.valid:
        for stage in (0, 1):
            wait(corr_epi.full[stage], phase)                                   # THE_WAIT
            for half in range(DV // 64):
                copy_s2g(O, {64*half, (2*m_block + stage)*128, head, batch}, sO + stage*o_stage_bytes + half*16384)
            bulk_commit_group()
            # instruction_selection: cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint x(DV/64)
            #   with column coordinate 0 / 64 and row coordinate (m_block<<8) | (stage*128), issued by ALL 32 lanes of
            #   warp 13 (no elect.sync: `:4433-4446` is a plain cute.copy from the whole warp); cp.async.bulk.commit_group;
            #   extent: one (64 cols x 128 rows) bf16 box per copy.
        for stage in (0, 1):
            bulk_wait_group_read(1 - stage); arrive(corr_epi.empty[stage])       # :4447-4448
            # instruction_selection: cp.async.bulk.wait_group.read 1 then 0; mbarrier.arrive.shared.b64 [mbar+192+8s]
            #   by all 32 lanes (the barrier is initialized with count 32); extent: scalar.
        phase ^= 1
        sched.advance(); tile = sched.current()

# ===========================================================================
# Schedulers (:1038-1078, ts:287-390, ts:393-634)
# ===========================================================================
# non-causal: StaticPersistentTileScheduler: linear = cta_id; block = linear % num_m; head = linear // num_m % H;
#   batch = linear // (num_m * H); advance += gridDim.x; valid = linear < tiles.
# causal: SingleTileLPTScheduler: one tile per CTA (advance -> invalid). l2_swizzle = 1 if 50 MiB < size_one_head
#   else 2^floor(log2(50 MiB // size_one_head)) with size_one_head = S_K * (D + DV) * max(width//8, 1)  (:1071);
#   bidhb, l2_mod = divmod(linear, l2_swizzle * num_m); block, residual = divmod(l2_mod, l2_swizzle | rem);
#   batch, head = divmod(bidhb * l2_swizzle + residual, H); block = num_m - 1 - block (LPT).
# instruction_selection: mul.hi.u32 / mul.lo.s32 / shr (FastDivmod) and cvt.u32.u16; extent: scalar bookkeeping.
```

## Logical GEMMs

| name | (M, N, K) per instruction | A | B | D | K phases per issue | issues per KV block | owner |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| QK nvfp4 | (128, 128, 64) | sQ stage, E2M1 K-major SW64B (SW32B d64), scale SFQ tmem | sK slot, same | S[stage] f32 | D/64 | 2 (one per stage) | warp 12 |
| QK mxfp8 | (128, 128, 32) | sQ stage, E4M3 K-major SW128B, scale SFQ, sf_id = k | sK slot | S[stage] | 4 | 2 | warp 12 |
| PV bf16 | (128, DV, 16) | P[stage] TMEM bf16 (+8 cols per k) | sV slot, bf16 MN-major SW128B | O[stage] f32 | 8 | 2 | warp 12 |
| PV fp8 | (128, DV, 32) | P TMEM e4m3 | sV e4m3 MN-major SW128B (SW64B d64) | O[stage] | 4 | 2 | warp 12 |
| PV nvfp4 | (128, 128, 64) | P TMEM e2m1 + SFP tmem | sV E2M1 K-major SW64B + SFV tmem | O[stage] | 2 | 2 | warp 12 |
| PV mxfp8 | (128, 128, 32) | P TMEM e4m3 + SFP (E8M0) at S[stage]+0..3, sf_id = k in idesc only | sV e4m3 K-major SW128B + SFV at S[stage]+4..7 | O[stage] | 4 | 2 | warp 12 |

## TensorMap fields

All maps are encoded on the host with 128-byte L2 promotion, no interleave, OOB
zero fill; dims and coordinates are listed innermost first, matching the export's operand
order `{col, row, head, batch}` for the 4-D maps (the head dimension is OUTSIDE the
sequence dimension: the 128-row box moves along dim 1, dim 2 selects the head).

| map | dtype | dims (inner -> outer) | box | swizzle |
| --- | --- | --- | --- | --- |
| Q, K (nvfp4) | u8 | (D/2, S, H or HKV, B) | (D/2, 128, 1, 1) | 64B (d128) / 32B (d64) |
| Q, K (mxfp8) | u8 | (D, S, H or HKV, B) | (D, 128, 1, 1) | 128B |
| V bf16 (MN-major) | bf16 | (DV, S_K, HKV, B) | (64, 128, 1, 1) x DV/64 issues | 128B |
| V e4m3 (MN-major) | u8 | (DV, S_K, HKV, B) | (DV, 128, 1, 1) | 128B (d128) / 64B (d64) |
| V nvfp4 (K-major) | u8 | (S_K/2, DV, HKV, B) | (64, 128, 1, 1); coords {n*64 B (the export issues n*128 in 4-bit elements), 0, kv_head, batch} | 64B |
| V mxfp8 (K-major) | u8 | (S_K, DV, HKV, B) | (128, 128, 1, 1); coords {n*128, 0, kv_head, batch} | 128B |
| SFQ / SFK (SF_TILE_K == 2: nvfp4 d128, modes 1-4) | u16 | (256, 2, S/128, H or HKV, B) | (256, 2, 1, 1, 1); coords {0, 0, block, head, batch} (nvfp4_bf16_d128 export `{%r, %r, block, head, batch}` with `%r = 0`) | none |
| SFQ / SFK (SF_TILE_K == 1: mxfp8 d128 modes 5-6, nvfp4 d64 modes 1-2) | u16 | (256, S/128, 1, H or HKV, B) | (256, 1, 1, 1, 1); coords {0, block, 0, head, batch} (mxfp8_fp8_d128 / nvfp4_fp8_d64 exports `{0, block, 0, head, batch}`; the single-chunk dimension collapses to dim 2) | none |
| SFV (nvfp4 PV, sf_vec 16) | u16 | (256, S_K/64, DV/128, HKV, B) | (256, 2, 1, 1, 1); coords {0, 2n, 0, kv_head, batch} (E3c `.loc 1 4589`: `{0, %r=n<<1, 0, kv_head, batch}`) | none |
| SFV (mxfp8 PV, sf_vec 32) | u16 | (256, DV/128 (=1), S_K/128, HKV, B) | (256, 1, 1, 1, 1); coords {0, 0, n, kv_head, batch} (mode-4 export `.loc 1 4589`) | none |
| O | bf16 | (DV, S_Q, H, B) | (64, 128, 1, 1) x DV/64 issues | 128B |

## Storage alias lifetimes

| storage | first meaning | second meaning | rule |
| --- | --- | --- | --- |
| KV ring slot | K_i (FP4 inside a wider V slot, or the slot itself) | V_i in the next slot | K and V alternate slots; K is released after QK of both stages, V after PV of stage 1 |
| S[stage] columns 64.. | S accumulator (read by ld.red) | P for the PV of the same block | P is written only after the whole S row is in registers |
| S[1-stage] columns 0..8*QK_KT | SFQ/SFK for stage's QK | S[1-stage] accumulator | the MMA warp copies the scales right before its QK and the other stage's QK overwrites them afterwards; ordered by `sfqk_load` |
| S[stage] columns 0.. (quant_pv) | SFP/SFV for the PV | S accumulator of the next QK | ordered by P_full / S_full |

## TIRx module and benchmark contract

- `get_kernel(qk_format, pv_format, batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal)`
  traces one `@K.kernel(warps=16, arch="sm_103a", min_blocks_per_sm=1, grid=...)`; `hardware_num_sms()` sizes
  the persistent grid.
- Inputs are quantized with flashinfer (`nvfp4_quantize` / `mxfp8_quantize`, `SfLayout.layout_128x4`) exactly as the
  fork's `bench_fp4.py`; the scale bytes are the contiguous `[b][h][s/128][d/(4 sf_vec)][32][4][4]` storage.
- `run_test` compares bit-for-bit against the fork kernel on identical bytes and against fp64 attention on the
  dequantized inputs (library-anchored); `run_gpu` benchmarks with proton against the fork's compiled callable.

## Instruction-selection summary

| pattern | PTX family | per-export count (nvfp4_bf16_d128) |
| --- | --- | ---: |
| TMA loads Q/K/V | `cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint` | 8 |
| TMA loads SFQ/SFK/(SFV) | same, `.5d` | 4 (6 with SFV) |
| TMA store O | `cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint` + `commit_group` + `wait_group.read` | 4 + 2 + 2 |
| scale SMEM -> TMEM | `tcgen05.cp.cta_group::1.32x128b.warpx4` | 16 (32 mode 3, 24 mode 4, 8 mxfp8 QK) |
| QK | `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X` / `kind::mxf8f6f4.block_scale.block32` | 8 / 16 |
| PV | `kind::f16` / `kind::f8f6f4` / `kind::mxf4nvf4...4X` / `kind::mxf8f6f4.block_scale.scale_vec::1X` | 32 / 16 / 8 / 16 |
| S load + row max | `tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max`, `tcgen05.wait::ld` | 8 (12 causal), 2 |
| P store | `tcgen05.st.sync.aligned.32x32b.x16/x8/x4.b32`, `tcgen05.wait::st` | 8 (+16 rescale), 6 |
| O rescale / epilogue reads | `tcgen05.ld.sync.aligned.32x32b.x16.b32` | 32 |
| completion signals | `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64` | 12 |
| exp | `ex2.approx.ftz.f32` | 257 |
| packed fp32 | `fma.rn.f32x2` / `add.rn.f32x2` / `mul.rn.f32x2` | 128 / 127 / 256 |
| conversions | `cvt.rn.bf16x2.f32`; `cvt.rn.satfinite.e4m3x2.f32`; `cvt.rn.satfinite.e2m1x2.f32`; `cvt.rz.satfinite.ue8m0x2.f32`; `cvt.rpi.f32.f32` | 256; 0; 0; 0; 0 |
| max / select / mask | `max.f32` (2/3-source), `selp.f32`, `shr.u32` | 70, 135, 66 |
| waits | `mbarrier.try_wait.parity.shared.b64` (+ 4 `.shared::cta` inside PV) | 38 + 4 |
| arrives | `mbarrier.arrive.shared.b64`, `mbarrier.arrive.expect_tx.shared.b64` | 28, 6 |
| register budgets | `setmaxnreg.dec/inc.sync.aligned.u32` | 5 / 1 |

The counts are audit evidence, not operands or issue-count hints in the sketch.
