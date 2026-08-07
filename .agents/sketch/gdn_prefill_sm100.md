# TIRx FP16 GDN prefill SM100: coarse WASP pipeline sketch

This file is a non-executable design sketch.  It is not a Python module, a new
IR, a builder API, or a mathematical reference implementation.  Its purpose is
to show the TIRx kernel as:

- an explicit runtime ABI and launch;
- explicit GMEM, SMEM, TMEM, and register tiles, including byte/column offsets;
- the persistent scheduler and 12-warp kernel role control flow;
- primitive directional copies and primitive computation inside every reachable
  helper;
- explicit stage/index/phase changes and publication/reuse edges;
- hardware instruction selection derived only after placement, shape, layout,
  and schedule have been stated.

The implementation represented by this sketch is maintained in
[`tirx_kernels/attention/gdn_prefill_sm100.py`](../../tirx_kernels/attention/gdn_prefill_sm100.py).
That module is the source of truth.

The specialization is fixed to FP16 Q/K/V/O and inverse storage, FP32
gate/beta/accumulator/state, initial and final state enabled, checkpoint and
state-index disabled, and persistent SM100 execution.  Only these static head
pairs are in scope: `(2,8), (4,16), (8,32), (16,64), (16,32), (16,48),
(16,16), (32,32)`.

## Pipeline at a glance

| Warps | Role-local tile program | Main publication/reuse edges |
| --- | --- | --- |
| 0..3 | form both 64x64 causal transfer tiles, consume KK0/KK1, seed and hierarchically invert two aliased Ainv tiles, then consume/store QK0/QK1 | `cg0_acc -> ainv_ready/qk_ready` |
| 4..7 | load/recur/store the 128x128 state; form V-KS, scale QS, publish NV and decay-V, and stage O | `kv_acc/state_inp/vks/nv/decay_v/o_store` |
| 8 | issue KK0, KK1, QK0, QK1 | `load_k/load_q -> cg0_acc` |
| 9 | copy/update Q/K/V TensorMaps and load every valid or pair-padding chunk | `load_k/load_q/load_v` |
| 10 | issue KS, QS, NV, QKV, KV in that order | all CG1 operand/result edges |
| 11 | copy/update O TensorMap, run four-chunk gate/beta lookahead, and store each O tile | `load_gate/load_beta/o_store` |

The kernel has two adjacent role chains: CG0 is an independent `if`; CG1,
warp 8, warp 10, and warp 9 are one `if/elif` chain; warp 11 is a final
independent `if`.  That exact shape remains visible below.

## Primitive vocabulary

Structural operations do not compute values:

```python
tile(...)             # declare storage, dtype, logical shape, and placement
view(...)             # change logical indexing without moving values
alias(...)            # declare exact storage aliasing and non-overlap lifetime
slice(...)            # select a logical interval
transpose(...)        # transpose view only
reg_tile(...)         # declare a role-local register tile
arange(...)           # construct logical row/column indices in registers
```

Copies always state their storage direction:

```python
copy_p2g(src, dst)                              # TensorMap parameter -> global slot
copy_g2s(src, dst, mask=None, completion=None)  # global -> shared
copy_s2g(src, dst, mask=None)                   # shared -> global
copy_g2r(src, dst, mask=None, cache=None)       # global -> register
copy_r2g(src, dst, mask=None, cache=None)       # register -> global
copy_s2r(src, dst)                              # shared -> register
copy_r2s(src, dst)                              # register -> shared
copy_r2r(src, dst)                              # register -> register
copy_t2r(src, dst)                              # tensor memory -> register
copy_r2t(src, dst)                              # register -> tensor memory
```

The complete computational vocabulary used below is:

```python
fill(dst, value)
cast(dst, src)
add(dst, lhs, rhs, lanes=1, rounding=None, ftz=None)
sub(dst, lhs, rhs)
mul(dst, lhs, rhs)
div(dst, lhs, rhs)
fma(dst, lhs, rhs, acc)
log2(dst, src)
exp2(dst, src)
shuffle_up(dst, src, lane_delta, mask, clamp)
shuffle_index(dst, src, source_lane, mask, clamp)
select(dst, predicate, true_value, false_value)
gemm(dst, lhs, rhs, accumulate=False)
```

For readability, `x = mul(a, b)` means the same primitive with an implicit
destination register.  `pipe`, `init_pipe`, `acquire`, `wait`,
`commit`, `release`, `tail`, `fence`, `barrier`, TensorMap field
replacement/fences, directional-copy groups, CTA synchronization, register
budget changes, TMEM lifetime, and cursor updates are schedule operations.
There are deliberately no computational primitives named `TMA`,
`cp.async`, `ldmatrix`, `stmatrix`, `mma.sync`, `TCGEN05`,
`TensorMap`, `inverse`, or `gdn_prefill`.

`add(..., lanes=2, rounding="rn", ftz=False)` below is one packed two-lane
f32 add operation with two ordered results.  It is not shorthand for two
independent scalar adds.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

HEAD_PAIRS = (
    (2, 8), (4, 16), (8, 32), (16, 64),
    (16, 32), (16, 48), (16, 16), (32, 32),
)

@specialize(
    HQ_HV=HEAD_PAIRS,
    DK=128,
    DV=128,
    IO_DTYPE="f16",
    ACC_DTYPE="f32",
    STATE_DTYPE="f32",
    INVERSE_DTYPE="f16",
    USE_INITIAL_STATE=True,
    STORE_FINAL_STATE=True,
    ENABLE_CHECKPOINTS=False,
    USE_STATE_INDICES=False,
    PERSISTENT=True,
)
@kernel(
    grid=(min(num_sequences * max(HQ, HV), num_sms), 1, 1),
    block=(384, 1, 1),
    num_warps=12,
    cluster=(1, 1, 1),
    cta_group=1,
    dynamic_smem_bytes=226048,
    tmem_columns=512,
    min_blocks_per_sm=1,
    target="sm_100a",
)
def gdn_prefill_sm100(
    q,                       # f16 [total_tokens,HQ,128], input
    k,                       # f16 [total_tokens,HQ,128], input
    v,                       # f16 [total_tokens,HV,128], input
    gate,                    # f32 [total_tokens,HV], input
    beta,                    # f32 [total_tokens,HV], input
    o,                       # f16 [total_tokens,HV,128], output
    cu_seqlens,              # i32 [num_sequences+1], input
    initial_state,           # f32 [num_sequences,HV,128,128], input [V,K]
    final_state,             # f32 [num_sequences,HV,128,128], output [V,K]
    q_map, k_map, v_map, o_map,  # four by-value, pre-encoded 64-byte maps
    descriptor_workspace,    # u8 [grid_x,4,128], in/out; Q,K,V,O slots
    total_tokens,            # i64 runtime shape/address extent
    num_sequences,           # i32
    num_sms,                 # i32
    scale,                   # f32; official value 1/sqrt(128)
):
    CHUNK = 64
    PAIR_CHUNKS = 2
    DIM = 128
    IS_GQA = HQ >= HV            # selected matrix: true only when HQ == HV
    H_OUT = HQ if IS_GQA else HV # equals max(HQ,HV) for all eight specializations
    H_RATIO = HV // HQ

    # The hierarchical head mode is colex: ratio coordinate is fastest.
    # For GVA, flat output head h maps to (ratio=h%H_RATIO,
    # qk_head=h//H_RATIO).  Q/K have zero stride in ratio; V/O/gate/beta/state
    # map the same pair back to physical HV head h.  Equal-head cases use ratio 1.
    def qk_head_for_output_head(head_i32):
        return head_i32 // H_RATIO

    tid = thread_id()
    warp = warp_uniform(tid // 32)
    lane = tid % 32
    block = block_id_x()
    grid_x = grid_dim_x()

    # Every token/head element and byte address is widened to i64 before
    # multiplication.  The largest official V/O has exactly 2^31 elements.
    def qk_elem(token_i64, head_i32, d_i32):
        return add(mul(add(mul(token_i64, i64(HQ)), i64(head_i32)), 128), i64(d_i32))

    def vo_elem(token_i64, head_i32, d_i32):
        return add(mul(add(mul(token_i64, i64(HV)), i64(head_i32)), 128), i64(d_i32))

    def scalar_elem(token_i64, head_i32):
        return add(mul(token_i64, i64(HV)), i64(head_i32))

    def state_elem(batch_i32, head_i32, value_i32, key_i32):
        x = add(mul(i64(batch_i32), i64(HV)), i64(head_i32))
        x = add(mul(x, 128), i64(value_i32))
        return add(mul(x, 128), i64(key_i32))

    Q = tile("gmem", q, "f16", [total_tokens,HQ,128],
             element_address=qk_elem, byte_scale=2, alignment=16)
    K = tile("gmem", k, "f16", [total_tokens,HQ,128],
             element_address=qk_elem, byte_scale=2, alignment=16)
    V = tile("gmem", v, "f16", [total_tokens,HV,128],
             element_address=vo_elem, byte_scale=2, alignment=16)
    Gate = tile("gmem", gate, "f32", [total_tokens,HV],
                element_address=scalar_elem, byte_scale=4, alignment=16)
    Beta = tile("gmem", beta, "f32", [total_tokens,HV],
                element_address=scalar_elem, byte_scale=4, alignment=16)
    O = tile("gmem", o, "f16", [total_tokens,HV,128],
             element_address=vo_elem, byte_scale=2, alignment=16)
    S0 = tile("gmem", initial_state, "f32", [num_sequences,HV,128,128],
              element_address=state_elem, byte_scale=4, alignment=16)
    S1 = tile("gmem", final_state, "f32", [num_sequences,HV,128,128],
              element_address=state_elem, byte_scale=4, alignment=16)
    Cu = tile("gmem", cu_seqlens, "i32", [num_sequences+1], alignment=4)

    # One CTA owns four 128-byte slots.  Only the first 64 bytes of each slot
    # are descriptor payload; the unused half is deliberately retained.
    desc_cta = view(descriptor_workspace, "u8", [4,128],
                    byte_offset=mul(i64(block), 512))
    desc_q = slice(desc_cta, [0,0:64])   # slot 0, CTA byte +0
    desc_k = slice(desc_cta, [1,0:64])   # slot 1, CTA byte +128
    desc_v = slice(desc_cta, [2,0:64])   # slot 2, CTA byte +256
    desc_o = slice(desc_cta, [3,0:64])   # slot 3, CTA byte +384

    # =======================================================================
    # Shared-memory pool, exact byte offsets, layouts, aliases, lifetimes
    # =======================================================================

    smem = tile("smem", "u8", [226048], byte_offset=0, alignment=1024)

    # 70 i64 mbarrier words, full half before empty half for each ring.
    barrier_words = view(smem, "i64", [70], byte_offset=0)
    tmem_holding = view(smem, "i32", [1], byte_offset=560)
    alignment_padding = view(smem, "u8", [460], byte_offset=564)

    # Physical layouts keep the stage mode last.  Stage-major aliases
    # below are indexing-only views used to make role code readable.
    sQ_storage = view(smem, "f16", [64,128,2], byte_offset=1024,
                      layout="SM100-A-b128-stage-last",
                      lifetime="Q load until warp8 and warp10 release")
    sQ = view(sQ_storage, "f16", [2,64,128],
              index_map="(stage,m,k)->(m,k,stage)")
    sK_storage = view(smem, "f16", [64,128,4], byte_offset=33792,
                      layout="SM100-B-b128-stage-last",
                      lifetime="K load until warp8 and warp10 release")
    sK = view(sK_storage, "f16", [4,64,128],
              index_map="(stage,m,k)->(m,k,stage)")
    sK_trans = alias(sK, "f16", [4,128,64], layout="SM100-KV-B-transpose")
    sV_storage = view(smem, "f16", [128,64,3], byte_offset=99328,
                      layout="SM100-MN-major-A-stage-last",
                      lifetime="V load until CG1 VKS publication")
    sV = view(sV_storage, "f16", [3,128,64],
              index_map="(stage,m,n)->(m,n,stage)")
    sAinv_storage = view(smem, "f16", [64,64,3], byte_offset=148480,
                         layout="SM100-K-major-B-stage-last",
                         lifetime="KK seed through hierarchical correction and NV issue")
    sAinv = view(sAinv_storage, "f16", [3,64,64],
                 index_map="(stage,m,n)->(m,n,stage)")
    sAinvCal = alias(
        sAinv, "f16", [3,64,64], layout="row-major inverse workspace",
        lifetime="seed store -> in-place inverse -> beta-scaled final overwrite",
    )
    sQk_storage = view(smem, "f16", [64,64,2], byte_offset=173056,
                       layout="SM100-K-major-B-stage-last",
                       lifetime="QK epilogue through QKV issue")
    sQk = view(sQk_storage, "f16", [2,64,64],
               index_map="(stage,m,n)->(m,n,stage)")
    sO_storage = view(smem, "f16", [128,64,2], byte_offset=189440,
                      layout="O-matrix-store/TensorMap-stage-last",
                      lifetime="CG1 O store through completed global bulk store")
    sO = view(sO_storage, "f16", [2,128,64],
              index_map="(stage,m,n)->(m,n,stage)")
    sCumsumlog_storage = view(smem, "f32", [64,1,5], byte_offset=222208,
                              layout="flat-stage-last")
    sCumsumlog = view(sCumsumlog_storage, "f32", [5,64],
                      index_map="(stage,m)->(m,0,stage)")
    sCumprod_storage = view(smem, "f32", [64,1,5], byte_offset=223488,
                            layout="flat-stage-last")
    sCumprod = view(sCumprod_storage, "f32", [5,64],
                    index_map="(stage,m)->(m,0,stage)")
    sBeta_storage = view(smem, "f32", [64,1,5], byte_offset=224768,
                         layout="flat-stage-last")
    sBeta = view(sBeta_storage, "f32", [5,64],
                 index_map="(stage,m)->(m,0,stage)")
    assert byte_end(sBeta_storage) == 226048

    # =======================================================================
    # Sixteen full/empty stage rings: physical bytes and persistent cursors
    # =======================================================================

    load_k = pipe("load_k", stages=4, full=[0,32], empty=[32,64],
                  producers=1, consumers=2, transaction_bytes=16384)
    load_q = pipe("load_q", stages=2, full=[64,80], empty=[80,96],
                  producers=1, consumers=2, transaction_bytes=16384)
    load_v = pipe("load_v", stages=3, full=[96,120], empty=[120,144],
                  producers=1, consumers=4, transaction_bytes=16384)
    load_gate = pipe("load_gate", stages=5, full=[144,184], empty=[184,224],
                     producers=32, consumers=256)
    load_beta = pipe("load_beta", stages=5, full=[224,264], empty=[264,304],
                     producers=32, consumers=128)
    q_state_acc = pipe("q_state_acc", stages=1, full=[304,312], empty=[312,320],
                       producers=1, consumers=128)
    kv_acc = pipe("kv_acc", stages=1, full=[320,328], empty=[328,336],
                  producers=1, consumers=128)
    cg0_acc = pipe("cg0_acc", stages=2, full=[336,352], empty=[352,368],
                   producers=1, consumers=128)
    cg1_acc = pipe("cg1_acc", stages=1, full=[368,376], empty=[376,384],
                   producers=1, consumers=128)
    ainv_ready = pipe("ainv_ready", stages=3, full=[384,408], empty=[408,432],
                      producers=128, consumers=1)
    qk_ready = pipe("qk_ready", stages=2, full=[432,448], empty=[448,464],
                    producers=128, consumers=1)
    state_inp_ready = pipe("state_inp_ready", stages=1,
                           full=[464,472], empty=[472,480],
                           producers=128, consumers=1)
    vks_ready = pipe("vks_ready", stages=1, full=[480,488], empty=[488,496],
                     producers=128, consumers=1, ready_only=True)
    nv_ready = pipe("nv_ready", stages=1, full=[496,504], empty=[504,512],
                    producers=128, consumers=1, ready_only=True)
    decay_v_ready = pipe("decay_v_ready", stages=1,
                         full=[512,520], empty=[520,528],
                         producers=128, consumers=1, ready_only=True)
    o_store = pipe("o_store", stages=2, full=[528,544], empty=[544,560],
                   producers=128, consumers=32)

    # Construct every participant once, before role selection.  The cursor is
    # (count,index,phase): producer starts (0,0,1), consumer starts (0,0,0).
    # advance increments count and index; wrapping index to zero toggles phase.
    # A one-stage ring therefore toggles phase on every advance.  No cursor is
    # reset when a persistent CTA advances to another (batch,head).
    for edge in (
        load_k, load_q, load_v, load_gate, load_beta, q_state_acc, kv_acc,
        cg0_acc, cg1_acc, ainv_ready, qk_ready, state_inp_ready,
        vks_ready, nv_ready, decay_v_ready, o_store,
    ):
        edge.producer = participant(edge, side="producer", state=(0,0,1),
                                    lifetime="whole CTA persistent loop")
        edge.consumer = participant(edge, side="consumer", state=(0,0,0),
                                    lifetime="whole CTA persistent loop")

    if warp == 0 and elected():
        for edge in (
            load_k, load_q, load_v, load_gate, load_beta, q_state_acc, kv_acc,
            cg0_acc, cg1_acc, ainv_ready, qk_ready, state_inp_ready,
            vks_ready, nv_ready, decay_v_ready, o_store,
        ):
            for stage in range(edge.stages):
                init_pipe(edge.full[stage], arrivals=edge.producers)
                init_pipe(edge.empty[stage], arrivals=edge.consumers)
        fence("mbarrier_init_release_cluster")
    cta_sync()

    # Named barriers are distinct from the 70-word pipeline pool.
    TMEM_ALLOC_BARRIER = named_barrier(1, threads=320)  # CG0+CG1+warps 8,10
    INVERSE_BARRIER = named_barrier(2, threads=128)     # CG0
    INIT_STATE_BARRIER = named_barrier(4, threads=128)  # CG1

    # Logical TMEM aliases.  Starts/extents are columns, not bytes.
    state_storage_t = tile("tmem", "f32", [128,128,1],
                           base_col=0, columns=128,
                           lifetime="one persistent work item; recurrent over chunks")
    state_t = view(state_storage_t, "f32", [1,128,128],
                   index_map="(stage,m,n)->(m,n,stage)")
    q_state_storage_t = tile("tmem", "f32", [128,64,1],
                             base_col=128, columns=64,
                             lifetime="QS then QKV for one chunk")
    q_state_t = view(q_state_storage_t, "f32", [1,128,64],
                     index_map="(stage,m,n)->(m,n,stage)")
    state_input_storage_t = tile("tmem", "f16", [128,128,1],
                                 base_col=192, columns=64,
                                 packing="two f16 per 32-bit cell")
    state_input_t = view(state_input_storage_t, "f16", [1,128,128],
                         index_map="(stage,m,n)->(m,n,stage)")
    cg0_acc_storage_t = tile("tmem", "f32", [64,64,2],
                             base_col=256, columns=128)
    cg0_acc_t = view(cg0_acc_storage_t, "f32", [2,64,64],
                     index_map="(stage,m,n)->(m,n,stage)")
    cg1_acc_storage_t = tile("tmem", "f32", [128,64,1],
                             base_col=384, columns=64)
    cg1_acc_t = view(cg1_acc_storage_t, "f32", [1,128,64],
                     index_map="(stage,m,n)->(m,n,stage)")
    shared_input_storage_t = tile("tmem", "f16", [128,64,2],
                                  base_col=448, columns=64,
                                  packing="two f16 per 32-bit cell")
    shared_input_t = view(shared_input_storage_t, "f16", [2,128,64],
                          index_map="(stage,m,n)->(m,n,stage)")
    vks_or_nv_t = alias(shared_input_t[0], lifetime="VKS then NV")
    decay_v_t = alias(shared_input_t[1], lifetime="decay-V")
    assert tmem_column_end(shared_input_storage_t) == 512
    assert 128*224 + 128*256 + 128*24 == 64512
    fresh_cubin_resources(registers_per_thread=168, stack_bytes=8,
                          local_memory_bytes=0)

    # Pipeline schedule semantics used literally below.
    # acquire_and_advance: wait empty[index] at producer.phase, clone handle,
    # then advance producer.  wait_and_advance: wait full[index] at
    # consumer.phase, clone handle, then advance consumer.  commit arrives on
    # full[handle.index]; release arrives on empty[handle.index].
    # TMA-like g2s acquire also expects exactly 16384 transaction bytes and the
    # copy completes the full barrier.  ready-only publications deliberately
    # use current_handle()+advance(), never an empty-stage acquire/release.

    # =======================================================================
    # Persistent fast-divmod scheduler, instantiated independently by each role
    # =======================================================================

    def make_scheduler():
        num_heads_fdd = fast_divmod_divisor(i32(H_OUT))
        num_seqs_fdd = fast_divmod_divisor(num_sequences)
        return scheduler_state(
            linear=i32(block),
            stride=i32(grid_x),
            total=i32(num_sequences * H_OUT),
            num_heads_fdd=num_heads_fdd,
            num_seqs_fdd=num_seqs_fdd,
        )

    def current_work(scheduler):
        valid = scheduler.linear < scheduler.total
        remain, head_idx = divmod(scheduler.linear, scheduler.num_heads_fdd)
        _, batch_idx = divmod(remain, scheduler.num_seqs_fdd)
        return work_tile(valid=valid, batch=i32(batch_idx), head=i32(head_idx))

    def advance_scheduler(scheduler):
        scheduler.linear = add(scheduler.linear, scheduler.stride)

    def chunk_geometry(batch_idx):
        batch_start = i64(copy_g2r(Cu[batch_idx], reg_tile([], "i32")))
        batch_end = i64(copy_g2r(Cu[batch_idx+1], reg_tile([], "i32")))
        seqlen = i32(sub(batch_end, batch_start))
        num_valid_chunks = ceil_div(seqlen, 64)
        num_pairs = ceil_div(seqlen, 128)
        num_chunks_padded = mul(num_pairs, 2)
        return batch_start, batch_end, seqlen, num_valid_chunks, num_pairs, num_chunks_padded

    # =======================================================================
    # Dynamic rank-five TensorMap replacement
    # =======================================================================

    def descriptor_fields(name, batch_end):
        if name in ("Q", "K"):
            dims = (128, i32(batch_end), HQ, 1, 1)
            strides = (2*128*HQ, 256, 0, 0)
        elif HQ == HV:
            dims = (128, i32(batch_end), HV, 1, 1)
            strides = (2*128*HV, 256, 0, 0)
        else:
            dims = (128, i32(batch_end), H_RATIO, HQ, 1)
            strides = (2*128*HV, 256, 2*128*H_RATIO, 0)
        return dims, strides

    def replace_descriptor(slot, base_address_i64, dims, strides):
        # One elected lane performs this exact instruction order.  Rank-five
        # dimension zero has implicit f16 element stride 2.
        replace_global_address(slot, base_address_i64)
        replace_global_dim(slot, 0, dims[0])
        replace_global_dim(slot, 1, dims[1])
        replace_global_stride(slot, 0, strides[0])
        replace_global_dim(slot, 2, dims[2])
        replace_global_stride(slot, 1, strides[1])
        replace_global_dim(slot, 3, dims[3])
        replace_global_stride(slot, 2, strides[2])
        replace_global_dim(slot, 4, dims[4])
        replace_global_stride(slot, 3, strides[3])

    def publish_descriptor_updates():
        warp_sync()
        tensormap_fence_release_gpu()

    def acquire_descriptor(slot):
        tensormap_fence_acquire_gpu(slot, bytes=128)

    # Warp 8 performs the kernel's descriptor-prefetch hints before role loops.
    if warp == 8:
        prefetch(q_map)
        prefetch(k_map)
        prefetch(v_map)
        prefetch(o_map)

    # The allocator object is visible to all roles, but warp 4 owns allocation
    # and CG1 is the last owner to relinquish/free it.
    tmem = tmem_allocator(tmem_holding, columns=512, allocator_warp=4,
                          retrieve_barrier=TMEM_ALLOC_BARRIER)

    # =======================================================================
    # Kernel warp-role control flow
    # =======================================================================

    # -----------------------------------------------------------------------
    # COMPUTE GROUP 0: warps 0..3
    # -----------------------------------------------------------------------
    if 0 <= warp <= 3:
        set_register_budget(direction="increase", count=224)
        tmem.wait_for_alloc()

        scheduler = make_scheduler()
        work = current_work(scheduler)
        while work.valid:
            batch = work.batch
            head = work.head
            batch_start, batch_end, seqlen, num_valid_chunks, num_pairs, padded = (
                chunk_geometry(batch)
            )

            # num_pairs is zero for an empty sequence.  The pair body is not
            # entered; this role only advances its persistent scheduler.
            for pair_idx in range(num_pairs):
                first_pair = pair_idx == 0
                has_next_pair = pair_idx < num_pairs - 1
                compute_group_0_pair(
                    pair_idx, first_pair, has_next_pair, scale,
                    load_gate.consumer, load_beta.consumer, cg0_acc.consumer,
                    ainv_ready.producer, qk_ready.producer,
                )

            advance_scheduler(scheduler)
            work = current_work(scheduler)

        tail(ainv_ready.producer)  # from its next state, visit all 3 empty stages
        tail(qk_ready.producer)    # then visit both QK empty stages

    # -----------------------------------------------------------------------
    # COMPUTE GROUP 1: warps 4..7
    # This starts a second top-level if/elif chain.
    # -----------------------------------------------------------------------
    if 4 <= warp <= 7:
        set_register_budget(direction="increase", count=256)
        if warp == 4:
            tmem.allocate(columns=512)
        tmem.wait_for_alloc()

        scheduler = make_scheduler()
        work = current_work(scheduler)
        while work.valid:
            batch = work.batch
            head = work.head
            batch_start, batch_end, seqlen, num_valid_chunks, num_pairs, padded = (
                chunk_geometry(batch)
            )

            if num_valid_chunks > 0:
                # USE_INITIAL_STATE=True is a fixed specialization.
                load_initial_state(
                    batch, head, kv_acc.producer,
                    copy_shape=[128,128], dtype="f32",
                    partition="128 threads x four 32-value subtiles",
                )

                is_first_chunk = True
                for chunk_idx in range(padded):
                    compute_group_1_chunk(
                        chunk_idx, num_pairs, head, seqlen, is_first_chunk, scale,
                        load_v.consumer, load_gate.consumer, cg1_acc.consumer,
                        kv_acc.consumer, q_state_acc.consumer, kv_acc.producer,
                        state_inp_ready.producer, vks_ready.producer,
                        nv_ready.producer, decay_v_ready.producer, o_store.producer,
                    )
                    is_first_chunk = False

                # STORE_FINAL_STATE=True.  This path is mutually exclusive with
                # the empty-sequence copy below.
                store_final_state(
                    batch, head, kv_acc.consumer,
                    copy_shape=[128,128], dtype="f32",
                    partition="128 threads x four 32-value subtiles",
                )
            else:
                # Runtime empty sequence: no TMEM state lifetime is opened.
                # Copy S0 to S1 directly; never execute initial/final TMEM paths.
                copy_empty_state(batch, head, shape=[128,128], dtype="f32")

            advance_scheduler(scheduler)
            work = current_work(scheduler)

        tmem.relinquish_alloc_permit()
        tmem.free(columns=512)
        tail(o_store.producer)         # drain both output empty stages first
        tail(state_inp_ready.producer) # then drain fixed state-input stage

    # -----------------------------------------------------------------------
    # CG0 MATRIX ISSUER: warp 8
    # -----------------------------------------------------------------------
    elif warp == 8:
        set_register_budget(direction="decrease", count=24)
        tmem.wait_for_alloc()

        scheduler = make_scheduler()
        work = current_work(scheduler)
        while work.valid:
            batch_start, batch_end, seqlen, num_valid_chunks, num_pairs, padded = (
                chunk_geometry(work.batch)
            )
            for pair_idx in range(num_pairs):
                mma_cg0_pair(
                    cg0_acc.producer, load_k.consumer, load_q.consumer,
                    gemms=("KK0", "KK1", "QK0", "QK1"),
                )
            advance_scheduler(scheduler)
            work = current_work(scheduler)

        tail(cg0_acc.producer)

    # -----------------------------------------------------------------------
    # CG1 MATRIX ISSUER: warp 10
    # -----------------------------------------------------------------------
    elif warp == 10:
        set_register_budget(direction="decrease", count=24)
        tmem.wait_for_alloc()

        scheduler = make_scheduler()
        work = current_work(scheduler)
        while work.valid:
            batch_start, batch_end, seqlen, num_valid_chunks, num_pairs, padded = (
                chunk_geometry(work.batch)
            )

            # With initial state enabled the kernel sets run_cg1_mma=True even
            # for an empty sequence; the padded loop itself still has zero trips.
            run_cg1_mma = True
            if run_cg1_mma:
                first_loop_chunk = 0  # no-initial-state peeled chunk eliminated
                for chunk_idx in range(first_loop_chunk, padded):
                    mma_cg1_chunk(
                        is_first_chunk=(chunk_idx == 0),
                        cg1_acc_producer=cg1_acc.producer,
                        q_state_producer=q_state_acc.producer,
                        kv_producer=kv_acc.producer,
                        k_consumer=load_k.consumer,
                        q_consumer=load_q.consumer,
                        ainv_consumer=ainv_ready.consumer,
                        qk_consumer=qk_ready.consumer,
                        state_input_consumer=state_inp_ready.consumer,
                        vks_consumer=vks_ready.consumer,
                        nv_consumer=nv_ready.consumer,
                        decay_v_consumer=decay_v_ready.consumer,
                    )

            advance_scheduler(scheduler)
            work = current_work(scheduler)

        tail(cg1_acc.producer)
        tail(q_state_acc.producer)
        tail(kv_acc.producer)

    # -----------------------------------------------------------------------
    # Q/K/V LOAD ISSUER: warp 9
    # -----------------------------------------------------------------------
    elif warp == 9:
        set_register_budget(direction="decrease", count=24)
        scheduler = make_scheduler()
        work = current_work(scheduler)

        if work.valid:
            if elected():
                copy_p2g(q_map[0:64], desc_q)
            warp_sync()
            if elected():
                copy_p2g(k_map[0:64], desc_k)
            warp_sync()
            if elected():
                copy_p2g(v_map[0:64], desc_v)
            warp_sync()
            fence("acq_rel_cta")  # base-payload initialization fence

        while work.valid:
            batch = work.batch
            head = work.head
            batch_start, batch_end, seqlen, num_valid_chunks, num_pairs, padded = (
                chunk_geometry(batch)
            )

            if padded > 0:
                if elected():
                    descriptor_bulk_read_drain()
                warp_sync()
                if elected():
                    q_dims, q_strides = descriptor_fields("Q", batch_end)
                    k_dims, k_strides = descriptor_fields("K", batch_end)
                    v_dims, v_strides = descriptor_fields("V", batch_end)
                    replace_descriptor(desc_q, i64(byte_address(q)), q_dims, q_strides)
                    replace_descriptor(desc_k, i64(byte_address(k)), k_dims, k_strides)
                    replace_descriptor(desc_v, i64(byte_address(v)), v_dims, v_strides)
                publish_descriptor_updates()

                # Preserve the three-part loop: all but last valid,
                # explicit last valid, then pair-padding chunks.  Each helper
                # still loads K, then Q, then V.
                for chunk_idx in range(num_valid_chunks - 1):
                    chunk_offset = add(batch_start, i64(chunk_idx * 64))
                    load_qkv_chunk(chunk_offset, chunk_idx, batch, head)

                chunk_idx = num_valid_chunks - 1
                chunk_offset = add(batch_start, i64(chunk_idx * 64))
                load_qkv_chunk(chunk_offset, chunk_idx, batch, head)

                for chunk_idx in range(num_valid_chunks, padded):
                    chunk_offset = add(batch_start, i64(chunk_idx * 64))
                    load_qkv_chunk(chunk_offset, chunk_idx, batch, head)

            advance_scheduler(scheduler)
            work = current_work(scheduler)

        tail(load_q.producer)
        tail(load_k.producer)
        tail(load_v.producer)

    # -----------------------------------------------------------------------
    # O STORE + GATE/BETA LOAD: warp 11, final independent top-level if
    # -----------------------------------------------------------------------
    if warp == 11:
        set_register_budget(direction="decrease", count=24)
        scheduler = make_scheduler()
        work = current_work(scheduler)

        if work.valid:
            if elected():
                copy_p2g(o_map[0:64], desc_o)
            warp_sync()
            fence("acq_rel_cta")

        while work.valid:
            batch = work.batch
            head = work.head
            batch_start, batch_end, seqlen, num_valid_chunks, num_pairs, padded = (
                chunk_geometry(batch)
            )

            if padded > 0:
                if elected():
                    descriptor_bulk_read_drain()
                warp_sync()
                if elected():
                    o_dims, o_strides = descriptor_fields("O", batch_end)
                    replace_descriptor(desc_o, i64(byte_address(o)), o_dims, o_strides)
                publish_descriptor_updates()
                if elected():
                    acquire_descriptor(desc_o)  # before any prefetch/store work

                # Fixed lookahead: 0 and 1 unconditionally; 2 and 3 together
                # when present; then chunk+4 immediately before current O store.
                for prefetch_idx in range(2):
                    prefetch_offset = add(batch_start, i64(prefetch_idx * 64))
                    is_last = prefetch_idx >= num_valid_chunks - 1
                    load_gate_beta_chunk(prefetch_offset, head, is_last, batch_end)

                if padded > 2:
                    for prefetch_idx in range(2, 4):
                        prefetch_offset = add(batch_start, i64(prefetch_idx * 64))
                        is_last = prefetch_idx >= num_valid_chunks - 1
                        load_gate_beta_chunk(prefetch_offset, head, is_last, batch_end)

                for chunk_idx in range(padded):
                    prefetch_idx = chunk_idx + 4
                    if prefetch_idx < padded:
                        prefetch_offset = add(batch_start, i64(prefetch_idx * 64))
                        is_last = prefetch_idx >= num_valid_chunks - 1
                        load_gate_beta_chunk(prefetch_offset, head, is_last, batch_end)

                    chunk_offset = add(batch_start, i64(chunk_idx * 64))
                    store_o_chunk(head, chunk_offset)

            advance_scheduler(scheduler)
            work = current_work(scheduler)

        tail(load_gate.producer)
        tail(load_beta.producer)

    # =======================================================================
    # Warp 9 per-chunk input helper: K, then Q, then V
    # =======================================================================

    def load_qkv_chunk(chunk_offset_i64, chunk_idx, batch_idx, head_idx):
        qk_head = qk_head_for_output_head(head_idx)
        # K: f16 [64,128], GMEM -> current 4-stage SMEM slot.
        k_handle = acquire_and_advance(load_k.producer)
        if chunk_idx == 0 and elected():
            acquire_descriptor(desc_k)
        copy_g2s(
            K[chunk_offset_i64:chunk_offset_i64+64, qk_head, 0:128],
            sK[k_handle.index, 0:64, 0:128],
            completion=k_handle.full_barrier,
            out_of_bounds="descriptor zero fill",
        )

        # Q: f16 [64,128], GMEM -> current 2-stage SMEM slot.
        q_handle = acquire_and_advance(load_q.producer)
        if chunk_idx == 0 and elected():
            acquire_descriptor(desc_q)
        copy_g2s(
            Q[chunk_offset_i64:chunk_offset_i64+64, qk_head, 0:128],
            sQ[q_handle.index, 0:64, 0:128],
            completion=q_handle.full_barrier,
            out_of_bounds="descriptor zero fill",
        )

        # V: f16 [128,64] logical transpose, GMEM -> current 3-stage SMEM slot.
        v_handle = acquire_and_advance(load_v.producer)
        if chunk_idx == 0 and elected():
            acquire_descriptor(desc_v)
        copy_g2s(
            transpose(V[chunk_offset_i64:chunk_offset_i64+64, head_idx, 0:128]),
            sV[v_handle.index, 0:128, 0:64],
            completion=v_handle.full_barrier,
            out_of_bounds="descriptor zero fill",
        )
        # Transaction completion publishes each full barrier; no software
        # commit occurs for these three copies.

    # =======================================================================
    # Warp 11 gate/beta helper
    # =======================================================================

    def load_gate_beta_chunk(chunk_offset_i64, head_idx, is_last_tile, batch_end_i64):
        # Each lane owns logical positions lane and lane+32.
        pos = reg_tile([2], "i64")
        pos[0] = add(chunk_offset_i64, i64(lane))
        pos[1] = add(chunk_offset_i64, i64(lane + 32))
        valid = reg_tile([2], "bool")

        gate_r = reg_tile([2], "f32")
        if is_last_tile:
            # Predicate storage exists for every call, but the kernel computes
            # both comparisons only on this branch, immediately before fill.
            for col in range(2):
                valid[col] = pos[col] < batch_end_i64
            fill(gate_r, 1.0)  # padded neutral gate
            copy_g2r(Gate[pos,head_idx], gate_r, mask=valid)
        else:
            copy_g2r(Gate[pos,head_idx], gate_r)

        # log2(gate + 1e-10), then an inclusive 32-lane scan independently
        # in each column, then add lane 31 of column 0 into column 1.
        for col in range(2):
            gate_r[col] = add(gate_r[col], 1.0e-10)
            gate_r[col] = log2(gate_r[col])
        for offset in (1, 2, 4, 8, 16):
            for col in range(2):
                prior = shuffle_up(
                    gate_r[col], lane_delta=offset,
                    mask=0xFFFFFFFF, clamp=0,
                )
                gate_r[col] = select(
                    lane >= offset, add(gate_r[col], prior), gate_r[col],
                )
        carry = shuffle_index(
            gate_r[0], source_lane=31,
            mask=0xFFFFFFFF, clamp=31,
        )
        gate_r[1] = add(gate_r[1], carry)

        cumprod_r = reg_tile([2], "f32")
        for col in range(2):
            cumprod_r[col] = exp2(gate_r[col])

        # The producer waits only when SMEM writes begin; the register scan
        # intentionally runs before stage acquisition.
        gate_handle = acquire_and_advance(load_gate.producer)
        copy_r2s(gate_r, sCumsumlog[gate_handle.index, (lane,lane+32)])
        copy_r2s(cumprod_r, sCumprod[gate_handle.index, (lane,lane+32)])
        commit(gate_handle)  # both f32 [64] arrays are now visible

        beta_handle = acquire_and_advance(load_beta.producer)
        if is_last_tile:
            fill(sBeta[beta_handle.index, (lane,lane+32)], 0.0)
            copy_g2s(
                Beta[pos,head_idx],
                sBeta[beta_handle.index, (lane,lane+32)],
                mask=valid,
            )
        else:
            copy_g2s(
                Beta[pos,head_idx],
                sBeta[beta_handle.index, (lane,lane+32)],
            )
        commit(beta_handle)

    # =======================================================================
    # Warp 8: fixed KK0, KK1, QK0, QK1 issue order
    # =======================================================================

    def mma_cg0_pair(cg0_acc_producer, k_consumer, q_consumer, gemms):
        # KK0: (M,N,K)=(64,64,128), f16 x f16 -> f32 TMEM.
        kk0 = acquire_and_advance(cg0_acc_producer)
        k0 = wait_and_advance(k_consumer)
        for kphase in range(8):
            gemm(
                cg0_acc_t[kk0.index],
                sK[k0.index], sK[k0.index],
                accumulate=(kphase != 0),
                phase=kphase, shape=[64,64,128],
            )
        commit(kk0)

        # KK1 is issued before either QK.
        kk1 = acquire_and_advance(cg0_acc_producer)
        k1 = wait_and_advance(k_consumer)
        for kphase in range(8):
            gemm(
                cg0_acc_t[kk1.index],
                sK[k1.index], sK[k1.index],
                accumulate=(kphase != 0),
                phase=kphase, shape=[64,64,128],
            )
        commit(kk1)

        # QK0: (64,64,128), current Q0 and held K0.
        q0 = wait_and_advance(q_consumer)
        qk0 = acquire_and_advance(cg0_acc_producer)
        for kphase in range(8):
            gemm(
                cg0_acc_t[qk0.index],
                sQ[q0.index], sK[k0.index],
                accumulate=(kphase != 0),
                phase=kphase, shape=[64,64,128],
            )
        commit(qk0)

        # QK1: (64,64,128), current Q1 and held K1.
        q1 = wait_and_advance(q_consumer)
        qk1 = acquire_and_advance(cg0_acc_producer)
        for kphase in range(8):
            gemm(
                cg0_acc_t[qk1.index],
                sQ[q1.index], sK[k1.index],
                accumulate=(kphase != 0),
                phase=kphase, shape=[64,64,128],
            )
        commit(qk1)

        # Exact release order after all four issues.
        release(q0)
        release(q1)
        release(k0)
        release(k1)

    # =======================================================================
    # Warp 10: KS, QS, NV, QKV, KV issue order
    # =======================================================================

    def mma_cg1_chunk(
        is_first_chunk,
        cg1_acc_producer, q_state_producer, kv_producer,
        k_consumer, q_consumer, ainv_consumer, qk_consumer,
        state_input_consumer, vks_consumer, nv_consumer, decay_v_consumer,
    ):
        k_handle = wait_and_advance(k_consumer)
        q_handle = wait_and_advance(q_consumer)
        valid_state = True  # USE_INITIAL_STATE=True compile-time fold

        # KS: (128,64,128), state_input TMEM x K SMEM -> cg1_acc TMEM.
        ks_handle = acquire_and_advance(cg1_acc_producer)
        state_handle = wait_and_advance(state_input_consumer)
        for kphase in range(8):
            gemm(
                cg1_acc_t[ks_handle.index],
                state_input_t[state_handle.index], sK[k_handle.index],
                accumulate=(kphase != 0),
                phase=kphase, shape=[128,64,128],
            )
        commit(ks_handle)

        # QS: (128,64,128), same state_input x Q -> q_state TMEM.
        qs_handle = acquire_and_advance(q_state_producer)
        for kphase in range(8):
            gemm(
                q_state_t[qs_handle.index],
                state_input_t[state_handle.index], sQ[q_handle.index],
                accumulate=(kphase != 0),
                phase=kphase, shape=[128,64,128],
            )
        commit(qs_handle)
        release(state_handle)
        release(q_handle)

        # NV: (128,64,64), fixed TMEM slot0 VKS x Ainv SMEM.
        nv_handle = acquire_and_advance(cg1_acc_producer)
        wait_and_advance(vks_consumer)   # ready-only, no returned release
        ainv_handle = wait_and_advance(ainv_consumer)
        for kphase in range(4):
            gemm(
                cg1_acc_t[nv_handle.index],
                vks_or_nv_t, sAinv[ainv_handle.index],
                accumulate=(kphase != 0),
                phase=kphase, shape=[128,64,64],
            )
        commit(nv_handle)
        release(ainv_handle)

        # QKV: (128,64,64), fixed TMEM slot0 NV x QK SMEM, accumulating QS.
        qkv_handle = acquire_and_advance(q_state_producer)
        qk_handle = wait_and_advance(qk_consumer)
        wait_and_advance(nv_consumer)    # ready-only
        for kphase in range(4):
            gemm(
                q_state_t[qkv_handle.index],
                vks_or_nv_t, sQk[qk_handle.index],
                accumulate=True,       # valid_state is fixed true
                phase=kphase, shape=[128,64,64],
            )
        release(qk_handle)
        commit(qkv_handle)

        # Initial-state publication occupies an earlier phase of the one-stage
        # KV ring.  The first chunk performs this exact extra advance.
        if is_first_chunk:
            advance(kv_producer)

        # KV: (128,128,64), fixed TMEM slot1 decay-V x transposed K SMEM,
        # accumulating the already decayed recurrent state.
        kv_handle = acquire_and_advance(kv_producer)
        wait_and_advance(decay_v_consumer)  # ready-only
        for kphase in range(4):
            gemm(
                state_t[kv_handle.index],
                decay_v_t, sK_trans[k_handle.index],
                accumulate=True,
                phase=kphase, shape=[128,128,64],
            )
        commit(kv_handle)
        release(k_handle)

    # =======================================================================
    # CG0 pair epilogue and inverse orchestration
    # =======================================================================

    def compute_group_0_pair(
        pair_idx, first_pair, has_next_pair, scale,
        gate_consumer, beta_consumer, shared_consumer,
        ainv_producer, qk_producer,
    ):
        cg0_tid = tid % 128
        row, col = cg0_partition_coord(cg0_tid, logical_shape=[64,64])

        # The predicates are passed exactly as in the kernel.  This selected
        # helper does not branch on them; one body serves first and steady pairs.
        preserve_predicate(first_pair)
        preserve_predicate(has_next_pair)

        # Two gate stages are consumed together to build two causal transfer
        # fragments T[row,col] = exp2(prefix[row]-prefix[col]) below diagonal.
        gate0 = wait_and_advance(gate_consumer)
        gate1 = wait_and_advance(gate_consumer)
        transfer0 = reg_tile("[64,64] partition over 128 threads", "f32")
        transfer1 = reg_tile("[64,64] partition over 128 threads", "f32")
        for coord in cg0_owned_coords(cg0_tid, [64,64]):
            r, c = coord
            is_lower = r >= c
            diff0 = sub(sCumsumlog[gate0.index,r], sCumsumlog[gate0.index,c])
            diff1 = sub(sCumsumlog[gate1.index,r], sCumsumlog[gate1.index,c])
            transfer0[coord] = select(is_lower, exp2(diff0), 0.0)
            transfer1[coord] = select(is_lower, exp2(diff1), 0.0)
        release(gate0)
        release(gate1)

        # Beta stages are acquired now and deliberately held until each final
        # beta-column-scaled inverse has been published.
        beta0 = wait_and_advance(beta_consumer)
        beta1 = wait_and_advance(beta_consumer)
        beta_row0 = sBeta[beta0.index,row]
        beta_row1 = sBeta[beta1.index,row]

        # ----------------------- KK0 seed ----------------------------------
        ainv0 = acquire_and_advance(ainv_producer)
        kk0 = wait_and_advance(shared_consumer)
        kk_r = reg_tile("[64,64] partition over 128 threads", "f32")
        copy_t2r(cg0_acc_t[kk0.index], kk_r)  # two 32-value subtiles/thread
        fence("async_tmem_load")
        release(kk0)
        seed0_f32 = reg_tile_like(kk_r, "f32")
        seed0_f16 = reg_tile_like(kk_r, "f16")
        for coord in cg0_owned_coords(cg0_tid, [64,64]):
            seed0_f32[coord] = mul(mul(kk_r[coord], transfer0[coord]), beta_row0)
        cast(seed0_f16, seed0_f32)
        copy_r2s(seed0_f16, sAinvCal[ainv0.index])
        # The f16 seed is in aliased SMEM here, before any inverse step.

        # ----------------------- KK1 seed ----------------------------------
        ainv1 = acquire_and_advance(ainv_producer)
        kk1 = wait_and_advance(shared_consumer)
        copy_t2r(cg0_acc_t[kk1.index], kk_r)
        fence("async_tmem_load")
        release(kk1)
        seed1_f32 = reg_tile_like(kk_r, "f32")
        seed1_f16 = reg_tile_like(kk_r, "f16")
        for coord in cg0_owned_coords(cg0_tid, [64,64]):
            seed1_f32[coord] = mul(mul(kk_r[coord], transfer1[coord]), beta_row1)
        cast(seed1_f16, seed1_f32)
        copy_r2s(seed1_f16, sAinvCal[ainv1.index])

        partial_pair_inverse(ainv0.index, ainv1.index)
        finish_pair_inverse_and_publish(
            ainv0, ainv1, beta0, beta1,
        )

        # ----------------------- QK0 epilogue ------------------------------
        qk_ready0 = acquire_and_advance(qk_producer)
        qk0 = wait_and_advance(shared_consumer)
        qk_r = reg_tile("[64,64] partition over 128 threads", "f32")
        qk0_f16 = reg_tile_like(qk_r, "f16")
        copy_t2r(cg0_acc_t[qk0.index], qk_r)
        for coord in cg0_owned_coords(cg0_tid, [64,64]):
            qk_r[coord] = mul(mul(qk_r[coord], transfer0[coord]), scale)
        cast(qk0_f16, qk_r)
        copy_r2s(qk0_f16, sQk[qk_ready0.index])
        fence("async_shared")
        fence("async_tmem_load")
        release(qk0)
        commit(qk_ready0)

        # ----------------------- QK1 epilogue ------------------------------
        qk_ready1 = acquire_and_advance(qk_producer)
        qk1 = wait_and_advance(shared_consumer)
        copy_t2r(cg0_acc_t[qk1.index], qk_r)
        for coord in cg0_owned_coords(cg0_tid, [64,64]):
            qk_r[coord] = mul(mul(qk_r[coord], transfer1[coord]), scale)
        qk1_f16 = reg_tile_like(qk_r, "f16")
        cast(qk1_f16, qk_r)
        copy_r2s(qk1_f16, sQk[qk_ready1.index])
        fence("async_shared")
        fence("async_tmem_load")
        release(qk1)
        commit(qk_ready1)

    # =======================================================================
    # FP16 hierarchical lower-triangular inverse
    # The FP32/TF32 alternatives are compile-time eliminated.
    # =======================================================================

    def invert_diagonal_8x8(block_f16, cg0_tid):
        # Eight-lane subgroups; every thread owns one complete row of 8 f16.
        row_in_group = cg0_tid % 8
        row_f16 = reg_tile([8], "f16")
        row = reg_tile([8], "f32")
        copy_s2r(block_f16[row_in_group,0:8], row_f16)
        cast(row, row_f16)

        for i in range(8):
            row[i] = select(row_in_group == i, 1.0, row[i])

        for src_row in range(7):
            row_scale = mul(row[src_row], -1.0)
            for i in range(src_row):
                pivot_value = shuffle_index(
                    row[i], source_lane=src_row,
                    mask=0xFFFFFFFF, clamp=0b1100000011111,
                )
                updated = fma(row_scale, pivot_value, row[i])
                row[i] = select(row_in_group > src_row, updated, row[i])
            # This pivot-slot write is required; omitting it changes the inverse.
            row[src_row] = select(
                row_in_group > src_row, row_scale, row[src_row],
            )

        cast(row_f16, row)
        copy_r2s(row_f16, block_f16[row_in_group,0:8])

    def block_8_to_16(block_f16, lane_id):
        # [A 0; C D]^-1 lower-left = (-D^-1*C)*A^-1.
        # x1 matrix loads: D non-transposed, C/A transposed.
        D = reg_tile([16,8], "f16", layout="lhs-x1-nontrans-broadcast")
        C = reg_tile([8,8], "f16", layout="rhs-x1-trans")
        A = reg_tile([8,8], "f16", layout="rhs-x1-trans")
        copy_s2r(block_f16[8:16,8:16], D)
        copy_s2r(transpose(block_f16[8:16,0:8]), C)

        dc_f32 = reg_tile([16,8], "f32", layout="warp-acc")
        fill(dc_f32, 0.0)
        gemm(dc_f32, D, C, accumulate=False, shape=[16,8,8])
        mul(dc_f32, dc_f32, -1.0)

        # Structural accumulator-to-A fragment bridge, then f32 -> f16.
        dc_a_view = view(dc_f32, layout="next-gemm-A-fragment")
        dc_f16 = reg_tile_like(dc_a_view, "f16")
        cast(dc_f16, dc_a_view)
        copy_s2r(transpose(block_f16[0:8,0:8]), A)

        out_f32 = reg_tile([16,8], "f32", layout="warp-acc")
        fill(out_f32, 0.0)
        gemm(out_f32, dc_f16, A, accumulate=False, shape=[16,8,8])
        out_f16 = reg_tile_like(out_f32, "f16", layout="x1-nontrans-store")
        cast(out_f16, out_f32)
        copy_r2s(out_f16, block_f16[8:16,0:8])

    def block_16_to_32(block_f16, lane_id):
        # x4 matrix loads: D non-transposed, C/A transposed.
        D = reg_tile([16,16], "f16", layout="lhs-x4-nontrans")
        C = reg_tile([16,16], "f16", layout="rhs-x4-trans")
        A = reg_tile([16,16], "f16", layout="rhs-x4-trans")
        copy_s2r(block_f16[16:32,16:32], D)
        copy_s2r(transpose(block_f16[16:32,0:16]), C)

        dc_f32 = reg_tile([16,16], "f32", layout="warp-acc")
        fill(dc_f32, 0.0)
        gemm(dc_f32, D, C, accumulate=False, shape=[16,16,16])
        mul(dc_f32, dc_f32, -1.0)
        dc_a_view = view(dc_f32, layout="next-gemm-A-fragment-ratio2")
        dc_f16 = reg_tile_like(dc_a_view, "f16")
        cast(dc_f16, dc_a_view)

        copy_s2r(transpose(block_f16[0:16,0:16]), A)
        out_f32 = reg_tile([16,16], "f32", layout="warp-acc")
        fill(out_f32, 0.0)
        gemm(out_f32, dc_f16, A, accumulate=False, shape=[16,16,16])
        out_f16 = reg_tile_like(out_f32, "f16", layout="x4-nontrans-store")
        cast(out_f16, out_f32)
        copy_r2s(out_f16, block_f16[16:32,0:16])

    def block_32_to_64(block_f16, local_warp, lane_id):
        # Two warps own disjoint [16,32] row slices of the bottom-left block.
        rows = slice(32 + local_warp*16, 32 + (local_warp+1)*16)
        D = reg_tile([16,32], "f16", layout="lhs-x4-nontrans")
        C = reg_tile([32,32], "f16", layout="rhs-x4-trans")
        A = reg_tile([32,32], "f16", layout="rhs-x4-trans")
        copy_s2r(block_f16[rows,32:64], D)
        copy_s2r(transpose(block_f16[32:64,0:32]), C)

        dc_f32 = reg_tile([16,32], "f32", layout="warp-acc")
        fill(dc_f32, 0.0)
        gemm(dc_f32, D, C, accumulate=False, shape=[16,32,32])
        mul(dc_f32, dc_f32, -1.0)
        dc_a_view = view(dc_f32, layout="next-gemm-A-fragment-ratio2")
        dc_f16 = reg_tile_like(dc_a_view, "f16")
        cast(dc_f16, dc_a_view)

        copy_s2r(transpose(block_f16[0:32,0:32]), A)
        out_f32 = reg_tile([16,32], "f32", layout="warp-acc")
        fill(out_f32, 0.0)
        gemm(out_f32, dc_f16, A, accumulate=False, shape=[16,32,32])
        out_f16 = reg_tile_like(out_f32, "f16", layout="x4-nontrans-store")
        cast(out_f16, out_f32)
        barrier(INVERSE_BARRIER)  # both warps finish reads before either writes
        copy_r2s(out_f16, block_f16[rows,0:32])

    def partial_pair_inverse(stage0, stage1):
        cg0_tid = tid % 128
        warp_local = cg0_tid // 32
        lane_local = cg0_tid % 32
        inverse_group = warp_local // 2
        inverse_local_warp = warp_local % 2
        selected_stage = select(inverse_group == 1, stage1, stage0)

        barrier(INVERSE_BARRIER)
        diagonal_block = (inverse_local_warp*32 + lane_local) // 8
        b = diagonal_block * 8
        invert_diagonal_8x8(
            sAinvCal[selected_stage,b:b+8,b:b+8], cg0_tid,
        )
        barrier(INVERSE_BARRIER)

        # Every CG0 warp builds one 16x16 diagonal block in each Ainv stage.
        b16 = warp_local * 16
        block_8_to_16(sAinvCal[stage0,b16:b16+16,b16:b16+16], lane_local)
        block_8_to_16(sAinvCal[stage1,b16:b16+16,b16:b16+16], lane_local)
        barrier(INVERSE_BARRIER)

    def finish_pair_inverse_and_publish(ainv0, ainv1, beta0, beta1):
        cg0_tid = tid % 128
        warp_local = cg0_tid // 32
        lane_local = cg0_tid % 32
        inverse_group = warp_local // 2
        inverse_local_warp = warp_local % 2
        selected_stage = select(inverse_group == 1, ainv1.index, ainv0.index)

        # One warp per 32x32 diagonal block, within its selected pair stage.
        b32 = inverse_local_warp * 32
        block_16_to_32(
            sAinvCal[selected_stage,b32:b32+32,b32:b32+32], lane_local,
        )
        barrier(INVERSE_BARRIER)

        # Two warps collaborate on the selected stage's full 64x64 correction.
        block_32_to_64(
            sAinvCal[selected_stage,0:64,0:64],
            inverse_local_warp, lane_local,
        )
        barrier(INVERSE_BARRIER)  # outer publication barrier after the store

        # Stage 0: reload the fully inverted f16 alias, widen, scale by the
        # beta COLUMN, narrow, overwrite the same physical sAinv storage,
        # fence, publish, and only then release beta0.
        inv0_f16 = reg_tile("[64,64] partition over 128 threads", "f16")
        inv0_f32 = reg_tile_like(inv0_f16, "f32")
        copy_s2r(sAinvCal[ainv0.index], inv0_f16)
        cast(inv0_f32, inv0_f16)
        for coord in cg0_owned_coords(cg0_tid, [64,64]):
            r, c = coord
            inv0_f32[coord] = mul(inv0_f32[coord], sBeta[beta0.index,c])
        cast(inv0_f16, inv0_f32)
        copy_r2s(inv0_f16, sAinv[ainv0.index])
        fence("async_shared")
        commit(ainv0)
        release(beta0)

        # Stage 1 follows stage 0 in this exact order.
        inv1_f16 = reg_tile("[64,64] partition over 128 threads", "f16")
        inv1_f32 = reg_tile_like(inv1_f16, "f32")
        copy_s2r(sAinvCal[ainv1.index], inv1_f16)
        cast(inv1_f32, inv1_f16)
        for coord in cg0_owned_coords(cg0_tid, [64,64]):
            r, c = coord
            inv1_f32[coord] = mul(inv1_f32[coord], sBeta[beta1.index,c])
        cast(inv1_f16, inv1_f32)
        copy_r2s(inv1_f16, sAinv[ainv1.index])
        fence("async_shared")
        commit(ainv1)
        release(beta1)

    # =======================================================================
    # Initial, empty, and final state movement
    # =======================================================================

    def load_initial_state(batch_idx, head_idx, kv_producer, copy_shape, dtype, partition):
        cg1_tid = tid % 128
        # Exact partition: St32x32b, repetition 32, 128 CG1 threads, four
        # 32-value subtiles per thread across the logical f32 [128,128] tile.
        state_r = reg_tile(
            "[128,128] partition over 128 threads; 4 subtiles/thread", "f32",
            layout="state-St32x32b-repetition32",
        )
        initial = acquire_and_advance(kv_producer)
        for sub in range(4):
            copy_g2r(
                S0[batch_idx,head_idx,state_gmem_slice(cg1_tid,sub)],
                state_r[sub],
                cache="NO_ALLOCATE",
            )
            copy_r2t(
                state_r[sub],
                state_t[initial.index,state_tmem_slice(cg1_tid,sub)],
            )
        fence("async_tmem_store")
        barrier(INIT_STATE_BARRIER)
        if cg1_tid == 0:
            # CG1 is not the matrix issuer.  The kernel uses a direct full-mbarrier
            # arrival, not the ordinary producer-handle commit operation.
            arrive(initial.full_barrier)

    def copy_empty_state(batch_idx, head_idx, shape, dtype):
        cg1_tid = tid % 128
        for linear in range(cg1_tid, 128*128, 128):
            key = linear // 128
            value = linear - key*128
            x = copy_g2r(S0[batch_idx,head_idx,value,key], reg_tile([], "f32"))
            copy_r2g(x, S1[batch_idx,head_idx,value,key])

    def store_final_state(batch_idx, head_idx, kv_consumer, copy_shape, dtype, partition):
        cg1_tid = tid % 128
        # Exact partition: Ld32x32b, repetition 32, matching the initial store.
        state_r = reg_tile(
            "[128,128] partition over 128 threads; 4 subtiles/thread", "f32",
            layout="state-Ld32x32b-repetition32",
        )
        final = wait_and_advance(kv_consumer)
        for sub in range(4):
            copy_t2r(
                state_t[final.index,state_tmem_slice(cg1_tid,sub)],
                state_r[sub],
            )
            copy_r2g(
                state_r[sub],
                S1[batch_idx,head_idx,state_gmem_slice(cg1_tid,sub)],
                cache="NO_ALLOCATE",
            )
        release(final)

    # =======================================================================
    # CG1 per-padded-chunk recurrence and O staging
    # =======================================================================

    def compute_group_1_chunk(
        chunk_idx, num_pairs, head_idx, seqlen, is_first_chunk, scale,
        v_consumer, gate_consumer, shared_consumer,
        kv_consumer, qstate_consumer, kv_producer,
        state_input_producer, vks_producer, nv_producer,
        decay_producer, output_producer,
    ):
        cg1_tid = tid % 128
        is_pair_first = (chunk_idx & 1) == 0
        valid_state = True  # fixed USE_INITIAL_STATE=True
        preserve_value(num_pairs)  # caller shape operand; checkpoint use eliminated
        preserve_value(seqlen)

        # One-stage KV cursor phase alignment at the start of every pair.
        if is_pair_first:
            advance(kv_producer)
            advance(kv_producer)

        gate_handle = wait_and_advance(gate_consumer)
        cumprod_total = sCumprod[gate_handle.index,63]

        # Snapshot then wait/advance the previous recurrent-state result.
        kv_previous = current_handle(kv_consumer)
        if valid_state:
            kv_previous = wait_and_advance(kv_consumer)

            # Fixed state-input slot publication: current+advance, intentionally
            # without an empty-stage acquire.  The first slot starts empty;
            # thereafter the preceding NV dependency proves warp10 released it.
            state_input = current_handle(state_input_producer)
            advance(state_input_producer)

            state_r = reg_tile(
                "[128,128] partition over 128 threads; 4x32 values/thread", "f32",
                layout="Ld32x32b-repetition32",
            )
            state_input_f16 = reg_tile_like(
                state_r, "f16", layout="St32x32b-repetition16",
            )
            copy_t2r(state_t[kv_previous.index], state_r)
            cast(state_input_f16, state_r)
            copy_r2t(state_input_f16, state_input_t[state_input.index])
            fence("async_tmem_store")
            commit(state_input)

            # ENABLE_CHECKPOINTS=False removes the intervening GMEM store and
            # checkpoint-offset update, but not this state-decay order.
            for x in owned_values(state_r):
                state_r[x] = mul(state_r[x], cumprod_total)
            copy_r2t(state_r, state_t[kv_previous.index])
            fence("async_tmem_store")
            release(kv_previous)

        # Register factors for all logical [128,64] CG1 fragment positions.
        cumprod_rows = reg_tile(
            "[128,64] partition over 128 threads", "f32",
        )
        decay_scale = reg_tile_like(cumprod_rows, "f32")
        last_log = sCumsumlog[gate_handle.index,63]
        for coord_pair in owned_coord_pairs(cg1_tid, [128,64]):
            (_, c0), (_, c1) = coord_pair
            cumprod_rows[coord_pair[0]] = sCumprod[gate_handle.index,c0]
            cumprod_rows[coord_pair[1]] = sCumprod[gate_handle.index,c1]
            decay_diff = reg_tile([2], "f32")
            add(
                decay_diff,
                (last_log, last_log),
                (
                    -sCumsumlog[gate_handle.index,c0],
                    -sCumsumlog[gate_handle.index,c1],
                ),
                lanes=2,
                rounding="rn",
                ftz=False,
            )
            decay_scale[coord_pair[0]] = exp2(decay_diff[0])
            decay_scale[coord_pair[1]] = exp2(decay_diff[1])
        release(gate_handle)

        # V and KS -> VKS.  V uses the current persistent 3-stage cursor; no
        # work item assumes that its first tile is stage zero.
        vks_handle = current_handle(vks_producer)
        advance(vks_producer)
        v_handle = wait_and_advance(v_consumer)
        v_f16 = reg_tile(
            "[128,64] partition over 128 threads", "f16",
            layout="SMEM-x4-trans-load",
        )
        copy_s2r(sV[v_handle.index], v_f16)

        if valid_state:
            ks_handle = wait_and_advance(shared_consumer)
            ks_f32 = reg_tile_like(v_f16, "f32")
            copy_t2r(cg1_acc_t[ks_handle.index], ks_f32)
            for x in owned_values(ks_f32):
                ks_f32[x] = mul(ks_f32[x], cumprod_rows[x])
            release(ks_handle)
            ks_f16 = reg_tile_like(ks_f32, "f16")
            cast(ks_f16, ks_f32)
            for x in owned_values(v_f16):
                v_f16[x] = sub(v_f16[x], ks_f16[x])

        copy_r2t(v_f16, vks_or_nv_t)  # St16x128b repetition 8
        fence("async_tmem_store")
        commit(vks_handle)

        # QS result is scaled in f32 and written back to the same q_state slot.
        if valid_state:
            qs_handle = wait_and_advance(qstate_consumer)
            qs_f32 = reg_tile(
                "[128,64] partition over 128 threads", "f32",
                layout="Ld16x256b/St16x256b-repetition8",
            )
            copy_t2r(q_state_t[qs_handle.index], qs_f32)
            for x in owned_values(qs_f32):
                qs_f32[x] = mul(qs_f32[x], cumprod_rows[x])
                qs_f32[x] = mul(qs_f32[x], scale)
            copy_r2t(qs_f32, q_state_t[qs_handle.index])
            fence("async_tmem_store")
            release(qs_handle)

        # Consume NV after VKS and QS publication.  Release V before reading NV,
        # exactly as in the kernel.  Preserve both the pre-decay f16 NV and an f32
        # alias that is then overwritten with decay-scaled values.
        nv_acc_handle = wait_and_advance(shared_consumer)
        release(v_handle)
        nv_f32 = reg_tile(
            "[128,64] partition over 128 threads; 2x32 subtiles", "f32",
            layout="Ld16x256b-repetition8",
        )
        nv_f16 = reg_tile_like(nv_f32, "f16")
        for sub in range(2):
            copy_t2r(cg1_acc_t[nv_acc_handle.index,sub], nv_f32[sub])
            cast(nv_f16[sub], nv_f32[sub])
        release(nv_acc_handle)

        decay_v_f32 = alias(nv_f32, lifetime="after nv_f16 snapshot")
        for x in owned_values(decay_v_f32):
            decay_v_f32[x] = mul(decay_v_f32[x], decay_scale[x])

        # Both are ready-only single-stage publications.  Take both handles and
        # advance both cursors before either fixed-slot store.
        nv_ready_handle = current_handle(nv_producer)
        advance(nv_producer)
        decay_ready_handle = current_handle(decay_producer)
        advance(decay_producer)

        copy_r2t(nv_f16, vks_or_nv_t)
        decay_v_f16 = reg_tile_like(decay_v_f32, "f16")
        cast(decay_v_f16, decay_v_f32)
        copy_r2t(decay_v_f16, decay_v_t)
        fence("async_tmem_store")  # one common fence for both slots
        commit(nv_ready_handle)    # NV is always published first
        commit(decay_ready_handle)

        # Same-chunk output: wait for QKV, convert f32 -> f16, stage to SMEM,
        # release QKV, then publish the O stage.
        o_handle = acquire_and_advance(output_producer)
        qkv_handle = wait_and_advance(qstate_consumer)
        o_f32 = reg_tile(
            "[128,64] partition over 128 threads", "f32",
            layout="Ld16x256b-repetition8",
        )
        o_f16 = reg_tile_like(o_f32, "f16", layout="x4-trans-store")
        copy_t2r(q_state_t[qkv_handle.index], o_f32)
        cast(o_f16, o_f32)
        copy_r2s(o_f16, sO[o_handle.index])
        fence("async_shared")
        release(qkv_handle)
        commit(o_handle)

    # =======================================================================
    # Warp 11 per-chunk O store
    # =======================================================================

    def store_o_chunk(head_idx, chunk_offset_i64):
        o_handle = wait_and_advance(o_store.consumer)
        copy_s2g(
            sO[o_handle.index,0:128,0:64],
            transpose(O[chunk_offset_i64:chunk_offset_i64+64,head_idx,0:128]),
            out_of_bounds="updated descriptor clips partial/padded rows",
        )
        directional_copy_commit_group()
        directional_copy_wait_group(0)
        release(o_handle)
```

## Seven logical GEMMs

The table states computation.  Issuer ownership, operand placement, stage waits,
and K-phase loops are visible in the sketch; instruction names are intentionally
absent here.

| Name | Logical (M,N,K) | A operand / dtype / orientation | B operand / dtype / orientation | C destination / dtype | K phases | first phase | owner |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| KK | (64,64,128) | staged K, f16, K-major A view | same staged K, f16, K-major B view | CG0 TMEM, f32 | 8 | overwrite | warp 8 |
| QK | (64,64,128) | staged Q, f16, K-major A view | held staged K, f16, K-major B view | CG0 TMEM, f32 | 8 | overwrite | warp 8 |
| KS | (128,64,128) | state-input TMEM, f16 packed, K-major | staged K, f16, K-major | CG1 TMEM, f32 | 8 | overwrite | warp 10 |
| QS | (128,64,128) | state-input TMEM, f16 packed, K-major | staged Q, f16, K-major | q-state TMEM, f32 | 8 | overwrite | warp 10 |
| NV | (128,64,64) | fixed TMEM slot 0 VKS, f16 packed, K-major | Ainv SMEM, f16, K-major | CG1 TMEM, f32 | 4 | overwrite | warp 10 |
| QKV | (128,64,64) | fixed TMEM slot 0 NV, f16 packed, K-major | QK SMEM, f16, K-major | q-state TMEM, f32 | 4 | accumulate scaled QS | warp 10 |
| KV | (128,128,64) | fixed TMEM slot 1 decay-V, f16 packed, K-major | transposed K SMEM, f16, MN-major | recurrent-state TMEM, f32 | 4 | accumulate decayed state | warp 10 |

## TensorMap fields for every static specialization

Every row is rank five.  Dimension zero has implicit two-byte f16 stride.
The four explicit stride values are byte strides.  Every update uses the exact
order shown in `replace_descriptor`: address, dim0, dim1, stride0, dim2,
stride1, dim3, stride2, dim4, stride3.

| HQ,HV | Q/K dimensions | Q/K strides | V/O dimensions | V/O strides |
| --- | --- | --- | --- | --- |
| 2,8 | (128,batch_end,2,1,1) | (512,256,0,0) | (128,batch_end,4,2,1) | (2048,256,1024,0) |
| 4,16 | (128,batch_end,4,1,1) | (1024,256,0,0) | (128,batch_end,4,4,1) | (4096,256,1024,0) |
| 8,32 | (128,batch_end,8,1,1) | (2048,256,0,0) | (128,batch_end,4,8,1) | (8192,256,1024,0) |
| 16,64 | (128,batch_end,16,1,1) | (4096,256,0,0) | (128,batch_end,4,16,1) | (16384,256,1024,0) |
| 16,32 | (128,batch_end,16,1,1) | (4096,256,0,0) | (128,batch_end,2,16,1) | (8192,256,512,0) |
| 16,48 | (128,batch_end,16,1,1) | (4096,256,0,0) | (128,batch_end,3,16,1) | (12288,256,768,0) |
| 16,16 | (128,batch_end,16,1,1) | (4096,256,0,0) | (128,batch_end,16,1,1) | (4096,256,0,0) |
| 32,32 | (128,batch_end,32,1,1) | (8192,256,0,0) | (128,batch_end,32,1,1) | (8192,256,0,0) |

## Ainv alias lifetime

The same 24576 bytes have three distinct ordered meanings per stage.  These are
not three buffers.

| Order | Physical storage | Value | Publication state |
| ---: | --- | --- | --- |
| 1 | `sAinvCal == sAinv` | f16 cast of `KK * causal_transfer * beta[row]` | reserved, not published |
| 2 | same bytes | unit-lower-triangular inverse after 8->16->32->64 correction | still private to CG0 |
| 3 | same bytes | reload to f32, multiply by `beta[column]`, narrow to f16, overwrite | shared fence, Ainv commit, then beta release |

## TIRx module and benchmark contract

The target module keeps these fixed host details:

- `_kernel` matches flat FP16 Q/K/V/O buffers, flat FP32 gate/beta/state
  buffers, int32 `cu_seqlens`, four by-value TensorMaps, and one int8
  descriptor workspace.  `total_tokens` is int64; `num_sequences` and
  `num_sms` are int32.
- `_build_tensor_maps` creates 64-byte-aligned host descriptors in 128-byte
  storage, using rank-three equal-head coordinates or rank-four GVA
  coordinates with 128-byte swizzle and 256-byte L2 promotion.
- `prepare_data` allocates `num_sms * 4 * 128` descriptor bytes, preallocates O
  and final state, and keeps `cu_seqlens` int32 throughout launch preparation.
- `get_kernel` specializes only HQ and HV.  All token counts, sequence counts,
  SM counts, descriptor addresses, and scale remain runtime arguments.
- Test data uses normal FP16 Q/V, normalized FP32-to-FP16 K, uniform FP32 gate,
  sigmoid of uniform FP32 beta, normal FP32 initial state, and
  `scale=1/sqrt(128)`.
- `CONFIGS` is the Cartesian product of the eight head pairs and 15 sequence
  cases: 120 labels `hq{HQ}_hv{HV}_s{normalized_sequence_label}`.

## Static specialization boundary

| Area | Status in this module | Reason |
| --- | --- | --- |
| constants, warp roles, stages, barriers, register and TMEM budgets | represented | fixed by the eight specializations |
| host layouts, seven GEMMs, TensorMaps, scheduler parameters, and launch | represented | required by the runtime ABI |
| device setup, all six role loops, loaders, and split matrix issuers | represented | reachable kernel program |
| CG0, FP16 inverse, state, CG1, and O epilogue helpers | represented | reachable kernel program |
| descriptor workspace sizing and position-independent views | represented | required by persistent TensorMap updates |
| inverse f32/TF32 variants | compile-time eliminated | inverse dtype is FP16 and equals I/O dtype |
| BF16 and non-FP32 state conversions | compile-time eliminated | FP16 I/O, FP32 state/accumulator |
| no-initial-state peel | compile-time eliminated | initial state enabled |
| final-state-disabled, checkpoint, and state-index variants | compile-time eliminated | final enabled; checkpoint/index disabled |
| non-persistent and two-CTA/multicast variants | compile-time eliminated | persistent, CTA group one |
| SM90, SM120, other dimensions or dtypes | outside scope | target is the eight SM100a FP16 specializations only |

## Instruction selection is a lowering consequence

The computational sketch above never requests a hardware instruction.  The
following lowering families follow from storage direction, tile shape, fragment
layout, and synchronization.  The SASS names are taken from generated cubins,
not guessed from semantic operation names.

| Primitive/schedule pattern | PTX family | Fresh SM100a SASS family |
| --- | --- | --- |
| 64-byte `copy_p2g(map payload, 128-byte GMEM slot)` | descriptor payload copy sequence | vector `LDG` / `STG` |
| ordered global address/dimension/byte-stride field replacement | `tensormap.replace.tile.global_address/global_dim/global_stride` | packed `LDG` / `STG` / `LOP` sequence, not `UTMALDG` |
| TensorMap release proxy fence after descriptor field replacement | `fence.proxy.tensormap::generic.release.gpu` | `UTMACMDFLUSH` |
| TensorMap acquire proxy fence at each descriptor's first use | `fence.proxy.tensormap::generic.acquire.gpu [addr], 128` | `CCTL.E.C.LDCU.IV.DEEP`, then `UTMACCTL.IV`, then `UTMACMDFLUSH` |
| full aligned `copy_g2s(GMEM f16 tile, staged SMEM tile, completion=edge)` | `cp.async.bulk.tensor.*.shared::cta.global.tile.mbarrier::complete_tx` | `UTMALDG` |
| predicated scalar beta `copy_g2s` | `cp.async.ca.shared.global` | `LDGSTS.E.LTC128B` |
| `copy_s2g(staged O, descriptor-bounded GMEM O)` plus copy group | `cp.async.bulk.tensor.*.global.shared::cta.tile.bulk_group` | `UTMASTG` |
| `copy_s2r(SMEM matrix layout, register fragment)` | `ldmatrix.sync.aligned.m8n8.x1/x4.shared.b16[.trans]` | `LDSM` |
| `copy_r2s(register fragment, SMEM matrix layout)` | `stmatrix.sync.aligned.m8n8.x1/x4.shared.b16[.trans]` | `STSM` |
| inverse `gemm(REG,REG,REG)` | `mma.sync.aligned.m16n8k8/k16.row.col.f32.f16.f16.f32` | `HMMA` |
| seven `gemm(TMEM,TMEM-or-SMEM,SMEM)` tile chains | `tcgen05.mma.cta_group::1.kind::f16` | `UTCHMMA` |
| `copy_t2r` from f32 TMEM | `tcgen05.ld.sync.aligned.*` | `LDTM` |
| `copy_r2t` to f32/f16-packed TMEM | `tcgen05.st.sync.aligned.*` | `STTM` |
| matrix-result commit to full mbarrier | `tcgen05.commit.cta_group::1.mbarrier::arrive::one` | matrix completion/control sequence |
| gate `log2` | `lg2.approx.ftz.f32` | `MUFU.LG2` |
| gate/decay `exp2` | `ex2.approx.ftz.f32` | `MUFU.EX2` |
| prefix/pivot register shuffle | `shfl.sync.up/index.b32` with stated masks/clamps | `SHFL` |
| one two-lane decay-difference add per coordinate pair, `rounding="rn"`, `ftz=False` | `add.rn.f32x2` | `FADD2` |
| scalar fill/cast/add/sub/mul/div/fma/select | native scalar/vector arithmetic and predicates | `F2F/FADD/FMUL/FFMA/SEL` families |
| pipeline acquire/wait/commit/release/tail | mbarrier wait/arrive/expect-tx | barrier/control families |
| register-budget schedule | `setmaxnreg.inc/dec.sync.aligned.u32` | register-control sequence |
| TMEM allocate/free/relinquish schedule | `tcgen05.alloc/dealloc/relinquish_alloc_permit` | TMEM-control sequence |

PTX and SASS counts are kept separate.  Every fresh specialization contains
24 `tcgen05.ld.sync`, 20 `tcgen05.st.sync`, and eight `add.rn.f32x2` PTX
instructions.  The decay adds account for eight of the 24 total `FADD2`
instructions in each SASS specialization; the other 16 are packed negations
in the FP16 inverse path.

All eight fresh specializations have the same audited SASS core opcode counts:

| Family | Count per specialization |
| --- | ---: |
| `LDTM` total: 8 `LDTM.x32` + 24 `LDTM.16...` | 32 |
| `STTM` | 20 |
| `UTMALDG` | 42 |
| `UTMASTG` | 2 |
| `UTMACCTL.IV` | 19 |
| `UTMACMDFLUSH` | 22 |
| `UTCHMMA` | 60 |
| `LDSM` | 27 |
| `STSM` | 13 |
| `HMMA` | 24 |
| `LDGSTS.E.LTC128B` | 20 |
| `FADD2` total (including 8 selected decay adds) | 24 |
| `MUFU.LG2` | 10 |
| `MUFU.EX2` | 90 |

The counts are audit evidence, not operands or issue-count hints in the sketch.
