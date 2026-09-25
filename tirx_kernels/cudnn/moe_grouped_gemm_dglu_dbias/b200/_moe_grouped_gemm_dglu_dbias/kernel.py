# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MoE BF16 grouped GEMM with a fused dGLU backward epilogue.

Upstream source:
``python/cudnn/gemm/cutedsl/grouped/dglu/moe_grouped_gemm_dglu_dbias.py``
(``MoEGroupedGemmDgluDbiasBf16Kernel``), with the tile scheduler from
``python/cudnn/gemm/cutedsl/grouped/moe_persistent_scheduler.py``, the per-expert
tensor-map workspace from ``python/cudnn/gemm/cutedsl/grouped/moe_utils.py``, and
the gmem addressing extensions from
``python/cudnn/gemm/cutedsl/grouped/moe_sched_extension.py``.

The kernel computes, for every expert ``g`` over its 256-aligned row range::

    ref   = alpha[g]^2 * A @ B[g]^T
    gate  = beta[g] * C[:, even 32-column blocks]
    up    = beta[g] * C[:, odd 32-column blocks]
    D     = interleave(dGLU_gate(ref, prob, gate, up), dGLU_up(ref, prob, gate, up))

plus the per-row ``dprob`` scalar and the optional per-expert ``dbias`` column
sums. Both are accumulated with global atomics into caller-zeroed buffers.

The operands are plain BF16, so the multiply is ``tcgen05.mma.kind::f16`` and
there are no scale-factor tensors, no output quantization and no amax pass. Two
consequences are load-bearing rather than cosmetic: tensor-memory columns are
derived per specialization instead of always reserving 512, and under a two-CTA
atom each CTA stages only its half of the N operand, so a B stage is
``(tile_n / atom_thr) * k_tile * 2`` bytes.
"""

from functools import cache

import tirx_kernels.tirx_lite as txl

from . import spec

# dGeGLU's constants are literals in the bf16 source rather than parameters:
# `sigmoid(1.702 * clamp(gate, max=7))` and `clamp(up, -7, 7)`.
GEGLU_ALPHA = 1.702
GEGLU_MAX = 7.0
GEGLU_MIN = -7.0


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _warp_uniform(value):
    """The source's `cute.arch.make_warp_uniform`: broadcast lane 0's copy.

    A value the compiler cannot prove is warp-invariant forces everything
    derived from it onto the per-thread datapath. Broadcasting through
    `shfl.sync.idx` states the invariance, which is what lets ptxas keep the
    warp-derived half of the address arithmetic on the uniform pipe -- the
    reference splits that arithmetic almost evenly between the two pipes where
    TIRx was running 4.25x as much per-thread.
    """
    uniform = txl.local_scalar("int32")
    txl.ptx["shfl_sync.idx.b32"](uniform, value, txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF))
    return uniform


def _elected():
    """`elect.sync` over the full warp, the source's single-issuer predicate."""
    elected_lane = txl.local_scalar("uint32")
    elected_pred = txl.local_scalar("uint32")
    txl.ptx.elect_sync(elected_lane, elected_pred, txl.uint32(0xFFFFFFFF))
    return elected_pred == txl.uint32(1)


def _try_wait_acquire(dst, barrier, phase):
    """The non-blocking look-ahead peek: acquire scope, and no suspend hint.

    Four of these in the anchor export, at `PTX 863, 919, 1170, 1241`; their
    status predicates the blocking acquire that follows.
    """
    txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
        dst, barrier, txl.cast(phase, "uint32")
    )


def _wait_plain(barrier, phase):
    """Spin on one mbarrier until its phase flips.

    The retry takes the hintless `try_wait`, the form the reference reserves for
    its look-ahead peeks. Given the suspend-time hint the reference carries at
    its 29 blocking acquires, ptxas expands the wait inline around a
    `NANOSLEEP.SYNCS 0x989680` -- four instructions, with the sleep and a
    re-check standing between the barrier and the load that depends on it.
    Without the hint it emits a two-instruction check and moves the retry out of
    line, so the dependent load issues directly behind the branch.

    This kernel runs one CTA per multiprocessor, so a sleeping warp has nothing
    to yield to and the sleep buys nothing the two extra instructions on every
    handshake do not cost back.
    """
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    with txl.While(ready == txl.uint32(0)):
        _try_wait_acquire(ready, barrier, phase)


def _wait_plain_if_needed(barrier, phase, speculative_ready):
    with txl.If(speculative_ready == txl.uint32(0)):
        with txl.Then():
            _wait_plain(barrier, phase)


_LOG2E_NEG = -1.4426950408889634  # exp(-x) == exp2(x * -log2(e))


def _arithmetic(vectorized, packed):
    """Elementwise arithmetic over two-element pairs.

    Under ``vectorized_f32`` the source pairs its FP32 arithmetic into
    ``mul.rn.f32x2`` / ``add.rn.f32x2``, halving the issue count for the same
    numbers -- each half of a packed line rounds exactly as its scalar sibling.
    The approximations (``ex2``, ``rcp``, ``tanh``) and the comparisons have no
    packed form and stay per-element in both modes.

    ``packed`` is the caller-owned 64-bit staging register the packed results
    land in; every helper here writes through caller-owned storage, because
    allocating fresh temporaries per element left one basic block holding
    roughly five hundred allocas and ptxas ran for over twelve minutes without
    finishing.
    """

    def binary(mnemonic, out, left, right):
        if vectorized:
            txl.ptx[f"{mnemonic}.rn.f32x2"](
                packed, txl.cuda.make_float2(left[0], left[1]), txl.cuda.make_float2(right[0], right[1])
            )
            txl.ptx["mov.b64"](out[0], out[1], packed)
        else:
            for half in range(2):
                txl.ptx[f"{mnemonic}.f32"](out[half], left[half], right[half])

    def unary(mnemonic, out, value):
        for half in range(2):
            txl.ptx[mnemonic](out[half], value[half])

    def fused(out, left, right, addend):
        """``left * right + addend`` as one instruction.

        Every arithmetic op here reaches ptxas as inline assembly, which is
        opaque to it: it will not contract a multiply feeding an add into an
        FFMA the way it does for ordinary instructions. The source's export
        writes the multiply and the add separately and lets its compiler fuse
        them -- its machine code carries the FFMA -- so the contraction has to
        be written out here to reach the same instruction count.
        """
        if vectorized:
            txl.ptx["fma.rn.f32x2"](
                packed,
                txl.cuda.make_float2(left[0], left[1]),
                txl.cuda.make_float2(right[0], right[1]),
                txl.cuda.make_float2(addend[0], addend[1]),
            )
            txl.ptx["mov.b64"](out[0], out[1], packed)
        else:
            for half in range(2):
                txl.ptx["fma.rn.f32"](out[half], left[half], right[half], addend[half])

    def scaled(out, value, constant):
        binary("mul", out, value, _spread(constant))

    def product(out, left, right):
        binary("mul", out, left, right)

    def offset(out, value, constant):
        binary("add", out, value, _spread(constant))

    def complement(out, constant, value, scratch):
        """``constant - value``.

        The vectorized export folds the subtraction into a ``neg.f32`` and an
        ``add.rn.f32x2`` rather than issuing a packed subtract.
        """
        if vectorized:
            unary("neg.f32", scratch, value)
            binary("add", out, _spread(constant), scratch)
        else:
            for half in range(2):
                txl.ptx["sub.f32"](out[half], txl.float32(constant), value[half])

    return {
        "binary": binary,
        "unary": unary,
        "scaled": scaled,
        "product": product,
        "offset": offset,
        "complement": complement,
        "fused": fused,
    }


def _spread(constant):
    """One FP32 immediate as a pair, so it can feed either arithmetic mode."""
    return (txl.float32(constant), txl.float32(constant))


def _sigmoid(ops, destination, value, scratch):
    """The source's fastmath sigmoid: 1 / (1 + exp(-x)), no library call.

    The negation rides in the multiplier immediate, which is why the export
    contains no ``neg.f32`` here.
    """
    ops["scaled"](scratch, value, _LOG2E_NEG)
    ops["unary"]("ex2.approx.ftz.f32", scratch, scratch)
    ops["offset"](scratch, scratch, 1.0)
    ops["unary"]("rcp.approx.ftz.f32", destination, scratch)


def _swizzled(row_offset, chunk, row_bytes):
    """One 16-byte chunk's byte offset under the epilogue TMA's row swizzle.

    A swizzled descriptor XORs bits `[4, 4 + B)` of the offset -- the 16-byte
    chunk index -- with bits `[7, 7 + B)`, where `B` counts the chunks in a row.
    The source bits sit at 128 bytes no matter how wide the row is, so a
    64-byte row twists on `row >> 1` and a 32-byte row on `row >> 2` rather than
    on the row index. Twisting on the row index reads the right bytes from the
    wrong places for every element narrower than 32 bits.
    """
    chunks = row_bytes // 16
    if chunks == 1:
        return row_offset + txl.int32(chunk * 16)
    # Every caller passes `row * row_bytes`, so the low `log2(row_bytes)` bits
    # of `row_offset` are zero and both `chunk * 16` and the twist fit entirely
    # inside them. The chunk therefore ORs in rather than adds, and the twist
    # only ever touches the chunk field -- which also means the twist can be
    # read off `row_offset` alone. Writing the whole thing as
    # `row_offset | (chunk ^ twist)` is one `LOP3` per chunk where the sum
    # followed by the XOR is two instructions, and this runs once per chunk per
    # fragment on every subtile.
    twist = txl.local_scalar(
        "int32", init=((row_offset // txl.int32(128)) % txl.int32(chunks)) * txl.int32(16)
    )
    return txl.local_scalar("int32", init=row_offset | (txl.int32(chunk * 16) ^ twist))


def _tcgen05_commit(barrier, mask, cta_group, cluster_size):
    """Asynchronous mbarrier arrival from the MMA pipeline.

    The multicast form carries the CTA mask; a singleton cluster has no peers to
    notify and takes the plain form.
    """
    if cluster_size > 1:
        txl.ptx[
            f"tcgen05.commit.{cta_group}.mbarrier::arrive::one"
            ".shared::cluster.multicast::cluster.b64"
        ](barrier, txl.cast(mask, "uint16"))
    else:
        txl.ptx[f"tcgen05.commit.{cta_group}.mbarrier::arrive::one.shared::cluster.b64"](barrier)


def _unpack_input(values, words, dtype, bits):
    """Widen one thread's row of a 16-bit epilogue tile into 32 FP32 values.

    A 32-bit C never reaches here: its shared words are the values already, and
    the caller loads them straight into the fragment.
    """
    converter = "cvt.f32.bf16" if dtype == "bfloat16" else "cvt.f32.f16"
    for word in range(16):
        low = txl.local_scalar("uint16")
        high = txl.local_scalar("uint16")
        txl.ptx["mov.b32"](low, high, words[word])
        txl.ptx[converter](values[2 * word + 0], low)
        txl.ptx[converter](values[2 * word + 1], high)


def _pack_output(words, values, dtype, bits):
    """Pack a 32-element FP32 fragment into the output dtype's storage words.

    A 16-bit output uses `cvt.rn.bf16x2.f32` / `cvt.rn.f16x2.f32`, two elements
    to a word. A 32-bit output never reaches here -- the gradient registers are
    already its representation, so the caller stages them directly.
    """
    converter = "cvt.rn.bf16x2.f32" if dtype == "bfloat16" else "cvt.rn.f16x2.f32"
    for word in range(16):
        txl.ptx[converter](words[word], values[2 * word + 1], values[2 * word + 0])


def _instruction_descriptor(M, N, a_major, b_major):
    """Fold the static fields of the ``kind::f16`` MMA instruction descriptor.

    Read off the PTX exports rather than adapted from the block-scaled encoder,
    whose kinds lay the same fields out differently. The anchor
    (``M = N = 256``, k-major B) emits ``0x10400490``; ``bmajor_n`` moves it to
    ``0x10410490``, ``tile128_c1x1`` to ``0x08400490`` and ``tile_n64`` to
    ``0x08100490``, which pins bit 16 and the two extent fields exactly.

    Bits 13/14 (negate A/B) arrive as ``TiledMMA`` kernel parameters that this
    kernel never sets, so both predicates are false and the constant below is
    the whole descriptor. Bit 23 (SF format) belongs to the block-scaled kinds
    and is always clear here.
    """
    value = 1 << 4  # D format: f32
    value |= 1 << 7  # A format: bf16
    value |= 1 << 10  # B format: bf16
    if a_major == "m":
        value |= 1 << 15
    if b_major == "n":
        value |= 1 << 16
    value |= ((N >> 3) & 0x3F) << 17
    value |= ((M >> 4) & 0x1F) << 24
    return value & 0xFFFFFFFF


def _descriptor_base(ldo, sdo, swizzle):
    """Fold the SM100 shared descriptor fields except its 14-bit address."""
    arrangement_type = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = 0
    value |= (ldo & 0x3FFF) << 16
    value |= (sdo & 0x3FFF) << 32
    value |= 1 << 46
    value |= (arrangement_type & 0x7) << 61
    return value & 0xFFFFFFFFFFFFFFFF


def _descriptor_with_address(base, shared_address):
    address_field = txl.cast(
        txl.bitwise_and(txl.shift_right(shared_address, txl.uint32(4)), txl.uint32(0x3FFF)), "uint64"
    )
    return txl.bitwise_or(txl.uint64(base), address_field)


_TMA_G2S_3D_CTA = (
    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_G2S_3D_CLUSTER = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_MCAST = ".multicast::cluster"


def _tma_load(destination, descriptor, coords, barrier, mask, *, two_cta, desc_ptr=None):
    """One `cp.async.bulk.tensor` load, multicast only when a mask is given.

    A copy takes the `shared::cluster` stem only when it actually addresses the
    cluster -- a two-CTA copy, or a multicast one. The anchor export carries two
    of each stem, its A and B loads against its two C loads;
    `tile128_c1x1`, whose single-CTA atom sits in a singleton cluster, carries
    four `shared::cta` and no `shared::cluster` at all.

    The two-CTA MMA adds `.cta_group::2`; the multicast form inserts its mask
    modifier before the cache hint, matching the operand order the export shows.
    """
    stem = _TMA_G2S_3D_CLUSTER if (two_cta or mask is not None) else _TMA_G2S_3D_CTA
    if mask is not None:
        head, _, tail = stem.partition(".mbarrier::complete_tx::bytes")
        stem = head + ".mbarrier::complete_tx::bytes" + _TMA_MCAST + tail
    if two_cta:
        stem = stem + ".cta_group::2"
    address = txl.address_of(descriptor) if desc_ptr is None else desc_ptr
    arguments = [destination, address, *[txl.cast(c, "int32") for c in coords], barrier]
    if mask is not None:
        arguments.append(txl.cast(mask, "uint16"))
    arguments.append(txl.uint64(0))
    txl.ptx[stem](*arguments)


def _entry_point(names, body):
    """Build an entry whose parameter names are this specialization's operands.

    Optional operands are absent from the signature entirely, which is what the
    upstream kernel's generated parameter list does when a tensor is ``None`` at
    compile time.
    """
    arguments = ", ".join(names)
    namespace = {"_body": body}
    exec(f"def kernel({arguments}, *, host):\n    _body(({arguments},), host)\n", namespace)
    return namespace["kernel"]


@cache
def _make_kernel(
    group_m_list,
    N,
    K_dim,
    weight_mode,
    sched,
    act,
    c_dtype,
    d_dtype,
    b_major,
    mma_tiler_mn,
    cluster_shape_mn,
    vectorized_f32,
    with_dbias,
    linear_offset,
):
    """Build the launch sequence for one static specialization.

    Returns ``[helper, main]`` when the discrete weight mode or the dynamic
    scheduler needs the upstream pre-kernel, and ``[main]`` otherwise. The
    entries are PrimFuncs, which is what the runner compiles and what
    ``check_low_level_ir`` inspects.
    """
    mode = {
        "weight_mode": weight_mode,
        "sched": sched,
        "act": act,
        "c_dtype": c_dtype,
        "d_dtype": d_dtype,
        "b_major": b_major,
        "mma_tiler_mn": tuple(mma_tiler_mn),
        "cluster_shape_mn": tuple(cluster_shape_mn),
        "vectorized_f32": vectorized_f32,
        "with_dbias": with_dbias,
    }
    derived = spec.derive(mode, group_m_list=list(group_m_list), N=N, K_dim=K_dim)

    ab_bits = 16
    c_bits = spec.dtype_bits(c_dtype)
    d_bits = spec.dtype_bits(d_dtype)
    tokens_total, N_out, L = derived["tokens_total"], derived["n_out"], derived["L"]

    def byte_count(rows, columns, bits):
        return rows * columns * bits // 8

    # Every payload crosses the launch boundary as a flat byte array; the logical
    # extents live in the tensor maps and in the scalar index arithmetic below.
    annotations = {
        "a": txl.gptr[txl.u8, (byte_count(tokens_total, K_dim, ab_bits),)],
        "b": (
            txl.gptr["int64", (L,)]
            if weight_mode == "discrete"
            else txl.gptr[txl.u8, (byte_count(N, K_dim, ab_bits) * L,)]
        ),
        "c": txl.gptr[txl.u8, (byte_count(tokens_total, N_out, c_bits),)],
        "d_row": txl.gptr[txl.u8, (byte_count(tokens_total, N_out, d_bits),)],
    }
    annotations["padded_offsets"] = txl.gptr["int32", (L,)]
    annotations["alpha"] = txl.gptr["float32", (L,)]
    annotations["beta"] = txl.gptr["float32", (L,)]
    # ``generate_dprob`` is hard-coded True upstream, so both routing tensors
    # are part of every specialization's signature.
    annotations["prob"] = txl.gptr["float32", (tokens_total,)]
    annotations["dprob"] = txl.gptr["float32", (tokens_total,)]
    if with_dbias:
        annotations["dbias"] = txl.gptr[txl.u8, (L * N_out * 2,)]
    if derived["workspace_bytes"]:
        annotations["workspace"] = txl.gptr[txl.u8, (derived["workspace_bytes"],)]

    # ---- shapes the launch geometry and the barrier protocol depend on -------
    offsets = derived["smem_offsets"]
    shared_bytes = derived["shared_bytes"]
    ab_stages = derived["num_ab_stage"]
    acc_stages = derived["num_acc_stage"]
    c_stages = derived["num_c_stage"]
    d_stages = derived["num_d_stage"]
    tile_stages = derived["num_tile_stage"]
    atom_thr = derived["atom_thr"]
    cluster_m, cluster_n = cluster_shape_mn
    cluster_size = derived["cluster_size"]
    cta_tile_m, cta_tile_n, k_tile = derived["cta_tile_shape_mnk"]
    k_tiles = derived["k_tiles"]
    epi_subtiles = cta_tile_n // 32
    generate_dbias = derived["generate_dbias"]

    # A two-CTA MMA splits B's N extent across the pair, so each CTA stages
    # `cta_tile_n // atom_thr` columns.
    b_tile_n = cta_tile_n // atom_thr
    # An n-major B is N-contiguous, and `get_smem_layout_atom_ab` picks the
    # widest MN swizzle atom whose contiguous extent divides that of the tile:
    # 128 bytes -- 64 BF16 elements -- when 64 divides `b_tile_n`, then 64, 32
    # and 16 bytes. A tile wider than one atom needs one TMA copy per atom side
    # by side in shared memory, which the `bmajor_n` export confirms: three
    # cluster-multicast loads where the anchor has two, the extra one being B's
    # second 64-column block.
    if b_major == "n":
        for elements, swizzle in ((64, 3), (32, 2), (16, 1), (8, 0)):
            if b_tile_n % elements == 0:
                b_atom_elements, b_atom_swizzle = elements, swizzle
                break
    else:
        b_atom_elements, b_atom_swizzle = k_tile, 3
    b_tma_copies = b_tile_n // b_atom_elements if b_major == "n" else 1

    # A is always k-major; B follows ``b_major``. The shared descriptors carry
    # everything but the 14-bit address, which the device body ORs in. The
    # anchor export builds both from 0x4000404000010000 -- leading offset 1,
    # stride offset 64, 128-byte swizzle -- and the n-major B from
    # 0x4000404002000000, whose leading offset 512 is the 16-byte distance
    # between the two N blocks of a stage.
    a_desc_base = _descriptor_base(ldo=1, sdo=64, swizzle=3)
    b_desc_base = _descriptor_base(
        ldo=(
            1
            if b_major == "k"
            else (derived["b_stage_bytes"] // b_tma_copies) // 16
            if b_tma_copies > 1
            else 0
        ),
        sdo=64,
        swizzle=b_atom_swizzle,
    )
    # The CTA group is 2 only when the MMA atom spans a CTA pair; asking for
    # `cta_group::2` with a single-CTA atom makes the launch itself invalid.
    cta_group = f"cta_group::{atom_thr}"
    mma_mnemonic = f"tcgen05.mma.{cta_group}.kind::f16"
    # The descriptor's M is the whole MMA tile, not the per-CTA half: the anchor
    # export's descriptor is 0x10400490, whose M field is 16 for a 256-row
    # two-CTA tile. Dividing by `atom_thr` here encodes 128 and is wrong on
    # every two-CTA specialization.
    instruction_descriptor = _instruction_descriptor(
        derived["mma_tiler"][0], cta_tile_n, "k", b_major
    )
    ab_empty_arrivals = max(1, cluster_n + (cluster_m // atom_thr) - 1)

    # TensorMaps the launch passes as grid constants, in the order
    # ``host_prelude`` returns them. Discrete weights read the B descriptor out
    # of the workspace instead, so they contribute no grid constant.
    map_names = ["a", "c", "d_row"]
    if weight_mode == "dense":
        map_names.append("b")

    # A multicasts over the cluster's N extent, B over the M extent the two-CTA
    # MMA has already halved.
    a_cluster_piece = cta_tile_m // cluster_n
    b_split = max(1, cluster_m // atom_thr)
    # The multicast split goes on the box's non-contiguous mode, which is N for
    # a k-major B and K for an n-major one.
    b_cluster_piece = (k_tile if b_major == "n" else b_tile_n) // b_split
    epi_m, epi_n = derived["epi_tile"]

    ab_tail = (1, 1, 1, 0, 3, 2, 0)

    def encode_map(descriptor, dtype, rank, data, *fields):
        txl.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, rank, data, *fields)

    def encode_weight_maps(descriptors, b_data, batch):
        """The B TensorMap.

        ``batch`` is the expert count for dense weights, whose single allocation
        the TMA indexes by a third coordinate, and 1 for discrete weights, where
        each expert is its own allocation and gets its own descriptor.
        """
        b_contiguous = (N if b_major == "n" else K_dim) * ab_bits // 8
        b_fields = (
            (
                N,
                K_dim,
                batch,
                b_contiguous,
                N * K_dim * ab_bits // 8,
                b_atom_elements,
                b_cluster_piece,
                1,
            )
            if b_major == "n"
            else (
                K_dim,
                N,
                batch,
                b_contiguous,
                N * K_dim * ab_bits // 8,
                k_tile,
                b_cluster_piece,
                1,
            )
        )
        encode_map(descriptors["b"], "bfloat16", 3, b_data, *b_fields, *ab_tail)

    def host_prelude(params):
        descriptors = {name: txl.stack_alloca("tensormap", 1) for name in map_names}
        encode = encode_map

        # A is (tokens, K) k-major, so K is the contiguous extent.
        a_contiguous = K_dim * ab_bits // 8
        encode(
            descriptors["a"],
            "bfloat16",
            3,
            params["a"].data,
            K_dim,
            tokens_total,
            1,
            a_contiguous,
            tokens_total * a_contiguous,
            k_tile,
            a_cluster_piece,
            1,
            *ab_tail,
        )

        # C and D are (tokens, 2N) n-major over the interleaved output.
        def encode_epilogue(name, tensor, dtype, bits):
            contiguous = N_out * bits // 8
            # One epilogue row is `epi_n` elements, and the source picks the
            # swizzle whose period matches that row exactly -- 128B for a 32-bit
            # element and 64B for a 16-bit one
            # (`get_smem_layout_atom_ab`'s K-major ladder). Leaving it
            # unswizzled costs a bank conflict per epilogue access whose width
            # scales with the element: a 128-byte row is exactly the 32 banks,
            # so every lane of a warp lands on bank 0.
            swizzle = {128: 3, 64: 2}[epi_n * bits // 8]
            encode(
                descriptors[name],
                dtype,
                3,
                tensor.data,
                N_out,
                tokens_total,
                1,
                contiguous,
                tokens_total * contiguous,
                epi_n,
                epi_m,
                1,
                1,
                1,
                1,
                0,
                swizzle,
                2,
                0,
            )

        encode_epilogue("c", params["c"], c_dtype, c_bits)
        encode_epilogue("d_row", params["d_row"], d_dtype, d_bits)

        if weight_mode == "dense":
            encode_weight_maps(descriptors, params["b"].data, L)
        return tuple(descriptors[name] for name in map_names)

    def body(operands, host):
        named = dict(zip(annotations, operands))
        maps = dict(zip(map_names, host))

        # ---- coordinates -------------------------------------------------
        block_x, block_y, cluster_work_id = txl.cta_id()
        cluster_x, cluster_y = txl.cta_id_in_cluster(
            [cluster_m, cluster_n], preferred=[cluster_m, cluster_n]
        )
        cluster_rank = _warp_uniform(cluster_x + cluster_m * cluster_y)
        del block_y
        warp = _warp_uniform(txl.warp_id())
        lane = txl.lane_id()

        # Position inside the two-CTA MMA pair, and the pair's coordinate in the
        # cluster. `cluster_v` is what distinguishes the two CTAs of a pair and
        # has to appear in every multicast mask.
        cluster_v = cluster_rank % atom_thr if atom_thr > 1 else 0
        cluster_m_coord = (cluster_rank // atom_thr) % max(1, cluster_m // atom_thr)
        cluster_n_coord = cluster_rank // cluster_m
        # Only the leader CTA of a two-CTA pair issues the MMA and the AB-full
        # arrival; with a single-CTA atom every CTA is its own leader.
        is_leader_cta = (
            (block_x % txl.int32(atom_thr)) == txl.int32(0)
            if atom_thr > 1
            else txl.int32(0) == txl.int32(0)
        )

        roles = txl.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4])
        tma_role = roles.role("tma", warps=[5])
        c_role = roles.role("c_load", warps=[6])
        sched_role = roles.role("scheduler", warps=[7])

        # ---- storage -----------------------------------------------------
        smem = txl.alloc_buffer((shared_bytes,), txl.u8, scope="shared.dyn", align=1024)
        protocol_pool = txl.smem_pool(base=smem)
        ab_pipe = txl.Pipeline(
            protocol_pool, ab_stages, full="tma", empty="tcgen05", leader=txl.bool(False)
        )
        acc_pipe = txl.Pipeline(
            protocol_pool,
            acc_stages,
            full="tcgen05",
            empty="mbar",
            init_empty=4 * atom_thr,
            leader=txl.bool(False),
        )
        tile_pipe = txl.Pipeline(
            protocol_pool, tile_stages, full="mbar", empty="mbar", leader=txl.bool(False)
        )
        if protocol_pool.bytes != offsets["sinfo"]:
            raise AssertionError("protocol storage order changed before sInfo")
        sinfo = protocol_pool.alloc((4 * tile_stages,), txl.i32, align=16)
        # The dynamic scheduler adds a one-stage cluster pipeline and the
        # 16-byte slot the elected CTA broadcasts the next work index into.
        # Both sit straight after sInfo, which is why every object below them
        # shifts by 32 bytes on that branch.
        sched_pipe = None
        sched_broadcast = None
        if sched == "dynamic":
            if protocol_pool.bytes != offsets["cluster_mbar"]:
                raise AssertionError("scheduler cluster pipeline is misplaced")
            sched_pipe = txl.Pipeline(
                protocol_pool, 1, full="mbar", empty="mbar", leader=txl.bool(False)
            )
            if protocol_pool.bytes != offsets["cluster_broadcast"]:
                raise AssertionError("scheduler broadcast slot is misplaced")
            sched_broadcast = protocol_pool.alloc((4,), txl.i32, align=16)
        if protocol_pool.bytes != offsets["c_full"]:
            raise AssertionError("protocol storage order changed before the C pipeline")
        c_pipe = txl.Pipeline(
            protocol_pool, c_stages, full="tma", empty="mbar", init_empty=4, leader=txl.bool(False)
        )
        if protocol_pool.bytes != offsets["tmem_dealloc"]:
            raise AssertionError("protocol storage order changed before the TMEM barrier")
        tmem_dealloc = protocol_pool.alloc((1,), txl.u64, align=8)
        tmem_slot = protocol_pool.alloc((1,), txl.u32, align=4)

        # ---- descriptor prefetch ----------------------------------------
        with tma_role:
            for name in map_names:
                txl.ptx.prefetch.tensormap(txl.address_of(maps[name]))

        # ---- barrier initialization, in the source's declaration order ----
        with txl.If(warp == 0):
            with txl.Then():
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, ab_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                ab_pipe.full.ptr_to([stage]), txl.uint32(1)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, ab_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                ab_pipe.empty.ptr_to([stage]), txl.uint32(ab_empty_arrivals)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, acc_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                acc_pipe.full.ptr_to([stage]), txl.uint32(1)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, acc_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                acc_pipe.empty.ptr_to([stage]), txl.uint32(4 * atom_thr)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, c_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(c_pipe.full.ptr_to([stage]), txl.uint32(1))
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, c_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                c_pipe.empty.ptr_to([stage]), txl.uint32(4)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, tile_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                tile_pipe.full.ptr_to([stage]), txl.uint32(32)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, tile_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                tile_pipe.empty.ptr_to([stage]), txl.uint32(224)
                            )

        # The tile-info pipeline is built without `defer_sync`, so it publishes
        # itself here; the TMEM barrier belongs to the second epoch below.
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0))

        # `TmemAllocator` is handed the deallocation barrier only under a
        # two-CTA atom, so a single-CTA specialization initializes one mbarrier
        # fewer: the exports count 23 for the anchor and 18 for `tile128_c1x1`,
        # one short of the 19 an unconditional init would leave.
        if sched == "dynamic" or atom_thr > 1:
            with txl.If(warp == 0):
                with txl.Then():
                    with txl.If(_elected()):
                        with txl.Then():
                            if sched == "dynamic":
                                # `internal_init` builds this pipeline with
                                # `defer_sync=True`, so it is published by the
                                # second epoch alongside the TMEM barrier.
                                txl.ptx.mbarrier.init.shared.b64(
                                    sched_pipe.full.ptr_to([0]), txl.uint32(1)
                                )
                                txl.ptx.mbarrier.init.shared.b64(
                                    sched_pipe.empty.ptr_to([0]), txl.uint32(32 * cluster_size)
                                )
                            if atom_thr > 1:
                                txl.ptx.mbarrier.init.shared.b64(
                                    tmem_dealloc.ptr_to([0]), txl.uint32(32)
                                )
        txl.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            txl.ptx.barrier.cluster.arrive.relaxed()

        # ---- scalar addresses, descriptors, and multicast masks ----------
        smem_base = txl.local_scalar("uint32")
        txl.assign(smem_base, txl.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        cluster_smem_base_u64 = txl.local_scalar("uint64")
        txl.ptx.cvta.to.shared__cluster.u64(cluster_smem_base_u64, smem.ptr_to([0]))
        cluster_smem_base = txl.local_scalar("uint32", init=txl.cast(cluster_smem_base_u64, "uint32"))
        a_descriptor = txl.local_scalar(
            "uint64", init=_descriptor_with_address(a_desc_base, smem_base + offsets["sA"])
        )
        b_descriptor = txl.local_scalar(
            "uint64", init=_descriptor_with_address(b_desc_base, smem_base + offsets["sB"])
        )

        # Every mask is the image of the vmnk cluster layout at this CTA's
        # coordinate with one mode varying; the flat rank of `(v, m, n)` is
        # `v + atom_thr * m + cluster_m * n`.
        def cta_bit(v, m, n):
            return txl.uint32(1) << txl.cast(v + atom_thr * m + cluster_m * n, "uint32")

        def mask_union(bits):
            accumulator = txl.local_scalar("uint32", init=txl.uint32(0))
            for bit in bits:
                txl.assign(accumulator, txl.bitwise_or(accumulator, bit))
            return accumulator

        a_mcast_mask = mask_union(
            [cta_bit(cluster_v, cluster_m_coord, n) for n in range(cluster_n)]
        )
        b_mcast_mask = mask_union(
            [cta_bit(cluster_v, m, cluster_n_coord) for m in range(cluster_m // atom_thr)]
        )
        peer_v = cluster_v ^ 1 if atom_thr > 1 else 0
        ab_consumer_mask = mask_union(
            [a_mcast_mask, b_mcast_mask]
            + [cta_bit(peer_v, cluster_m_coord, n) for n in range(cluster_n)]
            + [cta_bit(peer_v, m, cluster_n_coord) for m in range(cluster_m // atom_thr)]
        )
        acc_producer_mask = mask_union(
            [cta_bit(v, cluster_m_coord, cluster_n_coord) for v in range(atom_thr)]
        )

        if cluster_size > 1:
            txl.ptx.barrier.cluster.wait()
        else:
            txl.ptx.bar.sync(txl.uint32(1))

        total_tokens = txl.local_scalar("int32")
        txl.ptx.ld.global_.b32(total_tokens, named["padded_offsets"].ptr_to([L - 1]))
        with txl.If(total_tokens <= txl.int32(0)), txl.Then():
            txl.Return(txl.int32(0))

        # ---- Role 1: warp 7, MoE persistent tile scheduler ----------------
        # `offs` is cumulative, so this expert's token count is the difference
        # against its predecessor. The N tile count is a compile-time constant
        # because N and the cluster tile are both static; only the M count
        # varies per expert.
        cluster_tile_m = cta_tile_m * cluster_m
        cluster_tile_n = cta_tile_n * cluster_n
        n_tile_count = _ceil_div(N, cluster_tile_n)
        num_persistent_clusters = derived["grid"][2]

        def expert_tokens(expert):
            """Token count of one expert, from the cumulative offsets."""
            upper = txl.local_scalar("int32")
            txl.ptx.ld.global_.b32(upper, named["padded_offsets"].ptr_to([expert]))
            tokens = txl.local_scalar("int32", init=upper)
            with txl.If(expert > txl.int32(0)), txl.Then():
                lower = txl.local_scalar("int32")
                txl.ptx.ld.global_.b32(lower, named["padded_offsets"].ptr_to([expert - txl.int32(1)]))
                txl.assign(tokens, tokens - lower)
            return tokens

        def expert_m_tiles(expert):
            tokens = expert_tokens(expert)
            return (tokens + txl.int32(cluster_tile_m - 1)) // txl.int32(cluster_tile_m)

        with sched_role:
            info_prod = txl.PipelineState(tile_stages, phase=1)
            # Cached expert cursor: the walk stays inside one expert until the
            # linear index leaves it, which makes the common step O(1).
            current_expert = txl.local_scalar("int32", init=txl.int32(0))
            expert_tile_start = txl.local_scalar("int32", init=txl.int32(0))
            expert_tile_end = txl.local_scalar("int32", init=txl.int32(0))
            initialized = txl.local_scalar("uint32", init=txl.uint32(0))
            work_linear = txl.local_scalar("int32", init=cluster_work_id)

            record_expert = txl.local_scalar("int32")
            record_tile_m = txl.local_scalar("int32")
            record_tile_n = txl.local_scalar("int32")
            work_valid = txl.local_scalar("uint32")

            def resolve_work():
                """Source `_get_work_tile_for_linear_idx`, cursor included."""
                with txl.If(initialized == txl.uint32(0)), txl.Then():
                    txl.assign(expert_tile_end, expert_m_tiles(txl.int32(0)) * n_tile_count)
                    txl.assign(initialized, txl.uint32(1))
                with txl.While(txl.And(work_linear >= expert_tile_end, current_expert < txl.int32(L))):
                    txl.assign(current_expert, current_expert + txl.int32(1))
                    txl.assign(expert_tile_start, expert_tile_end)
                    with txl.If(current_expert < txl.int32(L)), txl.Then():
                        txl.assign(
                            expert_tile_end,
                            expert_tile_end + expert_m_tiles(current_expert) * n_tile_count,
                        )
                txl.assign(record_expert, txl.int32(-1))
                txl.assign(record_tile_m, txl.int32(0))
                txl.assign(record_tile_n, txl.int32(0))
                txl.assign(work_valid, txl.uint32(0))
                with txl.If(current_expert < txl.int32(L)), txl.Then():
                    txl.assign(work_valid, txl.uint32(1))
                    local_idx = txl.local_scalar("int32", init=work_linear - expert_tile_start)
                    m_count = txl.local_scalar("int32", init=expert_m_tiles(current_expert))
                    cluster_m_idx = txl.local_scalar("int32")
                    cluster_n_idx = txl.local_scalar("int32")
                    # Short side first: the shorter extent changes faster, so
                    # neighbouring clusters overlap in L2.
                    with txl.If(m_count <= txl.int32(n_tile_count)):
                        with txl.Then():
                            txl.assign(cluster_m_idx, local_idx % m_count)
                            txl.assign(cluster_n_idx, local_idx // m_count)
                        with txl.Else():
                            txl.assign(cluster_n_idx, local_idx % txl.int32(n_tile_count))
                            txl.assign(cluster_m_idx, local_idx // txl.int32(n_tile_count))
                    txl.assign(record_expert, current_expert)
                    txl.assign(record_tile_m, cluster_m_idx * txl.int32(cluster_m) + cluster_x)
                    txl.assign(record_tile_n, cluster_n_idx * txl.int32(cluster_n) + cluster_y)

            def publish_record(expert_value, tile_m_value, tile_n_value):
                _wait_plain(tile_pipe.empty.ptr_to([info_prod.stage]), info_prod.phase)
                with txl.If(_elected()), txl.Then():
                    base = info_prod.stage * 4
                    txl.ptx.st.shared.v4.b32(
                        sinfo.ptr_to([base]),
                        expert_value,
                        tile_m_value,
                        tile_n_value,
                        txl.int32(k_tiles),
                    )
                txl.ptx["fence.proxy.async.shared::cta"]()
                txl.ptx.bar.sync(txl.uint32(4), txl.uint32(32))
                # All 32 lanes of the scheduler warp arrive, matching the
                # barrier's 32 arrivals.
                txl.ptx.mbarrier.arrive.shared.b64(tile_pipe.full.ptr_to([info_prod.stage]))
                info_prod.advance()

            if sched == "dynamic":
                # The linear index comes from a global atomic instead of a
                # stride: the leader CTA claims the next one and broadcasts it
                # into every peer's shared slot, so a whole cluster agrees on
                # one work tile.
                counter_offset = L * 128 if weight_mode == "discrete" else 0
                sched_prod = txl.PipelineState(1, phase=1)
                sched_cons = txl.PipelineState(1, phase=0)
                is_leader_cluster = cluster_rank == txl.int32(0)

                def fetch_single_cta():
                    """The whole cluster is one CTA, so there is nobody to tell.

                    The broadcast below reaches its peers through
                    `mapa.shared::cluster` and `st_async.shared::cluster`, and a
                    launch with no cluster dimension has no such address space --
                    those instructions fault rather than degenerate. A lone CTA
                    just keeps the index it claimed.
                    """
                    claimed = txl.local_scalar("uint32", init=txl.uint32(0))
                    with txl.If(lane == txl.int32(0)), txl.Then():
                        txl.ptx["atom.global.add.u32"](
                            claimed, named["workspace"].ptr_to([counter_offset]), txl.uint32(1)
                        )
                    txl.ptx["shfl_sync.idx.b32"](
                        claimed, claimed, txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF)
                    )
                    txl.assign(work_linear, txl.cast(claimed, "int32"))

                def fetch_linear():
                    with txl.If(is_leader_cluster), txl.Then():
                        _wait_plain(sched_pipe.empty.ptr_to([0]), sched_prod.phase)
                        claimed = txl.local_scalar("uint32", init=txl.uint32(0))
                        with txl.If(lane == txl.int32(0)), txl.Then():
                            txl.ptx["atom.global.add.u32"](
                                claimed, named["workspace"].ptr_to([counter_offset]), txl.uint32(1)
                            )
                        txl.ptx["shfl_sync.idx.b32"](
                            claimed, claimed, txl.uint32(0), txl.uint32(31), txl.uint32(0xFFFFFFFF)
                        )
                        with txl.If(lane < txl.int32(cluster_size)), txl.Then():
                            peer = txl.local_scalar("uint32", init=txl.cast(lane, "uint32"))
                            remote_slot = txl.local_scalar("uint32")
                            remote_bar = txl.local_scalar("uint32")
                            local_slot = txl.local_scalar("uint32")
                            local_bar = txl.local_scalar("uint32")
                            txl.assign(
                                local_slot,
                                txl.cuda.cvta_generic_to_shared(sched_broadcast.ptr_to([0])),
                            )
                            txl.assign(
                                local_bar,
                                txl.cuda.cvta_generic_to_shared(sched_pipe.full.ptr_to([0])),
                            )
                            txl.ptx["mapa.shared::cluster.u32"](remote_slot, local_slot, peer)
                            txl.ptx["mapa.shared::cluster.u32"](remote_bar, local_bar, peer)
                            txl.ptx["st_async.shared::cluster.mbarrier::complete_tx::bytes.u32"](
                                remote_slot, claimed, remote_bar
                            )
                            txl.ptx["mbarrier.arrive.expect_tx.shared::cluster.b64"](
                                remote_bar, txl.uint32(4)
                            )
                    sched_prod.advance()
                    _wait_plain(sched_pipe.full.ptr_to([0]), sched_cons.phase)
                    fetched = txl.local_scalar("int32")
                    txl.ptx.ld.shared.b32(fetched, sched_broadcast.ptr_to([0]))
                    # Every CTA's scheduler warp releases onto the *leader's*
                    # empty barrier, which is why it is initialized with
                    # `32 * cluster_size` arrivals. Releasing locally leaves each
                    # CTA 32 short and the leader blocks on its next acquire.
                    release_local = txl.local_scalar("uint32")
                    release_peer = txl.local_scalar("uint32")
                    txl.assign(
                        release_local, txl.cuda.cvta_generic_to_shared(sched_pipe.empty.ptr_to([0]))
                    )
                    txl.ptx["mapa.shared::cluster.u32"](release_peer, release_local, txl.uint32(0))
                    txl.ptx["mbarrier.arrive.shared::cluster.b64"](release_peer, txl.uint32(1))
                    sched_cons.advance()
                    txl.assign(work_linear, fetched)

                claim_next = fetch_linear if cluster_size > 1 else fetch_single_cta
                claim_next()
                resolve_work()
                with txl.While(work_valid == txl.uint32(1)):
                    publish_record(record_expert, record_tile_m, record_tile_n)
                    claim_next()
                    resolve_work()
            else:
                resolve_work()
                with txl.While(work_valid == txl.uint32(1)):
                    publish_record(record_expert, record_tile_m, record_tile_n)
                    txl.assign(work_linear, work_linear + txl.int32(num_persistent_clusters))
                    resolve_work()

            # Termination record: expert_idx = -1 stops the other seven warps.
            publish_record(txl.int32(-1), txl.int32(0), txl.int32(0))
            with txl.unroll(0, tile_stages):
                _wait_plain(tile_pipe.empty.ptr_to([info_prod.stage]), info_prod.phase)
                info_prod.advance()

        # ---- consumer preamble shared by roles 2-5 -----------------------
        # Every consumer keeps its own cursor over the identical record stream.
        # The MMA warp needs only the expert index; nobody reads field 3, since
        # all four roles use the `k_tiles` computed once on the host.
        def take_tile_info(state, slots, want_tiles=True):
            _wait_plain(tile_pipe.full.ptr_to([state.stage]), state.phase)
            base = state.stage * 4
            txl.ptx.ld.shared.b32(slots[0], sinfo.ptr_to([base]))
            if want_tiles:
                txl.ptx.ld.shared.b32(slots[1], sinfo.ptr_to([base + 1]))
                txl.ptx.ld.shared.b32(slots[2], sinfo.ptr_to([base + 2]))
            txl.ptx["fence.proxy.async.shared::cta"]()
            # Every consumer thread arrives; the barrier carries 224 arrivals,
            # one per thread of the seven consumer warps.
            txl.ptx.mbarrier.arrive.shared.b64(tile_pipe.empty.ptr_to([state.stage]))
            state.advance()

        def expert_row_base(expert):
            """First padded row of an expert; the offsets are cumulative."""
            base = txl.local_scalar("int32", init=txl.int32(0))
            with txl.If(expert > txl.int32(0)), txl.Then():
                previous = txl.local_scalar("int32")
                txl.ptx.ld.global_.b32(
                    previous, named["padded_offsets"].ptr_to([expert - txl.int32(1)])
                )
                txl.assign(base, previous)
            return base

        # ---- Role 2: warp 5, persistent TMA producer ---------------------
        with tma_role:
            ab_prod = txl.PipelineState(ab_stages, phase=1)
            # A second cursor kept one step ahead, so the next stage's
            # speculative probe can be issued before this stage's copies and
            # overlap their latency, as the source does.
            ab_probe = txl.PipelineState(ab_stages, phase=1)
            ab_probe.advance()
            info_cons = txl.PipelineState(tile_stages, phase=0)
            tile_expert = txl.local_scalar("int32")
            tile_m_idx = txl.local_scalar("int32")
            tile_n_idx = txl.local_scalar("int32")
            slots = (tile_expert, tile_m_idx, tile_n_idx)
            speculative = txl.local_scalar("uint32")

            take_tile_info(info_cons, slots)
            with txl.While(tile_expert >= txl.int32(0)):
                row_base = expert_row_base(tile_expert)
                a_row = txl.local_scalar("int32", init=row_base + tile_m_idx * txl.int32(cta_tile_m))
                if cluster_n > 1:
                    txl.assign(a_row, a_row + cluster_y * txl.int32(a_cluster_piece))
                # The CTA pair splits B's N extent, so each CTA takes its own
                # half: the export's B coordinate carries the same `cta_in_pair
                # * 128` term that A's M coordinate does (PTX 920/926, both
                # built from `%r5 << 7`).
                n_base = txl.local_scalar("int32", init=tile_n_idx * txl.int32(cta_tile_n))
                if atom_thr > 1:
                    txl.assign(n_base, n_base + cluster_v * txl.int32(b_tile_n))
                # On top of the pair split, the cluster's M extent multicasts B:
                # each CTA fetches one `b_cluster_piece` slice and the pieces
                # together fill the stage. The destination address already
                # carried this term; the coordinate did not, so every CTA in the
                # M direction fetched the *same* slice into a different quarter
                # of SMEM and the tile's upper columns duplicated its lower
                # ones. A k-major B splits along N, an n-major B along K.
                if b_split > 1 and b_major == "k":
                    txl.assign(n_base, n_base + cluster_m_coord * txl.int32(b_cluster_piece))

                # Descriptor addresses: dense weights use the grid-constant
                # TensorMaps, discrete weights the per-expert image the
                # pre-kernel wrote into the workspace.
                if weight_mode == "discrete":
                    # The pre-kernel wrote this expert's B TensorMap image here;
                    # the TMA reads the descriptor straight out of global memory
                    # instead of from a grid constant. One 128-byte slot per
                    # expert -- the block-scaled sibling's second slot held SFB,
                    # which this kernel has no counterpart for.
                    b_desc = named["workspace"].ptr_to([txl.int32(128) * tile_expert])

                txl.assign(speculative, txl.uint32(1))
                with txl.If(txl.int32(k_tiles) > txl.int32(0)), txl.Then():
                    _try_wait_acquire(
                        speculative, ab_pipe.empty.ptr_to([ab_prod.stage]), ab_prod.phase
                    )
                counter = txl.local_scalar("int32", init=txl.int32(0))
                with txl.While(counter < txl.int32(k_tiles)):
                    _wait_plain_if_needed(
                        ab_pipe.empty.ptr_to([ab_prod.stage]), ab_prod.phase, speculative
                    )
                    # `num_tma_load_bytes` counts the whole CTA pair, so the
                    # leader alone arrives for `atom_thr` stages' worth.
                    with txl.If(is_leader_cta), txl.Then():
                        with txl.If(_elected()), txl.Then():
                            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                ab_pipe.full.ptr_to([ab_prod.stage]),
                                txl.uint32(derived["ab_expect_tx_bytes"]),
                            )
                    with txl.If(counter + txl.int32(1) < txl.int32(k_tiles)), txl.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.empty.ptr_to([ab_probe.stage]), ab_probe.phase
                        )

                    k_coord = txl.local_scalar("int32", init=counter * txl.int32(k_tile))
                    if atom_thr > 1:
                        # A two-CTA copy credits the *leader's* barrier: bit 24
                        # of a cluster shared address selects the CTA within the
                        # pair, and clearing it maps this CTA's barrier onto the
                        # even one. Crediting the local barrier instead leaves
                        # the leader permanently short of its expect_tx count.
                        full_bar = txl.local_scalar("uint32")
                        txl.assign(
                            full_bar,
                            cluster_smem_base
                            + txl.uint32(offsets["ab_full"])
                            + txl.cast(ab_prod.stage, "uint32") * txl.uint32(8),
                        )
                        txl.ptx["and.b32"](full_bar, full_bar, txl.uint32(0xFEFFFFFF))
                    else:
                        full_bar = ab_pipe.full.ptr_to([ab_prod.stage])
                    a_slot = smem.ptr_to([offsets["sA"] + ab_prod.stage * derived["a_stage_bytes"]])
                    two_cta = atom_thr > 1
                    # Multicast splits a stage across the cluster: each CTA
                    # issues its own piece into the owning CTA's *cluster*
                    # address, and the pieces together satisfy the stage's
                    # expect_tx byte count. Issuing the whole stage to the local
                    # address instead leaves the barrier permanently short.
                    a_destination = (
                        cluster_smem_base
                        + txl.uint32(offsets["sA"])
                        + txl.cast(ab_prod.stage, "uint32") * txl.uint32(derived["a_stage_bytes"])
                        + txl.cast(cluster_y, "uint32")
                        * txl.uint32(derived["a_stage_bytes"] // cluster_n)
                        if cluster_n > 1
                        else a_slot
                    )
                    with txl.If(_elected()), txl.Then():
                        _tma_load(
                            a_destination,
                            maps["a"],
                            (k_coord, a_row, txl.int32(0)),
                            full_bar,
                            a_mcast_mask if cluster_n > 1 else None,
                            two_cta=two_cta,
                        )
                    b_slot = smem.ptr_to([offsets["sB"] + ab_prod.stage * derived["b_stage_bytes"]])
                    b_block_bytes = derived["b_stage_bytes"] // b_tma_copies
                    b_k_coord = k_coord
                    if b_split > 1 and b_major == "n":
                        b_k_coord = txl.local_scalar(
                            "int32", init=k_coord + cluster_m_coord * txl.int32(b_cluster_piece)
                        )
                    # A discrete descriptor covers one expert's own allocation,
                    # so its batch extent is 1 and the coordinate is 0; the dense
                    # descriptor spans every expert and indexes by expert.
                    b_batch = tile_expert if weight_mode == "dense" else txl.int32(0)
                    for block in range(b_tma_copies):
                        b_destination = (
                            cluster_smem_base
                            + txl.uint32(offsets["sB"])
                            + txl.cast(ab_prod.stage, "uint32") * txl.uint32(derived["b_stage_bytes"])
                            + txl.uint32(block * b_block_bytes)
                            + txl.cast(cluster_m_coord, "uint32") * txl.uint32(b_block_bytes // b_split)
                            if (b_split > 1 or b_tma_copies > 1)
                            else b_slot
                        )
                        b_n_coord = (
                            n_base
                            if block == 0
                            else txl.local_scalar(
                                "int32", init=n_base + txl.int32(block * b_atom_elements)
                            )
                        )
                        with txl.If(_elected()), txl.Then():
                            b_coords = (
                                (b_n_coord, b_k_coord, b_batch)
                                if b_major == "n"
                                else (b_k_coord, b_n_coord, b_batch)
                            )
                            _tma_load(
                                b_destination,
                                maps["b"] if weight_mode == "dense" else None,
                                b_coords,
                                full_bar,
                                b_mcast_mask if b_split > 1 else None,
                                two_cta=two_cta,
                                desc_ptr=None if weight_mode == "dense" else b_desc,
                            )
                    txl.assign(counter, counter + txl.int32(1))
                    ab_prod.advance()
                    ab_probe.advance()

                take_tile_info(info_cons, slots)

            with txl.unroll(0, ab_stages):
                _wait_plain(ab_pipe.empty.ptr_to([ab_prod.stage]), ab_prod.phase)
                ab_prod.advance()

        # ---- Role 3: warp 4, persistent MMA -------------------------------
        with mma_role:
            txl.ptx.bar.sync(txl.uint32(3), txl.uint32(160))
            acc_tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(acc_tmem_base, tmem_slot.ptr_to([0]))

            ab_cons = txl.PipelineState(ab_stages, phase=0)
            ab_cons_probe = txl.PipelineState(ab_stages, phase=0)
            ab_cons_probe.advance()
            acc_prod = txl.PipelineState(acc_stages, phase=1)
            mma_info = txl.PipelineState(tile_stages, phase=0)
            mma_expert = txl.local_scalar("int32")
            mma_slots = (mma_expert, None, None)
            ab_full_ready = txl.local_scalar("uint32")

            take_tile_info(mma_info, mma_slots, want_tiles=False)
            with txl.While(mma_expert >= txl.int32(0)):
                txl.assign(ab_full_ready, txl.uint32(1))
                with txl.If(is_leader_cta), txl.Then():
                    _try_wait_acquire(
                        ab_full_ready, ab_pipe.full.ptr_to([ab_cons.stage]), ab_cons.phase
                    )
                    _wait_plain(acc_pipe.empty.ptr_to([acc_prod.stage]), acc_prod.phase)
                    accumulate = txl.local_scalar("uint32", init=txl.uint32(0))
                    # Two real accumulator stages, each its own `cta_tile_n`
                    # columns of tensor memory, indexed by the pipeline stage --
                    # the block-scaled sibling folded a second region into the
                    # slack its scale-factor columns left over, which the derived
                    # column count here removes the need for.
                    acc_column = txl.local_scalar(
                        "uint32", init=txl.cast(acc_prod.stage, "uint32") * txl.uint32(cta_tile_n)
                    )
                    counter = txl.local_scalar("int32", init=txl.int32(0))
                    with txl.While(counter < txl.int32(k_tiles)):
                        _wait_plain_if_needed(
                            ab_pipe.full.ptr_to([ab_cons.stage]), ab_cons.phase, ab_full_ready
                        )
                        with txl.If(counter + txl.int32(1) < txl.int32(k_tiles)), txl.Then():
                            _try_wait_acquire(
                                ab_full_ready,
                                ab_pipe.full.ptr_to([ab_cons_probe.stage]),
                                ab_cons_probe.phase,
                            )
                        # Four MMA issues per K tile -- the k tile is 64 and
                        # `kind::f16` contracts 16 at a time -- and only the
                        # first clears the accumulate field.
                        for kblock in range(4):
                            with txl.If(_elected()), txl.Then():
                                txl.ptx[mma_mnemonic](
                                    txl.cast(acc_tmem_base + acc_column, "uint32"),
                                    a_descriptor
                                    + txl.cast(
                                        ab_cons.stage * (derived["a_stage_bytes"] // 16)
                                        + kblock * 2,
                                        "uint64",
                                    ),
                                    # 16 rows of K advance 32 bytes along a
                                    # k-major operand and a whole 2 KiB swizzle
                                    # period along an n-major one; both steps are
                                    # read off the anchor and `bmajor_n` exports.
                                    b_descriptor
                                    + txl.cast(
                                        ab_cons.stage * (derived["b_stage_bytes"] // 16)
                                        + kblock * (2 if b_major == "k" else 128),
                                        "uint64",
                                    ),
                                    txl.uint32(instruction_descriptor),
                                    # `disable_output_lane`: this kernel writes
                                    # every lane. The mask is one bit per lane
                                    # of the atom, so a CTA pair spells out
                                    # eight zero words and a single CTA four --
                                    # exactly what the two exports carry.
                                    *([txl.uint32(0)] * (4 * atom_thr)),
                                    txl.ptx.pred(txl.cast(accumulate, "bool")),
                                )
                            txl.assign(accumulate, txl.uint32(1))
                        with txl.If(_elected()), txl.Then():
                            _tcgen05_commit(
                                ab_pipe.empty.ptr_to([ab_cons.stage]),
                                ab_consumer_mask,
                                cta_group,
                                cluster_size,
                            )
                        txl.assign(counter, counter + txl.int32(1))
                        ab_cons.advance()
                        ab_cons_probe.advance()
                    with txl.If(_elected()), txl.Then():
                        _tcgen05_commit(
                            acc_pipe.full.ptr_to([acc_prod.stage]),
                            acc_producer_mask,
                            cta_group,
                            cluster_size,
                        )
                acc_prod.advance()
                take_tile_info(mma_info, mma_slots, want_tiles=False)

            # Only the leader of a CTA pair issues MMA, owns the accumulator and
            # advances `acc_prod`, and every epilogue arrival is redirected to
            # the leader's barrier. The follower must not drain a pipeline it
            # never used: its own `acc_empty` receives nothing, so waiting on it
            # here is what hung every two-CTA specialization at the end of the
            # persistent loop.
            with txl.If(is_leader_cta), txl.Then():
                with txl.unroll(0, acc_stages):
                    _wait_plain(acc_pipe.empty.ptr_to([acc_prod.stage]), acc_prod.phase)
                    acc_prod.advance()

        # ---- Role 5: warp 6, epilogue C producer --------------------------
        # The C subtiles are issued in the same forward order the epilogue reads
        # them, so the two agree on which C subtile belongs to which accumulator
        # subtile with no extra handshake.
        with c_role:
            c_prod = txl.PipelineState(c_stages, phase=1)
            c_info = txl.PipelineState(tile_stages, phase=0)
            c_expert = txl.local_scalar("int32")
            c_tile_m = txl.local_scalar("int32")
            c_tile_n = txl.local_scalar("int32")
            c_slots = (c_expert, c_tile_m, c_tile_n)

            take_tile_info(c_info, c_slots)
            with txl.While(c_expert >= txl.int32(0)):
                c_row = txl.local_scalar(
                    "int32", init=expert_row_base(c_expert) + c_tile_m * txl.int32(cta_tile_m)
                )
                # The 2N interleaved column tile this work tile owns.
                d_tile_n = txl.local_scalar("int32", init=c_tile_n * txl.int32(cta_tile_n * 2))

                subtile = txl.local_scalar("int32", init=txl.int32(0))
                with txl.While(subtile < txl.int32(epi_subtiles)):
                    real_subtile = subtile
                    for half in range(2):
                        _wait_plain(c_pipe.empty.ptr_to([c_prod.stage]), c_prod.phase)
                        with txl.If(_elected()), txl.Then():
                            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                c_pipe.full.ptr_to([c_prod.stage]),
                                txl.uint32(derived["c_stage_bytes"]),
                            )
                        column = txl.local_scalar(
                            "int32",
                            init=d_tile_n
                            + (real_subtile * txl.int32(2) + txl.int32(half)) * txl.int32(32),
                        )
                        c_slot = smem.ptr_to(
                            [offsets["sC"] + c_prod.stage * derived["c_stage_bytes"]]
                        )
                        with txl.If(_elected()), txl.Then():
                            _tma_load(
                                c_slot,
                                maps["c"],
                                (column, c_row, txl.int32(0)),
                                c_pipe.full.ptr_to([c_prod.stage]),
                                None,
                                two_cta=False,
                            )
                        c_prod.advance()
                    txl.assign(subtile, subtile + txl.int32(1))

                take_tile_info(c_info, c_slots)

            with txl.unroll(0, c_stages):
                _wait_plain(c_pipe.empty.ptr_to([c_prod.stage]), c_prod.phase)
                c_prod.advance()

        # ---- Role 4: warps 0-3, dGLU backward epilogue --------------------
        with epilogue_role:
            with txl.If(warp == txl.int32(0)), txl.Then():
                # `.sync.aligned`, so every lane of warp 0 executes it; the warp
                # predicate is the only guard.
                txl.ptx[f"tcgen05.alloc.{cta_group}.sync.aligned.shared::cta.b32"](
                    tmem_slot.ptr_to([0]), txl.uint32(derived["num_tmem_alloc_cols"])
                )
            txl.ptx.bar.sync(txl.uint32(3), txl.uint32(160))
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))

            acc_cons = txl.PipelineState(acc_stages, phase=0)
            c_cons = txl.PipelineState(c_stages, phase=0)
            epi_info = txl.PipelineState(tile_stages, phase=0)
            epi_expert = txl.local_scalar("int32")
            epi_tile_m = txl.local_scalar("int32")
            epi_tile_n = txl.local_scalar("int32")
            epi_slots = (epi_expert, epi_tile_m, epi_tile_n)

            # The register tile is walked two elements at a time throughout the
            # epilogue, and the pairs are always issued packed: one
            # `mul.rn.f32x2` / `add.rn.f32x2` per pair where the scalar form
            # issues two, for identical numbers -- each half of a packed line
            # rounds exactly as its scalar sibling, and a `neg` plus a packed
            # `add` equals a `sub` bit for bit.
            #
            # `vectorized_f32` governs which intrinsics the reference's *author*
            # writes, not what its machine executes: its compiler pairs the
            # scalar arithmetic anyway, and on a scalar specialization the
            # reference retires roughly a quarter of a scalar lowering's FP32
            # instruction count. The larger half of the effect is register
            # pressure -- pairing halves the value-carrying registers in a wide
            # epilogue and takes the stack frame with it.
            #
            # The flag still selects the dGeGLU up-branch filter below, where
            # the reference's two arms genuinely disagree.
            packed_pair = txl.local_scalar("uint64")
            ops = _arithmetic(True, packed_pair)
            epi_lane = txl.local_scalar("int32", init=txl.thread_id() % txl.int32(32))
            epi_warp = _warp_uniform(txl.thread_id() // txl.int32(32))

            def dbias_warp_base():
                return txl.local_scalar(
                    "int32", init=offsets["sDbias"] + epi_warp * txl.int32(64 * 32 * 4)
                )

            def dbias_transpose():
                """Stage this subtile's 64 columns into `sDbias` as (column, row).

                Each thread holds one row's 32 gate and 32 up values, so a column
                sum is a reduction *across* threads; the transpose is what turns
                it into one. Each warp writes its own 8 KiB block.

                Rows are stored with the source's `((col >> 1) & 7) << 2` XOR on
                the row-group index. It cancels out of the sum -- it is there so
                that the 32 lanes reading 32 different columns hit 32 different
                banks instead of all landing on the same one.
                """
                warp_base = dbias_warp_base()
                group = txl.local_scalar("int32", init=epi_lane // txl.int32(4))
                sub = txl.local_scalar("int32", init=epi_lane % txl.int32(4))
                # `group ^ swizzle` for each of the eight constant swizzles.
                twisted = [
                    txl.local_scalar("int32", init=group ^ txl.int32(swizzle)) for swizzle in range(8)
                ]
                for n in range(32):
                    for column, fragment in ((n, rC1), (32 + n, rC2)):
                        slot = txl.local_scalar(
                            "int32",
                            init=warp_base
                            + (txl.int32(column * 32) + twisted[(column >> 1) & 7] * txl.int32(4) + sub)
                            * txl.int32(4),
                        )
                        txl.ptx.st.shared.b32(smem.ptr_to([slot]), fragment[n])

            def dbias_reduce(real_subtile, tile_base):
                """Reduce the staged transpose into this expert's column sums.

                One thread takes two whole columns and sums their 32 rows, the
                four warps' partial sums are combined through the front of the
                same buffer, and warp 0 atomically accumulates a BF16 pair per
                column.
                """
                warp_base = dbias_warp_base()
                txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))

                # Lanes 0-15 take the gate half's even columns, 16-31 the up
                # half's, and each also takes the odd column beside it.
                column_a = txl.local_scalar(
                    "int32",
                    init=(epi_lane % txl.int32(16)) * txl.int32(2)
                    + (epi_lane // txl.int32(16)) * txl.int32(32),
                )
                sums = [txl.local_scalar("float32", init=txl.float32(0.0)) for _ in range(2)]
                quad = txl.alloc_local((4,), "float32")
                # `column_a` is even, so both sides share one twist -- and once
                # the byte scale is folded in, the twist occupies bits 4 to 6
                # while the column and the warp base occupy bits 7 and above.
                # The chunk therefore ORs into the column base rather than
                # adding, which is the three-input `a | (b ^ c)` LOP3: one
                # instruction per address where the add-then-XOR was two, on
                # sixteen addresses every subtile.
                swizzle = txl.local_scalar(
                    "int32", init=((column_a // txl.int32(2)) % txl.int32(8)) * txl.int32(16)
                )
                for side in range(2):
                    column = txl.local_scalar("int32", init=column_a + txl.int32(side))
                    column_base = txl.local_scalar("int32", init=warp_base + column * txl.int32(128))
                    for chunk in range(8):
                        base = txl.local_scalar(
                            "int32", init=column_base | (txl.int32(chunk * 16) ^ swizzle)
                        )
                        # `sDbias` is placed 128-byte aligned and both terms of
                        # the offset are multiples of 16, so a chunk's four rows
                        # are one aligned 16-byte line. Issue the vector load
                        # rather than four scalar ones and let ptxas fuse them:
                        # it does not always choose to, and when it declines the
                        # cost is far out of proportion to the instruction count
                        # -- this read-back runs 64 loads per subtile.
                        txl.ptx["ld.shared.v4.b32"](
                            quad[0], quad[1], quad[2], quad[3], smem.ptr_to([base])
                        )
                        for j in range(4):
                            txl.ptx["add.f32"](sums[side], sums[side], quad[j])

                # Combine the four warps through the front of the same buffer,
                # which every warp has finished reading by now.
                txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))
                partial = txl.local_scalar(
                    "int32",
                    init=offsets["sDbias"]
                    + (epi_warp * txl.int32(64) + epi_lane * txl.int32(2)) * txl.int32(4),
                )
                for side in range(2):
                    txl.ptx.st.shared.b32(smem.ptr_to([partial + txl.int32(4 * side)]), sums[side])
                txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))
                with txl.If(epi_warp == txl.int32(0)), txl.Then():
                    totals = [txl.local_scalar("float32", init=txl.float32(0.0)) for _ in range(2)]
                    pair_in = txl.alloc_local((2,), "float32")
                    for other in range(4):
                        # The two sides sit next to each other and the address is
                        # 8-byte aligned, so one vector load covers both.
                        txl.ptx["ld.shared.v2.b32"](
                            pair_in[0],
                            pair_in[1],
                            smem.ptr_to(
                                [
                                    offsets["sDbias"]
                                    + (txl.int32(other * 64) + epi_lane * txl.int32(2)) * txl.int32(4)
                                ]
                            ),
                        )
                        for side in range(2):
                            txl.ptx["add.f32"](totals[side], totals[side], pair_in[side])
                    # `n_base_d2` is `n_base_d1 + 32` and the up half's columns
                    # start at 32, so one expression covers both halves.
                    n_offset = txl.local_scalar(
                        "int32",
                        init=tile_base + (real_subtile * txl.int32(2)) * txl.int32(32) + column_a,
                    )
                    packed = txl.local_scalar("uint32")
                    txl.ptx["cvt.rn.bf16x2.f32"](packed, totals[1], totals[0])
                    # A cluster covers `cluster_n` column tiles whether or not
                    # the output has that many, so the last cluster carries
                    # tiles past the end. The D store is a TMA and its
                    # descriptor drops those writes on its own; this accumulate
                    # is a plain reduction and needs the bound spelled out.
                    with txl.If(n_offset < txl.int32(N_out)), txl.Then():
                        txl.ptx["red.global.add.noftz.bf16x2"](
                            named["dbias"].ptr_to(
                                [(epi_expert * txl.int32(N_out) + n_offset) * txl.int32(2)]
                            ),
                            packed,
                        )

            rAcc = txl.alloc_local((32,), "float32")
            rC1 = txl.alloc_local((32,), "float32")
            rC2 = txl.alloc_local((32,), "float32")
            d_words = 32 * d_bits // 32
            if d_bits == 32:
                rD1, rD2 = rC1, rC2
            else:
                rD1 = txl.alloc_local((d_words,), "uint32")
                rD2 = txl.alloc_local((d_words,), "uint32")
            # Counts subtiles across the whole persistent loop, so the two D
            # slots keep alternating across work tiles.
            prev_subtiles = txl.local_scalar("int32", init=txl.int32(0))

            take_tile_info(epi_info, epi_slots)
            with txl.While(epi_expert >= txl.int32(0)):
                alpha_value = txl.local_scalar("float32")
                beta_value = txl.local_scalar("float32")
                txl.ptx.ld.global_.b32(alpha_value, named["alpha"].ptr_to([epi_expert]))
                txl.ptx.ld.global_.b32(beta_value, named["beta"].ptr_to([epi_expert]))
                square_alpha = txl.local_scalar("float32", init=alpha_value * alpha_value)

                row_base = expert_row_base(epi_expert)
                # Each accumulator stage owns its own `cta_tile_n` columns of
                # tensor memory, and the MMA warp fills the stage this consumer
                # cursor is about to read.
                acc_column = txl.local_scalar(
                    "uint32", init=txl.cast(acc_cons.stage, "uint32") * txl.uint32(cta_tile_n)
                )

                thread_row = txl.local_scalar(
                    "int32",
                    init=row_base
                    + (epi_tile_m // txl.int32(atom_thr)) * txl.int32(cta_tile_m * atom_thr)
                    + (block_x % txl.int32(atom_thr)) * txl.int32(cta_tile_m)
                    + txl.thread_id()
                    if atom_thr > 1
                    else row_base + epi_tile_m * txl.int32(cta_tile_m) + txl.thread_id(),
                )
                prob_value = txl.local_scalar("float32")
                txl.ptx.ld.global_.b32(prob_value, named["prob"].ptr_to([thread_row]))
                dprob_acc = txl.local_scalar("float32", init=txl.float32(0.0))

                _wait_plain(acc_pipe.full.ptr_to([acc_cons.stage]), acc_cons.phase)

                d_tile_base = txl.local_scalar("int32", init=epi_tile_n * txl.int32(cta_tile_n * 2))

                # The source pins `unroll=1` on this loop and its export
                # carries `.pragma "nounroll"`; a `While` lowers with
                # `#pragma unroll 1`, which is the shape the source has.
                real_subtile = txl.local_scalar("int32", init=txl.int32(0))
                subtile = txl.local_scalar("int32", init=txl.int32(0))
                with txl.While(subtile < txl.int32(epi_subtiles)):
                    # A TMEM address is (lane << 16) | column. The source folds
                    # `(tid << 16) & 0xE00000` in so each epilogue warp reads its
                    # own row group; without it all four warps read warp 0's
                    # rows (source 3617, 3121-3122; PTX 1771-1770).
                    # `(tid << 16) & 0xE00000` keeps bits 21 to 23, and the
                    # column term below never reaches bit 16, so the two fields
                    # are disjoint and the lane bits OR in. That folds the mask
                    # and the combine into one `LOP3` where the mask, the add
                    # and the move were three.
                    tmem_lane = txl.local_scalar("uint32")
                    txl.ptx["shl.b32"](tmem_lane, txl.cast(txl.thread_id(), "uint32"), txl.uint32(16))
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                        *[rAcc[i] for i in range(32)],
                        (tmem_lane & txl.uint32(0xE00000))
                        | (tmem_base + acc_column + txl.cast(real_subtile * txl.int32(32), "uint32")),
                    )

                    # Two C stages per accumulator subtile: gate then up.
                    c_row_bytes = 32 * c_bits // 8
                    c_words = c_row_bytes // 4
                    for fragment in (rC1, rC2):
                        _wait_plain(c_pipe.full.ptr_to([c_cons.stage]), c_cons.phase)
                        c_slot = offsets["sC"] + c_cons.stage * derived["c_stage_bytes"]
                        # Each thread owns one row of the 128x32 subtile, so
                        # its slice starts at row * 32 * c_bits/8 bytes.
                        #
                        # An FP32 C needs no widening at all -- its shared words
                        # already are the values -- so the vector loads land
                        # straight in the fragment. Going through a staging
                        # array first cost 32 `mov.b32` per fragment, and a
                        # register-to-register move written as inline assembly
                        # is opaque to ptxas: it cannot coalesce them away. The
                        # generated code carried 65 such call sites on an FP32-C
                        # specialization against none on a 16-bit one.
                        raw = fragment if c_bits == 32 else txl.alloc_local((c_words,), "uint32")
                        c_row = txl.local_scalar("int32", init=txl.thread_id() * txl.int32(c_row_bytes))
                        for word in range(0, c_words, 4):
                            # The descriptor swizzled this box on the way in, so
                            # the read walks the same permutation.
                            txl.ptx["ld.shared.v4.b32"](
                                raw[word],
                                raw[word + 1],
                                raw[word + 2],
                                raw[word + 3],
                                smem.ptr_to([c_slot + _swizzled(c_row, word // 4, c_row_bytes)]),
                            )
                        # The stage is handed back the moment its bytes are in
                        # registers, before they are widened. The reference's
                        # C fragments are the C dtype, so it releases with
                        # nothing between the load and the fence; widening first
                        # would hold the buffer for another forty-odd
                        # instructions, and with only two C stages the producer
                        # has no slack to absorb that.
                        txl.ptx["fence.proxy.async.shared::cta"]()
                        txl.cuda.warp_sync()
                        with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                            txl.ptx.mbarrier.arrive.shared.b64(c_pipe.empty.ptr_to([c_cons.stage]))
                        c_cons.advance()
                        if c_bits != 32:
                            _unpack_input(fragment, raw, c_dtype, c_bits)

                    # ---- dGLU derivative, elementwise over the subtile ----
                    # The register tile is walked two elements at a time: a
                    # `vectorized_f32` specialization issues one packed
                    # `mul.rn.f32x2` / `add.rn.f32x2` per pair where the scalar
                    # one issues two, for identical numbers. One scratch set is
                    # reused for every pair -- fresh scalars per element make
                    # ptxas superlinear (see `_arithmetic`).
                    def pair():
                        return (txl.local_scalar("float32"), txl.local_scalar("float32"))

                    acc = pair()
                    gate = pair()
                    up = pair()
                    d_gate = pair()
                    d_up = pair()
                    sig = pair()
                    work = pair()
                    step = pair()
                    helper = pair()
                    scratch = {name: pair() for name in "abcdefgh"}
                    prob_pair = (prob_value, prob_value)
                    alpha_pair = (square_alpha, square_alpha)
                    beta_pair = (beta_value, beta_value)
                    # The source reduces dProb as a packed pair and folds the
                    # two halves in once per subtile, not once per element.
                    dprob_pair = pair()
                    for half in range(2):
                        txl.assign(dprob_pair[half], txl.float32(0.0))

                    for element in range(0, 32, 2):
                        span = (element, element + 1)
                        ops["product"](acc, tuple(rAcc[j] for j in span), alpha_pair)
                        ops["product"](gate, tuple(rC1[j] for j in span), beta_pair)
                        ops["product"](up, tuple(rC2[j] for j in span), beta_pair)

                        if act == "dswiglu":
                            _sigmoid(ops, sig, gate, work)
                            # The source spells this out as `swish = gate * sig`
                            # followed by three independent product chains, and
                            # its compiler reassociates them around the shared
                            # `acc * sig` before contracting the last multiply
                            # and add: its machine code retires eleven packed
                            # multiplies, two adds and one FFMA where the
                            # written form has thirteen and three. Every
                            # operation here is inline assembly, so nothing
                            # rewrites it on the way down -- the reassociated
                            # form is what has to be written. `swish` itself is
                            # never formed, and neither is `up * sig`.
                            ops["product"](step, acc, sig)  # acc * sig
                            ops["product"](helper, step, prob_pair)  # acc_prob * sig
                            ops["product"](d_up, helper, gate)  # acc_prob * swish
                            ops["product"](helper, helper, up)
                            ops["product"](step, step, up)
                            # The accumulation absorbs the last multiply:
                            # `acc * up * swish` is never materialised.
                            ops["fused"](dprob_pair, step, gate, dprob_pair)
                            ops["complement"](work, 1.0, sig, scratch["h"])
                            ops["fused"](work, gate, work, _spread(1.0))
                            ops["product"](d_gate, helper, work)
                        else:
                            # dGeGLU. Both gradients are scaled by a *value*,
                            # not by a 0/1 mask: upstream's `x1_filter` is the
                            # clamped gate itself and `x2_filter` the clamped
                            # up, zeroed only where the raw operand falls
                            # outside the bound its branch tests. The block-
                            # scaled sibling carries a later revision of this
                            # function whose filters are 1.0/0.0; transcribing
                            # that here would drop a whole factor.
                            y_gate = scratch["a"]
                            y_up = scratch["b"]
                            # `min`/`max` reach the clamp in one instruction
                            # where the source's `setp`/`selp` pair costs two;
                            # the two forms differ only on a NaN input, which a
                            # C-matrix element cannot be.
                            for half in range(2):
                                txl.ptx["min.f32"](y_gate[half], gate[half], txl.float32(GEGLU_MAX))
                                txl.ptx["min.f32"](y_up[half], up[half], txl.float32(GEGLU_MAX))
                                txl.ptx["max.f32"](y_up[half], y_up[half], txl.float32(GEGLU_MIN))
                            ops["scaled"](work, y_gate, GEGLU_ALPHA)  # 1.702 * y_gate
                            _sigmoid(ops, sig, work, scratch["c"])
                            offset_up = scratch["e"]
                            ops["offset"](offset_up, y_up, linear_offset)
                            step = scratch["d"]
                            ops["product"](step, acc, sig)  # acc * sigmoid
                            # dProb reads the pre-`prob` product, which is why
                            # it is folded in before the routing factor.
                            helper = scratch["f"]
                            ops["product"](helper, offset_up, step)
                            ops["fused"](dprob_pair, helper, y_gate, dprob_pair)
                            inner = scratch["g"]
                            ops["complement"](inner, 1.0, sig, scratch["h"])
                            ops["fused"](inner, work, inner, _spread(1.0))
                            ops["product"](d_gate, helper, inner)
                            ops["product"](d_gate, d_gate, prob_pair)
                            ops["product"](d_up, step, prob_pair)
                            ops["product"](d_up, d_up, y_gate)
                            # The filters. `gate` keeps `y_gate` at or below the
                            # upper bound and zero above it. `up` runs the two
                            # bounds in the source's order, which leaves the
                            # upper one inert on the packed path -- a raw value
                            # above +7 zeroes the intermediate, and zero passes
                            # the lower test -- so only the lower bound
                            # survives. The scalar path applies both, and the
                            # two disagree above +7; each specialization gets
                            # the arm its `vectorized_f32` selects.
                            keep = scratch["c"]
                            for half in range(2):
                                predicate = txl.local_scalar("bool")
                                txl.assign(
                                    predicate, txl.cast(gate[half] <= txl.float32(GEGLU_MAX), "bool")
                                )
                                txl.ptx["selp.f32"](
                                    keep[half], y_gate[half], txl.float32(0.0), predicate
                                )
                            ops["product"](d_gate, d_gate, keep)
                            for half in range(2):
                                predicate = txl.local_scalar("bool")
                                if vectorized_f32:
                                    txl.assign(
                                        predicate, txl.cast(up[half] >= txl.float32(GEGLU_MIN), "bool")
                                    )
                                    txl.ptx["selp.f32"](
                                        keep[half], y_up[half], txl.float32(0.0), predicate
                                    )
                                else:
                                    txl.assign(
                                        predicate,
                                        txl.cast(
                                            txl.And(
                                                up[half] >= txl.float32(GEGLU_MIN),
                                                up[half] <= txl.float32(GEGLU_MAX),
                                            ),
                                            "bool",
                                        ),
                                    )
                                    txl.ptx["selp.f32"](
                                        keep[half], up[half], txl.float32(0.0), predicate
                                    )
                            ops["product"](d_up, d_up, keep)

                        for half in range(2):
                            txl.assign(rC1[span[half]], d_gate[half])
                            txl.assign(rC2[span[half]], d_up[half])

                    folded_prob = txl.local_scalar("float32")
                    txl.ptx["add.f32"](folded_prob, dprob_pair[0], dprob_pair[1])
                    txl.ptx["add.f32"](dprob_acc, dprob_acc, folded_prob)

                    # ---- convert D and stage it through SMEM -------------
                    # An FP32 D is already the gradient's own representation, so
                    # the packing is an identity copy and the staging stores read
                    # the gradient registers directly. The conversion registers
                    # exist only for a narrower D.
                    if d_bits != 32:
                        _pack_output(rD1, rC1, d_dtype, d_bits)
                        _pack_output(rD2, rC2, d_dtype, d_bits)

                    with txl.If(warp == txl.int32(0)), txl.Then():
                        # The D pipeline is a bulk-group counter, not an
                        # mbarrier: there is no D barrier in the storage map.
                        txl.ptx["cp.async.bulk.wait_group.read"](txl.uint32(0))
                    txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))

                    slot1 = txl.local_scalar("int32", init=prev_subtiles % txl.int32(d_stages))
                    txl.assign(prev_subtiles, prev_subtiles + txl.int32(1))
                    slot2 = txl.local_scalar("int32", init=prev_subtiles % txl.int32(d_stages))
                    txl.assign(prev_subtiles, prev_subtiles + txl.int32(1))

                    d_row_bytes = 32 * d_bits // 8

                    def stage_fragment(fragment, slot, region="sD"):
                        # Mirror of the C load: this thread owns one row of the
                        # 128x32 subtile.
                        base = offsets[region] + slot * txl.int32(derived["d_stage_bytes"])
                        d_row = txl.local_scalar("int32", init=txl.thread_id() * txl.int32(d_row_bytes))
                        for word in range(0, d_words, 4):
                            # Written through the same permutation the store
                            # descriptor reads back.
                            txl.ptx["st.shared.v4.b32"](
                                smem.ptr_to([base + _swizzled(d_row, word // 4, d_row_bytes)]),
                                fragment[word],
                                fragment[word + 1],
                                fragment[word + 2],
                                fragment[word + 3],
                            )

                    # Both halves of one slot go out together, D then D_col.
                    stage_fragment(rD1, slot1)
                    stage_fragment(rD2, slot2)
                    txl.ptx["fence.proxy.async.shared::cta"]()
                    txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))

                    with txl.If((warp == txl.int32(0)) & (txl.lane_id() == 0)), txl.Then():
                        # The two D subtiles of one accumulator subtile land in
                        # adjacent halves of the 2N region, which is why the
                        # column index is 2 * real_subtile + {0, 1}.
                        d_row_coord = txl.local_scalar(
                            "int32", init=row_base + epi_tile_m * txl.int32(cta_tile_m)
                        )
                        for map_name, region in (("d_row", "sD"),):
                            for half, slot in ((0, slot1), (1, slot2)):
                                column = txl.local_scalar(
                                    "int32",
                                    init=d_tile_base
                                    + (real_subtile * txl.int32(2) + txl.int32(half)) * txl.int32(32),
                                )
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.global.shared::cta.tile"
                                    ".bulk_group.L2::cache_hint"
                                ](
                                    txl.address_of(maps[map_name]),
                                    txl.cast(column, "int32"),
                                    txl.cast(d_row_coord, "int32"),
                                    txl.int32(0),
                                    smem.ptr_to(
                                        [offsets[region] + slot * txl.int32(derived["d_stage_bytes"])]
                                    ),
                                    txl.uint64(0),
                                )
                        txl.ptx["cp.async.bulk.commit_group"]()
                    # The dBias column sums are taken with the tile's D store
                    # already in flight. They read the activation fragments,
                    # which the packing only copied, and they touch their own
                    # shared region, so nothing in them depends on the store.
                    # The placement has to be after the *barrier* that precedes
                    # the store, not merely after the shared writes: ptxas is
                    # free to hoist across plain stores and is not free to hoist
                    # across `bar.sync`, so the earlier position bought nothing.
                    if generate_dbias:
                        dbias_transpose()
                        dbias_reduce(real_subtile, d_tile_base)
                    txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))
                    txl.assign(real_subtile, real_subtile + txl.int32(1))
                    txl.assign(subtile, subtile + txl.int32(1))

                # Release the accumulator stage once every subtile has been read
                # out of it. One elected lane per epilogue warp arrives, and
                # both CTAs of a pair arrive on the *leader's* barrier, which is
                # where the accumulator lives -- hence its `4 * atom_thr`
                # arrivals. Releasing locally leaves the leader's MMA waiting on
                # a stage nobody frees.
                with txl.If(_elected()), txl.Then():
                    peer_bar = txl.local_scalar("uint32")
                    local_bar = txl.local_scalar("uint32")
                    txl.assign(
                        local_bar,
                        txl.cuda.cvta_generic_to_shared(acc_pipe.empty.ptr_to([acc_cons.stage])),
                    )
                    txl.ptx["mapa.shared::cluster.u32"](
                        peer_bar,
                        local_bar,
                        txl.cast(
                            cluster_rank - cluster_v if atom_thr > 1 else cluster_rank, "uint32"
                        ),
                    )
                    txl.ptx["mbarrier.arrive.shared::cluster.b64"](peer_bar, txl.uint32(1))
                acc_cons.advance()

                # The next record is taken before the dProb tail, as the source
                # does, so the tile-info slot is freed as early as possible.
                take_tile_info(epi_info, epi_slots)
                # One reduction per thread per work tile. The returning form
                # would put every one of these on the scoreboard for a value that
                # is immediately discarded.
                txl.ptx["red.global.add.f32"](named["dprob"].ptr_to([thread_row]), dprob_acc)

            # Release the TMEM allocation permit, then synchronize the four
            # epilogue warps, then free the columns. Both tensor-memory
            # instructions are warp 0's -- each is `.sync.aligned`, and the
            # export predicates both on the allocator warp -- but the permit
            # goes *before* the barrier and the deallocation after it.
            with txl.If(warp == txl.int32(0)), txl.Then():
                txl.ptx[f"tcgen05.relinquish_alloc_permit.{cta_group}.sync.aligned"]()
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))
            with txl.If(warp == txl.int32(0)), txl.Then():
                if atom_thr > 1:
                    # A CTA pair frees its TMEM collectively: each CTA arrives on
                    # its peer's deallocation barrier and waits on its own before
                    # issuing the free.
                    peer_dealloc = txl.local_scalar("uint32")
                    own_dealloc = txl.local_scalar("uint32")
                    txl.assign(own_dealloc, txl.cuda.cvta_generic_to_shared(tmem_dealloc.ptr_to([0])))
                    txl.ptx["mapa.shared::cluster.u32"](
                        peer_dealloc, own_dealloc, txl.cast(cluster_rank ^ txl.int32(1), "uint32")
                    )
                    txl.ptx["mbarrier.arrive.shared::cluster.b64"](peer_dealloc, txl.uint32(1))
                    _wait_plain(tmem_dealloc.ptr_to([0]), txl.uint32(0))
                txl.ptx[f"tcgen05.dealloc.{cta_group}.sync.aligned.b32"](
                    tmem_base, txl.uint32(derived["num_tmem_alloc_cols"])
                )
            txl.ptx["cp.async.bulk.wait_group.read"](txl.uint32(0))

        del (
            block_x,
            lane,
            epilogue_role,
            mma_role,
            c_role,
            a_descriptor,
            b_descriptor,
            ab_consumer_mask,
            acc_producer_mask,
            cluster_smem_base,
            tmem_slot,
        )

    def build_helper():
        """Pre-kernel launched before the main kernel on the same stream.

        For the dynamic scheduler it resets the global work counter. For discrete
        weights each block publishes one expert's B TensorMap image into the
        workspace, which is where the main kernel's TMA reads it from.

        A discrete expert's weights are its own allocation, so only the global
        address differs between the experts' images (probe/descriptor_images.txt
        shows the other fifteen words identical across all four). The host
        prelude therefore encodes one template -- correct in every field but the
        address -- and each block copies it and patches the address with
        ``tensormap.replace``.
        """

        def helper_prelude(params):
            descriptors = {"b": txl.stack_alloca("tensormap", 1)}
            # The template's address is a placeholder: it has to be a real
            # 16-byte-aligned allocation for the encode call to accept it, and
            # the copy below overwrites it per expert. The pointer array itself
            # is the convenient stand-in.
            encode_weight_maps(descriptors, params["b"].data, 1)
            return (descriptors["b"],)

        def read_tensormap_image(source_map):
            """Read one TensorMap image out of the host-encoded template.

            Only the first 64 bytes of a 128-byte image carry anything -- the
            encoder leaves the tail zero (probe/descriptor_images.txt) and the
            workspace starts zeroed -- so the image moves as two
            `ld.global.v4.b64` / `st.global.v4.b64` pairs.

            The source builds these words as immediates because CuTeDSL
            constructs the descriptor inside the device compiler. TIRx's encoder
            is the host-side `cuTensorMapEncodeTiled`, which cannot produce
            immediates at trace time, so the words are read from the template the
            host prelude encoded. That read is the one part of this that the
            source has no counterpart for.
            """
            source: txl.uint64 = txl.reinterpret("uint64", txl.address_of(source_map))
            groups = []
            for group in range(2):
                payload = txl.alloc_local((4,), "uint64")
                offset: txl.uint64 = txl.uint64(group * 32)
                txl.ptx.ld.global_.v4.b64(
                    payload[0],
                    payload[1],
                    payload[2],
                    payload[3],
                    txl.reinterpret("handle", source + offset),
                )
                groups.append(payload)
            return groups

        def write_tensormap_image(groups, destination, address):
            """Publish one image, substituting word 0, the global address."""
            target: txl.uint64 = txl.reinterpret("uint64", destination)
            for group, payload in enumerate(groups):
                if group == 0:
                    txl.assign(payload[0], address)
                txl.ptx.st.global_.v4.b64(
                    txl.reinterpret("handle", target + txl.uint64(group * 32)),
                    payload[0],
                    payload[1],
                    payload[2],
                    payload[3],
                )

        def helper(operands, host):
            b, workspace = operands
            expert = txl.cta_id()[0]
            if weight_mode == "discrete":
                slot = txl.local_scalar("int32", init=txl.int32(128) * expert)
                with txl.If(_elected()), txl.Then():
                    # This kernel is nothing but memory latency: three global
                    # reads and two writes on one lane. The expert pointer is
                    # loaded before the template image so the two round trips
                    # overlap instead of adding.
                    address = txl.local_scalar("uint64")
                    txl.ptx.ld.global_.b64(address, b.ptr_to([expert]))
                    groups = read_tensormap_image(host[0])
                    write_tensormap_image(groups, workspace.ptr_to([slot]), address)
                txl.cuda.warp_sync()
            if sched == "dynamic":
                counter_offset = L * 128 if weight_mode == "discrete" else 0
                with txl.If(expert == txl.int32(0)), txl.Then():
                    with txl.If(_elected()), txl.Then():
                        txl.ptx.st.global_.b32(workspace.ptr_to([counter_offset]), txl.int32(0))

        # The launch passes the per-expert pointer array for discrete weights
        # and `padded_offsets` (int32) for dense ones, so the first parameter
        # changes dtype with the mode.
        pointer_dtype = "int64" if weight_mode == "discrete" else "int32"
        if weight_mode == "discrete":
            helper_body = _entry_point(["b", "workspace"], helper)
        else:
            # Only the discrete branch has a host prelude, and an entry may not
            # take the keyword-only `host` parameter without one.
            def helper_body(b, workspace):
                helper((b, workspace), ())

        helper_body.__annotations__ = {
            "b": txl.gptr[pointer_dtype, (L,)],
            "workspace": txl.gptr[txl.u8, (max(1, derived["workspace_bytes"]),)],
        }
        return txl.kernel(
            warps=1,
            arch="sm_100a",
            min_blocks_per_sm=1,
            grid=list(derived["helper_grid"]),
            host_prelude=helper_prelude if weight_mode == "discrete" else None,
        )(helper_body)

    kernel = _entry_point(list(annotations), body)
    kernel.__annotations__ = dict(annotations)
    main = txl.kernel(
        warps=8,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid=list(derived["grid"]),
        host_prelude=host_prelude,
    )(kernel)
    if derived["needs_helper"]:
        return [build_helper().func, main.func]
    return [main.func]


def get_kernel(**config):
    config = {key: value for key, value in config.items() if key != "label"}
    return _make_kernel(
        group_m_list=tuple(config["group_m_list"]),
        N=config["N"],
        K_dim=config["K_dim"],
        weight_mode=config["weight_mode"],
        sched=config["sched"],
        act=config["act"],
        c_dtype=config["c_dtype"],
        d_dtype=config["d_dtype"],
        b_major=config["b_major"],
        mma_tiler_mn=tuple(config["mma_tiler_mn"]),
        cluster_shape_mn=tuple(config["cluster_shape_mn"]),
        vectorized_f32=config["vectorized_f32"],
        with_dbias=config["with_dbias"],
        linear_offset=config["linear_offset"],
    )
