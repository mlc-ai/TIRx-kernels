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

import tirx_kernels.kern as K

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
    uniform = K.local_scalar("int32")
    K.ptx["shfl_sync.idx.b32"](uniform, value, K.uint32(0), K.uint32(31), K.uint32(0xFFFFFFFF))
    return uniform


def _elected():
    """`elect.sync` over the full warp, the source's single-issuer predicate."""
    elected_lane = K.local_scalar("uint32")
    elected_pred = K.local_scalar("uint32")
    K.ptx.elect_sync(elected_lane, elected_pred, K.uint32(0xFFFFFFFF))
    return elected_pred == K.uint32(1)


def _try_wait_acquire(dst, barrier, phase):
    """The non-blocking look-ahead peek: acquire scope, and no suspend hint.

    Four of these in the anchor export, at `PTX 863, 919, 1170, 1241`; their
    status predicates the blocking acquire that follows.
    """
    K.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
        dst, barrier, K.cast(phase, "uint32")
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
    ready = K.local_scalar("uint32", init=K.uint32(0))
    with K.While(ready == K.uint32(0)):
        _try_wait_acquire(ready, barrier, phase)


def _wait_plain_if_needed(barrier, phase, speculative_ready):
    with K.If(speculative_ready == K.uint32(0)):
        with K.Then():
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
            K.ptx[f"{mnemonic}.rn.f32x2"](
                packed, K.cuda.make_float2(left[0], left[1]), K.cuda.make_float2(right[0], right[1])
            )
            K.ptx["mov.b64"](out[0], out[1], packed)
        else:
            for half in range(2):
                K.ptx[f"{mnemonic}.f32"](out[half], left[half], right[half])

    def unary(mnemonic, out, value):
        for half in range(2):
            K.ptx[mnemonic](out[half], value[half])

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
            K.ptx["fma.rn.f32x2"](
                packed,
                K.cuda.make_float2(left[0], left[1]),
                K.cuda.make_float2(right[0], right[1]),
                K.cuda.make_float2(addend[0], addend[1]),
            )
            K.ptx["mov.b64"](out[0], out[1], packed)
        else:
            for half in range(2):
                K.ptx["fma.rn.f32"](out[half], left[half], right[half], addend[half])

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
                K.ptx["sub.f32"](out[half], K.float32(constant), value[half])

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
    return (K.float32(constant), K.float32(constant))


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
        return row_offset + K.int32(chunk * 16)
    # Every caller passes `row * row_bytes`, so the low `log2(row_bytes)` bits
    # of `row_offset` are zero and both `chunk * 16` and the twist fit entirely
    # inside them. The chunk therefore ORs in rather than adds, and the twist
    # only ever touches the chunk field -- which also means the twist can be
    # read off `row_offset` alone. Writing the whole thing as
    # `row_offset | (chunk ^ twist)` is one `LOP3` per chunk where the sum
    # followed by the XOR is two instructions, and this runs once per chunk per
    # fragment on every subtile.
    twist = K.local_scalar(
        "int32", init=((row_offset // K.int32(128)) % K.int32(chunks)) * K.int32(16)
    )
    return K.local_scalar("int32", init=row_offset | (K.int32(chunk * 16) ^ twist))


def _tcgen05_commit(barrier, mask, cta_group, cluster_size):
    """Asynchronous mbarrier arrival from the MMA pipeline.

    The multicast form carries the CTA mask; a singleton cluster has no peers to
    notify and takes the plain form.
    """
    if cluster_size > 1:
        K.ptx[
            f"tcgen05.commit.{cta_group}.mbarrier::arrive::one"
            ".shared::cluster.multicast::cluster.b64"
        ](barrier, K.cast(mask, "uint16"))
    else:
        K.ptx[f"tcgen05.commit.{cta_group}.mbarrier::arrive::one.shared::cluster.b64"](barrier)


def _unpack_input(values, words, dtype, bits):
    """Expand one thread's row of an epilogue tile into 32 FP32 values."""
    if bits == 16:
        converter = "cvt.f32.bf16" if dtype == "bfloat16" else "cvt.f32.f16"
        for word in range(16):
            low = K.local_scalar("uint16")
            high = K.local_scalar("uint16")
            K.ptx["mov.b32"](low, high, words[word])
            K.ptx[converter](values[2 * word + 0], low)
            K.ptx[converter](values[2 * word + 1], high)
    else:
        for word in range(32):
            K.ptx["mov.b32"](values[word], words[word])


def _pack_output(words, values, dtype, bits):
    """Pack a 32-element FP32 fragment into the output dtype's storage words.

    A 16-bit output uses `cvt.rn.bf16x2.f32` / `cvt.rn.f16x2.f32`, two elements
    to a word; FP32 is a move.
    """
    if bits == 16:
        converter = "cvt.rn.bf16x2.f32" if dtype == "bfloat16" else "cvt.rn.f16x2.f32"
        for word in range(16):
            K.ptx[converter](words[word], values[2 * word + 1], values[2 * word + 0])
    else:
        for word in range(32):
            K.ptx["mov.b32"](words[word], values[word])


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
    address_field = K.cast(
        K.bitwise_and(K.shift_right(shared_address, K.uint32(4)), K.uint32(0x3FFF)), "uint64"
    )
    return K.bitwise_or(K.uint64(base), address_field)


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
    address = K.address_of(descriptor) if desc_ptr is None else desc_ptr
    arguments = [destination, address, *[K.cast(c, "int32") for c in coords], barrier]
    if mask is not None:
        arguments.append(K.cast(mask, "uint16"))
    arguments.append(K.uint64(0))
    K.ptx[stem](*arguments)


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
        "a": K.gptr[K.u8, (byte_count(tokens_total, K_dim, ab_bits),)],
        "b": (
            K.gptr["int64", (L,)]
            if weight_mode == "discrete"
            else K.gptr[K.u8, (byte_count(N, K_dim, ab_bits) * L,)]
        ),
        "c": K.gptr[K.u8, (byte_count(tokens_total, N_out, c_bits),)],
        "d_row": K.gptr[K.u8, (byte_count(tokens_total, N_out, d_bits),)],
    }
    annotations["padded_offsets"] = K.gptr["int32", (L,)]
    annotations["alpha"] = K.gptr["float32", (L,)]
    annotations["beta"] = K.gptr["float32", (L,)]
    # ``generate_dprob`` is hard-coded True upstream, so both routing tensors
    # are part of every specialization's signature.
    annotations["prob"] = K.gptr["float32", (tokens_total,)]
    annotations["dprob"] = K.gptr["float32", (tokens_total,)]
    if with_dbias:
        annotations["dbias"] = K.gptr[K.u8, (L * N_out * 2,)]
    if derived["workspace_bytes"]:
        annotations["workspace"] = K.gptr[K.u8, (derived["workspace_bytes"],)]

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
        K.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, rank, data, *fields)

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
        descriptors = {name: K.stack_alloca("tensormap", 1) for name in map_names}
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
        block_x, block_y, cluster_work_id = K.cta_id()
        cluster_x, cluster_y = K.cta_id_in_cluster(
            [cluster_m, cluster_n], preferred=[cluster_m, cluster_n]
        )
        cluster_rank = _warp_uniform(cluster_x + cluster_m * cluster_y)
        del block_y
        warp = _warp_uniform(K.warp_id())
        lane = K.lane_id()

        # Position inside the two-CTA MMA pair, and the pair's coordinate in the
        # cluster. `cluster_v` is what distinguishes the two CTAs of a pair and
        # has to appear in every multicast mask.
        cluster_v = cluster_rank % atom_thr if atom_thr > 1 else 0
        cluster_m_coord = (cluster_rank // atom_thr) % max(1, cluster_m // atom_thr)
        cluster_n_coord = cluster_rank // cluster_m
        # Only the leader CTA of a two-CTA pair issues the MMA and the AB-full
        # arrival; with a single-CTA atom every CTA is its own leader.
        is_leader_cta = (
            (block_x % K.int32(atom_thr)) == K.int32(0)
            if atom_thr > 1
            else K.int32(0) == K.int32(0)
        )

        roles = K.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4])
        tma_role = roles.role("tma", warps=[5])
        c_role = roles.role("c_load", warps=[6])
        sched_role = roles.role("scheduler", warps=[7])

        # ---- storage -----------------------------------------------------
        smem = K.alloc_buffer((shared_bytes,), K.u8, scope="shared.dyn", align=1024)
        protocol_pool = K.smem_pool(base=smem)
        ab_pipe = K.Pipeline(
            protocol_pool, ab_stages, full="tma", empty="tcgen05", leader=K.bool(False)
        )
        acc_pipe = K.Pipeline(
            protocol_pool,
            acc_stages,
            full="tcgen05",
            empty="mbar",
            init_empty=4 * atom_thr,
            leader=K.bool(False),
        )
        tile_pipe = K.Pipeline(
            protocol_pool, tile_stages, full="mbar", empty="mbar", leader=K.bool(False)
        )
        if protocol_pool.bytes != offsets["sinfo"]:
            raise AssertionError("protocol storage order changed before sInfo")
        sinfo = protocol_pool.alloc((4 * tile_stages,), K.i32, align=16)
        # The dynamic scheduler adds a one-stage cluster pipeline and the
        # 16-byte slot the elected CTA broadcasts the next work index into.
        # Both sit straight after sInfo, which is why every object below them
        # shifts by 32 bytes on that branch.
        sched_pipe = None
        sched_broadcast = None
        if sched == "dynamic":
            if protocol_pool.bytes != offsets["cluster_mbar"]:
                raise AssertionError("scheduler cluster pipeline is misplaced")
            sched_pipe = K.Pipeline(
                protocol_pool, 1, full="mbar", empty="mbar", leader=K.bool(False)
            )
            if protocol_pool.bytes != offsets["cluster_broadcast"]:
                raise AssertionError("scheduler broadcast slot is misplaced")
            sched_broadcast = protocol_pool.alloc((4,), K.i32, align=16)
        if protocol_pool.bytes != offsets["c_full"]:
            raise AssertionError("protocol storage order changed before the C pipeline")
        c_pipe = K.Pipeline(
            protocol_pool, c_stages, full="tma", empty="mbar", init_empty=4, leader=K.bool(False)
        )
        if protocol_pool.bytes != offsets["tmem_dealloc"]:
            raise AssertionError("protocol storage order changed before the TMEM barrier")
        tmem_dealloc = protocol_pool.alloc((1,), K.u64, align=8)
        tmem_slot = protocol_pool.alloc((1,), K.u32, align=4)

        # ---- descriptor prefetch ----------------------------------------
        with tma_role:
            for name in map_names:
                K.ptx.prefetch.tensormap(K.address_of(maps[name]))

        # ---- barrier initialization, in the source's declaration order ----
        with K.If(warp == 0):
            with K.Then():
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, ab_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                ab_pipe.full.ptr_to([stage]), K.uint32(1)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, ab_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                ab_pipe.empty.ptr_to([stage]), K.uint32(ab_empty_arrivals)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, acc_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                acc_pipe.full.ptr_to([stage]), K.uint32(1)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, acc_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                acc_pipe.empty.ptr_to([stage]), K.uint32(4 * atom_thr)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, c_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(c_pipe.full.ptr_to([stage]), K.uint32(1))
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, c_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                c_pipe.empty.ptr_to([stage]), K.uint32(4)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, tile_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                tile_pipe.full.ptr_to([stage]), K.uint32(32)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, tile_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                tile_pipe.empty.ptr_to([stage]), K.uint32(224)
                            )

        # The tile-info pipeline is built without `defer_sync`, so it publishes
        # itself here; the TMEM barrier belongs to the second epoch below.
        K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.sync(K.uint32(0))

        # `TmemAllocator` is handed the deallocation barrier only under a
        # two-CTA atom, so a single-CTA specialization initializes one mbarrier
        # fewer: the exports count 23 for the anchor and 18 for `tile128_c1x1`,
        # one short of the 19 an unconditional init would leave.
        if sched == "dynamic" or atom_thr > 1:
            with K.If(warp == 0):
                with K.Then():
                    with K.If(_elected()):
                        with K.Then():
                            if sched == "dynamic":
                                # `internal_init` builds this pipeline with
                                # `defer_sync=True`, so it is published by the
                                # second epoch alongside the TMEM barrier.
                                K.ptx.mbarrier.init.shared.b64(
                                    sched_pipe.full.ptr_to([0]), K.uint32(1)
                                )
                                K.ptx.mbarrier.init.shared.b64(
                                    sched_pipe.empty.ptr_to([0]), K.uint32(32 * cluster_size)
                                )
                            if atom_thr > 1:
                                K.ptx.mbarrier.init.shared.b64(
                                    tmem_dealloc.ptr_to([0]), K.uint32(32)
                                )
        K.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            K.ptx.barrier.cluster.arrive.relaxed()

        # ---- scalar addresses, descriptors, and multicast masks ----------
        smem_base = K.local_scalar("uint32")
        K.assign(smem_base, K.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        cluster_smem_base_u64 = K.local_scalar("uint64")
        K.ptx.cvta.to.shared__cluster.u64(cluster_smem_base_u64, smem.ptr_to([0]))
        cluster_smem_base = K.local_scalar("uint32", init=K.cast(cluster_smem_base_u64, "uint32"))
        a_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(a_desc_base, smem_base + offsets["sA"])
        )
        b_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(b_desc_base, smem_base + offsets["sB"])
        )

        # Every mask is the image of the vmnk cluster layout at this CTA's
        # coordinate with one mode varying; the flat rank of `(v, m, n)` is
        # `v + atom_thr * m + cluster_m * n`.
        def cta_bit(v, m, n):
            return K.uint32(1) << K.cast(v + atom_thr * m + cluster_m * n, "uint32")

        def mask_union(bits):
            accumulator = K.local_scalar("uint32", init=K.uint32(0))
            for bit in bits:
                K.assign(accumulator, K.bitwise_or(accumulator, bit))
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
            K.ptx.barrier.cluster.wait()
        else:
            K.ptx.bar.sync(K.uint32(1))

        total_tokens = K.local_scalar("int32")
        K.ptx.ld.global_.b32(total_tokens, named["padded_offsets"].ptr_to([L - 1]))
        with K.If(total_tokens <= K.int32(0)), K.Then():
            K.Return(K.int32(0))

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
            upper = K.local_scalar("int32")
            K.ptx.ld.global_.b32(upper, named["padded_offsets"].ptr_to([expert]))
            tokens = K.local_scalar("int32", init=upper)
            with K.If(expert > K.int32(0)), K.Then():
                lower = K.local_scalar("int32")
                K.ptx.ld.global_.b32(lower, named["padded_offsets"].ptr_to([expert - K.int32(1)]))
                K.assign(tokens, tokens - lower)
            return tokens

        def expert_m_tiles(expert):
            tokens = expert_tokens(expert)
            return (tokens + K.int32(cluster_tile_m - 1)) // K.int32(cluster_tile_m)

        with sched_role:
            info_prod = K.PipelineState(tile_stages, phase=1)
            # Cached expert cursor: the walk stays inside one expert until the
            # linear index leaves it, which makes the common step O(1).
            current_expert = K.local_scalar("int32", init=K.int32(0))
            expert_tile_start = K.local_scalar("int32", init=K.int32(0))
            expert_tile_end = K.local_scalar("int32", init=K.int32(0))
            initialized = K.local_scalar("uint32", init=K.uint32(0))
            work_linear = K.local_scalar("int32", init=cluster_work_id)

            record_expert = K.local_scalar("int32")
            record_tile_m = K.local_scalar("int32")
            record_tile_n = K.local_scalar("int32")
            work_valid = K.local_scalar("uint32")

            def resolve_work():
                """Source `_get_work_tile_for_linear_idx`, cursor included."""
                with K.If(initialized == K.uint32(0)), K.Then():
                    K.assign(expert_tile_end, expert_m_tiles(K.int32(0)) * n_tile_count)
                    K.assign(initialized, K.uint32(1))
                with K.While(K.And(work_linear >= expert_tile_end, current_expert < K.int32(L))):
                    K.assign(current_expert, current_expert + K.int32(1))
                    K.assign(expert_tile_start, expert_tile_end)
                    with K.If(current_expert < K.int32(L)), K.Then():
                        K.assign(
                            expert_tile_end,
                            expert_tile_end + expert_m_tiles(current_expert) * n_tile_count,
                        )
                K.assign(record_expert, K.int32(-1))
                K.assign(record_tile_m, K.int32(0))
                K.assign(record_tile_n, K.int32(0))
                K.assign(work_valid, K.uint32(0))
                with K.If(current_expert < K.int32(L)), K.Then():
                    K.assign(work_valid, K.uint32(1))
                    local_idx = K.local_scalar("int32", init=work_linear - expert_tile_start)
                    m_count = K.local_scalar("int32", init=expert_m_tiles(current_expert))
                    cluster_m_idx = K.local_scalar("int32")
                    cluster_n_idx = K.local_scalar("int32")
                    # Short side first: the shorter extent changes faster, so
                    # neighbouring clusters overlap in L2.
                    with K.If(m_count <= K.int32(n_tile_count)):
                        with K.Then():
                            K.assign(cluster_m_idx, local_idx % m_count)
                            K.assign(cluster_n_idx, local_idx // m_count)
                        with K.Else():
                            K.assign(cluster_n_idx, local_idx % K.int32(n_tile_count))
                            K.assign(cluster_m_idx, local_idx // K.int32(n_tile_count))
                    K.assign(record_expert, current_expert)
                    K.assign(record_tile_m, cluster_m_idx * K.int32(cluster_m) + cluster_x)
                    K.assign(record_tile_n, cluster_n_idx * K.int32(cluster_n) + cluster_y)

            def publish_record(expert_value, tile_m_value, tile_n_value):
                _wait_plain(tile_pipe.empty.ptr_to([info_prod.stage]), info_prod.phase)
                with K.If(_elected()), K.Then():
                    base = info_prod.stage * 4
                    K.ptx.st.shared.v4.b32(
                        sinfo.ptr_to([base]),
                        expert_value,
                        tile_m_value,
                        tile_n_value,
                        K.int32(k_tiles),
                    )
                K.ptx["fence.proxy.async.shared::cta"]()
                K.ptx.bar.sync(K.uint32(4), K.uint32(32))
                # All 32 lanes of the scheduler warp arrive, matching the
                # barrier's 32 arrivals.
                K.ptx.mbarrier.arrive.shared.b64(tile_pipe.full.ptr_to([info_prod.stage]))
                info_prod.advance()

            if sched == "dynamic":
                # The linear index comes from a global atomic instead of a
                # stride: the leader CTA claims the next one and broadcasts it
                # into every peer's shared slot, so a whole cluster agrees on
                # one work tile.
                counter_offset = L * 128 if weight_mode == "discrete" else 0
                sched_prod = K.PipelineState(1, phase=1)
                sched_cons = K.PipelineState(1, phase=0)
                is_leader_cluster = cluster_rank == K.int32(0)

                def fetch_single_cta():
                    """The whole cluster is one CTA, so there is nobody to tell.

                    The broadcast below reaches its peers through
                    `mapa.shared::cluster` and `st_async.shared::cluster`, and a
                    launch with no cluster dimension has no such address space --
                    those instructions fault rather than degenerate. A lone CTA
                    just keeps the index it claimed.
                    """
                    claimed = K.local_scalar("uint32", init=K.uint32(0))
                    with K.If(lane == K.int32(0)), K.Then():
                        K.ptx["atom.global.add.u32"](
                            claimed, named["workspace"].ptr_to([counter_offset]), K.uint32(1)
                        )
                    K.ptx["shfl_sync.idx.b32"](
                        claimed, claimed, K.uint32(0), K.uint32(31), K.uint32(0xFFFFFFFF)
                    )
                    K.assign(work_linear, K.cast(claimed, "int32"))

                def fetch_linear():
                    with K.If(is_leader_cluster), K.Then():
                        _wait_plain(sched_pipe.empty.ptr_to([0]), sched_prod.phase)
                        claimed = K.local_scalar("uint32", init=K.uint32(0))
                        with K.If(lane == K.int32(0)), K.Then():
                            K.ptx["atom.global.add.u32"](
                                claimed, named["workspace"].ptr_to([counter_offset]), K.uint32(1)
                            )
                        K.ptx["shfl_sync.idx.b32"](
                            claimed, claimed, K.uint32(0), K.uint32(31), K.uint32(0xFFFFFFFF)
                        )
                        with K.If(lane < K.int32(cluster_size)), K.Then():
                            peer = K.local_scalar("uint32", init=K.cast(lane, "uint32"))
                            remote_slot = K.local_scalar("uint32")
                            remote_bar = K.local_scalar("uint32")
                            local_slot = K.local_scalar("uint32")
                            local_bar = K.local_scalar("uint32")
                            K.assign(
                                local_slot,
                                K.cuda.cvta_generic_to_shared(sched_broadcast.ptr_to([0])),
                            )
                            K.assign(
                                local_bar,
                                K.cuda.cvta_generic_to_shared(sched_pipe.full.ptr_to([0])),
                            )
                            K.ptx["mapa.shared::cluster.u32"](remote_slot, local_slot, peer)
                            K.ptx["mapa.shared::cluster.u32"](remote_bar, local_bar, peer)
                            K.ptx["st_async.shared::cluster.mbarrier::complete_tx::bytes.u32"](
                                remote_slot, claimed, remote_bar
                            )
                            K.ptx["mbarrier.arrive.expect_tx.shared::cluster.b64"](
                                remote_bar, K.uint32(4)
                            )
                    sched_prod.advance()
                    _wait_plain(sched_pipe.full.ptr_to([0]), sched_cons.phase)
                    fetched = K.local_scalar("int32")
                    K.ptx.ld.shared.b32(fetched, sched_broadcast.ptr_to([0]))
                    # Every CTA's scheduler warp releases onto the *leader's*
                    # empty barrier, which is why it is initialized with
                    # `32 * cluster_size` arrivals. Releasing locally leaves each
                    # CTA 32 short and the leader blocks on its next acquire.
                    release_local = K.local_scalar("uint32")
                    release_peer = K.local_scalar("uint32")
                    K.assign(
                        release_local, K.cuda.cvta_generic_to_shared(sched_pipe.empty.ptr_to([0]))
                    )
                    K.ptx["mapa.shared::cluster.u32"](release_peer, release_local, K.uint32(0))
                    K.ptx["mbarrier.arrive.shared::cluster.b64"](release_peer, K.uint32(1))
                    sched_cons.advance()
                    K.assign(work_linear, fetched)

                claim_next = fetch_linear if cluster_size > 1 else fetch_single_cta
                claim_next()
                resolve_work()
                with K.While(work_valid == K.uint32(1)):
                    publish_record(record_expert, record_tile_m, record_tile_n)
                    claim_next()
                    resolve_work()
            else:
                resolve_work()
                with K.While(work_valid == K.uint32(1)):
                    publish_record(record_expert, record_tile_m, record_tile_n)
                    K.assign(work_linear, work_linear + K.int32(num_persistent_clusters))
                    resolve_work()

            # Termination record: expert_idx = -1 stops the other seven warps.
            publish_record(K.int32(-1), K.int32(0), K.int32(0))
            with K.unroll(0, tile_stages):
                _wait_plain(tile_pipe.empty.ptr_to([info_prod.stage]), info_prod.phase)
                info_prod.advance()

        # ---- consumer preamble shared by roles 2-5 -----------------------
        # Every consumer keeps its own cursor over the identical record stream.
        # The MMA warp needs only the expert index; nobody reads field 3, since
        # all four roles use the `k_tiles` computed once on the host.
        def take_tile_info(state, slots, want_tiles=True):
            _wait_plain(tile_pipe.full.ptr_to([state.stage]), state.phase)
            base = state.stage * 4
            K.ptx.ld.shared.b32(slots[0], sinfo.ptr_to([base]))
            if want_tiles:
                K.ptx.ld.shared.b32(slots[1], sinfo.ptr_to([base + 1]))
                K.ptx.ld.shared.b32(slots[2], sinfo.ptr_to([base + 2]))
            K.ptx["fence.proxy.async.shared::cta"]()
            # Every consumer thread arrives; the barrier carries 224 arrivals,
            # one per thread of the seven consumer warps.
            K.ptx.mbarrier.arrive.shared.b64(tile_pipe.empty.ptr_to([state.stage]))
            state.advance()

        def expert_row_base(expert):
            """First padded row of an expert; the offsets are cumulative."""
            base = K.local_scalar("int32", init=K.int32(0))
            with K.If(expert > K.int32(0)), K.Then():
                previous = K.local_scalar("int32")
                K.ptx.ld.global_.b32(
                    previous, named["padded_offsets"].ptr_to([expert - K.int32(1)])
                )
                K.assign(base, previous)
            return base

        # ---- Role 2: warp 5, persistent TMA producer ---------------------
        with tma_role:
            ab_prod = K.PipelineState(ab_stages, phase=1)
            # A second cursor kept one step ahead, so the next stage's
            # speculative probe can be issued before this stage's copies and
            # overlap their latency, as the source does.
            ab_probe = K.PipelineState(ab_stages, phase=1)
            ab_probe.advance()
            info_cons = K.PipelineState(tile_stages, phase=0)
            tile_expert = K.local_scalar("int32")
            tile_m_idx = K.local_scalar("int32")
            tile_n_idx = K.local_scalar("int32")
            slots = (tile_expert, tile_m_idx, tile_n_idx)
            speculative = K.local_scalar("uint32")

            take_tile_info(info_cons, slots)
            with K.While(tile_expert >= K.int32(0)):
                row_base = expert_row_base(tile_expert)
                a_row = K.local_scalar("int32", init=row_base + tile_m_idx * K.int32(cta_tile_m))
                if cluster_n > 1:
                    K.assign(a_row, a_row + cluster_y * K.int32(a_cluster_piece))
                # The CTA pair splits B's N extent, so each CTA takes its own
                # half: the export's B coordinate carries the same `cta_in_pair
                # * 128` term that A's M coordinate does (PTX 920/926, both
                # built from `%r5 << 7`).
                n_base = K.local_scalar("int32", init=tile_n_idx * K.int32(cta_tile_n))
                if atom_thr > 1:
                    K.assign(n_base, n_base + cluster_v * K.int32(b_tile_n))
                # On top of the pair split, the cluster's M extent multicasts B:
                # each CTA fetches one `b_cluster_piece` slice and the pieces
                # together fill the stage. The destination address already
                # carried this term; the coordinate did not, so every CTA in the
                # M direction fetched the *same* slice into a different quarter
                # of SMEM and the tile's upper columns duplicated its lower
                # ones. A k-major B splits along N, an n-major B along K.
                if b_split > 1 and b_major == "k":
                    K.assign(n_base, n_base + cluster_m_coord * K.int32(b_cluster_piece))

                # Descriptor addresses: dense weights use the grid-constant
                # TensorMaps, discrete weights the per-expert image the
                # pre-kernel wrote into the workspace.
                if weight_mode == "discrete":
                    # The pre-kernel wrote this expert's B TensorMap image here;
                    # the TMA reads the descriptor straight out of global memory
                    # instead of from a grid constant. One 128-byte slot per
                    # expert -- the block-scaled sibling's second slot held SFB,
                    # which this kernel has no counterpart for.
                    b_desc = named["workspace"].ptr_to([K.int32(128) * tile_expert])

                K.assign(speculative, K.uint32(1))
                with K.If(K.int32(k_tiles) > K.int32(0)), K.Then():
                    _try_wait_acquire(
                        speculative, ab_pipe.empty.ptr_to([ab_prod.stage]), ab_prod.phase
                    )
                counter = K.local_scalar("int32", init=K.int32(0))
                with K.While(counter < K.int32(k_tiles)):
                    _wait_plain_if_needed(
                        ab_pipe.empty.ptr_to([ab_prod.stage]), ab_prod.phase, speculative
                    )
                    # `num_tma_load_bytes` counts the whole CTA pair, so the
                    # leader alone arrives for `atom_thr` stages' worth.
                    with K.If(is_leader_cta), K.Then():
                        with K.If(_elected()), K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                ab_pipe.full.ptr_to([ab_prod.stage]),
                                K.uint32(derived["ab_expect_tx_bytes"]),
                            )
                    with K.If(counter + K.int32(1) < K.int32(k_tiles)), K.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.empty.ptr_to([ab_probe.stage]), ab_probe.phase
                        )

                    k_coord = K.local_scalar("int32", init=counter * K.int32(k_tile))
                    if atom_thr > 1:
                        # A two-CTA copy credits the *leader's* barrier: bit 24
                        # of a cluster shared address selects the CTA within the
                        # pair, and clearing it maps this CTA's barrier onto the
                        # even one. Crediting the local barrier instead leaves
                        # the leader permanently short of its expect_tx count.
                        full_bar = K.local_scalar("uint32")
                        K.assign(
                            full_bar,
                            cluster_smem_base
                            + K.uint32(offsets["ab_full"])
                            + K.cast(ab_prod.stage, "uint32") * K.uint32(8),
                        )
                        K.ptx["and.b32"](full_bar, full_bar, K.uint32(0xFEFFFFFF))
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
                        + K.uint32(offsets["sA"])
                        + K.cast(ab_prod.stage, "uint32") * K.uint32(derived["a_stage_bytes"])
                        + K.cast(cluster_y, "uint32")
                        * K.uint32(derived["a_stage_bytes"] // cluster_n)
                        if cluster_n > 1
                        else a_slot
                    )
                    with K.If(_elected()), K.Then():
                        _tma_load(
                            a_destination,
                            maps["a"],
                            (k_coord, a_row, K.int32(0)),
                            full_bar,
                            a_mcast_mask if cluster_n > 1 else None,
                            two_cta=two_cta,
                        )
                    b_slot = smem.ptr_to([offsets["sB"] + ab_prod.stage * derived["b_stage_bytes"]])
                    b_block_bytes = derived["b_stage_bytes"] // b_tma_copies
                    b_k_coord = k_coord
                    if b_split > 1 and b_major == "n":
                        b_k_coord = K.local_scalar(
                            "int32", init=k_coord + cluster_m_coord * K.int32(b_cluster_piece)
                        )
                    # A discrete descriptor covers one expert's own allocation,
                    # so its batch extent is 1 and the coordinate is 0; the dense
                    # descriptor spans every expert and indexes by expert.
                    b_batch = tile_expert if weight_mode == "dense" else K.int32(0)
                    for block in range(b_tma_copies):
                        b_destination = (
                            cluster_smem_base
                            + K.uint32(offsets["sB"])
                            + K.cast(ab_prod.stage, "uint32") * K.uint32(derived["b_stage_bytes"])
                            + K.uint32(block * b_block_bytes)
                            + K.cast(cluster_m_coord, "uint32") * K.uint32(b_block_bytes // b_split)
                            if (b_split > 1 or b_tma_copies > 1)
                            else b_slot
                        )
                        b_n_coord = (
                            n_base
                            if block == 0
                            else K.local_scalar(
                                "int32", init=n_base + K.int32(block * b_atom_elements)
                            )
                        )
                        with K.If(_elected()), K.Then():
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
                    K.assign(counter, counter + K.int32(1))
                    ab_prod.advance()
                    ab_probe.advance()

                take_tile_info(info_cons, slots)

            with K.unroll(0, ab_stages):
                _wait_plain(ab_pipe.empty.ptr_to([ab_prod.stage]), ab_prod.phase)
                ab_prod.advance()

        # ---- Role 3: warp 4, persistent MMA -------------------------------
        with mma_role:
            K.ptx.bar.sync(K.uint32(3), K.uint32(160))
            acc_tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(acc_tmem_base, tmem_slot.ptr_to([0]))

            ab_cons = K.PipelineState(ab_stages, phase=0)
            ab_cons_probe = K.PipelineState(ab_stages, phase=0)
            ab_cons_probe.advance()
            acc_prod = K.PipelineState(acc_stages, phase=1)
            mma_info = K.PipelineState(tile_stages, phase=0)
            mma_expert = K.local_scalar("int32")
            mma_slots = (mma_expert, None, None)
            ab_full_ready = K.local_scalar("uint32")

            take_tile_info(mma_info, mma_slots, want_tiles=False)
            with K.While(mma_expert >= K.int32(0)):
                K.assign(ab_full_ready, K.uint32(1))
                with K.If(is_leader_cta), K.Then():
                    _try_wait_acquire(
                        ab_full_ready, ab_pipe.full.ptr_to([ab_cons.stage]), ab_cons.phase
                    )
                    _wait_plain(acc_pipe.empty.ptr_to([acc_prod.stage]), acc_prod.phase)
                    accumulate = K.local_scalar("uint32", init=K.uint32(0))
                    # Two real accumulator stages, each its own `cta_tile_n`
                    # columns of tensor memory, indexed by the pipeline stage --
                    # the block-scaled sibling folded a second region into the
                    # slack its scale-factor columns left over, which the derived
                    # column count here removes the need for.
                    acc_column = K.local_scalar(
                        "uint32", init=K.cast(acc_prod.stage, "uint32") * K.uint32(cta_tile_n)
                    )
                    counter = K.local_scalar("int32", init=K.int32(0))
                    with K.While(counter < K.int32(k_tiles)):
                        _wait_plain_if_needed(
                            ab_pipe.full.ptr_to([ab_cons.stage]), ab_cons.phase, ab_full_ready
                        )
                        with K.If(counter + K.int32(1) < K.int32(k_tiles)), K.Then():
                            _try_wait_acquire(
                                ab_full_ready,
                                ab_pipe.full.ptr_to([ab_cons_probe.stage]),
                                ab_cons_probe.phase,
                            )
                        # Four MMA issues per K tile -- the k tile is 64 and
                        # `kind::f16` contracts 16 at a time -- and only the
                        # first clears the accumulate field.
                        for kblock in range(4):
                            with K.If(_elected()), K.Then():
                                K.ptx[mma_mnemonic](
                                    K.cast(acc_tmem_base + acc_column, "uint32"),
                                    a_descriptor
                                    + K.cast(
                                        ab_cons.stage * (derived["a_stage_bytes"] // 16)
                                        + kblock * 2,
                                        "uint64",
                                    ),
                                    # 16 rows of K advance 32 bytes along a
                                    # k-major operand and a whole 2 KiB swizzle
                                    # period along an n-major one; both steps are
                                    # read off the anchor and `bmajor_n` exports.
                                    b_descriptor
                                    + K.cast(
                                        ab_cons.stage * (derived["b_stage_bytes"] // 16)
                                        + kblock * (2 if b_major == "k" else 128),
                                        "uint64",
                                    ),
                                    K.uint32(instruction_descriptor),
                                    # `disable_output_lane`: this kernel writes
                                    # every lane. The mask is one bit per lane
                                    # of the atom, so a CTA pair spells out
                                    # eight zero words and a single CTA four --
                                    # exactly what the two exports carry.
                                    *([K.uint32(0)] * (4 * atom_thr)),
                                    K.ptx.pred(K.cast(accumulate, "bool")),
                                )
                            K.assign(accumulate, K.uint32(1))
                        with K.If(_elected()), K.Then():
                            _tcgen05_commit(
                                ab_pipe.empty.ptr_to([ab_cons.stage]),
                                ab_consumer_mask,
                                cta_group,
                                cluster_size,
                            )
                        K.assign(counter, counter + K.int32(1))
                        ab_cons.advance()
                        ab_cons_probe.advance()
                    with K.If(_elected()), K.Then():
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
            with K.If(is_leader_cta), K.Then():
                with K.unroll(0, acc_stages):
                    _wait_plain(acc_pipe.empty.ptr_to([acc_prod.stage]), acc_prod.phase)
                    acc_prod.advance()

        # ---- Role 5: warp 6, epilogue C producer --------------------------
        # The C subtiles are issued in the same forward order the epilogue reads
        # them, so the two agree on which C subtile belongs to which accumulator
        # subtile with no extra handshake.
        with c_role:
            c_prod = K.PipelineState(c_stages, phase=1)
            c_info = K.PipelineState(tile_stages, phase=0)
            c_expert = K.local_scalar("int32")
            c_tile_m = K.local_scalar("int32")
            c_tile_n = K.local_scalar("int32")
            c_slots = (c_expert, c_tile_m, c_tile_n)

            take_tile_info(c_info, c_slots)
            with K.While(c_expert >= K.int32(0)):
                c_row = K.local_scalar(
                    "int32", init=expert_row_base(c_expert) + c_tile_m * K.int32(cta_tile_m)
                )
                # The 2N interleaved column tile this work tile owns.
                d_tile_n = K.local_scalar("int32", init=c_tile_n * K.int32(cta_tile_n * 2))

                subtile = K.local_scalar("int32", init=K.int32(0))
                with K.While(subtile < K.int32(epi_subtiles)):
                    real_subtile = subtile
                    for half in range(2):
                        _wait_plain(c_pipe.empty.ptr_to([c_prod.stage]), c_prod.phase)
                        with K.If(_elected()), K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                c_pipe.full.ptr_to([c_prod.stage]),
                                K.uint32(derived["c_stage_bytes"]),
                            )
                        column = K.local_scalar(
                            "int32",
                            init=d_tile_n
                            + (real_subtile * K.int32(2) + K.int32(half)) * K.int32(32),
                        )
                        c_slot = smem.ptr_to(
                            [offsets["sC"] + c_prod.stage * derived["c_stage_bytes"]]
                        )
                        with K.If(_elected()), K.Then():
                            _tma_load(
                                c_slot,
                                maps["c"],
                                (column, c_row, K.int32(0)),
                                c_pipe.full.ptr_to([c_prod.stage]),
                                None,
                                two_cta=False,
                            )
                        c_prod.advance()
                    K.assign(subtile, subtile + K.int32(1))

                take_tile_info(c_info, c_slots)

            with K.unroll(0, c_stages):
                _wait_plain(c_pipe.empty.ptr_to([c_prod.stage]), c_prod.phase)
                c_prod.advance()

        # ---- Role 4: warps 0-3, dGLU backward epilogue --------------------
        with epilogue_role:
            with K.If(warp == K.int32(0)), K.Then():
                # `.sync.aligned`, so every lane of warp 0 executes it; the warp
                # predicate is the only guard.
                K.ptx[f"tcgen05.alloc.{cta_group}.sync.aligned.shared::cta.b32"](
                    tmem_slot.ptr_to([0]), K.uint32(derived["num_tmem_alloc_cols"])
                )
            K.ptx.bar.sync(K.uint32(3), K.uint32(160))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))

            acc_cons = K.PipelineState(acc_stages, phase=0)
            c_cons = K.PipelineState(c_stages, phase=0)
            epi_info = K.PipelineState(tile_stages, phase=0)
            epi_expert = K.local_scalar("int32")
            epi_tile_m = K.local_scalar("int32")
            epi_tile_n = K.local_scalar("int32")
            epi_slots = (epi_expert, epi_tile_m, epi_tile_n)

            # The register tile is walked two elements at a time throughout the
            # epilogue. Under `vectorized_f32` each pair issues one
            # `mul.rn.f32x2` / `add.rn.f32x2` where the scalar form issues two,
            # for identical numbers -- each half of a packed line rounds exactly
            # as its scalar sibling -- and `scalar_f32` is its own export branch.
            packed_pair = K.local_scalar("uint64")
            ops = _arithmetic(vectorized_f32, packed_pair)
            epi_lane = K.local_scalar("int32", init=K.thread_id() % K.int32(32))
            epi_warp = _warp_uniform(K.thread_id() // K.int32(32))

            def dbias_warp_base():
                return K.local_scalar(
                    "int32", init=offsets["sDbias"] + epi_warp * K.int32(64 * 32 * 4)
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
                group = K.local_scalar("int32", init=epi_lane // K.int32(4))
                sub = K.local_scalar("int32", init=epi_lane % K.int32(4))
                # `group ^ swizzle` for each of the eight constant swizzles.
                twisted = [
                    K.local_scalar("int32", init=group ^ K.int32(swizzle)) for swizzle in range(8)
                ]
                for n in range(32):
                    for column, fragment in ((n, rC1), (32 + n, rC2)):
                        slot = K.local_scalar(
                            "int32",
                            init=warp_base
                            + (K.int32(column * 32) + twisted[(column >> 1) & 7] * K.int32(4) + sub)
                            * K.int32(4),
                        )
                        K.ptx.st.shared.b32(smem.ptr_to([slot]), fragment[n])

            def dbias_reduce(real_subtile, tile_base):
                """Reduce the staged transpose into this expert's column sums.

                One thread takes two whole columns and sums their 32 rows, the
                four warps' partial sums are combined through the front of the
                same buffer, and warp 0 atomically accumulates a BF16 pair per
                column.
                """
                warp_base = dbias_warp_base()
                K.ptx.bar.sync(K.uint32(2), K.uint32(128))

                # Lanes 0-15 take the gate half's even columns, 16-31 the up
                # half's, and each also takes the odd column beside it.
                column_a = K.local_scalar(
                    "int32",
                    init=(epi_lane % K.int32(16)) * K.int32(2)
                    + (epi_lane // K.int32(16)) * K.int32(32),
                )
                sums = [K.local_scalar("float32", init=K.float32(0.0)) for _ in range(2)]
                quad = K.alloc_local((4,), "float32")
                # `column_a` is even, so both sides share one twist -- and once
                # the byte scale is folded in, the twist occupies bits 4 to 6
                # while the column and the warp base occupy bits 7 and above.
                # The chunk therefore ORs into the column base rather than
                # adding, which is the three-input `a | (b ^ c)` LOP3: one
                # instruction per address where the add-then-XOR was two, on
                # sixteen addresses every subtile.
                swizzle = K.local_scalar(
                    "int32", init=((column_a // K.int32(2)) % K.int32(8)) * K.int32(16)
                )
                for side in range(2):
                    column = K.local_scalar("int32", init=column_a + K.int32(side))
                    column_base = K.local_scalar("int32", init=warp_base + column * K.int32(128))
                    for chunk in range(8):
                        base = K.local_scalar(
                            "int32", init=column_base | (K.int32(chunk * 16) ^ swizzle)
                        )
                        # `sDbias` is placed 128-byte aligned and both terms of
                        # the offset are multiples of 16, so a chunk's four rows
                        # are one aligned 16-byte line. Issue the vector load
                        # rather than four scalar ones and let ptxas fuse them:
                        # it does not always choose to, and when it declines the
                        # cost is far out of proportion to the instruction count
                        # -- this read-back runs 64 loads per subtile.
                        K.ptx["ld.shared.v4.b32"](
                            quad[0], quad[1], quad[2], quad[3], smem.ptr_to([base])
                        )
                        for j in range(4):
                            K.ptx["add.f32"](sums[side], sums[side], quad[j])

                # Combine the four warps through the front of the same buffer,
                # which every warp has finished reading by now.
                K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                partial = K.local_scalar(
                    "int32",
                    init=offsets["sDbias"]
                    + (epi_warp * K.int32(64) + epi_lane * K.int32(2)) * K.int32(4),
                )
                for side in range(2):
                    K.ptx.st.shared.b32(smem.ptr_to([partial + K.int32(4 * side)]), sums[side])
                K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                with K.If(epi_warp == K.int32(0)), K.Then():
                    totals = [K.local_scalar("float32", init=K.float32(0.0)) for _ in range(2)]
                    pair_in = K.alloc_local((2,), "float32")
                    for other in range(4):
                        # The two sides sit next to each other and the address is
                        # 8-byte aligned, so one vector load covers both.
                        K.ptx["ld.shared.v2.b32"](
                            pair_in[0],
                            pair_in[1],
                            smem.ptr_to(
                                [
                                    offsets["sDbias"]
                                    + (K.int32(other * 64) + epi_lane * K.int32(2)) * K.int32(4)
                                ]
                            ),
                        )
                        for side in range(2):
                            K.ptx["add.f32"](totals[side], totals[side], pair_in[side])
                    # `n_base_d2` is `n_base_d1 + 32` and the up half's columns
                    # start at 32, so one expression covers both halves.
                    n_offset = K.local_scalar(
                        "int32",
                        init=tile_base + (real_subtile * K.int32(2)) * K.int32(32) + column_a,
                    )
                    packed = K.local_scalar("uint32")
                    K.ptx["cvt.rn.bf16x2.f32"](packed, totals[1], totals[0])
                    # A cluster covers `cluster_n` column tiles whether or not
                    # the output has that many, so the last cluster carries
                    # tiles past the end. The D store is a TMA and its
                    # descriptor drops those writes on its own; this accumulate
                    # is a plain reduction and needs the bound spelled out.
                    with K.If(n_offset < K.int32(N_out)), K.Then():
                        K.ptx["red.global.add.noftz.bf16x2"](
                            named["dbias"].ptr_to(
                                [(epi_expert * K.int32(N_out) + n_offset) * K.int32(2)]
                            ),
                            packed,
                        )

            rAcc = K.alloc_local((32,), "float32")
            rC1 = K.alloc_local((32,), "float32")
            rC2 = K.alloc_local((32,), "float32")
            d_words = 32 * d_bits // 32
            rD1 = K.alloc_local((d_words,), "uint32")
            rD2 = K.alloc_local((d_words,), "uint32")
            # Counts subtiles across the whole persistent loop, so the two D
            # slots keep alternating across work tiles.
            prev_subtiles = K.local_scalar("int32", init=K.int32(0))

            take_tile_info(epi_info, epi_slots)
            with K.While(epi_expert >= K.int32(0)):
                alpha_value = K.local_scalar("float32")
                beta_value = K.local_scalar("float32")
                K.ptx.ld.global_.b32(alpha_value, named["alpha"].ptr_to([epi_expert]))
                K.ptx.ld.global_.b32(beta_value, named["beta"].ptr_to([epi_expert]))
                square_alpha = K.local_scalar("float32", init=alpha_value * alpha_value)

                row_base = expert_row_base(epi_expert)
                # Each accumulator stage owns its own `cta_tile_n` columns of
                # tensor memory, and the MMA warp fills the stage this consumer
                # cursor is about to read.
                acc_column = K.local_scalar(
                    "uint32", init=K.cast(acc_cons.stage, "uint32") * K.uint32(cta_tile_n)
                )

                thread_row = K.local_scalar(
                    "int32",
                    init=row_base
                    + (epi_tile_m // K.int32(atom_thr)) * K.int32(cta_tile_m * atom_thr)
                    + (block_x % K.int32(atom_thr)) * K.int32(cta_tile_m)
                    + K.thread_id()
                    if atom_thr > 1
                    else row_base + epi_tile_m * K.int32(cta_tile_m) + K.thread_id(),
                )
                prob_value = K.local_scalar("float32")
                K.ptx.ld.global_.b32(prob_value, named["prob"].ptr_to([thread_row]))
                dprob_acc = K.local_scalar("float32", init=K.float32(0.0))

                _wait_plain(acc_pipe.full.ptr_to([acc_cons.stage]), acc_cons.phase)

                d_tile_base = K.local_scalar("int32", init=epi_tile_n * K.int32(cta_tile_n * 2))

                # The source pins `unroll=1` on this loop and its export
                # carries `.pragma "nounroll"`; a `While` lowers with
                # `#pragma unroll 1`, which is the shape the source has.
                real_subtile = K.local_scalar("int32", init=K.int32(0))
                subtile = K.local_scalar("int32", init=K.int32(0))
                with K.While(subtile < K.int32(epi_subtiles)):
                    # A TMEM address is (lane << 16) | column. The source folds
                    # `(tid << 16) & 0xE00000` in so each epilogue warp reads its
                    # own row group; without it all four warps read warp 0's
                    # rows (source 3617, 3121-3122; PTX 1771-1770).
                    # `(tid << 16) & 0xE00000` keeps bits 21 to 23, and the
                    # column term below never reaches bit 16, so the two fields
                    # are disjoint and the lane bits OR in. That folds the mask
                    # and the combine into one `LOP3` where the mask, the add
                    # and the move were three.
                    tmem_lane = K.local_scalar("uint32")
                    K.ptx["shl.b32"](tmem_lane, K.cast(K.thread_id(), "uint32"), K.uint32(16))
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                        *[rAcc[i] for i in range(32)],
                        (tmem_lane & K.uint32(0xE00000))
                        | (tmem_base + acc_column + K.cast(real_subtile * K.int32(32), "uint32")),
                    )

                    # Two C stages per accumulator subtile: gate then up.
                    c_row_bytes = 32 * c_bits // 8
                    c_words = c_row_bytes // 4
                    for fragment in (rC1, rC2):
                        _wait_plain(c_pipe.full.ptr_to([c_cons.stage]), c_cons.phase)
                        c_slot = offsets["sC"] + c_cons.stage * derived["c_stage_bytes"]
                        # Each thread owns one row of the 128x32 subtile, so
                        # its slice starts at row * 32 * c_bits/8 bytes.
                        raw = K.alloc_local((c_words,), "uint32")
                        c_row = K.local_scalar("int32", init=K.thread_id() * K.int32(c_row_bytes))
                        for word in range(0, c_words, 4):
                            # The descriptor swizzled this box on the way in, so
                            # the read walks the same permutation.
                            K.ptx["ld.shared.v4.b32"](
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
                        K.ptx["fence.proxy.async.shared::cta"]()
                        with K.If(K.lane_id() == K.int32(0)), K.Then():
                            K.ptx.mbarrier.arrive.shared.b64(c_pipe.empty.ptr_to([c_cons.stage]))
                        c_cons.advance()
                        _unpack_input(fragment, raw, c_dtype, c_bits)

                    # ---- dGLU derivative, elementwise over the subtile ----
                    # The register tile is walked two elements at a time: a
                    # `vectorized_f32` specialization issues one packed
                    # `mul.rn.f32x2` / `add.rn.f32x2` per pair where the scalar
                    # one issues two, for identical numbers. One scratch set is
                    # reused for every pair -- fresh scalars per element make
                    # ptxas superlinear (see `_arithmetic`).
                    def pair():
                        return (K.local_scalar("float32"), K.local_scalar("float32"))

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
                        K.assign(dprob_pair[half], K.float32(0.0))

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
                                K.ptx["min.f32"](y_gate[half], gate[half], K.float32(GEGLU_MAX))
                                K.ptx["min.f32"](y_up[half], up[half], K.float32(GEGLU_MAX))
                                K.ptx["max.f32"](y_up[half], y_up[half], K.float32(GEGLU_MIN))
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
                                predicate = K.local_scalar("bool")
                                K.assign(
                                    predicate, K.cast(gate[half] <= K.float32(GEGLU_MAX), "bool")
                                )
                                K.ptx["selp.f32"](
                                    keep[half], y_gate[half], K.float32(0.0), predicate
                                )
                            ops["product"](d_gate, d_gate, keep)
                            for half in range(2):
                                predicate = K.local_scalar("bool")
                                if vectorized_f32:
                                    K.assign(
                                        predicate, K.cast(up[half] >= K.float32(GEGLU_MIN), "bool")
                                    )
                                    K.ptx["selp.f32"](
                                        keep[half], y_up[half], K.float32(0.0), predicate
                                    )
                                else:
                                    K.assign(
                                        predicate,
                                        K.cast(
                                            K.And(
                                                up[half] >= K.float32(GEGLU_MIN),
                                                up[half] <= K.float32(GEGLU_MAX),
                                            ),
                                            "bool",
                                        ),
                                    )
                                    K.ptx["selp.f32"](
                                        keep[half], up[half], K.float32(0.0), predicate
                                    )
                            ops["product"](d_up, d_up, keep)

                        for half in range(2):
                            K.assign(rC1[span[half]], d_gate[half])
                            K.assign(rC2[span[half]], d_up[half])

                    folded_prob = K.local_scalar("float32")
                    K.ptx["add.f32"](folded_prob, dprob_pair[0], dprob_pair[1])
                    K.ptx["add.f32"](dprob_acc, dprob_acc, folded_prob)

                    # ---- convert D and stage it through SMEM -------------
                    _pack_output(rD1, rC1, d_dtype, d_bits)
                    _pack_output(rD2, rC2, d_dtype, d_bits)

                    with K.If(warp == K.int32(0)), K.Then():
                        # The D pipeline is a bulk-group counter, not an
                        # mbarrier: there is no D barrier in the storage map.
                        K.ptx["cp.async.bulk.wait_group.read"](K.uint32(0))
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))

                    slot1 = K.local_scalar("int32", init=prev_subtiles % K.int32(d_stages))
                    K.assign(prev_subtiles, prev_subtiles + K.int32(1))
                    slot2 = K.local_scalar("int32", init=prev_subtiles % K.int32(d_stages))
                    K.assign(prev_subtiles, prev_subtiles + K.int32(1))

                    d_row_bytes = 32 * d_bits // 8

                    def stage_fragment(fragment, slot, region="sD"):
                        # Mirror of the C load: this thread owns one row of the
                        # 128x32 subtile.
                        base = offsets[region] + slot * K.int32(derived["d_stage_bytes"])
                        d_row = K.local_scalar("int32", init=K.thread_id() * K.int32(d_row_bytes))
                        for word in range(0, d_words, 4):
                            # Written through the same permutation the store
                            # descriptor reads back.
                            K.ptx["st.shared.v4.b32"](
                                smem.ptr_to([base + _swizzled(d_row, word // 4, d_row_bytes)]),
                                fragment[word],
                                fragment[word + 1],
                                fragment[word + 2],
                                fragment[word + 3],
                            )

                    # Both halves of one slot go out together, D then D_col.
                    stage_fragment(rD1, slot1)
                    stage_fragment(rD2, slot2)
                    K.ptx["fence.proxy.async.shared::cta"]()
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))

                    with K.If(warp == K.int32(0)), K.Then():
                        # The two D subtiles of one accumulator subtile land in
                        # adjacent halves of the 2N region, which is why the
                        # column index is 2 * real_subtile + {0, 1}.
                        d_row_coord = K.local_scalar(
                            "int32", init=row_base + epi_tile_m * K.int32(cta_tile_m)
                        )
                        for map_name, region in (("d_row", "sD"),):
                            for half, slot in ((0, slot1), (1, slot2)):
                                column = K.local_scalar(
                                    "int32",
                                    init=d_tile_base
                                    + (real_subtile * K.int32(2) + K.int32(half)) * K.int32(32),
                                )
                                K.ptx[
                                    "cp.async.bulk.tensor.3d.global.shared::cta.tile"
                                    ".bulk_group.L2::cache_hint"
                                ](
                                    K.address_of(maps[map_name]),
                                    K.cast(column, "int32"),
                                    K.cast(d_row_coord, "int32"),
                                    K.int32(0),
                                    smem.ptr_to(
                                        [offsets[region] + slot * K.int32(derived["d_stage_bytes"])]
                                    ),
                                    K.uint64(0),
                                )
                        K.ptx["cp.async.bulk.commit_group"]()
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
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    K.assign(real_subtile, real_subtile + K.int32(1))
                    K.assign(subtile, subtile + K.int32(1))

                # Release the accumulator stage once every subtile has been read
                # out of it. One elected lane per epilogue warp arrives, and
                # both CTAs of a pair arrive on the *leader's* barrier, which is
                # where the accumulator lives -- hence its `4 * atom_thr`
                # arrivals. Releasing locally leaves the leader's MMA waiting on
                # a stage nobody frees.
                with K.If(_elected()), K.Then():
                    peer_bar = K.local_scalar("uint32")
                    local_bar = K.local_scalar("uint32")
                    K.assign(
                        local_bar,
                        K.cuda.cvta_generic_to_shared(acc_pipe.empty.ptr_to([acc_cons.stage])),
                    )
                    K.ptx["mapa.shared::cluster.u32"](
                        peer_bar,
                        local_bar,
                        K.cast(
                            cluster_rank - cluster_v if atom_thr > 1 else cluster_rank, "uint32"
                        ),
                    )
                    K.ptx["mbarrier.arrive.shared::cluster.b64"](peer_bar, K.uint32(1))
                acc_cons.advance()

                # The next record is taken before the dProb tail, as the source
                # does, so the tile-info slot is freed as early as possible.
                take_tile_info(epi_info, epi_slots)
                # One reduction per thread per work tile. The returning form
                # would put every one of these on the scoreboard for a value that
                # is immediately discarded.
                K.ptx["red.global.add.f32"](named["dprob"].ptr_to([thread_row]), dprob_acc)

            # Release the TMEM allocation permit, then synchronize the four
            # epilogue warps, then free the columns. Both tensor-memory
            # instructions are warp 0's -- each is `.sync.aligned`, and the
            # export predicates both on the allocator warp -- but the permit
            # goes *before* the barrier and the deallocation after it.
            with K.If(warp == K.int32(0)), K.Then():
                K.ptx[f"tcgen05.relinquish_alloc_permit.{cta_group}.sync.aligned"]()
            K.ptx.bar.sync(K.uint32(2), K.uint32(128))
            with K.If(warp == K.int32(0)), K.Then():
                if atom_thr > 1:
                    # A CTA pair frees its TMEM collectively: each CTA arrives on
                    # its peer's deallocation barrier and waits on its own before
                    # issuing the free.
                    peer_dealloc = K.local_scalar("uint32")
                    own_dealloc = K.local_scalar("uint32")
                    K.assign(own_dealloc, K.cuda.cvta_generic_to_shared(tmem_dealloc.ptr_to([0])))
                    K.ptx["mapa.shared::cluster.u32"](
                        peer_dealloc, own_dealloc, K.cast(cluster_rank ^ K.int32(1), "uint32")
                    )
                    K.ptx["mbarrier.arrive.shared::cluster.b64"](peer_dealloc, K.uint32(1))
                    _wait_plain(tmem_dealloc.ptr_to([0]), K.uint32(0))
                K.ptx[f"tcgen05.dealloc.{cta_group}.sync.aligned.b32"](
                    tmem_base, K.uint32(derived["num_tmem_alloc_cols"])
                )
            K.ptx["cp.async.bulk.wait_group.read"](K.uint32(0))

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
            descriptors = {"b": K.stack_alloca("tensormap", 1)}
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
            source: K.uint64 = K.reinterpret("uint64", K.address_of(source_map))
            groups = []
            for group in range(2):
                payload = K.alloc_local((4,), "uint64")
                offset: K.uint64 = K.uint64(group * 32)
                K.ptx.ld.global_.v4.b64(
                    payload[0],
                    payload[1],
                    payload[2],
                    payload[3],
                    K.reinterpret("handle", source + offset),
                )
                groups.append(payload)
            return groups

        def write_tensormap_image(groups, destination, address):
            """Publish one image, substituting word 0, the global address."""
            target: K.uint64 = K.reinterpret("uint64", destination)
            for group, payload in enumerate(groups):
                if group == 0:
                    K.assign(payload[0], address)
                K.ptx.st.global_.v4.b64(
                    K.reinterpret("handle", target + K.uint64(group * 32)),
                    payload[0],
                    payload[1],
                    payload[2],
                    payload[3],
                )

        def helper(operands, host):
            b, workspace = operands
            expert = K.cta_id()[0]
            if weight_mode == "discrete":
                slot = K.local_scalar("int32", init=K.int32(128) * expert)
                with K.If(_elected()), K.Then():
                    # This kernel is nothing but memory latency: three global
                    # reads and two writes on one lane. The expert pointer is
                    # loaded before the template image so the two round trips
                    # overlap instead of adding.
                    address = K.local_scalar("uint64")
                    K.ptx.ld.global_.b64(address, b.ptr_to([expert]))
                    groups = read_tensormap_image(host[0])
                    write_tensormap_image(groups, workspace.ptr_to([slot]), address)
                K.cuda.warp_sync()
            if sched == "dynamic":
                counter_offset = L * 128 if weight_mode == "discrete" else 0
                with K.If(expert == K.int32(0)), K.Then():
                    with K.If(_elected()), K.Then():
                        K.ptx.st.global_.b32(workspace.ptr_to([counter_offset]), K.int32(0))

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
            "b": K.gptr[pointer_dtype, (L,)],
            "workspace": K.gptr[K.u8, (max(1, derived["workspace_bytes"]),)],
        }
        return K.kernel(
            warps=1,
            arch="sm_100a",
            min_blocks_per_sm=1,
            grid=list(derived["helper_grid"]),
            host_prelude=helper_prelude if weight_mode == "discrete" else None,
        )(helper_body)

    kernel = _entry_point(list(annotations), body)
    kernel.__annotations__ = dict(annotations)
    main = K.kernel(
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
