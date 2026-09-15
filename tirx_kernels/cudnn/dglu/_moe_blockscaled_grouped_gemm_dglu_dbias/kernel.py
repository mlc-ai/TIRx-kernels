# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MoE block-scaled grouped GEMM with a fused dGLU backward epilogue.

Upstream source:
``python/cudnn/gemm/cutedsl/grouped/dglu/moe_blockscaled_grouped_gemm_dglu_dbias.py``
(``BlockScaledMoEGroupedGemmDgluDbiasKernel``), with the tile scheduler from
``python/cudnn/gemm/cutedsl/grouped/moe_persistent_scheduler.py``, the per-expert
tensor-map workspace from ``python/cudnn/gemm/cutedsl/grouped/moe_utils.py``, and
the gmem addressing extensions from
``python/cudnn/gemm/cutedsl/grouped/moe_sched_extension.py``.

The kernel computes, for every expert ``g`` over its 256-aligned row range::

    ref   = alpha[g]^2 * dequant(A, SFA) @ dequant(B[g], SFB)^T
    gate  = beta[g] * C[:, even 32-column blocks]
    up    = beta[g] * C[:, odd 32-column blocks]
    D     = interleave(dGLU_gate(ref, prob, gate, up), dGLU_up(ref, prob, gate, up))

plus the optional per-row ``dprob`` scalar, per-expert ``dbias`` column sums,
per-expert ``amax``, and FP8 row/column output scale factors.

One deliberate deviation. Upstream's ``discrete_col_sfd`` declares the column
scale-factor tensor with the *row* layout (``__call__`` 826-833) and then stores
into it through an MN-chunk partition (``create_and_partition_new_SFDCol``), so
the bytes land in the positions of neither layout; upstream's own test leaves
that buffer unchecked for exactly that reason. This port always writes the
column scale factors in the transposed layout the FP32 reference defines, and
``data.validate_outputs`` holds it to that exactly, including where
``discrete_col_sfd`` is set.
"""

from functools import cache

import tirx_kernels.tirx_lite as txl

from . import spec

_TRY_WAIT_TICKS = 0x989680  # 10,000,000, the source's mbarrier spin bound


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
    txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
        dst, barrier, txl.cast(phase, "uint32")
    )


def _wait_plain(barrier, phase):
    """Spin on one mbarrier until its phase flips.

    The retry takes the hintless `try_wait` -- the same form the source's own
    hot waits use. Given a suspend-time hint, ptxas expands the wait inline into
    four instructions: the check, a `NANOSLEEP`, a second check and the branch.
    That is where 48.6% of this port's not-issued samples were parked, against
    12.8% for the reference, and it puts three instructions between the barrier
    and the load that depends on it. Without the hint the wait is two
    instructions -- the check and one predicated branch to an out-of-line retry
    -- and the export shows the reference's dependent `LDS.128` issuing directly
    behind its branch.
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


def _situglu(ops, out, gate, up, beta1, beta2, tmp):
    """dSiTU-GLU terms into `out`, a dict of caller-owned pairs.

    `beta1 == 4` takes the source's closed form: with t = tanh(gate/4) the
    identity sigmoid(g) = 0.5 + t/(1 + t^2) removes the exponential entirely,
    which is why that export contains no `ex2`.
    """
    gate_tanh, up_tanh = tmp["a"], tmp["b"]
    ops["scaled"](gate_tanh, gate, 1.0 / beta1)
    ops["unary"]("tanh.approx.f32", gate_tanh, gate_tanh)
    ops["scaled"](up_tanh, up, 1.0 / beta2)
    ops["unary"]("tanh.approx.f32", up_tanh, up_tanh)

    # Every multiply below that feeds an add is written as one `fma`. The
    # arithmetic reaches ptxas as inline assembly, which it will not contract on
    # its own, so the contraction has to be spelled out to reach the machine
    # code the reference's compiler produces. `1 - t^2` goes the same way: a
    # negate and an `fma` where a multiply, a negate and an add were three.
    gate_sq, work = tmp["c"], tmp["e"]
    ops["product"](gate_sq, gate_tanh, gate_tanh)
    ops["unary"]("neg.f32", tmp["d"], up_tanh)
    ops["fused"](out["up_grad"], tmp["d"], up_tanh, _spread(1.0))
    ops["scaled"](out["up_value"], up_tanh, beta2)

    sigmoid = tmp["f"]
    if beta1 == 4.0:
        ops["offset"](work, gate_sq, 1.0)
        ops["unary"]("rcp.approx.ftz.f32", work, work)
        ops["fused"](sigmoid, gate_tanh, work, _spread(0.5))
        ops["scaled"](out["gate_value"], gate_tanh, beta1)
        ops["product"](out["gate_value"], out["gate_value"], sigmoid)
        inner = tmp["g"]
        ops["product"](inner, work, work)
        ops["product"](inner, gate_tanh, inner)
        ops["fused"](inner, inner, _spread(2.0), _spread(0.5))
        ops["complement"](work, 1.0, gate_sq, tmp["h"])
        ops["product"](out["gate_grad"], work, inner)
    else:
        _sigmoid(ops, sigmoid, gate, work)
        ops["scaled"](out["gate_value"], gate_tanh, beta1)
        ops["product"](out["gate_value"], out["gate_value"], sigmoid)
        ops["complement"](work, 1.0, gate_sq, tmp["h"])
        ops["product"](out["gate_grad"], work, sigmoid)
        extra = tmp["g"]
        ops["complement"](extra, 1.0, sigmoid, tmp["h"])
        ops["fused"](out["gate_grad"], out["gate_value"], extra, out["gate_grad"])


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
    """Expand one thread's row of an epilogue tile into 32 FP32 values."""
    if bits == 8:
        # Four FP8 per word: split into halves, widen each pair through
        # `cvt.rn.f16x2.<f8>x2`, then to FP32.
        converter = "cvt.rn.f16x2.e4m3x2" if dtype == "float8_e4m3fn" else "cvt.rn.f16x2.e5m2x2"
        for word in range(8):
            low = txl.local_scalar("uint16")
            high = txl.local_scalar("uint16")
            txl.ptx["mov.b32"](low, high, words[word])
            for half, source in enumerate((low, high)):
                pair = txl.local_scalar("uint32")
                txl.ptx[converter](pair, source)
                first = txl.local_scalar("uint16")
                second = txl.local_scalar("uint16")
                txl.ptx["mov.b32"](first, second, pair)
                txl.ptx["cvt.f32.f16"](values[4 * word + 2 * half + 0], first)
                txl.ptx["cvt.f32.f16"](values[4 * word + 2 * half + 1], second)
        return
    if bits == 16:
        converter = "cvt.f32.bf16" if dtype == "bfloat16" else "cvt.f32.f16"
        for word in range(16):
            low = txl.local_scalar("uint16")
            high = txl.local_scalar("uint16")
            txl.ptx["mov.b32"](low, high, words[word])
            txl.ptx[converter](values[2 * word + 0], low)
            txl.ptx[converter](values[2 * word + 1], high)
    else:
        for word in range(32):
            txl.ptx["mov.b32"](values[word], words[word])


def _pack_output(words, values, dtype, bits):
    """Pack a 32-element FP32 fragment into the output dtype's storage words.

    FP8 goes through the source's inline `cvt.rn.satfinite.<f8>x2.f32` pairs
    joined by `mov.b32`, four elements to a word; 16-bit outputs use
    `cvt.rn.bf16x2.f32` / `cvt.rn.f16x2.f32`, two to a word; FP32 is a move.
    """
    if bits == 8:
        converter = (
            "cvt.rn.satfinite.e4m3x2.f32"
            if dtype == "float8_e4m3fn"
            else "cvt.rn.satfinite.e5m2x2.f32"
        )
        for word in range(8):
            low = txl.local_scalar("uint16")
            high = txl.local_scalar("uint16")
            txl.ptx[converter](low, values[4 * word + 1], values[4 * word + 0])
            txl.ptx[converter](high, values[4 * word + 3], values[4 * word + 2])
            txl.ptx["mov.b32"](words[word], low, high)
    elif bits == 16:
        converter = "cvt.rn.bf16x2.f32" if dtype == "bfloat16" else "cvt.rn.f16x2.f32"
        for word in range(16):
            txl.ptx[converter](words[word], values[2 * word + 1], values[2 * word + 0])
    else:
        for word in range(32):
            txl.ptx["mov.b32"](words[word], values[word])


def _instruction_descriptor(M, N, ab_dtype, sf_dtype, a_major, b_major):
    """Fold the static fields of the block-scaled MMA instruction descriptor."""
    sf_format = {"float8_e4m3fn": 0, "float8_e8m0fnu": 1}[sf_dtype]
    value = 0
    if ab_dtype in {"float4_e2m1fn", "float8_e5m2"}:
        value |= 1 << 7
        value |= 1 << 10
    if a_major == "m":
        value |= 1 << 15
    if b_major == "n":
        value |= 1 << 16
    value |= ((N >> 3) & 0x3F) << 17
    value |= (sf_format & 1) << 23
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


_TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_MCAST = ".multicast::cluster"


def _tma_load(destination, descriptor, coords, barrier, mask, *, two_cta, desc_ptr=None):
    """One `cp.async.bulk.tensor` load, multicast only when a mask is given.

    The two-CTA MMA adds `.cta_group::2`; the multicast form inserts its mask
    modifier before the cache hint, matching the operand order the export shows.
    """
    stem = _TMA_G2S_3D if len(coords) == 3 else _TMA_G2S_4D
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
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    c_dtype,
    d_dtype,
    b_major,
    mma_tiler_mn,
    cluster_shape_mn,
    vectorized_f32,
    with_dbias,
    with_prob,
    with_amax,
    discrete_col_sfd,
    linear_offset,
    geglu_alpha,
    glu_clamp_max,
    glu_clamp_min,
    situ_beta1,
    situ_beta2,
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
        "ab_dtype": ab_dtype,
        "sf_dtype": sf_dtype,
        "sf_vec_size": sf_vec_size,
        "c_dtype": c_dtype,
        "d_dtype": d_dtype,
        "b_major": b_major,
        "mma_tiler_mn": tuple(mma_tiler_mn),
        "cluster_shape_mn": tuple(cluster_shape_mn),
        "vectorized_f32": vectorized_f32,
        "with_dbias": with_dbias,
        "with_prob": with_prob,
        "with_amax": with_amax,
        "discrete_col_sfd": discrete_col_sfd,
    }
    derived = spec.derive(mode, group_m_list=list(group_m_list), N=N, K_dim=K_dim)

    ab_bits = spec.dtype_bits(ab_dtype)
    c_bits = spec.dtype_bits(c_dtype)
    d_bits = spec.dtype_bits(d_dtype)
    tokens_total, N_out, L = derived["tokens_total"], derived["n_out"], derived["L"]

    def byte_count(rows, columns, bits):
        return rows * columns * bits // 8

    def sf_bytes(shape):
        total = 1
        for extent in shape:
            total *= extent
        return total

    # Every payload crosses the launch boundary as a flat byte array; the logical
    # extents live in the tensor maps and in the scalar index arithmetic below.
    annotations = {
        "a": txl.gptr[txl.u8, (byte_count(tokens_total, K_dim, ab_bits),)],
        "b": (
            txl.gptr["int64", (L,)]
            if weight_mode == "discrete"
            else txl.gptr[txl.u8, (byte_count(N, K_dim, ab_bits) * L,)]
        ),
        "sfa": txl.gptr[txl.u8, (sf_bytes(derived["sf_shape_a"]),)],
        "sfb": (
            txl.gptr["int64", (L,)]
            if weight_mode == "discrete"
            else txl.gptr[txl.u8, (sf_bytes(derived["sf_shape_b"]),)]
        ),
        "c": txl.gptr[txl.u8, (byte_count(tokens_total, N_out, c_bits),)],
        "d_row": txl.gptr[txl.u8, (byte_count(tokens_total, N_out, d_bits),)],
    }
    if derived["generate_sfd"]:
        annotations["d_col"] = txl.gptr[txl.u8, (byte_count(tokens_total, N_out, d_bits),)]
        annotations["sfd_row"] = txl.gptr[txl.u8, (sf_bytes(derived["sf_shape_d_row"]),)]
        annotations["sfd_col"] = txl.gptr[txl.u8, (sf_bytes(derived["sf_shape_d_col"]),)]
        annotations["norm_const"] = txl.gptr["float32", (1,)]
    if derived["generate_amax"]:
        annotations["amax"] = txl.gptr["float32", (L * 2,)]
    annotations["padded_offsets"] = txl.gptr["int32", (L,)]
    annotations["alpha"] = txl.gptr["float32", (L,)]
    annotations["beta"] = txl.gptr["float32", (L,)]
    if with_prob:
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
    generate_sfd = derived["generate_sfd"]
    generate_amax = derived["generate_amax"]
    generate_dbias = derived["generate_dbias"]
    generate_dprob = derived["generate_dprob"]
    early_release = derived["iter_acc_early_release"]

    # A is always k-major; B follows ``b_major``. The shared descriptors carry
    # everything but the 14-bit address, which the device body ORs in.
    a_desc_base = _descriptor_base(ldo=1, sdo=64, swizzle=3)
    b_desc_base = _descriptor_base(
        ldo=(
            derived["b_stage_bytes"] // 32
            if b_major == "n" and cta_tile_n == 256
            else 1
            if b_major == "k"
            else 0
        ),
        sdo=64,
        swizzle=3,
    )
    # One arrival per participating CTA of the A and B multicast images, less
    # this CTA itself.
    # The CTA group is 2 only when the MMA atom spans a CTA pair; asking for
    # `cta_group::2` with a single-CTA atom makes the launch itself invalid.
    cta_group = f"cta_group::{atom_thr}"
    sf_desc_base = _descriptor_base(ldo=1, sdo=8, swizzle=0)
    mma_mnemonic = (
        f"tcgen05.mma.{cta_group}.kind::"
        + ("mxf4nvf4" if ab_dtype == "float4_e2m1fn" else "mxf8f6f4")
        + ".block_scale.block"
        + str(sf_vec_size)
    )
    # The descriptor's M is the whole MMA tile, not the per-CTA half: the
    # anchor export's base descriptor is 0x10C00000, whose M field is 16 for a
    # 256-row two-CTA tile. Dividing by `atom_thr` here encodes 128 and is wrong
    # on every two-CTA specialization.
    instruction_descriptor = _instruction_descriptor(
        derived["mma_tiler"][0], cta_tile_n, ab_dtype, sf_dtype, "k", b_major
    )
    ab_empty_arrivals = max(1, cluster_n + (cluster_m // atom_thr) - 1)

    # TensorMaps the launch passes as grid constants, in the order
    # ``host_prelude`` returns them. Discrete weights read B/SFB descriptors out
    # of the workspace instead, so they contribute no grid constant.
    map_names = ["a", "sfa", "c", "d_row"]
    if generate_sfd:
        map_names.append("d_col")
    if weight_mode == "dense":
        map_names.extend(["b", "sfb"])

    # A and SFA multicast over the cluster's N extent; B and SFB over the M
    # extent the two-CTA MMA has already halved.
    a_cluster_piece = cta_tile_m // cluster_n
    # A two-CTA MMA splits B's N extent across the pair: `b_stage_bytes` is 128
    # columns per CTA for a 256-column tile, so the box divides by `atom_thr`
    # before the cluster's own B multicast split. Without it the stage delivers
    # twice its bytes and overruns the expect_tx count.
    b_tile_n = cta_tile_n // atom_thr
    b_split = max(1, cluster_m // atom_thr)
    b_cluster_piece = (k_tile if b_major == "n" else b_tile_n) // b_split
    # An n-major B is N-contiguous, and a 128-byte swizzle atom caps a TMA box's
    # contiguous extent at 128 columns, so a 256-column per-CTA tile needs two
    # copies side by side in SMEM. `b_desc_base`'s leading offset of
    # `b_stage_bytes // 32` is exactly the 16-byte-unit distance between them,
    # which is how the MMA reaches the second block. A k-major B is K-contiguous
    # and needs one copy whatever its N extent.
    b_tma_copies = b_tile_n // 128 if b_major == "n" else 1
    sf_k_box = k_tile // (4 * sf_vec_size)
    sfa_piece_values = 256 * sf_k_box // cluster_n
    # SFB carries one 128-column group per 128 output columns, and the cluster's
    # M extent splits it. Dropping either factor leaves the stage's expect_tx
    # count short and the AB barrier never completes.
    sfb_n_box = cta_tile_n // 128
    sfb_piece_values = 256 * sf_k_box * sfb_n_box // cluster_m
    epi_m, epi_n = derived["epi_tile"]

    ab_tail = (1, 1, 1, 0, 3, 2, 0)
    if ab_dtype == "float4_e2m1fn":
        ab_tail = (*ab_tail, 13)
    sf_k_groups = _ceil_div(K_dim, 4 * sf_vec_size)

    def encode_map(descriptor, dtype, rank, data, *fields):
        txl.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, rank, data, *fields)

    def encode_weight_maps(descriptors, b_data, sfb_data, batch):
        """B and SFB TensorMaps.

        ``batch`` is the expert count for dense weights, whose single allocation
        the TMA indexes by a fourth coordinate, and 1 for discrete weights, where
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
                b_tile_n // b_tma_copies,
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
        encode_map(descriptors["b"], ab_dtype, 3, b_data, *b_fields, *ab_tail)
        sfb_box_0 = min(256, sfb_piece_values)
        sfb_remaining = sfb_piece_values // sfb_box_0
        sfb_box_1 = min(sf_k_box, sfb_remaining)
        sfb_box_2 = sfb_remaining // sfb_box_1
        encode_map(
            descriptors["sfb"],
            "uint16",
            4,
            sfb_data,
            256,
            sf_k_groups,
            _ceil_div(N, 128),
            batch,
            512,
            sf_k_groups * 512,
            _ceil_div(N, 128) * sf_k_groups * 512,
            sfb_box_0,
            sfb_box_1,
            sfb_box_2,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            2,
            0,
        )

    def host_prelude(params):
        descriptors = {name: txl.stack_alloca("tensormap", 1) for name in map_names}
        encode = encode_map

        # A is (tokens, K) k-major, so K is the contiguous extent.
        a_contiguous = K_dim * ab_bits // 8
        encode(
            descriptors["a"],
            ab_dtype,
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

        sfa_box_0 = min(256, sfa_piece_values)
        sfa_box_1 = sfa_piece_values // sfa_box_0
        encode(
            descriptors["sfa"],
            "uint16",
            4,
            params["sfa"].data,
            256,
            sf_k_groups,
            _ceil_div(tokens_total, 128),
            1,
            512,
            sf_k_groups * 512,
            _ceil_div(tokens_total, 128) * sf_k_groups * 512,
            sfa_box_0,
            sfa_box_1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            2,
            0,
        )

        # C, D and D_col are (tokens, 2N) n-major over the interleaved output.
        def encode_epilogue(name, tensor, dtype, bits):
            contiguous = N_out * bits // 8
            # One epilogue row is `epi_n` elements, and the source picks the
            # swizzle whose period matches that row exactly -- 128B for a 32-bit
            # element, 64B for 16-bit, 32B for 8-bit
            # (`get_smem_layout_atom_ab`'s K-major ladder). Leaving it
            # unswizzled costs a bank conflict per epilogue access whose width
            # scales with the element: a 128-byte row is exactly the 32 banks,
            # so every lane of a warp lands on bank 0.
            swizzle = {128: 3, 64: 2, 32: 1}[epi_n * bits // 8]
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
        if generate_sfd:
            encode_epilogue("d_col", params["d_col"], d_dtype, d_bits)

        if weight_mode == "dense":
            encode_weight_maps(descriptors, params["b"].data, params["sfb"].data, L)
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

        with txl.If(warp == 0):
            with txl.Then():
                with txl.If(_elected()):
                    with txl.Then():
                        if sched == "dynamic":
                            # `internal_init` builds this pipeline with
                            # `defer_sync=True`, so it is published by the second
                            # epoch alongside the TMEM barrier.
                            txl.ptx.mbarrier.init.shared.b64(sched_pipe.full.ptr_to([0]), txl.uint32(1))
                            txl.ptx.mbarrier.init.shared.b64(
                                sched_pipe.empty.ptr_to([0]), txl.uint32(32 * cluster_size)
                            )
                        txl.ptx.mbarrier.init.shared.b64(tmem_dealloc.ptr_to([0]), txl.uint32(32))
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

        sfa_mcast_mask = None
        a_mcast_mask = mask_union(
            [cta_bit(cluster_v, cluster_m_coord, n) for n in range(cluster_n)]
        )
        sfa_mcast_mask = a_mcast_mask
        b_mcast_mask = mask_union(
            [cta_bit(cluster_v, m, cluster_n_coord) for m in range(cluster_m // atom_thr)]
        )
        sfb_mcast_mask = mask_union(
            [
                txl.uint32(1) << txl.cast(m + cluster_m * cluster_n_coord, "uint32")
                for m in range(cluster_m)
            ]
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
                counter_offset = 2 * L * 128 if weight_mode == "discrete" else 0
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
                    # The pre-kernel wrote this expert's B and SFB TensorMap
                    # images here; the TMA reads the descriptor straight out of
                    # global memory instead of from a grid constant.
                    b_desc = named["workspace"].ptr_to([txl.int32(256) * tile_expert])
                    sfb_desc = named["workspace"].ptr_to(
                        [txl.int32(256) * tile_expert + txl.int32(128)]
                    )

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
                                txl.uint32(derived["ab_stage_bytes"] * atom_thr),
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
                            else txl.local_scalar("int32", init=n_base + txl.int32(block * 128))
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
                    sfa_slot = smem.ptr_to(
                        [offsets["sSFA"] + ab_prod.stage * derived["sfa_stage_bytes"]]
                    )
                    # One k tile spans `sf_k_box` scale-factor groups, and a
                    # multicast splits the stage across the cluster, so each CTA
                    # starts at its own linear offset inside the tile. Loading
                    # the same slice on every CTA leaves three quarters of the
                    # scale factors wrong even though the byte count adds up.
                    sfa_linear = cluster_y * sfa_piece_values
                    sfa_coord_0 = txl.int32(sfa_linear % 256)
                    sfa_coord_1 = txl.local_scalar(
                        "int32", init=counter * txl.int32(sf_k_box) + txl.int32(sfa_linear // 256)
                    )
                    sfa_row = txl.local_scalar("int32", init=a_row // txl.int32(128))
                    sfa_destination = (
                        cluster_smem_base
                        + txl.uint32(offsets["sSFA"])
                        + txl.cast(ab_prod.stage, "uint32") * txl.uint32(derived["sfa_stage_bytes"])
                        + txl.cast(cluster_y, "uint32")
                        * txl.uint32(derived["sfa_stage_bytes"] // cluster_n)
                        if cluster_n > 1
                        else sfa_slot
                    )
                    with txl.If(_elected()), txl.Then():
                        _tma_load(
                            sfa_destination,
                            maps["sfa"],
                            (sfa_coord_0, sfa_coord_1, sfa_row, txl.int32(0)),
                            full_bar,
                            sfa_mcast_mask if cluster_n > 1 else None,
                            two_cta=two_cta,
                        )
                    sfb_slot = smem.ptr_to(
                        [offsets["sSFB"] + ab_prod.stage * derived["sfb_stage_bytes"]]
                    )
                    sfb_linear = cluster_x * sfb_piece_values
                    sfb_quotient = sfb_linear // 256
                    sfb_coord_0 = txl.int32(sfb_linear % 256)
                    sfb_coord_1 = txl.local_scalar(
                        "int32", init=counter * txl.int32(sf_k_box) + txl.int32(sfb_quotient % sf_k_box)
                    )
                    # The row-group coordinate is built from the *tile* index,
                    # not from the column base: `n_base` already carries the
                    # tile's 256 columns, so dividing it by 128 and multiplying
                    # by `sfb_n_box` counts the tile twice and runs off the end
                    # of the scale-factor tensor for every tile past the first.
                    sfb_row = txl.local_scalar(
                        "int32",
                        init=tile_n_idx * txl.int32(sfb_n_box) + txl.int32(sfb_quotient // sf_k_box),
                    )
                    sfb_destination = (
                        cluster_smem_base
                        + txl.uint32(offsets["sSFB"])
                        + txl.cast(ab_prod.stage, "uint32") * txl.uint32(derived["sfb_stage_bytes"])
                        + txl.cast(cluster_x, "uint32")
                        * txl.uint32(derived["sfb_stage_bytes"] // cluster_m)
                        if cluster_m > 1
                        else sfb_slot
                    )
                    with txl.If(_elected()), txl.Then():
                        _tma_load(
                            sfb_destination,
                            maps["sfb"] if weight_mode == "dense" else None,
                            (
                                sfb_coord_0,
                                sfb_coord_1,
                                sfb_row,
                                tile_expert if weight_mode == "dense" else txl.int32(0),
                            ),
                            full_bar,
                            sfb_mcast_mask if cluster_m > 1 else None,
                            two_cta=two_cta,
                            desc_ptr=None if weight_mode == "dense" else sfb_desc,
                        )
                    txl.assign(counter, counter + txl.int32(1))
                    ab_prod.advance()
                    ab_probe.advance()

                take_tile_info(info_cons, slots)

            with txl.unroll(0, ab_stages):
                _wait_plain(ab_pipe.empty.ptr_to([ab_prod.stage]), ab_prod.phase)
                ab_prod.advance()

        # ---- Role 3: warp 4, persistent block-scaled MMA ------------------
        with mma_role:
            txl.ptx.bar.sync(txl.uint32(3), txl.uint32(160))
            acc_tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(acc_tmem_base, tmem_slot.ptr_to([0]))
            sfa_tmem = txl.local_scalar(
                "uint32", init=acc_tmem_base + txl.uint32(derived["num_accumulator_tmem_cols"])
            )
            sfb_tmem = txl.local_scalar(
                "uint32",
                init=acc_tmem_base
                + txl.uint32(derived["num_accumulator_tmem_cols"] + derived["num_sfa_tmem_cols"]),
            )
            sfa_descriptor = txl.local_scalar(
                "uint64", init=_descriptor_with_address(sf_desc_base, smem_base + offsets["sSFA"])
            )
            sfb_descriptor = txl.local_scalar(
                "uint64", init=_descriptor_with_address(sf_desc_base, smem_base + offsets["sSFB"])
            )

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
                    # One accumulator mbarrier stage, two TMEM regions: the
                    # stage index is the producer phase's complement, so
                    # successive tiles alternate between the strided regions.
                    _wait_plain(acc_pipe.empty.ptr_to([acc_prod.stage]), acc_prod.phase)
                    accumulate = txl.local_scalar("uint32", init=txl.uint32(0))
                    # Overlapping accumulator: one mbarrier stage, two TMEM
                    # regions strided by (256 - sf_cols) columns, selected by
                    # the producer phase's complement.
                    acc_column = txl.local_scalar("uint32", init=txl.uint32(0))
                    if derived["overlapping_accum"]:
                        with txl.If(acc_prod.phase == txl.uint32(0)), txl.Then():
                            txl.assign(acc_column, txl.uint32(cta_tile_n - derived["num_sf_tmem_cols"]))
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
                        # One `tcgen05.cp` moves a 32x128b block, i.e. 512
                        # bytes, so a stage wider than that needs one copy per
                        # chunk with TMEM destinations four columns apart --
                        # exactly as SFB does. Emitting a single copy leaves the
                        # upper scale factors zero, and every k block that reads
                        # them contributes nothing.
                        for chunk in range(derived["sfa_stage_bytes"] // 512):
                            stage_sfa = txl.local_scalar(
                                "uint64",
                                init=sfa_descriptor
                                + txl.uint64(derived["sfa_stage_bytes"] // 16)
                                * txl.cast(ab_cons.stage, "uint64")
                                + txl.uint64(chunk * 512 // 16),
                            )
                            with txl.If(_elected()), txl.Then():
                                txl.ptx[f"tcgen05.cp.{cta_group}.32x128b.warpx4"](
                                    sfa_tmem + txl.uint32(4 * chunk), stage_sfa
                                )
                        for chunk in range(derived["sfb_stage_bytes"] // 512):
                            # The TMA writes SFB's chunks in (k, n) order while
                            # TMEM wants them in (n, k) order, so the SMEM chunk
                            # index is transposed. Reading them straight through
                            # feeds each k block a duplicated pair of scale
                            # factor groups.
                            shared_chunk = (chunk % sfb_n_box) * sf_k_box + chunk // sfb_n_box
                            stage_sfb = txl.local_scalar(
                                "uint64",
                                init=sfb_descriptor
                                + txl.uint64(derived["sfb_stage_bytes"] // 16)
                                * txl.cast(ab_cons.stage, "uint64")
                                + txl.uint64(shared_chunk * 512 // 16),
                            )
                            with txl.If(_elected()), txl.Then():
                                txl.ptx[f"tcgen05.cp.{cta_group}.32x128b.warpx4"](
                                    sfb_tmem + txl.uint32(4 * chunk), stage_sfb
                                )
                        # Four block-scaled MMA issues per K tile; only the
                        # first clears the accumulate field.
                        for kblock in range(4):
                            # Each k-block consumes its own slice of the scale
                            # factors: vec-16 walks four TMEM columns per block,
                            # FP4 pairs two blocks per slice and selects the half
                            # with the high bit, and vec-32 FP8 keeps one slice
                            # and selects the block. The chosen SF TMEM addresses
                            # are then folded into the instruction descriptor --
                            # without this every k-block multiplies by the same
                            # scale factors.
                            if sf_vec_size == 16:
                                sfa_offset = kblock * 4
                                sfb_offset = kblock * 4 * (cta_tile_n // 128)
                                selector = 0
                            elif ab_dtype == "float4_e2m1fn":
                                sfa_offset = (kblock // 2) * 4
                                sfb_offset = (kblock // 2) * 4 * (cta_tile_n // 128)
                                selector = (kblock % 2) * 0x80000000
                            else:
                                sfa_offset = 0
                                sfb_offset = 0
                                selector = kblock * 0x40000000
                            sfa_address = txl.cast(
                                sfa_tmem + txl.uint32(sfa_offset) + txl.uint32(selector), "uint32"
                            )
                            sfb_address = txl.cast(
                                sfb_tmem + txl.uint32(sfb_offset) + txl.uint32(selector), "uint32"
                            )
                            runtime_desc = txl.bitwise_and(
                                txl.uint32(instruction_descriptor), txl.uint32(0x9FFFFFCF)
                            )
                            runtime_desc = txl.bitwise_or(
                                runtime_desc,
                                txl.bitwise_and(
                                    txl.shift_right(sfa_address, txl.uint32(1)), txl.uint32(0x60000000)
                                ),
                            )
                            runtime_desc = txl.bitwise_or(
                                runtime_desc,
                                txl.bitwise_and(
                                    txl.shift_right(sfb_address, txl.uint32(26)), txl.uint32(0x30)
                                ),
                            )
                            with txl.If(_elected()), txl.Then():
                                txl.ptx[mma_mnemonic](
                                    txl.cast(acc_tmem_base + acc_column, "uint32"),
                                    a_descriptor
                                    + txl.cast(
                                        ab_cons.stage * (derived["a_stage_bytes"] // 16)
                                        + kblock * 2,
                                        "uint64",
                                    ),
                                    b_descriptor
                                    + txl.cast(
                                        ab_cons.stage * (derived["b_stage_bytes"] // 16)
                                        + kblock * (2 if b_major == "k" else 256),
                                        "uint64",
                                    ),
                                    runtime_desc,
                                    sfa_address,
                                    sfb_address,
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
        # This warp keeps its own copy of the epilogue's alternating direction,
        # so the two agree on which C subtile belongs to which accumulator
        # subtile with no extra handshake.
        with c_role:
            c_prod = txl.PipelineState(c_stages, phase=1)
            c_info = txl.PipelineState(tile_stages, phase=0)
            c_expert = txl.local_scalar("int32")
            c_tile_m = txl.local_scalar("int32")
            c_tile_n = txl.local_scalar("int32")
            c_slots = (c_expert, c_tile_m, c_tile_n)
            c_reverse = txl.local_scalar("uint32", init=txl.uint32(1))

            take_tile_info(c_info, c_slots)
            with txl.While(c_expert >= txl.int32(0)):
                reverse_now = txl.local_scalar("uint32", init=c_reverse)
                txl.assign(c_reverse, c_reverse ^ txl.uint32(1))
                c_row = txl.local_scalar(
                    "int32", init=expert_row_base(c_expert) + c_tile_m * txl.int32(cta_tile_m)
                )
                # The 2N interleaved column tile this work tile owns.
                d_tile_n = txl.local_scalar("int32", init=c_tile_n * txl.int32(cta_tile_n * 2))

                subtile = txl.local_scalar("int32", init=txl.int32(0))
                with txl.While(subtile < txl.int32(epi_subtiles)):
                    real_subtile = txl.local_scalar("int32", init=subtile)
                    with txl.If(reverse_now == txl.uint32(1)), txl.Then():
                        txl.assign(real_subtile, txl.int32(epi_subtiles - 1) - subtile)
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

            if generate_sfd:
                norm_const = txl.local_scalar("float32")
                txl.ptx.ld.global_.b32(norm_const, named["norm_const"].ptr_to([0]))
                # The source's `get_dtype_rcp_limits` caps E5M2 at 128 rather
                # than at its largest finite value; matching it matters because
                # this factor scales every stored scale factor.
                rcp_limit = 1.0 / (448.0 if d_dtype == "float8_e4m3fn" else 128.0)

            acc_cons = txl.PipelineState(acc_stages, phase=0)
            c_cons = txl.PipelineState(c_stages, phase=0)
            d_prod = txl.PipelineState(max(1, d_stages // 2), phase=1)
            epi_info = txl.PipelineState(tile_stages, phase=0)
            epi_expert = txl.local_scalar("int32")
            epi_tile_m = txl.local_scalar("int32")
            epi_tile_n = txl.local_scalar("int32")
            epi_slots = (epi_expert, epi_tile_m, epi_tile_n)

            # The register tile is walked two elements at a time throughout the
            # epilogue, and the pairs are always issued packed: one
            # `mul.rn.f32x2` / `add.rn.f32x2` per pair where the scalar form
            # issues two, for identical numbers -- each half of a packed line
            # rounds exactly as its scalar sibling.
            #
            # The source only writes packed intrinsics under `vectorized_f32`,
            # but its scalar branch is packed for it: on a `vectorized_f32=False`
            # shape the reference still retires roughly a quarter of TIRx's FP32
            # instruction count, which a scalar lowering cannot do. Emitting the
            # packed form on every specialization is what actually matches the
            # reference's machine code.
            packed_pair = txl.local_scalar("uint64")
            ops = _arithmetic(True, packed_pair)
            epi_lane = txl.local_scalar("int32", init=txl.thread_id() % txl.int32(32))
            epi_warp = _warp_uniform(txl.thread_id() // txl.int32(32))

            def sf_atom_offset(row, group, groups_per_row):
                """Byte offset of one scale factor in the interleaved SF layout.

                The buffer is ``(1, ceil(rows/128), ceil(groups/4), 32, 4, 4)``,
                so logical row ``m`` lands at ``(m % 32, (m % 128) // 32, m //
                128)`` and logical group ``g`` at ``(g % 4, g // 4)``.
                """
                rest_k = _ceil_div(groups_per_row, 4)
                return (
                    ((row // txl.int32(128)) * txl.int32(rest_k) + group // txl.int32(4)) * txl.int32(512)
                    + (row % txl.int32(32)) * txl.int32(16)
                    + ((row % txl.int32(128)) // txl.int32(32)) * txl.int32(4)
                    + group % txl.int32(4)
                )

            # The row path multiplies `amax * rcp_limit * norm_const` in that
            # order while the column path folds the two constants first and
            # multiplies once. Under E8M0 that is not a distinction without a
            # difference: `cvt.rp` rounds toward positive infinity, so a single
            # ULP between the two orderings moves the stored scale a whole
            # binade whenever the exact product lands on a power of two.
            combined_scale = None
            if generate_sfd:
                combined_scale = txl.local_scalar("float32")
                txl.ptx["mul.f32"](combined_scale, txl.float32(rcp_limit), norm_const)

            def pack_scales(destination, scales):
                """Four FP32 scales into one word of four E8M0 bytes.

                The round-toward-positive-infinity family, not the round-nearest
                one the D values take. These are the only `cvt.rp` in the export
                with no matching upcast partner -- every other one is half of a
                dequantize round trip.
                """
                low = txl.local_scalar("uint16")
                high = txl.local_scalar("uint16")
                converter = (
                    "cvt.rp.satfinite.ue8m0x2.f32"
                    if sf_dtype == "float8_e8m0fnu"
                    else (
                        "cvt.rn.satfinite.e4m3x2.f32"
                        if sf_dtype == "float8_e4m3fn"
                        else "cvt.rn.satfinite.e5m2x2.f32"
                    )
                )
                txl.ptx[converter](low, scales[1], scales[0])
                txl.ptx[converter](high, scales[3], scales[2])
                txl.ptx["mov.b32"](destination, low, high)

            def round_scales(destinations, scales):
                """Round four scales into the scale dtype and read them back.

                The conversions are inherently two-at-a-time, so a group of four
                costs two of each rather than the four pairs a per-element round
                trip would spend. Widening a BF16 to FP32 is a shift, so reading
                the pair back is masking rather than another conversion.
                """
                if sf_dtype == "float8_e8m0fnu":
                    packed = txl.local_scalar("uint32")
                    pack_scales(packed, scales)
                    piece = txl.local_scalar("uint16")
                    wide = txl.local_scalar("uint32")
                    bits = txl.local_scalar("uint32")
                    shifted = txl.local_scalar("uint32")
                    for half in range(2):
                        txl.ptx["shr.b32"](bits, packed, txl.uint32(16 * half))
                        txl.ptx["cvt.u16.u32"](piece, bits)
                        txl.ptx["cvt.rn.bf16x2.ue8m0x2"](wide, piece)
                        txl.ptx["shl.b32"](shifted, wide, txl.uint32(16))
                        txl.ptx["mov.b32"](destinations[2 * half], shifted)
                        txl.ptx["and.b32"](shifted, wide, txl.uint32(0xFFFF0000))
                        txl.ptx["mov.b32"](destinations[2 * half + 1], shifted)
                else:
                    byte = txl.local_scalar("uint32")
                    for j in range(4):
                        round_scale(destinations[j], byte, scales[j])

            def round_scale(destination, byte, scaled):
                """Round a scaled maximum into the scale dtype and read it back.

                E8M0 rounds toward positive infinity -- the next power of two at
                or above the value -- and the round trip is the pair of
                conversions the export uses rather than arithmetic on the
                exponent field.
                """
                if sf_dtype == "float8_e8m0fnu":
                    rounded = txl.local_scalar("uint16")
                    txl.ptx["cvt.rp.satfinite.ue8m0x2.f32"](rounded, txl.float32(0.0), scaled)
                    txl.ptx["cvt.u32.u16"](byte, rounded)
                    txl.ptx["and.b32"](byte, byte, txl.uint32(0xFF))
                    wide = txl.local_scalar("uint32")
                    txl.ptx["cvt.rn.bf16x2.ue8m0x2"](wide, rounded)
                    # The low BF16 carries the low scale factor, and widening a
                    # BF16 to FP32 is a shift into the high half.
                    txl.ptx["shl.b32"](wide, wide, txl.uint32(16))
                    txl.ptx["mov.b32"](destination, wide)
                else:
                    converter = (
                        "cvt.rn.satfinite.e4m3x2.f32"
                        if sf_dtype == "float8_e4m3fn"
                        else "cvt.rn.satfinite.e5m2x2.f32"
                    )
                    reader = (
                        "cvt.rn.f16x2.e4m3x2"
                        if sf_dtype == "float8_e4m3fn"
                        else "cvt.rn.f16x2.e5m2x2"
                    )
                    packed_byte = txl.local_scalar("uint16")
                    txl.ptx[converter](packed_byte, txl.float32(0.0), scaled)
                    txl.ptx["cvt.u32.u16"](byte, packed_byte)
                    txl.ptx["and.b32"](byte, byte, txl.uint32(0xFF))
                    half_pair = txl.local_scalar("uint32")
                    low_half = txl.local_scalar("uint16")
                    txl.ptx[reader](half_pair, txl.cast(byte, "uint16"))
                    txl.ptx["cvt.u16.u32"](low_half, half_pair)
                    txl.ptx["cvt.f32.f16"](destination, low_half)

            def reciprocal_scale(destination, scale):
                """``norm_const * rcp_approx(scale)``, clamped at FP32_MAX.

                The source clamps with the NaN-propagating `min` rather than
                special-casing a zero scale. A zero scale only arises when every
                value in the vector is zero, and zero times FP32_MAX is still
                zero, so the clamp and the reference's explicit zero agree.
                """
                txl.ptx["rcp.approx.ftz.f32"](destination, scale)
                txl.ptx["mul.f32"](destination, destination, norm_const)
                txl.ptx["min.NaN.f32"](destination, destination, txl.float32(3.4028234663852886e38))

            def quantize_row(fragment, words, slot):
                """Row-direction quantization of one 32-column fragment.

                A scale-factor vector runs along the row, and every
                specialization that produces SFD uses a vector of 32, so one
                fragment is exactly one vector and the reduction never leaves the
                thread. The scale itself goes to a register slot, not to memory:
                four consecutive subtile halves are packed and stored together.
                `rScaled` doubles as the reduction scratch before it holds the
                rescaled values.
                """
                for i in range(32):
                    txl.ptx["abs.f32"](rScaled[i], fragment[i])
                width = 32
                while width > 1:
                    width //= 2
                    for i in range(width):
                        txl.ptx["max.NaN.f32"](rScaled[i], rScaled[i], rScaled[i + width])
                txl.ptx["mul.f32"](rSFDr[slot], rScaled[0], txl.float32(rcp_limit))
                txl.ptx["mul.f32"](rSFDr[slot], rSFDr[slot], norm_const)
                rounded = txl.local_scalar("float32")
                unused = txl.local_scalar("uint32")
                round_scale(rounded, unused, rSFDr[slot])
                factor = txl.local_scalar("float32")
                reciprocal_scale(factor, rounded)
                uniform = (factor, factor)
                for element in range(0, 32, 2):
                    ops["product"](
                        (rScaled[element], rScaled[element + 1]),
                        (fragment[element], fragment[element + 1]),
                        uniform,
                    )
                _pack_output(words, rScaled, d_dtype, d_bits)

            def quantize_column(fragment, words, slot):
                """Column-direction quantization of the same fragment.

                Here a vector runs down 32 consecutive rows, which is exactly one
                warp's worth of threads, so each column's maximum is a warp
                reduction. The eight groups of four serve both purposes at once:
                the four reductions give this lane the factors for its own four
                elements, and the lane whose index matches keeps that column's
                scale because it is the one that will store it. There is no
                second reduction pass.
                """
                magnitude = txl.local_scalar("float32")
                column_max = [txl.local_scalar("float32") for _ in range(4)]
                column_scale = [txl.local_scalar("float32") for _ in range(4)]
                rounded_scale = [txl.local_scalar("float32") for _ in range(4)]
                folded = (combined_scale, combined_scale)
                txl.assign(rSFDc[slot], txl.float32(0.0))
                for group in range(8):
                    for j in range(4):
                        txl.ptx["abs.f32"](magnitude, fragment[4 * group + j])
                        txl.ptx["redux_sync.max.NaN.f32"](
                            column_max[j], magnitude, txl.uint32(0xFFFFFFFF)
                        )
                    for j in range(0, 4, 2):
                        ops["product"](
                            (column_scale[j], column_scale[j + 1]),
                            (column_max[j], column_max[j + 1]),
                            folded,
                        )
                    for j in range(4):
                        txl.ptx["selp.f32"](
                            rSFDc[slot],
                            column_scale[j],
                            rSFDc[slot],
                            txl.cast(epi_lane == txl.int32(4 * group + j), "bool"),
                        )
                    # The group's four scales round-trip together. Converting
                    # them one at a time costs four times the conversions, and
                    # the scale-factor round trip is the epilogue's dominant
                    # fixed cost -- it is what a short K loop stops hiding.
                    round_scales(rounded_scale, column_scale)
                    for j in range(4):
                        reciprocal_scale(column_scale[j], rounded_scale[j])
                    for j in range(0, 4, 2):
                        ops["product"](
                            (rScaled[4 * group + j], rScaled[4 * group + j + 1]),
                            (fragment[4 * group + j], fragment[4 * group + j + 1]),
                            (column_scale[j], column_scale[j + 1]),
                        )
                _pack_output(words, rScaled, d_dtype, d_bits)

            def store_scale_factors(real_subtile, tile_base):
                """Flush four subtile halves' scale factors, every second subtile.

                Both packs are unconditional and precede both guards; only the
                stores are predicated, and over two different extents. The four
                row scales are four consecutive scale-factor groups of one token
                row, which is four contiguous bytes. The four column scales
                belong to four output columns 32 apart, which the interleaved
                atom puts at a stride of four bytes -- close, but not one store.

                A cluster covers `cluster_n` column tiles whether or not the
                output has that many, so the last cluster carries tiles past the
                end. The D store is a TMA and its descriptor drops those writes
                on its own; these go out as plain stores and need the bound
                spelled out, or they overwrite a later row's scale factors -- or
                run off the end of the buffer entirely.
                """
                packed_row = txl.local_scalar("uint32")
                packed_col = txl.local_scalar("uint32")
                pack_scales(packed_row, rSFDr)
                pack_scales(packed_col, rSFDc)
                group_base = txl.local_scalar(
                    "int32",
                    init=tile_base // txl.int32(sf_vec_size)
                    + txl.int32(4) * (real_subtile // txl.int32(2)),
                )
                first_column = txl.local_scalar("int32", init=group_base * txl.int32(sf_vec_size))
                with txl.If(first_column < txl.int32(N_out)), txl.Then():
                    txl.ptx["st.global.b32"](
                        named["sfd_row"].ptr_to(
                            [sf_atom_offset(thread_row, group_base, _ceil_div(N_out, sf_vec_size))]
                        ),
                        packed_row,
                    )
                # The column buffer is the transpose of the row one: its rows are
                # output columns and its vectors are groups of 32 token rows.
                # This holds for `discrete_col_sfd` too -- see the module
                # docstring for why that flag does not change what is written
                # here even though it changes what upstream's buffer holds.
                with txl.If(first_column + epi_lane < txl.int32(N_out)), txl.Then():
                    column_base = txl.local_scalar(
                        "int32",
                        init=sf_atom_offset(
                            first_column + epi_lane,
                            thread_row // txl.int32(sf_vec_size),
                            _ceil_div(tokens_total, sf_vec_size),
                        ),
                    )
                    byte = txl.local_scalar("uint32")
                    for j in range(4):
                        txl.ptx["shr.b32"](byte, packed_col, txl.uint32(8 * j))
                        txl.ptx.st.global_.b8(
                            named["sfd_col"].ptr_to([column_base + txl.int32(4 * j)]),
                            txl.cast(byte, "uint8"),
                        )

            def dbias_reduce(real_subtile, tile_base):
                """One subtile's contribution to this expert's dBias column sums.

                Each thread holds one row's 32 gate and 32 up values, so a column
                sum is a reduction *across* threads. The 64 columns go to SMEM as
                (column, row) -- the transpose -- each warp into its own 8 KiB
                block; then one thread takes two whole columns and sums their 32
                rows, the four warps' partial sums are combined, and warp 0
                atomically accumulates a bf16 pair per column.

                Rows are stored with the source's `((col >> 1) & 7) << 2` XOR on
                the row-group index. It cancels out of the sum -- it is there so
                that the 32 lanes reading 32 different columns hit 32 different
                banks instead of all landing on the same one.
                """
                warp_base = txl.local_scalar(
                    "int32", init=offsets["sDbias"] + epi_warp * txl.int32(64 * 32 * 4)
                )
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
                    # Same over-range tiles as the scale factors, and the source
                    # guards this accumulate for the same reason.
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
            rD1 = txl.alloc_local((d_words,), "uint32")
            rD2 = txl.alloc_local((d_words,), "uint32")
            if generate_sfd:
                # The row- and column-quantized outputs are different numbers,
                # so both live at once: one goes to D_row, the other to D_col.
                rD1_col = txl.alloc_local((d_words,), "uint32")
                rD2_col = txl.alloc_local((d_words,), "uint32")
                rScaled = txl.alloc_local((32,), "float32")
                # One slot per subtile half; four halves share a store.
                rSFDr = txl.alloc_local((4,), "float32")
                rSFDc = txl.alloc_local((4,), "float32")
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
                # The accumulator stage alternates with the consumer phase, and
                # the subtile walk reverses on the tiles that land in stage 0.
                acc_column = txl.local_scalar("uint32", init=txl.uint32(0))
                reverse_subtile = txl.local_scalar("uint32", init=txl.uint32(0))
                if derived["overlapping_accum"]:
                    with txl.If(acc_cons.phase == txl.uint32(1)), txl.Then():
                        txl.assign(acc_column, txl.uint32(cta_tile_n - derived["num_sf_tmem_cols"]))
                    with txl.If(acc_cons.phase == txl.uint32(0)), txl.Then():
                        txl.assign(reverse_subtile, txl.uint32(1))

                thread_row = txl.local_scalar(
                    "int32",
                    init=row_base
                    + (epi_tile_m // txl.int32(atom_thr)) * txl.int32(cta_tile_m * atom_thr)
                    + (block_x % txl.int32(atom_thr)) * txl.int32(cta_tile_m)
                    + txl.thread_id()
                    if atom_thr > 1
                    else row_base + epi_tile_m * txl.int32(cta_tile_m) + txl.thread_id(),
                )
                prob_value = txl.local_scalar("float32", init=txl.float32(1.0))
                if with_prob:
                    txl.ptx.ld.global_.b32(prob_value, named["prob"].ptr_to([thread_row]))
                dprob_acc = txl.local_scalar("float32", init=txl.float32(0.0))
                amax_gate = txl.local_scalar("float32", init=txl.float32(0.0))
                amax_up = txl.local_scalar("float32", init=txl.float32(0.0))

                _wait_plain(acc_pipe.full.ptr_to([acc_cons.stage]), acc_cons.phase)

                d_tile_base = txl.local_scalar("int32", init=epi_tile_n * txl.int32(cta_tile_n * 2))

                # The source pins `unroll=1` on this loop and its export
                # carries `.pragma "nounroll"`; a `While` lowers with
                # `#pragma unroll 1`, which is the shape the source has.
                # The reversed walk is a direction, not a subtraction. Deriving
                # `epi_subtiles - 1 - subtile` inside the loop costs a negate, a
                # select and the two predicates that feed it on every subtile;
                # stepping a counter costs one add, and the whole per-subtile
                # address prologue hangs off this value.
                real_subtile = txl.local_scalar("int32", init=txl.int32(0))
                subtile_step = txl.local_scalar("int32", init=txl.int32(1))
                with txl.If(reverse_subtile == txl.uint32(1)), txl.Then():
                    txl.assign(real_subtile, txl.int32(epi_subtiles - 1))
                    txl.assign(subtile_step, txl.int32(-1))
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

                    if derived["overlapping_accum"]:
                        with txl.If(subtile == txl.int32(early_release)), txl.Then():
                            txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                            with txl.If(_elected()), txl.Then():
                                peer_bar = txl.local_scalar("uint32")
                                local_bar = txl.local_scalar("uint32")
                                txl.assign(
                                    local_bar,
                                    txl.cuda.cvta_generic_to_shared(
                                        acc_pipe.empty.ptr_to([acc_cons.stage])
                                    ),
                                )
                                # The accumulator lives on the leader CTA of
                                # this MMA pair, so the release goes there, not
                                # to CTA 0. Targeting rank 0 leaves every other
                                # CTA's MMA waiting on an accumulator nobody
                                # ever frees.
                                txl.ptx["mapa.shared::cluster.u32"](
                                    peer_bar,
                                    local_bar,
                                    txl.cast(
                                        cluster_rank - cluster_v if atom_thr > 1 else cluster_rank,
                                        "uint32",
                                    ),
                                )
                                txl.ptx["mbarrier.arrive.shared::cluster.b64"](peer_bar, txl.uint32(1))
                            acc_cons.advance()

                    # Two C stages per accumulator subtile: gate then up.
                    c_row_bytes = 32 * c_bits // 8
                    c_words = c_row_bytes // 4
                    for fragment in (rC1, rC2):
                        _wait_plain(c_pipe.full.ptr_to([c_cons.stage]), c_cons.phase)
                        c_slot = offsets["sC"] + c_cons.stage * derived["c_stage_bytes"]
                        # Each thread owns one row of the 128x32 subtile, so
                        # its slice starts at row * 32 * c_bits/8 bytes.
                        raw = txl.alloc_local((c_words,), "uint32")
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
                        # instructions, and with only two C stages on a
                        # four-bit specialization the producer has no slack to
                        # absorb that.
                        txl.ptx["fence.proxy.async.shared::cta"]()
                        txl.cuda.warp_sync()
                        with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                            txl.ptx.mbarrier.arrive.shared.b64(c_pipe.empty.ptr_to([c_cons.stage]))
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
                        return (txl.local_scalar("float32"), txl.local_scalar("float32"))

                    acc = pair()
                    gate = pair()
                    up = pair()
                    d_gate = pair()
                    d_up = pair()
                    d_prob = pair()
                    sig = pair()
                    work = pair()
                    step = pair()
                    helper = pair()
                    prob_pair = (prob_value, prob_value)
                    alpha_pair = (square_alpha, square_alpha)
                    beta_pair = (beta_value, beta_value)
                    situ = {
                        name: pair() for name in ("gate_value", "up_value", "gate_grad", "up_grad")
                    }
                    situ_tmp = {name: pair() for name in "abcdefgh"}
                    if act == "dgeglu":
                        # The gradient filters stay predicates rather than
                        # 1.0/0.0 factors; see where they are applied below.
                        keep_gate = (txl.local_scalar("bool"), txl.local_scalar("bool"))
                        keep_up = (txl.local_scalar("bool"), txl.local_scalar("bool"))
                    if generate_dprob:
                        # The source reduces dProb as a packed pair and folds the
                        # two halves in once per subtile, not once per element.
                        dprob_pair = pair()
                        for half in range(2):
                            txl.assign(dprob_pair[half], txl.float32(0.0))

                    for element in range(0, 32, 2):
                        span = (element, element + 1)
                        ops["product"](acc, tuple(rAcc[j] for j in span), alpha_pair)
                        if act == "dgeglu":
                            # dGeGLU clamps instead of scaling by beta.
                            for half in range(2):
                                txl.assign(gate[half], rC1[span[half]])
                                txl.assign(up[half], rC2[span[half]])
                        else:
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
                            if generate_dprob:
                                ops["product"](helper, step, prob_pair)  # acc_prob * sig
                                ops["product"](d_up, helper, gate)  # acc_prob * swish
                                ops["product"](helper, helper, up)
                                ops["product"](step, step, up)
                                # The accumulation absorbs the last multiply:
                                # `acc * up * swish` is never materialised.
                                ops["fused"](dprob_pair, step, gate, dprob_pair)
                            else:
                                # Without dProb the accumulator carries no
                                # per-row factor, so `acc_prob` is `acc` and the
                                # multiply by a register that holds 1.0 -- which
                                # inline assembly keeps -- goes away with it.
                                ops["product"](d_up, step, gate)
                                ops["product"](helper, step, up)
                            ops["complement"](work, 1.0, sig, situ_tmp["h"])
                            ops["fused"](work, gate, work, _spread(1.0))
                            ops["product"](d_gate, helper, work)
                        elif act == "dgeglu":
                            for half in range(2):
                                txl.assign(
                                    keep_gate[half],
                                    txl.cast(gate[half] <= txl.float32(glu_clamp_max), "bool"),
                                )
                                txl.assign(
                                    keep_up[half],
                                    txl.cast(
                                        txl.And(
                                            up[half] >= txl.float32(glu_clamp_min),
                                            up[half] <= txl.float32(glu_clamp_max),
                                        ),
                                        "bool",
                                    ),
                                )
                            y_gate = situ_tmp["a"]
                            y_up = situ_tmp["b"]
                            # The source's export clamps with `setp` feeding
                            # `selp`, which costs two instructions per bound.
                            # `min`/`max` reach the same value in one: the two
                            # forms differ only when the input is NaN, which the
                            # clamped operand -- a C-matrix element that has
                            # already survived quantization -- cannot be.
                            for half in range(2):
                                txl.ptx["min.f32"](y_gate[half], gate[half], txl.float32(glu_clamp_max))
                                txl.ptx["min.f32"](y_up[half], up[half], txl.float32(glu_clamp_max))
                                txl.ptx["max.f32"](y_up[half], y_up[half], txl.float32(glu_clamp_min))
                            ops["scaled"](work, y_gate, geglu_alpha)
                            _sigmoid(ops, sig, work, situ_tmp["c"])
                            grad = situ_tmp["d"]
                            ops["product"](grad, acc, prob_pair)
                            offset_up = situ_tmp["e"]
                            ops["offset"](offset_up, y_up, linear_offset)
                            inner = situ_tmp["f"]
                            ops["complement"](inner, 1.0, sig, situ_tmp["h"])
                            ops["scaled"](work, y_gate, geglu_alpha)
                            ops["fused"](inner, work, inner, _spread(1.0))
                            ops["product"](d_gate, grad, sig)
                            ops["product"](d_gate, d_gate, inner)
                            ops["product"](d_gate, d_gate, offset_up)
                            ops["product"](d_up, grad, y_gate)
                            ops["product"](d_up, d_up, sig)
                            # Zeroing a filtered gradient is a select, not a
                            # multiply by a mask that had to be materialized
                            # first: one instruction per element instead of two.
                            for half in range(2):
                                txl.ptx["selp.f32"](
                                    d_gate[half], d_gate[half], txl.float32(0.0), keep_gate[half]
                                )
                                txl.ptx["selp.f32"](
                                    d_up[half], d_up[half], txl.float32(0.0), keep_up[half]
                                )
                            if generate_dprob:
                                ops["product"](work, y_gate, sig)
                                ops["product"](work, work, offset_up)
                                ops["fused"](dprob_pair, work, acc, dprob_pair)
                        else:
                            _situglu(ops, situ, gate, up, situ_beta1, situ_beta2, situ_tmp)
                            ops["product"](helper, acc, prob_pair)
                            ops["product"](d_gate, helper, situ["up_value"])
                            ops["product"](d_gate, d_gate, situ["gate_grad"])
                            ops["product"](d_up, helper, situ["gate_value"])
                            ops["product"](d_up, d_up, situ["up_grad"])
                            if generate_dprob:
                                ops["product"](d_prob, acc, situ["gate_value"])
                                ops["fused"](dprob_pair, d_prob, situ["up_value"], dprob_pair)

                        for half in range(2):
                            txl.assign(rC1[span[half]], d_gate[half])
                            txl.assign(rC2[span[half]], d_up[half])

                    if generate_dprob:
                        folded_prob = txl.local_scalar("float32")
                        txl.ptx["add.f32"](folded_prob, dprob_pair[0], dprob_pair[1])
                        txl.ptx["add.f32"](dprob_acc, dprob_acc, folded_prob)

                    # ---- convert D and stage it through SMEM -------------
                    if generate_sfd:
                        for half, (fragment, row_words, col_words) in enumerate(
                            ((rC1, rD1, rD1_col), (rC2, rD2, rD2_col))
                        ):
                            # Four subtile halves share one scale-factor store,
                            # so each writes its scale into its own register slot
                            # and the store happens every second subtile.
                            slot = txl.local_scalar(
                                "int32",
                                init=(real_subtile * txl.int32(2) + txl.int32(half)) % txl.int32(4),
                            )
                            quantize_row(fragment, row_words, slot)
                            quantize_column(fragment, col_words, slot)
                        with txl.If(subtile % txl.int32(2) == txl.int32(1)), txl.Then():
                            store_scale_factors(real_subtile, d_tile_base)
                    else:
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
                    if generate_sfd:
                        stage_fragment(rD1_col, slot1, "sD_col")
                    stage_fragment(rD2, slot2)
                    if generate_sfd:
                        stage_fragment(rD2_col, slot2, "sD_col")
                    txl.ptx["fence.proxy.async.shared::cta"]()
                    txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))

                    with txl.If((warp == txl.int32(0)) & (txl.lane_id() == 0)), txl.Then():
                        # The two D subtiles of one accumulator subtile land in
                        # adjacent halves of the 2N region, which is why the
                        # column index is 2 * real_subtile + {0, 1}.
                        d_row_coord = txl.local_scalar(
                            "int32", init=row_base + epi_tile_m * txl.int32(cta_tile_m)
                        )
                        stores = [("d_row", "sD")]
                        if generate_sfd:
                            stores.append(("d_col", "sD_col"))
                        # The two D stores precede the two D_col stores.
                        for map_name, region in stores:
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
                        d_prod.advance()
                    # The absolute-maximum tree runs with the tile's D store
                    # already in flight, for the same reason the dBias
                    # reduction below does: it reads the activation fragments,
                    # which the packing only copied, so nothing in it depends on
                    # the store. It has to sit after the barrier that precedes
                    # the store rather than merely after the shared writes --
                    # put in front of that barrier it is simply scheduled back
                    # above them.
                    if generate_amax:
                        # An in-register tree per subtile, then one running
                        # maximum across subtiles. The tree propagates NaN and
                        # the accumulation does not, which is what the source's
                        # `reduce(MAX)` and `cute.arch.fmax` respectively select.
                        # `rAcc` is spent by now and doubles as the scratch.
                        for fragment, running in ((rC1, amax_gate), (rC2, amax_up)):
                            for i in range(32):
                                txl.ptx["abs.f32"](rAcc[i], fragment[i])
                            span = 32
                            while span > 1:
                                span //= 2
                                for i in range(span):
                                    txl.ptx["max.NaN.f32"](rAcc[i], rAcc[i], rAcc[i + span])
                            txl.ptx["max.NaN.f32"](rAcc[0], rAcc[0], txl.float32(0.0))
                            txl.ptx["max.f32"](running, running, rAcc[0])

                    # The dBias column sums are taken with the tile's D store
                    # already in flight. They read the activation fragments,
                    # which the packing only copied, and they touch their own
                    # shared region, so nothing in them depends on the store --
                    # and the reduction is a transpose through shared memory
                    # with two barriers in it, which is exactly the kind of work
                    # a bulk store wants underneath it.
                    if generate_dbias:
                        dbias_reduce(real_subtile, d_tile_base)
                    txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))
                    txl.assign(real_subtile, real_subtile + subtile_step)
                    txl.assign(subtile, subtile + txl.int32(1))

                # The next record is taken before the amax and dProb tails, as
                # the source does, so the tile-info slot is freed as early as
                # possible. That overwrites `epi_expert`, so the tails below use
                # the copy taken here rather than the record they no longer own.
                finished_expert = txl.local_scalar("int32", init=epi_expert)
                take_tile_info(epi_info, epi_slots)
                if generate_amax:
                    # The warp reduction propagates NaN and the four-warp combine
                    # does not, matching `warp_redux_sync(nan=True)` and
                    # `cute.arch.fmax`. Only the global accumulate is an integer
                    # maximum: every value here is an absolute value, so the IEEE
                    # patterns are non-negative and order like signed integers,
                    # and the output starts at -inf, whose pattern is negative.
                    warp_max = txl.local_scalar("float32")
                    block_max = txl.local_scalar("float32")
                    other_max = txl.local_scalar("float32")
                    block_bits = txl.local_scalar("int32")
                    for index, running in ((0, amax_gate), (1, amax_up)):
                        txl.ptx["redux_sync.max.NaN.f32"](warp_max, running, txl.uint32(0xFFFFFFFF))
                        with txl.If(epi_lane == txl.int32(0)), txl.Then():
                            txl.ptx.st.shared.b32(
                                smem.ptr_to([offsets["sAmax"] + epi_warp * txl.int32(4)]), warp_max
                            )
                        txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))
                        with txl.If(txl.And(epi_warp == txl.int32(0), epi_lane == txl.int32(0))), txl.Then():
                            txl.ptx.ld.shared.b32(block_max, smem.ptr_to([offsets["sAmax"]]))
                            for other in range(1, 4):
                                txl.ptx.ld.shared.b32(
                                    other_max, smem.ptr_to([offsets["sAmax"] + other * 4])
                                )
                                txl.ptx["max.f32"](block_max, block_max, other_max)
                            txl.ptx["mov.b32"](block_bits, block_max)
                            # A reduction, not a returning atomic: the old
                            # value is not wanted, and asking for it puts the
                            # accumulate on the scoreboard so the warp waits for
                            # a round trip it has no use for.
                            txl.ptx["red.global.max.s32"](
                                named["amax"].ptr_to(
                                    [finished_expert * txl.int32(2) + txl.int32(index)]
                                ),
                                block_bits,
                            )
                        txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))

                if generate_dprob:
                    # One reduction per thread per work tile. The returning form
                    # would put every one of these on the scoreboard for a value
                    # that is immediately discarded.
                    txl.ptx["red.global.add.f32"](named["dprob"].ptr_to([thread_row]), dprob_acc)

            # Release the TMEM allocation permit and free the columns. Both are
            # warp 0's: the permit is a warp-wide instruction and the source
            # issues it under the same predicate as the deallocation.
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(128))
            with txl.If(warp == txl.int32(0)), txl.Then():
                txl.ptx[f"tcgen05.relinquish_alloc_permit.{cta_group}.sync.aligned"]()
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
            sfb_mcast_mask,
            ab_consumer_mask,
            acc_producer_mask,
            cluster_smem_base,
            tmem_slot,
        )

    def build_helper():
        """Pre-kernel launched before the main kernel on the same stream.

        For the dynamic scheduler it resets the global work counter. For discrete
        weights each block publishes one expert's B and SFB TensorMap image into
        the workspace, which is where the main kernel's TMA reads them from.

        A discrete expert's weights are its own allocation, so only the global
        address differs between the experts' images (probe/descriptor_images.txt
        shows the other fifteen words identical across all four). The host
        prelude therefore encodes one template -- correct in every field but the
        address -- and each block copies it and patches the address with
        ``tensormap.replace``.
        """

        def helper_prelude(params):
            descriptors = {name: txl.stack_alloca("tensormap", 1) for name in ("b", "sfb")}
            # The template's address is a placeholder: it has to be a real
            # 16-byte-aligned allocation for the encode call to accept it, and
            # `tensormap.replace` overwrites it per expert below. The pointer
            # arrays themselves are the convenient stand-in.
            encode_weight_maps(descriptors, params["b"].data, params["sfb"].data, 1)
            return (descriptors["b"], descriptors["sfb"])

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
            b, sfb, workspace = operands
            expert = txl.cta_id()[0]
            if weight_mode == "discrete":
                slot = txl.local_scalar("int32", init=txl.int32(256) * expert)
                with txl.If(_elected()), txl.Then():
                    # This kernel is nothing but memory latency: six global
                    # reads and four writes on one lane. The B and SFB images
                    # are independent, so every read is issued before any write
                    # rather than running the two descriptors one after the
                    # other -- otherwise the second expert-pointer load does not
                    # start until the first image has been stored, and the two
                    # round trips add instead of overlapping.
                    addresses = []
                    for pointers in (b, sfb):
                        address = txl.local_scalar("uint64")
                        txl.ptx.ld.global_.b64(address, pointers.ptr_to([expert]))
                        addresses.append(address)
                    images = [read_tensormap_image(template) for template in host]
                    for index, groups in enumerate(images):
                        write_tensormap_image(
                            groups,
                            workspace.ptr_to([slot + txl.int32(128 * index)]),
                            addresses[index],
                        )
                txl.cuda.warp_sync()
            if sched == "dynamic":
                counter_offset = 2 * L * 128 if weight_mode == "discrete" else 0
                with txl.If(expert == txl.int32(0)), txl.Then():
                    with txl.If(_elected()), txl.Then():
                        txl.ptx.st.global_.b32(workspace.ptr_to([counter_offset]), txl.int32(0))

        # The launch passes the per-expert pointer arrays for discrete weights
        # and `padded_offsets` (int32) for dense ones, so the first two
        # parameters change dtype with the mode.
        pointer_dtype = "int64" if weight_mode == "discrete" else "int32"
        if weight_mode == "discrete":
            helper_body = _entry_point(["b", "sfb", "workspace"], helper)
        else:
            # Only the discrete branch has a host prelude, and an entry may not
            # take the keyword-only `host` parameter without one.
            def helper_body(b, sfb, workspace):
                helper((b, sfb, workspace), ())

        helper_body.__annotations__ = {
            "b": txl.gptr[pointer_dtype, (L,)],
            "sfb": txl.gptr[pointer_dtype, (L,)],
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
        K_dim=config["K"],
        weight_mode=config["weight_mode"],
        sched=config["sched"],
        act=config["act"],
        ab_dtype=config["ab_dtype"],
        sf_dtype=config["sf_dtype"],
        sf_vec_size=config["sf_vec_size"],
        c_dtype=config["c_dtype"],
        d_dtype=config["d_dtype"],
        b_major=config["b_major"],
        mma_tiler_mn=tuple(config["mma_tiler_mn"]),
        cluster_shape_mn=tuple(config["cluster_shape_mn"]),
        vectorized_f32=config["vectorized_f32"],
        with_dbias=config["with_dbias"],
        with_prob=config["with_prob"],
        with_amax=config["with_amax"],
        discrete_col_sfd=config["discrete_col_sfd"],
        linear_offset=config["linear_offset"],
        geglu_alpha=config["geglu_alpha"],
        glu_clamp_max=config["glu_clamp_max"],
        glu_clamp_min=config["glu_clamp_min"],
        situ_beta1=config["situ_beta1"],
        situ_beta2=config["situ_beta2"],
    )
