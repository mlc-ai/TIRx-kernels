# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import math
from typing import Any

import numpy as np

import tvm
from tirx_kernels.runner import bench
from tvm.ir.type import PointerType, PrimType
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T
from tvm.script.ir_builder.tirx import tile as Tx


def _named(obj, name):
    """Give a builder-created IR object its parser-compatible name."""
    IRBuilder.current().name(name, obj)
    return obj


def _scalar(name, dtype, value=None):
    """Allocate a named mutable local scalar and optionally initialize it."""
    result = T.local_scalar(dtype)
    _named(result.scalar.buffer, name)
    if value is not None:
        T.buffer_store(result.scalar.buffer, value, [0])
    return result


def _store(lhs, value):
    T.buffer_store(lhs.scalar.buffer, value, [0])


def _bind(name, value, type_annotation=None):
    return _named(T.Bind(value, type_annotation), name)


def _expr(value):
    """Unwrap an IRBuilder mutable scalar where an exact PrimExpr is required."""
    return value.scalar if hasattr(value, "scalar") else value


def _match_global_buffers(input_ptr, weight_ptr, output_ptr, batch_size, dim):
    input_global = _named(
        T.match_buffer(input_ptr, [batch_size, dim], "float16", scope="global"), "input_global"
    )
    weight_global = _named(
        T.match_buffer(weight_ptr, [dim], "float16", scope="global"), "weight_global"
    )
    output_global = _named(
        T.match_buffer(output_ptr, [batch_size, dim], "float16", scope="global"), "output_global"
    )
    return input_global, weight_global, output_global


def _cluster_and_thread_ids(cluster_n, num_clusters, b_dx, b_dy, bind_cluster_id):
    cbx = _scalar("cbx", "int32", 0)
    if cluster_n > 1:
        cta_rank = _named(T.cta_id_in_cluster([cluster_n]), "v")
        _store(cbx, cta_rank)
    b_id = _named(T.cta_id([num_clusters * cluster_n]), "b_id")
    cluster_id_value = b_id // cluster_n
    cluster_id = (
        _scalar("cluster_id", "int32", cluster_id_value) if bind_cluster_id else cluster_id_value
    )
    t_idx, t_idy = T.thread_id([b_dx, b_dy])
    _named(t_idx, "t_idx")
    _named(t_idy, "t_idy")
    return cbx, b_id, cluster_id, t_idx, t_idy


def _rmsnorm_vectors(vector_size):
    return (
        _named(T.alloc_local([vector_size], "float16"), "input_vec"),
        _named(T.alloc_local([vector_size], "float16"), "weight_vec"),
        _named(T.alloc_local([vector_size], "float32"), "x_vec"),
        _named(T.alloc_local([vector_size], "float32"), "weight_vec_f32"),
        _named(T.alloc_local([vector_size], "float32"), "mul_result"),
    )


def _emit_cluster_reduce(
    cluster_n, b_dx, b_dy, t_idx, t_idy, cluster_reduce_smem, sum_sq_smem, sum_sq, norm_factor, dim
):
    if cluster_n > 1:
        with T.If(T.And(t_idy == 0, t_idx == 0)), T.Then():
            T.buffer_store(cluster_reduce_smem, sum_sq_smem[0], [0])
        T.evaluate(T.cuda.cluster_sync())
        with T.If(t_idy == 0), T.Then():
            with T.If(t_idx < cluster_n):
                with T.Then():
                    remote_ptr = _bind(
                        "remote_ptr",
                        T.reinterpret(
                            PointerType(PrimType("float32")),
                            _mapa_u64(cluster_reduce_smem.ptr_to([0]), t_idx),
                        ),
                        PointerType(PrimType("float32")),
                    )
                    remote_buf = _named(
                        T.decl_buffer([1], "float32", scope="shared", data=remote_ptr), "remote_buf"
                    )
                    _store(sum_sq, remote_buf[0])
                with T.Else():
                    _store(sum_sq, T.float32(0))
            _store(sum_sq, T.cuda.warp_sum(_expr(sum_sq), width=cluster_n))
            with T.If(t_idx == 0), T.Then():
                T.buffer_store(sum_sq_smem, _expr(sum_sq), [0])
        T.evaluate(T.ptx.bar.sync(1, T.uint32(b_dx * b_dy)))
        T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
        _store(norm_factor, T.rsqrt(sum_sq_smem[0] / dim + eps))
    else:
        _store(norm_factor, T.rsqrt(sum_sq / dim + eps))


def _emit_output(
    dim_per_cta,
    vector_size,
    b_dx,
    b_dy,
    t_idx,
    t_idy,
    col_offset,
    batch_idx,
    input_smem,
    weight_global,
    output_global,
    input_vec,
    weight_vec,
    x_vec,
    weight_vec_f32,
    mul_result,
    norm_factor,
):
    with T.serial(ceildiv(dim_per_cta, vector_size * b_dx * b_dy)) as ki:
        _named(ki, "ki")
        st = (ki * b_dx * b_dy + b_dx * t_idy + t_idx) * vector_size
        with T.If(st < dim_per_cta), T.Then():
            Tx.copy(
                weight_vec[0:vector_size],
                weight_global[col_offset + st : col_offset + st + vector_size],
            )
            Tx.cast(weight_vec_f32[0:vector_size], weight_vec[0:vector_size])
            Tx.copy(input_vec[0:vector_size], input_smem[st : st + vector_size])
            Tx.cast(x_vec[0:vector_size], input_vec[0:vector_size])
            Tx.mul(mul_result[0:vector_size], x_vec[0:vector_size], _expr(norm_factor))
            Tx.mul(
                mul_result[0:vector_size], mul_result[0:vector_size], weight_vec_f32[0:vector_size]
            )
            Tx.cast(input_vec[0:vector_size], mul_result[0:vector_size])
            Tx.copy(
                output_global[batch_idx, col_offset + st : col_offset + st + vector_size],
                input_vec[0:vector_size],
            )


def _mapa_u64(ptr, rank):
    """`mapa.u64` into a declared register, returned as an ordinary value.

    PTX has no defining form, so mapa writes a register the caller declares;
    a one-element local buffer gives both a writable lvalue and an Expr.
    """
    mapped = T.alloc_local([1], "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    return mapped[0]


eps = 1e-06
F16_BYTES = 2
F32_BYTES = 4
SM_COUNT = 152
SMEM_SIZE = 232448


def ceildiv(a, b):
    return (a + b - 1) // b


def get_cluster_n(elem, smem_capacity=220, dtype_width=16):
    if dtype_width != 16:
        raise ValueError(f"Unsupported dtype width: {dtype_width}")
    perSMLimit = smem_capacity
    thresholds = [
        (perSMLimit * 1 * 512, 1),
        (perSMLimit * 2 * 512, 2),
        (perSMLimit * 4 * 512, 4),
        (perSMLimit * 8 * 512, 8),
    ]
    for limit, cluster in thresholds:
        if elem <= limit:
            return cluster
    return 16


def prepare_data(batch_size, dim):
    import torch

    torch.manual_seed(42)
    input = torch.randn(batch_size, dim, dtype=torch.float16, device="cuda")
    weights = torch.randn(dim, dtype=torch.float16, device="cuda")
    return (input, weights)


def torch_impl(input, weights):
    import torch

    input_naive = input.clone().to(dtype=torch.float32, device="cuda")
    weights_naive = weights.clone().to(dtype=torch.float32, device="cuda")

    def func():
        variance = input_naive.pow(2).mean(dim=-1, keepdim=True)
        norm_factor = torch.rsqrt(variance + eps)
        scaled = input_naive * norm_factor
        output = (scaled * weights_naive).to(torch.float16)
        return output

    result = bench({"naive": func}, timer="event")
    ms = result["impls"].get("naive", float("nan"))
    print(f"torch time: {ms:.3f} ms")
    return func()


def flashinfer_impl(input, weights, batch_size, dim):
    import flashinfer
    import torch

    out = torch.empty((batch_size, dim), dtype=torch.float16, device="cuda")
    flashinfer_input = input.clone().to(dtype=torch.float16, device="cuda")
    flashinfer_weights = weights.clone().to(dtype=torch.float16, device="cuda")

    def func():
        return flashinfer.norm.rmsnorm(
            flashinfer_input, flashinfer_weights, eps, enable_pdl=False, out=out
        )

    result = bench({"flashinfer": func}, timer="event")
    ms = result["impls"].get("flashinfer", float("nan"))
    print(f"FlashInfer time: {ms:.3f} ms")
    return out


def quack_impl(input, weights, batch_size, dim):
    import quack
    import torch

    quack_input = input.clone().to(dtype=torch.float16, device="cuda")
    quack_weights = weights.clone().to(dtype=torch.float16, device="cuda")

    def func():
        return quack.rmsnorm(quack_input, quack_weights, eps=eps)

    result = bench({"quack": func}, timer="event")
    ms = result["impls"].get("quack", float("nan"))
    print(f"Quack time: {ms:.3f} ms")
    return func()


def tirx_dispatch_rmsnorm(dim: int, batch_size: int, SMEM_PER_CTA=220, MAX_THREADS=256):
    if dim % 256 == 0 and dim <= 8192:
        CLUSTER_N = get_cluster_n(dim, 40)
        MAX_THREADS = 128
        useTMA = 1
    else:
        MAX_THREADS = 512
        useTMA = 0
        CLUSTER_N = get_cluster_n(dim, 110)
        if CLUSTER_N >= 16:
            MAX_THREADS = 1024
            CLUSTER_N = get_cluster_n(dim, 220)
            if CLUSTER_N >= 16:
                raise ValueError(
                    f"Dimension {dim} is too large to fit within SMEM constraints with current cluster reduction scheme"
                )
    print("CLUSTER_N =", CLUSTER_N)
    if useTMA:
        print("Using TMA SMEM load for input")
    else:
        print("Using synchronous SMEM load for input")
    if dim % CLUSTER_N != 0:
        raise ValueError(f"Dimension {dim} must be divisible by cluster size {CLUSTER_N}")
    dim_per_cta = ceildiv(dim, CLUSTER_N)
    if useTMA and dim_per_cta % 256 != 0:
        raise ValueError(f"dim_per_cta={dim_per_cta} must be divisible by 256 for TMA")
    num_clusters = batch_size
    VECTOR_SIZE = math.gcd(16 // F16_BYTES, dim_per_cta, dim - dim_per_cta * (CLUSTER_N - 1))
    BLOCK_SIZE = min(MAX_THREADS, max(32, dim_per_cta // VECTOR_SIZE))
    b_dx = 32
    b_dy = ceildiv(BLOCK_SIZE, b_dx)
    TMA_TILE = min(256, dim_per_cta)
    NUM_TMA_CHUNKS = dim_per_cta // TMA_TILE
    NUM_INPUT_BARS = 1

    def build_dispatch_kernel(use_tma):
        """
        RMSNorm: output = x * rsqrt(mean(x^2) + eps) * weight
        Uses TMA to load input/weight from GMEM to SMEM.
        For large dim, shards N across a cluster of CTAs with cross-CTA reduction.
        """
        with IRBuilder() as ib:
            with T.prim_func():
                T.func_name("input_SMEM_TMA" if use_tma else "input_SMEM_sync")
                input_ptr = T.arg("input_ptr", T.handle())
                weight_ptr = T.arg("weight_ptr", T.handle())
                output_ptr = T.arg("output_ptr", T.handle())
                input_global, weight_global, output_global = _match_global_buffers(
                    input_ptr, weight_ptr, output_ptr, batch_size, dim
                )
                cbx, b_id, cluster_id, t_idx, t_idy = _cluster_and_thread_ids(
                    CLUSTER_N, num_clusters, b_dx, b_dy, bind_cluster_id=use_tma
                )
                col_offset = _expr(cbx) * dim_per_cta
                input_smem_bytes = dim_per_cta * F16_BYTES
                smem_total = 128 + input_smem_bytes + b_dy * F32_BYTES + F32_BYTES
                buf = _named(T.alloc_buffer([smem_total], "uint8", scope="shared.dyn"), "buf")
                with T.attr({"tirx.dyn_smem_bytes": smem_total}):
                    pool = T.SMEMPool(buf.data)
                    input_bar = (
                        _named(pool.alloc([NUM_INPUT_BARS], "uint64", align=8), "buffer")
                        if use_tma
                        else None
                    )
                    pool.move_base_to(128)
                    input_smem = _named(
                        pool.alloc([dim_per_cta], "float16", align=128), "input_smem"
                    )
                    sum_sq_smem = _named(pool.alloc([b_dy], "float32"), "sum_sq_smem")
                    cluster_reduce_smem = _named(pool.alloc([1], "float32"), "cluster_reduce_smem")
                    if use_tma:
                        with T.If(T.cuda.thread_rank() == 0), T.Then():
                            with T.unroll(NUM_INPUT_BARS) as i:
                                _named(i, "i")
                                T.evaluate(
                                    T.ptx.mbarrier.init.shared.b64(
                                        input_bar.ptr_to([i]), T.uint32(1)
                                    )
                                )
                        if CLUSTER_N > 1:
                            T.evaluate(T.cuda.cluster_sync())
                        else:
                            T.evaluate(T.ptx.bar.sync(1, T.uint32(b_dx * b_dy)))
                    input_vec, weight_vec, x_vec, weight_vec_f32, mul_result = _rmsnorm_vectors(
                        VECTOR_SIZE
                    )
                    sum_sq = _scalar("sum_sq", "float32")
                    norm_factor = _scalar("norm_factor", "float32")
                    batch_idx = (
                        _scalar("batch_idx", "int32", _expr(cluster_id)) if use_tma else cluster_id
                    )
                    if use_tma:
                        zero_and_tidy = T.bitwise_and(0, t_idy)
                        with T.If(T.And(t_idx == zero_and_tidy, zero_and_tidy == 0)), T.Then():
                            with T.serial(NUM_TMA_CHUNKS) as tma_chunk:
                                _named(tma_chunk, "tma_chunk")
                                tma_off = tma_chunk * TMA_TILE
                                Tx.copy_async(
                                    input_smem[tma_off : tma_off + TMA_TILE],
                                    input_global[
                                        _expr(batch_idx),
                                        col_offset + tma_off : col_offset + tma_off + TMA_TILE,
                                    ],
                                    dispatch="tma_auto",
                                    mbar=input_bar.ptr_to([0]),
                                )
                            T.evaluate(
                                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                    input_bar.ptr_to([0]), T.uint32(input_smem_bytes)
                                )
                            )
                        T.evaluate(T.cuda.mbarrier_wait(input_bar.ptr_to([0]), 0))
                    _store(sum_sq, T.float32(0.0))
                    with T.serial(ceildiv(dim_per_cta, VECTOR_SIZE * b_dx * b_dy)) as ki:
                        _named(ki, "ki")
                        st = (ki * b_dx * b_dy + b_dx * t_idy + t_idx) * VECTOR_SIZE
                        with T.If(st < dim_per_cta), T.Then():
                            if not use_tma:
                                Tx.copy(
                                    input_smem[st : st + VECTOR_SIZE],
                                    input_global[
                                        batch_idx, col_offset + st : col_offset + st + VECTOR_SIZE
                                    ],
                                )
                            Tx.copy(input_vec[0:VECTOR_SIZE], input_smem[st : st + VECTOR_SIZE])
                            Tx.cast(x_vec[0:VECTOR_SIZE], input_vec[0:VECTOR_SIZE])
                            if use_tma:
                                with T.unroll(VECTOR_SIZE) as v_id:
                                    _named(v_id, "v_id")
                                    T.buffer_store(
                                        weight_vec_f32, x_vec[v_id] * x_vec[v_id], [v_id]
                                    )
                                with T.unroll(VECTOR_SIZE) as v_id:
                                    _named(v_id, "v_id")
                                    _store(sum_sq, sum_sq + weight_vec_f32[v_id])
                            else:
                                with T.unroll(VECTOR_SIZE) as v_id:
                                    _named(v_id, "v_id")
                                    _store(sum_sq, sum_sq + x_vec[v_id] * x_vec[v_id])
                    _store(sum_sq, T.cuda.cta_sum(_expr(sum_sq), b_dy, sum_sq_smem.ptr_to([0])))
                    _emit_cluster_reduce(
                        CLUSTER_N,
                        b_dx,
                        b_dy,
                        t_idx,
                        t_idy,
                        cluster_reduce_smem,
                        sum_sq_smem,
                        sum_sq,
                        norm_factor,
                        dim,
                    )
                    _emit_output(
                        dim_per_cta,
                        VECTOR_SIZE,
                        b_dx,
                        b_dy,
                        t_idx,
                        t_idy,
                        col_offset,
                        _expr(batch_idx),
                        input_smem,
                        weight_global,
                        output_global,
                        input_vec,
                        weight_vec,
                        x_vec,
                        weight_vec_f32,
                        mul_result,
                        norm_factor,
                    )
        return ib.get()

    return build_dispatch_kernel(bool(useTMA))


def tirx_original_impl(hidden_size, batch_size, SMEM_PER_CTA=220, MAX_THREADS=256):
    vec_size = math.gcd(16 // F16_BYTES, hidden_size)
    block_size = min(256, hidden_size // vec_size)
    bdx = 32
    bdy = ceildiv(block_size, 32)
    smem_size = (bdy + hidden_size) * F32_BYTES
    if smem_size > SMEM_SIZE:
        raise ValueError(
            f"SMEM usage for this dim exceeds limit of {SMEM_SIZE} bytes. Consider using a smaller dim."
        )

    with IRBuilder() as ib:
        with T.prim_func():
            T.func_name("rmsnorm")
            input_ptr = T.arg("input_ptr", T.handle())
            weight_ptr = T.arg("weight_ptr", T.handle())
            out_ptr = T.arg("out_ptr", T.handle())
            input_global = _named(
                T.match_buffer(input_ptr, [batch_size, hidden_size], "float16", scope="global"),
                "input_global",
            )
            weight_global = _named(
                T.match_buffer(weight_ptr, [hidden_size], "float16", scope="global"),
                "weight_global",
            )
            out_global = _named(
                T.match_buffer(out_ptr, [batch_size, hidden_size], "float16", scope="global"),
                "out_global",
            )
            bx = _named(T.cta_id([SM_COUNT]), "bx")
            tx, ty = T.thread_id([bdx, bdy])
            _named(tx, "tx")
            _named(ty, "ty")
            thread_id = ty * bdx + tx
            buf = _named(T.alloc_buffer([smem_size], "uint8", scope="shared.dyn"), "buf")
            with T.attr({"tirx.dyn_smem_bytes": smem_size}):
                pool = T.SMEMPool(buf.data)
                x_smem = _named(pool.alloc([hidden_size], "float32"), "x_smem")
                sum_sq_smem = _named(pool.alloc([bdy], "float32"), "sum_sq_smem")
                input_vec = _named(T.alloc_local([vec_size], "float16"), "input_vec")
                weight_vec = _named(T.alloc_local([vec_size], "float16"), "weight_vec")
                input_vec_f32 = _named(T.alloc_local([vec_size], "float32"), "input_vec_f32")
                weight_vec_f32 = _named(T.alloc_local([vec_size], "float32"), "weight_vec_f32")
                x_vec = _named(T.alloc_local([vec_size], "float32"), "x_vec")
                x_tmp = _scalar("x_tmp", "float32")
                sum_sq = _scalar("sum_sq", "float32")
                rms_norm = _scalar("rms_norm", "float32")
                idx = _scalar("idx", "int32", bx)
                with T.While(idx < batch_size):
                    _store(sum_sq, T.float32(0.0))
                    with T.serial(ceildiv(hidden_size, vec_size * bdx * bdy)) as ki:
                        _named(ki, "ki")
                        with T.unroll(vec_size) as kv:
                            _named(kv, "kv")
                            T.buffer_store(input_vec, T.float16(0.0), [kv])
                            T.buffer_store(x_vec, T.float32(0.0), [kv])
                        st = (ki * bdx * bdy + thread_id) * vec_size
                        with T.If(st < hidden_size), T.Then():
                            Tx.copy(
                                input_vec[0:vec_size], input_global[_expr(idx), st : st + vec_size]
                            )
                            Tx.cast(input_vec_f32[0:vec_size], input_vec[0:vec_size])
                            with T.unroll(vec_size) as kv:
                                _named(kv, "kv")
                                _store(x_tmp, input_vec_f32[kv])
                                _store(sum_sq, sum_sq + x_tmp * x_tmp)
                                T.buffer_store(x_vec, _expr(x_tmp), [kv])
                            Tx.copy(x_smem[st : st + vec_size], x_vec[0:vec_size])
                    _store(sum_sq, T.cuda.warp_sum(_expr(sum_sq)))
                    T.buffer_store(sum_sq_smem, _expr(sum_sq), [ty])
                    T.evaluate(T.ptx.bar.sync(1, T.uint32(bdx * bdy)))
                    T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                    with T.If(ty == 0), T.Then():
                        with T.If(tx < bdy):
                            with T.Then():
                                _store(sum_sq, sum_sq_smem[tx])
                            with T.Else():
                                _store(sum_sq, T.float32(0.0))
                        _store(sum_sq, T.cuda.warp_sum(_expr(sum_sq)))
                        T.buffer_store(sum_sq_smem, _expr(sum_sq), [0])
                    T.evaluate(T.ptx.bar.sync(1, T.uint32(bdx * bdy)))
                    T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                    _store(rms_norm, T.rsqrt(sum_sq_smem[0] / hidden_size + eps))
                    with T.serial(ceildiv(hidden_size, vec_size * bdx * bdy)) as ki:
                        _named(ki, "ki")
                        with T.unroll(vec_size) as kv:
                            _named(kv, "kv")
                            T.buffer_store(input_vec, T.float16(0.0), [kv])
                            T.buffer_store(weight_vec_f32, T.float32(0.0), [kv])
                            T.buffer_store(x_vec, T.float32(0.0), [kv])
                        st = (ki * bdx * bdy + thread_id) * vec_size
                        with T.If(st < hidden_size), T.Then():
                            Tx.copy(weight_vec[0:vec_size], weight_global[st : st + vec_size])
                            Tx.copy(x_vec[0:vec_size], x_smem[st : st + vec_size])
                            Tx.cast(weight_vec_f32[0:vec_size], weight_vec[0:vec_size])
                        Tx.mul(input_vec_f32[0:vec_size], x_vec[0:vec_size], _expr(rms_norm))
                        Tx.mul(
                            input_vec_f32[0:vec_size],
                            input_vec_f32[0:vec_size],
                            weight_vec_f32[0:vec_size],
                        )
                        with T.If(st < hidden_size), T.Then():
                            Tx.cast(input_vec[0:vec_size], input_vec_f32[0:vec_size])
                            Tx.copy(
                                out_global[_expr(idx), st : st + vec_size], input_vec[0:vec_size]
                            )
                    T.evaluate(T.ptx.bar.sync(1, T.uint32(bdx * bdy)))
                    _store(idx, idx + SM_COUNT)

    return ib.get()


def tirx_input_DSMEM_write_TMA_wts_GMEM(
    dim: int, batch_size: int, SMEM_PER_CTA=220, MAX_THREADS=256
):
    CLUSTER_N = get_cluster_n(dim, SMEM_PER_CTA)
    print("tirx_input_DSMEM_TMA_wts_GMEM: CLUSTER_N =", CLUSTER_N)
    if dim % CLUSTER_N != 0:
        raise ValueError(f"Dimension {dim} must be divisible by cluster size {CLUSTER_N}")
    dim_per_cta = ceildiv(dim, CLUSTER_N)
    if dim_per_cta % 256 != 0:
        raise ValueError(f"dim_per_cta={dim_per_cta} must be divisible by 256 for TMA")
    num_clusters = batch_size
    VECTOR_SIZE = math.gcd(16 // F16_BYTES, dim_per_cta, dim - dim_per_cta * (CLUSTER_N - 1))
    BLOCK_SIZE = min(MAX_THREADS, max(32, dim_per_cta // VECTOR_SIZE))
    b_dx = 32
    b_dy = ceildiv(BLOCK_SIZE, b_dx)
    TMA_TILE = min(256, dim_per_cta)
    NUM_TMA_CHUNKS = dim_per_cta // TMA_TILE
    NUM_INPUT_BARS = 1
    NUM_CLUSTER_BARS = 1

    with IRBuilder() as ib:
        with T.prim_func():
            T.func_name("rms_norm")
            input_ptr = T.arg("input_ptr", T.handle())
            weight_ptr = T.arg("weight_ptr", T.handle())
            output_ptr = T.arg("output_ptr", T.handle())
            input_global, weight_global, output_global = _match_global_buffers(
                input_ptr, weight_ptr, output_ptr, batch_size, dim
            )
            cbx, b_id, cluster_id, t_idx, t_idy = _cluster_and_thread_ids(
                CLUSTER_N, num_clusters, b_dx, b_dy, bind_cluster_id=True
            )
            cta_rank = _expr(cbx)
            col_offset = _expr(cbx) * dim_per_cta
            input_smem_bytes = dim_per_cta * F16_BYTES
            smem_total = 128 + input_smem_bytes + b_dy * F32_BYTES + CLUSTER_N * F32_BYTES
            buf = _named(T.alloc_buffer([smem_total], "uint8", scope="shared.dyn"), "buf")
            with T.attr({"tirx.dyn_smem_bytes": smem_total}):
                pool = T.SMEMPool(buf.data)
                input_bar_buf = _named(pool.alloc([NUM_INPUT_BARS], "uint64", align=8), "buffer")
                cluster_bar_buf = _named(
                    pool.alloc([NUM_CLUSTER_BARS], "uint64", align=8), "cluster_bar_buf"
                )
                pool.move_base_to(128)
                input_smem = _named(pool.alloc([dim_per_cta], "float16", align=128), "input_smem")
                sum_sq_smem = _named(pool.alloc([b_dy], "float32"), "sum_sq_smem")
                cluster_reduce_smem = _named(
                    pool.alloc([CLUSTER_N], "float32"), "cluster_reduce_smem"
                )
                with T.If(T.cuda.thread_rank() == 0), T.Then():
                    with T.unroll(NUM_INPUT_BARS) as i:
                        _named(i, "i")
                        T.evaluate(
                            T.ptx.mbarrier.init.shared.b64(input_bar_buf.ptr_to([i]), T.uint32(1))
                        )
                if CLUSTER_N > 1:
                    with T.If(T.cuda.thread_rank() == 0), T.Then():
                        with T.unroll(NUM_CLUSTER_BARS) as i:
                            _named(i, "i")
                            T.evaluate(
                                T.ptx.mbarrier.init.shared.b64(
                                    cluster_bar_buf.ptr_to([i]), T.uint32(CLUSTER_N)
                                )
                            )
                    T.evaluate(T.ptx.fence.mbarrier_init.release.cluster())
                    T.evaluate(T.cuda.cluster_sync())
                else:
                    T.evaluate(T.ptx.bar.sync(1, T.uint32(b_dx * b_dy)))
                input_vec, weight_vec, x_vec, weight_vec_f32, mul_result = _rmsnorm_vectors(
                    VECTOR_SIZE
                )
                sum_sq = _scalar("sum_sq", "float32")
                norm_factor = _scalar("norm_factor", "float32")
                batch_idx = _scalar("batch_idx", "int32", _expr(cluster_id))
                zero_and_tidy = T.bitwise_and(0, t_idy)
                with T.If(T.And(t_idx == zero_and_tidy, zero_and_tidy == 0)), T.Then():
                    with T.serial(NUM_TMA_CHUNKS) as tma_chunk:
                        _named(tma_chunk, "tma_chunk")
                        tma_off = tma_chunk * TMA_TILE
                        Tx.copy_async(
                            input_smem[tma_off : tma_off + TMA_TILE],
                            input_global[
                                _expr(batch_idx),
                                col_offset + tma_off : col_offset + tma_off + TMA_TILE,
                            ],
                            dispatch="tma_auto",
                            mbar=input_bar_buf.ptr_to([0]),
                        )
                    T.evaluate(
                        T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                            input_bar_buf.ptr_to([0]), T.uint32(input_smem_bytes)
                        )
                    )
                T.evaluate(T.cuda.mbarrier_wait(input_bar_buf.ptr_to([0]), 0))
                _store(sum_sq, T.float32(0.0))
                with T.serial(ceildiv(dim_per_cta, VECTOR_SIZE * b_dx * b_dy)) as ki:
                    _named(ki, "ki")
                    st = (ki * b_dx * b_dy + b_dx * t_idy + t_idx) * VECTOR_SIZE
                    with T.If(st < dim_per_cta), T.Then():
                        Tx.copy(input_vec[0:VECTOR_SIZE], input_smem[st : st + VECTOR_SIZE])
                        Tx.cast(x_vec[0:VECTOR_SIZE], input_vec[0:VECTOR_SIZE])
                        with T.unroll(VECTOR_SIZE) as v_id:
                            _named(v_id, "v_id")
                            _store(sum_sq, sum_sq + x_vec[v_id] * x_vec[v_id])
                _store(sum_sq, T.cuda.warp_sum(_expr(sum_sq)))
                T.buffer_store(sum_sq_smem, _expr(sum_sq), [t_idy])
                T.evaluate(T.ptx.bar.sync(1, T.uint32(b_dx * b_dy)))
                T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                with T.If(t_idy == 0), T.Then():
                    with T.If(t_idx < b_dy):
                        with T.Then():
                            _store(sum_sq, sum_sq_smem[t_idx])
                        with T.Else():
                            _store(sum_sq, T.float32(0.0))
                    _store(sum_sq, T.cuda.warp_sum(_expr(sum_sq)))
                    T.buffer_store(sum_sq_smem, _expr(sum_sq), [0])
                T.evaluate(T.ptx.bar.sync(1, T.uint32(b_dx * b_dy)))
                T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                if CLUSTER_N > 1:
                    with T.If(t_idy == 0), T.Then():
                        with T.If(t_idx < CLUSTER_N), T.Then():
                            remote_ptr = _bind(
                                "remote_ptr",
                                T.reinterpret(
                                    PointerType(PrimType("float32")),
                                    _mapa_u64(cluster_reduce_smem.ptr_to([cta_rank]), t_idx),
                                ),
                                PointerType(PrimType("float32")),
                            )
                            remote_buf = _named(
                                T.decl_buffer([1], "float32", scope="shared", data=remote_ptr),
                                "remote_buf",
                            )
                            T.buffer_store(remote_buf, sum_sq_smem[0], [0])
                            rem1 = _named(T.alloc_local([1], "uint64"), "_rem1")
                            T.evaluate(
                                T.ptx.mapa.shared__cluster.u64(
                                    rem1[0], cluster_bar_buf.ptr_to([0]), T.uint32(t_idx)
                                )
                            )
                            T.evaluate(
                                T.ptx.mbarrier.arrive.b64(rem1[0], T.uint32(1), pred=T.bool(True))
                            )
                    T.evaluate(T.cuda.mbarrier_wait(cluster_bar_buf.ptr_to([0]), 0))
                    with T.If(t_idy == 0), T.Then():
                        with T.If(t_idx < CLUSTER_N):
                            with T.Then():
                                _store(sum_sq, cluster_reduce_smem[t_idx])
                            with T.Else():
                                _store(sum_sq, T.float32(0))
                        _store(sum_sq, T.cuda.warp_sum(_expr(sum_sq), width=CLUSTER_N))
                        with T.If(t_idx == 0), T.Then():
                            T.buffer_store(sum_sq_smem, _expr(sum_sq), [0])
                    T.evaluate(T.ptx.bar.sync(1, T.uint32(b_dx * b_dy)))
                    T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                _store(norm_factor, T.rsqrt(sum_sq_smem[0] / dim + eps))
                _emit_output(
                    dim_per_cta,
                    VECTOR_SIZE,
                    b_dx,
                    b_dy,
                    t_idx,
                    t_idy,
                    col_offset,
                    _expr(batch_idx),
                    input_smem,
                    weight_global,
                    output_global,
                    input_vec,
                    weight_vec,
                    x_vec,
                    weight_vec_f32,
                    mul_result,
                    norm_factor,
                )

    return ib.get()


def build_tirx_soln(
    func, input_cat, weights, funcstr: str, dim: int, batch_size: int
) -> tuple[np.ndarray, tvm.runtime.Executable]:
    import torch

    from tirx_kernels.runner import cuda_target

    input_cat_tir = input_cat.cuda() if not input_cat.is_cuda else input_cat
    weights_tir = weights.cuda() if not weights.is_cuda else weights
    output_tir = torch.empty((batch_size, dim), dtype=torch.float16, device="cuda")
    target = cuda_target()
    with target:
        mod = tvm.IRModule({"main": func(dim, batch_size)})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        def run():
            return mod(input_cat_tir, weights_tir, output_tir)

        result = bench({f"tirx_soln_{funcstr}": run}, timer="event")
        ms = result["impls"].get(f"tirx_soln_{funcstr}", float("nan"))
        print(f"{funcstr} time: {ms:.3f} ms")

    return (output_tir, mod)


def test(batch_size: int, dim: int = 16384):
    import torch

    input, weights = prepare_data(batch_size, dim)
    print(f"----Testing Batch Size {batch_size}, Dim {dim}----")
    output_torch = torch_impl(input, weights)
    output_flashinfer = flashinfer_impl(input, weights, batch_size, dim)
    output_quack = quack_impl(input, weights, batch_size, dim)
    output_tirx_original, tirx_primfunc_1 = build_tirx_soln(
        tirx_original_impl, input, weights, "TIRX_original_impl", dim, batch_size
    )
    output_tirx_dispatch_rmsnorm, tirx_primfunc_2 = build_tirx_soln(
        tirx_dispatch_rmsnorm, input, weights, "TIRX_dispatch_rmsnorm", dim, batch_size
    )
    torch.testing.assert_close(output_flashinfer, output_torch, rtol=0.005, atol=0.005)
    torch.testing.assert_close(output_quack, output_torch, rtol=0.005, atol=0.005)
    torch.testing.assert_close(output_tirx_original, output_torch, rtol=0.005, atol=0.005)
    torch.testing.assert_close(output_tirx_dispatch_rmsnorm, output_torch, rtol=0.005, atol=0.005)


KERNEL_META = {"name": "rmsnorm", "category": "basic", "compute_capability": 10}
CONFIGS = [
    {"hidden_size": hs, "batch_size": bs, "label": f"hs{hs}_bs{bs}"}
    for hs in [128, 4096, 5120, 8192]
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 4113]
]


def _get_rmsnorm_kernel(hidden_size):
    """Registry-compatible kernel factory (dynamic batch_size)."""
    vec_size = math.gcd(16 // F16_BYTES, hidden_size)
    block_size = min(256, hidden_size // vec_size)
    bdx = 32
    bdy = ceildiv(block_size, 32)
    with IRBuilder() as ib:
        with T.prim_func():
            T.func_name("rmsnorm")
            input_ptr = T.arg("input_ptr", T.handle())
            weight_ptr = T.arg("weight_ptr", T.handle())
            out_ptr = T.arg("out_ptr", T.handle())
            batch_size = _named(T.int32(), "batch_size")
            input_global = _named(
                T.match_buffer(input_ptr, [batch_size, hidden_size], "float16", scope="global"),
                "input_global",
            )
            weight_global = _named(
                T.match_buffer(weight_ptr, [hidden_size], "float16", scope="global"),
                "weight_global",
            )
            out_global = _named(
                T.match_buffer(out_ptr, [batch_size, hidden_size], "float16", scope="global"),
                "out_global",
            )
            T.device_entry()
            bx = _named(T.cta_id([SM_COUNT]), "bx")
            tx, ty = T.thread_id([bdx, bdy])
            _named(tx, "tx")
            _named(ty, "ty")
            thread_id = ty * bdx + tx
            pool = T.SMEMPool()
            x_smem = _named(pool.alloc([hidden_size], "float32"), "x_smem")
            sum_sq_smem = _named(pool.alloc([bdy], "float32"), "sum_sq_smem")
            pool.commit()
            input_words = _named(T.alloc_local([vec_size // 2], "uint32"), "input_words")
            weight_words = _named(T.alloc_local([vec_size // 2], "uint32"), "weight_words")
            output_words = _named(T.alloc_local([vec_size // 2], "uint32"), "output_words")
            input_vec_f32 = _named(T.alloc_local([vec_size], "float32"), "input_vec_f32")
            weight_vec_f32 = _named(T.alloc_local([vec_size], "float32"), "weight_vec_f32")
            x_vec = _named(T.alloc_local([vec_size], "float32"), "x_vec")
            packed_mul = _scalar("packed_mul", "uint64")
            x_tmp = _scalar("x_tmp", "float32")
            sum_sq = _scalar("sum_sq", "float32")
            rms_norm = _scalar("rms_norm", "float32")
            idx = _scalar("idx", "int32", bx)
            with T.While(idx < batch_size):
                _store(sum_sq, T.float32(0.0))
                with T.serial(ceildiv(hidden_size, vec_size * bdx * bdy)) as ki:
                    _named(ki, "ki")
                    with T.unroll(vec_size) as kv:
                        _named(kv, "kv")
                        T.buffer_store(x_vec, T.float32(0.0), [kv])
                    st = (ki * bdx * bdy + thread_id) * vec_size
                    with T.If(st < hidden_size), T.Then():
                        T.evaluate(
                            T.ptx.ld.global_.v4.b32(
                                input_words[0],
                                input_words[1],
                                input_words[2],
                                input_words[3],
                                input_global.ptr_to([_expr(idx), st]),
                            )
                        )
                        with T.unroll(vec_size // 2) as pair:
                            _named(pair, "pair")
                            T.buffer_store(
                                input_vec_f32,
                                T.cast(
                                    T.reinterpret(
                                        "float16",
                                        T.cast(
                                            T.bitwise_and(input_words[pair], T.uint32(0xFFFF)),
                                            "uint16",
                                        ),
                                    ),
                                    "float32",
                                ),
                                [pair * 2],
                            )
                            T.buffer_store(
                                input_vec_f32,
                                T.cast(
                                    T.reinterpret(
                                        "float16",
                                        T.cast(
                                            T.shift_right(input_words[pair], T.uint32(16)), "uint16"
                                        ),
                                    ),
                                    "float32",
                                ),
                                [pair * 2 + 1],
                            )
                        with T.unroll(vec_size) as kv:
                            _named(kv, "kv")
                            _store(x_tmp, input_vec_f32[kv])
                            _store(sum_sq, sum_sq + x_tmp * x_tmp)
                            T.buffer_store(x_vec, _expr(x_tmp), [kv])
                        T.evaluate(
                            T.ptx.st.shared.v4.f32(
                                x_smem.ptr_to([st]), x_vec[0], x_vec[1], x_vec[2], x_vec[3]
                            )
                        )
                        T.evaluate(
                            T.ptx.st.shared.v4.f32(
                                x_smem.ptr_to([st + 4]), x_vec[4], x_vec[5], x_vec[6], x_vec[7]
                            )
                        )

                for delta in (16, 8, 4, 2, 1):
                    _store(
                        sum_sq,
                        sum_sq
                        + T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), _expr(sum_sq), delta, bdx),
                    )
                with T.If(tx == 0), T.Then():
                    T.evaluate(T.ptx.st.shared.f32(sum_sq_smem.ptr_to([ty]), _expr(sum_sq)))
                T.evaluate(T.ptx.bar.sync(0, T.uint32(bdx * bdy)))
                with T.If(ty == 0), T.Then():
                    with T.If(tx < bdy):
                        with T.Then():
                            T.evaluate(T.ptx.ld.shared.f32(_expr(sum_sq), sum_sq_smem.ptr_to([tx])))
                        with T.Else():
                            _store(sum_sq, T.float32(0.0))
                    for delta in (16, 8, 4, 2, 1):
                        _store(
                            sum_sq,
                            sum_sq
                            + T.cuda.__shfl_xor_sync(
                                T.uint32(0xFFFFFFFF), _expr(sum_sq), delta, bdx
                            ),
                        )
                    with T.If(tx == 0), T.Then():
                        T.evaluate(T.ptx.st.shared.f32(sum_sq_smem.ptr_to([0]), _expr(sum_sq)))
                T.evaluate(T.ptx.bar.sync(0, T.uint32(bdx * bdy)))
                T.evaluate(T.ptx.ld.shared.f32(_expr(sum_sq), sum_sq_smem.ptr_to([0])))
                _store(rms_norm, T.rsqrt(sum_sq / hidden_size + eps))
                with T.serial(ceildiv(hidden_size, vec_size * bdx * bdy)) as ki:
                    _named(ki, "ki")
                    with T.unroll(vec_size) as kv:
                        _named(kv, "kv")
                        T.buffer_store(weight_vec_f32, T.float32(0.0), [kv])
                        T.buffer_store(x_vec, T.float32(0.0), [kv])
                    st = (ki * bdx * bdy + thread_id) * vec_size
                    with T.If(st < hidden_size), T.Then():
                        T.evaluate(
                            T.ptx.ld.global_.v4.b32(
                                weight_words[0],
                                weight_words[1],
                                weight_words[2],
                                weight_words[3],
                                weight_global.ptr_to([st]),
                            )
                        )
                        T.evaluate(
                            T.ptx.ld.shared.v4.f32(
                                x_vec[0], x_vec[1], x_vec[2], x_vec[3], x_smem.ptr_to([st])
                            )
                        )
                        T.evaluate(
                            T.ptx.ld.shared.v4.f32(
                                x_vec[4], x_vec[5], x_vec[6], x_vec[7], x_smem.ptr_to([st + 4])
                            )
                        )
                        with T.unroll(vec_size // 2) as pair:
                            _named(pair, "pair")
                            T.buffer_store(
                                weight_vec_f32,
                                T.cast(
                                    T.reinterpret(
                                        "float16",
                                        T.cast(
                                            T.bitwise_and(weight_words[pair], T.uint32(0xFFFF)),
                                            "uint16",
                                        ),
                                    ),
                                    "float32",
                                ),
                                [pair * 2],
                            )
                            T.buffer_store(
                                weight_vec_f32,
                                T.cast(
                                    T.reinterpret(
                                        "float16",
                                        T.cast(
                                            T.shift_right(weight_words[pair], T.uint32(16)),
                                            "uint16",
                                        ),
                                    ),
                                    "float32",
                                ),
                                [pair * 2 + 1],
                            )
                    with T.unroll(vec_size // 2) as pair:
                        _named(pair, "pair")
                        T.evaluate(
                            T.ptx.mul.rz.ftz.f32x2(
                                _expr(packed_mul),
                                T.cuda.make_float2(x_vec[pair * 2], x_vec[pair * 2 + 1]),
                                T.cuda.make_float2(_expr(rms_norm), _expr(rms_norm)),
                            )
                        )
                        T.buffer_store(
                            input_vec_f32, T.cuda.float2_x(_expr(packed_mul)), [pair * 2]
                        )
                        T.buffer_store(
                            input_vec_f32, T.cuda.float2_y(_expr(packed_mul)), [pair * 2 + 1]
                        )
                    with T.unroll(vec_size // 2) as pair:
                        _named(pair, "pair")
                        T.evaluate(
                            T.ptx.mul.rz.ftz.f32x2(
                                _expr(packed_mul),
                                T.cuda.make_float2(
                                    input_vec_f32[pair * 2], input_vec_f32[pair * 2 + 1]
                                ),
                                T.cuda.make_float2(
                                    weight_vec_f32[pair * 2], weight_vec_f32[pair * 2 + 1]
                                ),
                            )
                        )
                        T.buffer_store(
                            input_vec_f32, T.cuda.float2_x(_expr(packed_mul)), [pair * 2]
                        )
                        T.buffer_store(
                            input_vec_f32, T.cuda.float2_y(_expr(packed_mul)), [pair * 2 + 1]
                        )
                    with T.If(st < hidden_size), T.Then():
                        with T.unroll(vec_size // 2) as pair:
                            _named(pair, "pair")
                            T.evaluate(
                                T.ptx.cvt.rn.f16x2.f32(
                                    output_words[pair],
                                    input_vec_f32[pair * 2 + 1],
                                    input_vec_f32[pair * 2],
                                )
                            )
                        T.evaluate(
                            T.ptx.st.global_.v4.b32(
                                out_global.ptr_to([_expr(idx), st]),
                                output_words[0],
                                output_words[1],
                                output_words[2],
                                output_words[3],
                            )
                        )
                T.evaluate(T.ptx.bar.sync(1, T.uint32(bdx * bdy)))
                _store(idx, idx + SM_COUNT)

    return ib.get()


def get_kernel(hidden_size, **kwargs):
    return _get_rmsnorm_kernel(hidden_size)


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(hidden_size, batch_size, **kwargs):
    """Compile, run, and verify rmsnorm kernel."""
    import torch

    from tirx_kernels.runner import compile_kernel

    input_data, weights = prepare_data(batch_size, hidden_size)
    kernel = _get_rmsnorm_kernel(hidden_size)
    ex = compile_kernel(kernel)
    output_tir = torch.empty((batch_size, hidden_size), dtype=torch.float16, device="cuda")
    ex(input_data, weights, output_tir)
    torch.cuda.synchronize()
    input_f32 = input_data.to(torch.float32).cuda()
    variance = input_f32.pow(2).mean(dim=-1, keepdim=True)
    ref = (input_f32 * torch.rsqrt(variance + eps) * weights.float().cuda()).to(torch.float16)
    torch.testing.assert_close(output_tir.cpu(), ref.cpu(), rtol=0.001, atol=0.001)


# timer=None inherits the global default (proton). Proton matters here: rmsnorm is a
# tiny (~2µs) kernel whose event wall is ~3x inflated by launch overhead, and its
# reference is flashinfer (Python-dispatch-heavy). Proton measures the true ~2µs kernel
# time and an undistorted ratio.
def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, **kwargs):
    """Allocate, validate, and measure after GPU assignment."""
    return _run_gpu(
        prepared["executable"],
        **prepared["config"],
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        **kwargs,
    )


def _run_gpu(ex, hidden_size, batch_size, warmup=None, repeat=None, timer=None, **kwargs):
    """GPU-stage implementation shared by suite and standalone execution."""

    import torch

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    input_data, weights = prepare_data(batch_size, hidden_size)
    input_cuda = input_data.cuda()
    weights_cuda = weights.cuda()
    output_cuda = torch.empty((batch_size, hidden_size), dtype=torch.float16, device="cuda")

    funcs = {"tir": lambda: ex(input_cuda, weights_cuda, output_cuda)}

    def _flashinfer():
        import flashinfer

        out_fi = torch.zeros_like(input_cuda)
        return lambda: flashinfer.norm.rmsnorm(
            input_cuda, weights_cuda, eps, enable_pdl=False, out=out_fi
        )

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer": _flashinfer},
        **kwargs,
    )


def run_bench(hidden_size, batch_size, warmup=None, repeat=None, timer=None, **kwargs):
    """Standalone wrapper over the same explicit prepare and GPU stages."""
    return prepare_bench(hidden_size=hidden_size, batch_size=batch_size).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, **kwargs
    )


if __name__ == "__main__":
    for batch_size, dim in [(2048, 8192)]:
        test(batch_size, dim)
