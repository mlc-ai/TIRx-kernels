# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 CSA compressor forward kernel.

Upstream source:
``python/cudnn/csa/compressor/compressor_sm100.py::_compressor_fwd_kernel``.
"""

from typing import Any

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cudnn_sm100_csa_compressor_fwd",
    "category": "cudnn",
    "compute_capability": 10,
}

_RATIO = 4
_SUPPORTED_HEAD_DIMS = (65, 128, 512)
_SUPPORTED_COFF = (1, 2)


def _cfg(label: str, seq_lens: tuple[int, ...], head_dim: int, coff: int, capacity_pad: int = 0):
    return {
        "label": label,
        "seq_lens": seq_lens,
        "head_dim": head_dim,
        "coff": coff,
        "capacity_pad": capacity_pad,
    }


CONFIGS = [
    _cfg("b1_s2048_d128_c2", (2048,), 128, 2),
    _cfg("ragged_1023_2048_509_d128_c2", (1023, 2048, 509), 128, 2),
    _cfg("b1_s2048_d512_c2", (2048,), 512, 2),
    _cfg("short_3_515_1024_129_d128_c2", (3, 515, 1024, 129), 128, 2),
    _cfg("b1_s260_d65_c2", (260,), 65, 2),
    _cfg("emptyseg_64_0_253_3_d128_c2", (64, 0, 253, 3), 128, 2),
    _cfg("b1_s2048_d128_c1", (2048,), 128, 1),
    _cfg("ragged_1023_2048_509_d128_c1", (1023, 2048, 509), 128, 1),
    _cfg("b1_s2048_d512_c1", (2048,), 512, 1),
    _cfg("short_3_515_1024_129_d128_c1", (3, 515, 1024, 129), 128, 1),
    _cfg("b1_s260_d65_c1", (260,), 65, 1),
    _cfg("emptyseg_64_0_253_3_d128_c1", (64, 0, 253, 3), 128, 1),
    _cfg("padding_short_d128_c2_p8", (3, 515, 1024, 129), 128, 2, 8),
    _cfg("padding_ragged_d128_c2_p8", (1023, 2048, 509), 128, 2, 8),
    _cfg("padding_short_d128_c1_p8", (3, 515, 1024, 129), 128, 1, 8),
    _cfg("padding_ragged_d128_c1_p8", (1023, 2048, 509), 128, 1, 8),
    _cfg("empty_output_d128_c2", (2,), 128, 2),
    _cfg("empty_output_d128_c1", (2,), 128, 1),
]

BENCH_CONFIGS = [
    _cfg("perf_b1_s8192_d128_c2", (8192,), 128, 2),
    _cfg("perf_b3_s8192_d128_c2", (8192, 8192, 8192), 128, 2),
    _cfg("perf_b1_s8192_d512_c2", (8192,), 512, 2),
    _cfg("perf_b3_s8192_d512_c2", (8192, 8192, 8192), 512, 2),
]


def _validate(head_dim: int, coff: int) -> None:
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim={head_dim} is outside the ported source specializations")
    if coff not in _SUPPORTED_COFF:
        raise ValueError(f"coff={coff} is outside the source dispatch domain")


def _f32_bits(bits: int):
    return K.reinterpret("float32", K.uint32(bits))


def _source_exp(value, magic_bias, magic_scale):
    """Reproduce the source expansion with its literal PTX immediates."""
    rounded = K.local_scalar("float32")
    exponent = K.local_scalar("float32")
    offset = K.local_scalar("float32")
    negated = K.local_scalar("float32")
    reduced = K.local_scalar("float32")
    fraction = K.local_scalar("float32")
    exponent_bits = K.local_scalar("float32")
    result = K.local_scalar("float32")
    K.ptx.fma.rn.f32(rounded, value, _f32_bits(0x3BBB989D), _f32_bits(0x3F000000))
    K.ptx.cvt.sat.f32.f32(rounded, rounded)
    K.ptx.fma.rm.f32(exponent, rounded, magic_scale, magic_bias)
    K.ptx["add.f32"](offset, exponent, _f32_bits(0xCB40007F))
    K.ptx.neg.f32(negated, offset)
    K.ptx.fma.rn.f32(reduced, value, _f32_bits(0x3FB8AA3B), negated)
    K.ptx.fma.rn.f32(reduced, value, _f32_bits(0x32A57060), reduced)
    K.ptx.shl.b32(exponent_bits, exponent, K.uint32(23))
    K.ptx.ex2.approx.ftz.f32(fraction, reduced)
    K.ptx["mul.f32"](result, fraction, exponent_bits)
    return result


def _ordered_max(candidate, current):
    """Select with the source kernel's strict ordered floating-point predicate."""
    greater = K.local_scalar("uint32")
    result = K.local_scalar("float32")
    K.ptx.setp.gt.f32(greater, candidate, current)
    K.ptx.selp.f32(result, candidate, current, K.ptx.pred(greater))
    return result


def _load4_bf16x2(score_words, value_words, start, score_pointer, value_pointer, stride_bytes):
    for index in range(4):
        byte_offset = K.int32(index * stride_bytes)
        K.ptx.ld.global_.b32(score_words[start + index], K.ptx.addr(score_pointer, byte_offset))
        K.ptx.ld.global_.b32(value_words[start + index], K.ptx.addr(value_pointer, byte_offset))


def get_kernel(head_dim: int, coff: int, **kwargs):
    """Return the static ``(head_dim, coff)`` specialization."""
    _validate(head_dim, coff)
    vec = 2 if head_dim % 2 == 0 else 1
    ncol = head_dim // vec
    width = coff * head_dim
    win = 8 if coff == 2 else 4

    @K.kernel(warps=2, arch="sm_100a", grid=lambda p: [p["nb_total"], K.ceildiv(ncol, 64), 1])
    def compressor_fwd(
        kv: K.gptr[K.bf16],
        score: K.gptr[K.bf16],
        ape: K.gptr[K.f32],
        cu_seqlens: K.gptr[K.i32],
        cu_seqlens_comp: K.gptr[K.i32],
        out: K.gptr[K.bf16],
        nb_total: K.i32,
        n_seq: K.i32,
    ):
        bb, block_y, _ = K.cta_id()
        thread = K.thread_id()
        col = block_y * K.int32(64) + thread
        with K.If(col < ncol), K.Then():
            column = col * K.int32(vec)

            # The source hoists all APE traffic ahead of row validity checks.
            ape_values = K.alloc_local([win * vec], "float32")
            for k in range(win):
                column_base = column
                if coff == 2 and k >= 4:
                    column_base = K.int32(head_dim) + column
                ape_offset = K.int32((k % 4) * width) + column_base
                if vec == 1:
                    K.ptx.ld.global_.b32(ape_values[k], ape.ptr_to([ape_offset]))
                else:
                    K.ptx.ld.global_.v2.b32(
                        ape_values[k * 2], ape_values[k * 2 + 1], ape.ptr_to([ape_offset])
                    )

            nb_valid = K.local_scalar("int32")
            K.ptx.ld.global_.b32(nb_valid, cu_seqlens_comp.ptr_to([n_seq]))

            with K.If(bb < nb_total), K.Then():
                seq_idx = K.local_scalar("int32", init=K.int32(0))
                block_in_sequence = K.local_scalar("int32", init=K.int32(0))
                with K.If(bb < nb_valid), K.Then():
                    K.assign(block_in_sequence, bb)
                    sequence = K.local_scalar("int32", init=K.int32(0))

                    sequence_begin = K.local_scalar("int32")
                    K.ptx.ld.global_.b32(sequence_begin, cu_seqlens_comp.ptr_to([K.int32(0)]))

                    def scan_boundary(sequence_number):
                        sequence_end = K.local_scalar("int32")
                        K.ptx.ld.global_.b32(
                            sequence_end, cu_seqlens_comp.ptr_to([sequence_number + 1])
                        )
                        before_begin = K.local_scalar("uint32")
                        before_end = K.local_scalar("uint32")
                        candidate_sequence = K.local_scalar("int32")
                        candidate_block = K.local_scalar("int32")
                        K.ptx.setp.lt.s32(before_begin, bb, sequence_begin)
                        K.ptx.setp.lt.s32(before_end, bb, sequence_end)
                        K.ptx.selp.b32(
                            candidate_sequence, sequence_number, seq_idx, K.ptx.pred(before_end)
                        )
                        K.ptx.sub.s32(candidate_block, bb, sequence_begin)
                        K.ptx.selp.b32(
                            candidate_block,
                            candidate_block,
                            block_in_sequence,
                            K.ptx.pred(before_end),
                        )
                        K.ptx.selp.b32(
                            seq_idx, seq_idx, candidate_sequence, K.ptx.pred(before_begin)
                        )
                        K.ptx.selp.b32(
                            block_in_sequence,
                            block_in_sequence,
                            candidate_block,
                            K.ptx.pred(before_begin),
                        )
                        K.assign(sequence_begin, sequence_end)

                    # Preserve the source compiler's nounroll 8/4/2/1 boundary
                    # grouping rather than a scalar two-load loop.
                    with K.While(sequence + 8 <= n_seq):
                        for offset in range(8):
                            scan_boundary(sequence + K.int32(offset))
                        K.assign(sequence, sequence + 8)
                    with K.If(sequence + 4 <= n_seq), K.Then():
                        K.ptx.ld.global_.b32(sequence_begin, cu_seqlens_comp.ptr_to([sequence]))
                        for offset in range(4):
                            scan_boundary(sequence + K.int32(offset))
                        K.assign(sequence, sequence + 4)
                    with K.If(sequence + 2 <= n_seq), K.Then():
                        K.ptx.ld.global_.b32(sequence_begin, cu_seqlens_comp.ptr_to([sequence]))
                        for offset in range(2):
                            scan_boundary(sequence + K.int32(offset))
                        K.assign(sequence, sequence + 2)
                    with K.If(sequence < n_seq), K.Then():
                        K.ptx.ld.global_.b32(sequence_begin, cu_seqlens_comp.ptr_to([sequence]))
                        scan_boundary(sequence)

                token_base = K.local_scalar("int32")
                K.ptx.ld.global_.b32(token_base, cu_seqlens.ptr_to([seq_idx]))
                token = token_base + block_in_sequence * K.int32(4)

                scores = K.alloc_local([win * vec], "float32")
                values = K.alloc_local([win * vec], "float32")
                if coff == 2 and vec == 2:
                    score_words = K.alloc_local([win], "uint32")
                    value_words = K.alloc_local([win], "uint32")
                    own_offset = token * K.int32(width) + K.int32(head_dim) + column
                    _load4_bf16x2(
                        score_words,
                        value_words,
                        4,
                        score.ptr_to([own_offset]),
                        kv.ptr_to([own_offset]),
                        width * 2,
                    )
                for k in range(win):
                    score_bits = K.alloc_local([vec], "uint16")
                    value_bits = K.alloc_local([vec], "uint16")
                    if coff == 2 and k < 4:
                        with K.If(block_in_sequence > 0), K.Then():
                            offset = (token - K.int32(4) + K.int32(k)) * K.int32(width) + column
                            if vec == 1:
                                K.ptx.ld.global_.b16(score_bits[0], score.ptr_to([offset]))
                                K.ptx.ld.global_.b16(value_bits[0], kv.ptr_to([offset]))
                            else:
                                if k == 0:
                                    _load4_bf16x2(
                                        score_words,
                                        value_words,
                                        0,
                                        score.ptr_to([offset]),
                                        kv.ptr_to([offset]),
                                        width * 2,
                                    )
                                K.ptx.mov.b32(score_bits[0], score_bits[1], score_words[k])
                                K.ptx.mov.b32(value_bits[0], value_bits[1], value_words[k])
                        for lane in range(vec):
                            K.ptx.mov.b32(scores[k * vec + lane], K.float32(float("-inf")))
                            K.ptx.mov.b32(values[k * vec + lane], K.float32(0.0))
                            with K.If(block_in_sequence > 0), K.Then():
                                K.ptx.add.rn.f32.bf16(
                                    scores[k * vec + lane],
                                    score_bits[lane],
                                    ape_values[k * vec + lane],
                                )
                                K.ptx.cvt.f32.bf16(values[k * vec + lane], value_bits[lane])
                    else:
                        if coff == 2:
                            offset = (
                                (token + K.int32(k - 4)) * K.int32(width)
                                + K.int32(head_dim)
                                + column
                            )
                        else:
                            offset = (token + K.int32(k)) * K.int32(width) + column
                        if vec == 1:
                            K.ptx.ld.global_.b16(score_bits[0], score.ptr_to([offset]))
                            K.ptx.ld.global_.b16(value_bits[0], kv.ptr_to([offset]))
                        else:
                            if coff == 2:
                                K.ptx.mov.b32(score_bits[0], score_bits[1], score_words[k])
                                K.ptx.mov.b32(value_bits[0], value_bits[1], value_words[k])
                            else:
                                K.ptx.ld.global_.v2.b16(
                                    score_bits[0], score_bits[1], score.ptr_to([offset])
                                )
                                K.ptx.ld.global_.v2.b16(
                                    value_bits[0], value_bits[1], kv.ptr_to([offset])
                                )
                        for lane in range(vec):
                            K.ptx.add.rn.f32.bf16(
                                scores[k * vec + lane], score_bits[lane], ape_values[k * vec + lane]
                            )
                            K.ptx.cvt.f32.bf16(values[k * vec + lane], value_bits[lane])

                # CuTe materializes these two constants once and shares them across
                # every scalar exponential expansion in the thread.
                magic_bias = K.local_scalar("float32")
                magic_scale = K.local_scalar("float32")
                K.ptx.mov.b32(magic_bias, K.uint32(0x4B400001))
                K.ptx.mov.b32(magic_scale, K.uint32(0x437C0000))

                outputs = K.alloc_local([vec], "float32")
                for lane in range(vec):
                    maximum = K.local_scalar("float32")
                    K.ptx.mov.b32(maximum, scores[lane])
                    for k in range(1, win):
                        K.assign(maximum, _ordered_max(scores[k * vec + lane], maximum))

                    denominator = K.local_scalar("float32")
                    K.ptx.mov.b32(denominator, K.float32(0.0))
                    exponentials = K.alloc_local([win], "float32")
                    for k in range(win):
                        difference = K.local_scalar("float32")
                        K.ptx["sub.f32"](difference, scores[k * vec + lane], maximum)
                        K.assign(exponentials[k], _source_exp(difference, magic_bias, magic_scale))
                        K.ptx["add.f32"](denominator, denominator, exponentials[k])

                    accumulator = K.local_scalar("float32")
                    K.ptx.mov.b32(accumulator, K.float32(0.0))
                    for k in range(win):
                        probability = K.local_scalar("float32")
                        K.ptx.div.rn.f32(probability, exponentials[k], denominator)
                        product = K.local_scalar("float32")
                        K.ptx.mul.rn.f32(product, values[k * vec + lane], probability)
                        K.ptx["add.f32"](accumulator, accumulator, product)
                    K.ptx.mov.b32(outputs[lane], accumulator)

                output_offset = bb * K.int32(head_dim) + column
                if vec == 1:
                    output_bits = K.local_scalar("uint16")
                    K.ptx.cvt.rn.bf16.f32(output_bits, outputs[0])
                    K.ptx.st.global_.b16(out.ptr_to([output_offset]), output_bits)
                else:
                    output_word = K.local_scalar("uint32")
                    K.ptx.cvt.rn.bf16x2.f32(output_word, outputs[1], outputs[0])
                    K.ptx.st.global_.b32(out.ptr_to([output_offset]), output_word)

    return compressor_fwd.func


def prepare_data(
    seq_lens: tuple[int, ...],
    head_dim: int,
    coff: int,
    capacity_pad: int = 0,
    *,
    seed: int = 1234,
    **kwargs,
):
    """Create deterministic flat source-layout tensors and segment metadata."""
    import torch

    _validate(head_dim, coff)
    if any(length < 0 for length in seq_lens):
        raise ValueError("sequence lengths must be non-negative")
    if capacity_pad < 0:
        raise ValueError("capacity_pad must be non-negative")

    total_tokens = sum(seq_lens)
    width = coff * head_dim
    token_prefix = [0]
    block_prefix = [0]
    for length in seq_lens:
        token_prefix.append(token_prefix[-1] + length)
        block_prefix.append(block_prefix[-1] + length // _RATIO)
    nb_valid = block_prefix[-1]
    nb_total = nb_valid + capacity_pad

    generator = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.randn(total_tokens, width, generator=generator, dtype=torch.float32).to(
        device="cuda", dtype=torch.bfloat16
    )
    score = torch.randn(total_tokens, width, generator=generator, dtype=torch.float32).to(
        device="cuda", dtype=torch.bfloat16
    )
    ape = torch.randn(_RATIO, width, generator=generator, dtype=torch.float32).cuda()
    cu_seqlens = torch.tensor(token_prefix, dtype=torch.int32, device="cuda")
    cu_seqlens_comp = torch.tensor(block_prefix, dtype=torch.int32, device="cuda")
    guard_elements = 16
    guard_value = -123.0

    def guarded_output():
        backing = torch.full(
            (nb_total * head_dim + 2 * guard_elements,),
            guard_value,
            dtype=torch.bfloat16,
            device="cuda",
        )
        view = backing[guard_elements : guard_elements + nb_total * head_dim]
        return backing, view

    out_backing, out = guarded_output()
    source_out_backing, source_out = guarded_output()
    repeat_out_backing, repeat_out = guarded_output()
    return {
        "kv": kv.reshape(-1),
        "score": score.reshape(-1),
        "ape": ape.reshape(-1),
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_comp": cu_seqlens_comp,
        "out": out,
        "out_backing": out_backing,
        "source_out": source_out,
        "source_out_backing": source_out_backing,
        "repeat_out": repeat_out,
        "repeat_out_backing": repeat_out_backing,
        "nb_total": nb_total,
        "nb_valid": nb_valid,
        "n_seq": len(seq_lens),
        "seq_lens": seq_lens,
        "head_dim": head_dim,
        "coff": coff,
        "guard_elements": guard_elements,
        "guard_value": guard_value,
    }


def _launch(executable, data, output_key="out"):
    if data["nb_total"] == 0:
        return
    executable(
        data["kv"],
        data["score"],
        data["ape"],
        data["cu_seqlens"],
        data["cu_seqlens_comp"],
        data[output_key],
        data["nb_total"],
        data["n_seq"],
    )


_REFERENCE_SOURCE = None


def _load_reference_source():
    global _REFERENCE_SOURCE
    if _REFERENCE_SOURCE is not None:
        return _REFERENCE_SOURCE

    import importlib.util
    import os
    from pathlib import Path

    root = os.environ.get("CUDNN_FRONTEND_PATH")
    if root is None:
        raise RuntimeError("CUDNN_FRONTEND_PATH must point to a cuDNN Frontend source checkout")
    source_path = Path(root) / "python/cudnn/csa/compressor/compressor_sm100.py"
    if not source_path.is_file():
        raise RuntimeError(f"CUDNN_FRONTEND_PATH does not contain {source_path.relative_to(root)}")
    spec = importlib.util.spec_from_file_location(
        "tirx_cudnn_frontend_csa_compressor_source", source_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load cuDNN Frontend compressor source from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _REFERENCE_SOURCE = module
    return module


def _source_launch(data):
    import torch

    source = _load_reference_source()
    device = torch.device("cuda", torch.cuda.current_device())
    source.precompile_fwd(_RATIO, data["head_dim"], data["coff"], device)

    def launch():
        if data["nb_total"] == 0:
            return
        source.run_fwd(
            data["kv"],
            data["score"],
            data["ape"],
            data["cu_seqlens"],
            data["cu_seqlens_comp"],
            data["source_out"],
            data["nb_total"],
            _RATIO,
            data["head_dim"],
            data["coff"],
        )

    return launch


def _eager_reference(data):
    """Vectorized FP32 oracle, including source-defined capacity-padding rows."""
    import torch

    d = data["head_dim"]
    coff = data["coff"]
    width = coff * d
    kv = data["kv"].view(-1, width).float()
    score = data["score"].view(-1, width).float()
    ape = data["ape"].view(_RATIO, width)
    rows = []
    token_base = 0
    for length in data["seq_lens"]:
        blocks = length // _RATIO
        if blocks:
            block_tokens = blocks * _RATIO
            if coff == 1:
                block_values = kv[token_base : token_base + block_tokens, :d].view(
                    blocks, _RATIO, d
                )
                block_scores = score[token_base : token_base + block_tokens, :d].view(
                    blocks, _RATIO, d
                ) + ape[:, :d].unsqueeze(0)
            else:
                own_values = kv[token_base : token_base + block_tokens, d:].view(blocks, _RATIO, d)
                own_scores = score[token_base : token_base + block_tokens, d:].view(
                    blocks, _RATIO, d
                ) + ape[:, d:].unsqueeze(0)
                previous_values = torch.zeros_like(own_values)
                previous_scores = torch.full_like(own_scores, float("-inf"))
                if blocks > 1:
                    previous_values[1:] = kv[
                        token_base : token_base + (blocks - 1) * _RATIO, :d
                    ].view(blocks - 1, _RATIO, d)
                    previous_scores[1:] = score[
                        token_base : token_base + (blocks - 1) * _RATIO, :d
                    ].view(blocks - 1, _RATIO, d) + ape[:, :d].unsqueeze(0)
                block_values = torch.cat((previous_values, own_values), dim=1)
                block_scores = torch.cat((previous_scores, own_scores), dim=1)
            rows.append((block_values * torch.softmax(block_scores, dim=1)).sum(dim=1))
        token_base += length

    padding = data["nb_total"] - data["nb_valid"]
    if padding:
        if coff == 1:
            pad_values = kv[:_RATIO, :d]
            pad_scores = score[:_RATIO, :d] + ape[:, :d]
        else:
            pad_values = torch.cat((torch.zeros_like(kv[:_RATIO, :d]), kv[:_RATIO, d:]), dim=0)
            pad_scores = torch.cat(
                (
                    torch.full_like(score[:_RATIO, :d], float("-inf")),
                    score[:_RATIO, d:] + ape[:, d:],
                ),
                dim=0,
            )
        pad_row = (pad_values * torch.softmax(pad_scores, dim=0)).sum(dim=0)
        rows.append(pad_row.unsqueeze(0).expand(padding, d))
    if not rows:
        return torch.empty((0, d), dtype=torch.bfloat16, device=kv.device)
    return torch.cat(rows, dim=0).to(torch.bfloat16)


def _assert_guard(data, backing_key):
    import torch

    guard = data["guard_elements"]
    expected = torch.full(
        (guard,), data["guard_value"], dtype=torch.bfloat16, device=data[backing_key].device
    )
    backing = data[backing_key]
    if not torch.equal(backing[:guard], expected) or not torch.equal(backing[-guard:], expected):
        raise AssertionError(f"compressor forward wrote outside {backing_key}'s output view")


def _validate_outputs(data, *, include_source: bool, include_oracle: bool):
    import torch

    actual = data["out"].view(data["nb_total"], data["head_dim"])
    repeated = data["repeat_out"].view(data["nb_total"], data["head_dim"])
    if not torch.equal(actual, repeated):
        mismatch = int((actual != repeated).sum().item())
        raise AssertionError(f"TIRx compressor forward is not deterministic ({mismatch} values)")
    if include_source:
        source = data["source_out"].view(data["nb_total"], data["head_dim"])
        if not torch.equal(actual, source):
            mismatch = int((actual != source).sum().item())
            max_abs = float((actual.float() - source.float()).abs().max().item())
            raise AssertionError(
                f"TIRx differs from cuDNN Frontend source: mismatch={mismatch}, max_abs={max_abs}"
            )
    if include_oracle:
        oracle = _eager_reference(data)
        mismatch = int((actual != oracle).sum().item())
        allowed = max(1, actual.numel() // 1000)
        max_abs = (
            float((actual.float() - oracle.float()).abs().max().item()) if actual.numel() else 0.0
        )
        if mismatch > allowed or max_abs > 1.6e-2:
            raise AssertionError(
                f"TIRx differs from FP32 eager oracle: mismatch={mismatch}/{allowed}, "
                f"max_abs={max_abs}"
            )
    _assert_guard(data, "out_backing")
    _assert_guard(data, "repeat_out_backing")
    if include_source:
        _assert_guard(data, "source_out_backing")


def _kernel_config(config):
    return {key: value for key, value in config.items() if key != "label"}


def prepare_bench(**config: Any):
    """Compile the static TIRx specialization before GPU assignment."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    config = _kernel_config(config)
    state = {"config": config, "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(**config: Any):
    """Compare TIRx with the pinned source and a standalone FP32 eager oracle."""
    import torch

    from tirx_kernels.runner import compile_kernel

    config = _kernel_config(config)
    data = prepare_data(**config)
    snapshots = {
        key: data[key].clone() for key in ("kv", "score", "ape", "cu_seqlens", "cu_seqlens_comp")
    }
    executable = compile_kernel(get_kernel(**config))
    _launch(executable, data)
    _launch(executable, data, "repeat_out")
    source_launch = _source_launch(data)
    source_launch()
    torch.cuda.synchronize()
    _validate_outputs(data, include_source=True, include_oracle=True)
    for key, snapshot in snapshots.items():
        if not torch.equal(data[key], snapshot):
            raise AssertionError(f"compressor forward modified input {key}")
    return {"rows": data["nb_total"], "head_dim": data["head_dim"], "coff": data["coff"]}


def run_gpu(
    prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs: Any
):
    """Validate once, then expose only the two launch closures to bench_suite."""
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = _kernel_config({**prepared["config"], **kwargs})
    data = prepare_data(**config)
    executable = prepared["executable"]

    def tirx_launch():
        _launch(executable, data)

    def tirx_repeat_launch():
        _launch(executable, data, "repeat_out")

    tirx_launch()
    tirx_repeat_launch()
    torch.cuda.synchronize()

    references = None
    include_source = external_references_enabled()
    if include_source:
        source_launch = _source_launch(data)
        source_launch()
        torch.cuda.synchronize()
        references = {"cudnn_frontend": lambda: source_launch}
    _validate_outputs(data, include_source=include_source, include_oracle=False)
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config: Any):
    """Standalone wrapper over the bench-suite preparation and GPU stages."""
    return prepare_bench(**config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]
