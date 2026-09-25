# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Lazy direct three-kernel loader for the pinned SM100 blk64 BSA backward."""

import torch

from tirx_kernels.cudnn._reference import import_cutlass_reference, load_reference_module


def compile_reference(data):
    """Compile the direct class ABI; CSR construction and allocations stay outside timing."""
    import_cutlass_reference()
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    interface = load_reference_module("cudnn.block_sparse_attention._interface")
    source = load_reference_module(
        "cudnn.block_sparse_attention.csrc.bwd.sm100_blk64.bsa_bwd_sm100"
    )
    config = data["config"]
    inputs = data["inputs"]
    output = data["source"]
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    problem_shape = (seqlen_q, seqlen_kv, 128, (heads, batch))

    def dynamic_tensor(tensor):
        value = interface.from_dlpack(
            tensor.detach(), assumed_align=16, enable_tvm_ffi=True
        ).mark_layout_dynamic()
        try:
            return value.mark_compact_shape_dynamic(
                mode=3, stride_order=tensor.dim_order(), divisibility=128
            )
        except RuntimeError as error:
            if "stride_order is not consistent" not in str(error):
                raise
            # The source class itself carries Int64 layout arithmetic.  Its
            # public helper's optional compact-shape annotation rejects the
            # deliberate singleton-dimension padding probe, so retain the same
            # class ABI with a fully dynamic layout for that host-only case.
            return interface.from_dlpack(
                tensor.detach(), assumed_align=16, enable_tvm_ffi=True
            ).mark_layout_dynamic()

    kernel = source.BlockSparseAttnBackwardSm100Blk64(
        sparse_block_size=64, has_block_sizes=bool(config["has_block_sizes"])
    )
    compiled = cute.compile(
        kernel,
        problem_shape,
        dynamic_tensor(inputs["do"]),
        dynamic_tensor(inputs["o"]),
        dynamic_tensor(inputs["q"]),
        dynamic_tensor(inputs["k"]),
        dynamic_tensor(inputs["v"]),
        interface._to_cute_tensor(inputs["lse"], leading_dim=2),
        dynamic_tensor(output["dq"]),
        dynamic_tensor(output["dk"]),
        dynamic_tensor(output["dv"]),
        interface._to_cute_tensor(inputs["bucketed_offsets"], leading_dim=3),
        interface._to_cute_tensor(inputs["bucketed_indices"], leading_dim=2),
        interface._to_cute_tensor(inputs["block_sizes"], leading_dim=1),
        interface._to_cute_tensor(output["workspace"], fully_dynamic=True),
        float(inputs["softmax_scale"]),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )

    def launch():
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(
            problem_shape,
            inputs["do"],
            inputs["o"],
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["lse"],
            output["dq"],
            output["dk"],
            output["dv"],
            inputs["bucketed_offsets"],
            inputs["bucketed_indices"],
            inputs["block_sizes"],
            output["workspace"],
            float(inputs["softmax_scale"]),
            stream,
        )

    launch.source_kernel = kernel
    launch.compiled = compiled
    return launch
