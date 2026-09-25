# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Lazy loader for the pinned cuDNN Frontend blk128 BSA backward program."""

from tirx_kernels.cudnn._reference import load_reference_module


def compile_reference(data):
    """Compile/cache the pinned source path and return a preallocated launch closure."""
    interface = load_reference_module("cudnn.block_sparse_attention._interface")
    inputs = data["inputs"]
    output = data["source"]

    def launch():
        interface.bsa_attn_bwd(
            inputs["do"],
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["o"],
            inputs["lse"],
            inputs["block_index"],
            int(inputs["block_sparse_num"]),
            block_sizes=None,
            q2k_block_nums=inputs["block_nums"],
            softmax_scale=float(inputs["softmax_scale"]),
            dq=output["dq"],
            dk=output["dk"],
            dv=output["dv"],
            bucket_size_blocks=int(data["derived"]["bucket_size_blocks"]),
            sparse_block_size=128,
            layout=data["config"]["tensor_layout"],
        )

    return launch
