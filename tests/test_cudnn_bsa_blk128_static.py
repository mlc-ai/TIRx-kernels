# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Static contracts for the SM100 blk128 BSA backward kernel."""

import ast
import inspect
from collections import Counter

from tirx_kernels.cudnn.bsa import block_sparse_attention_backward_sm100_blk128 as bsa
from tirx_kernels.cudnn.bsa._block_sparse_attention_backward_sm100_blk128 import (
    kernel as bsa_kernel,
)
from tvm.tirx.stmt_functor import StmtExprVisitor


def test_main_kernel_uses_uniform_register_targets_per_physical_warpgroup():
    kernels = bsa.get_kernel(
        batch=1, num_heads=1, seqlen_q=1, seqlen_kv=128, head_dim=64, tensor_layout="bhsd"
    )

    class Scanner(StmtExprVisitor):
        def __init__(self):
            super().__init__()
            self.calls = []

        def visit_call_(self, op):
            if getattr(op.op, "name", "") == "tirx.ptx.setmaxnreg":
                self.calls.append((int(op.args[0]), op.args[1].value))
            super().visit_call_(op)

    scanner = Scanner()
    scanner(kernels[1].body)
    assert Counter(scanner.calls) == Counter({(88, "dec"): 4, (136, "inc"): 1, (152, "inc"): 1})


def test_direct_epilogue_bridges_tcgen_reads_before_generic_smem_reuse():
    tree = ast.parse(inspect.getsource(bsa_kernel))
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "is_dk"
    ]
    assert len(loops) == 1

    statements = [ast.unparse(statement) for statement in loops[0].body]
    assert statements[0].startswith("dkdv_pipe.full.wait(")
    assert statements[1] == "K.ptx.fence.proxy.async_.shared__cta()"
