# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Static synchronization contracts for the SM100 GDN backward kernel."""

import ast
import inspect

from tirx_kernels.cudnn.linear_attention import gdn_bprop_f16 as gdn


def _statement_sequences(tree):
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(node, field, None)
            if isinstance(statements, list):
                yield [ast.unparse(statement) for statement in statements]


def _has_adjacent_contract(sequences, *fragments):
    for statements in sequences:
        for start in range(len(statements) - len(fragments) + 1):
            if all(
                fragment in statements[start + offset] for offset, fragment in enumerate(fragments)
            ):
                return True
    return False


def test_kk_scratch_reuse_waits_for_all_cg0_readers():
    sequences = list(_statement_sequences(ast.parse(inspect.getsource(gdn))))

    assert _has_adjacent_contract(
        sequences, "K.ptx.bar.sync(K.uint32(2), K.uint32(128))", "token0 =", "K.ptx.st.shared.f32"
    )


def test_tmem_consumers_drain_loads_before_publishing_reuse():
    sequences = list(_statement_sequences(ast.parse(inspect.getsource(gdn))))

    contracts = (
        ("_tcgen_wait_load()", "_BAR_KK_DONE"),
        ("_tcgen_wait_load()", "K.ptx.fence.proxy.async_.shared__cta()", "_BAR_A_READY"),
        ("_tcgen_wait_load()", "_tcgen_wait_store()", "_BAR_DU_INP_READY"),
        ("_tcgen_wait_load()", "K.ptx.fence.proxy.async_.shared__cta()", "_BAR_U_READY"),
        ("_tcgen_wait_load()", "_tcgen_wait_store()", "_BAR_DYP_INP_READY"),
        ("_tcgen_wait_load()", "K.ptx.bar.sync(K.uint32(5), K.uint32(128))"),
    )
    assert all(_has_adjacent_contract(sequences, *contract) for contract in contracts)
