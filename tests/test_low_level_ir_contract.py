# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

import pytest

import tvm
from tirx_kernels.low_level_ir import (
    LowLevelIRContractError,
    check_low_level_ir,
    inspect_low_level_ir,
)
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


@T.prim_func
def _local_accesses_are_legal():
    values = T.alloc_local((2,), "float32")
    tmem_value = T.alloc_buffer((1,), "float32", scope="tmem")
    values[0] = T.float32(1)
    values[1] = values[0]
    tmem_value[0] = values[1]


@T.prim_func
def _ptx_global_and_shared_accesses_are_legal(
    source: T.Buffer((1,), "float32"), destination: T.Buffer((1,), "float32")
):
    scratch = T.alloc_buffer((1,), "float32", scope="shared")
    value = T.alloc_local((1,), "float32")
    T.ptx.ld.global_.f32(value[0], source.ptr_to([0]))
    T.ptx.st.shared.f32(scratch.ptr_to([0]), value[0])
    T.ptx.ld.shared.f32(value[0], scratch.ptr_to([0]))
    T.ptx.st.global_.f32(destination.ptr_to([0]), value[0])


def test_local_and_explicit_ptx_memory_accesses_pass() -> None:
    report = check_low_level_ir(
        [_local_accesses_are_legal, _ptx_global_and_shared_accesses_are_legal]
    )

    assert report.ok
    assert len(report.checked_functions) == 2


@T.prim_func
def _address_only_accesses_are_legal(global_buffer: T.Buffer((2,), "float32")):
    shared_buffer = T.alloc_buffer((2,), "float32", scope="shared.dyn")
    local_buffer = T.alloc_local((1,), "float32")
    T.evaluate(T.address_of(global_buffer[1]))
    T.evaluate(T.address_of(shared_buffer[1]))
    T.evaluate(T.address_of(local_buffer[0]))


def test_direct_address_operands_are_reported_without_being_rejected() -> None:
    report = check_low_level_ir(_address_only_accesses_are_legal)

    assert report.ok
    assert {finding.scope for finding in report.address_only_loads} == {"global", "shared.dyn"}
    assert {finding.kind for finding in report.address_only_loads} == {"address_only_buffer_load"}


def test_supported_function_containers_are_checked_recursively() -> None:
    module = tvm.IRModule({"module_entry": _local_accesses_are_legal})
    nested = {
        "direct": _local_accesses_are_legal,
        "module": module,
        "sequence": [_local_accesses_are_legal, (_local_accesses_are_legal,)],
    }

    report = check_low_level_ir(nested)

    assert report.ok
    assert len(report.checked_functions) == 4
    assert any("module_entry" in path for path in report.checked_functions)


@T.prim_func
def _tile_operation_is_forbidden(
    source: T.Buffer((1,), "float32"), destination: T.Buffer((1,), "float32")
):
    Tx.copy(destination[:], source[:])


def test_tile_operation_is_rejected() -> None:
    with pytest.raises(LowLevelIRContractError) as caught:
        check_low_level_ir(_tile_operation_is_forbidden)

    assert {finding.kind for finding in caught.value.report.violations} == {"tile_primitive"}


def _real_load(scope: str):
    @T.prim_func
    def kernel(source_ptr: T.handle):
        source = T.match_buffer(source_ptr, (1,), "float32", scope=scope)
        value = T.alloc_local((1,), "float32")
        value[0] = source[0]

    return kernel


@pytest.mark.parametrize("scope", ["global", "global.texture", "shared", "shared.dyn"])
def test_real_loads_from_forbidden_scopes_are_rejected(scope: str) -> None:
    report = inspect_low_level_ir(_real_load(scope))

    assert not report.ok
    assert [(finding.kind, finding.scope) for finding in report.violations] == [
        ("buffer_load", scope)
    ]


def _store(scope: str):
    @T.prim_func
    def kernel(destination_ptr: T.handle):
        destination = T.match_buffer(destination_ptr, (1,), "float32", scope=scope)
        destination[0] = T.float32(1)

    return kernel


@pytest.mark.parametrize("scope", ["global", "global.texture", "shared", "shared.dyn"])
def test_stores_to_forbidden_scopes_are_rejected(scope: str) -> None:
    report = inspect_low_level_ir(_store(scope))

    assert not report.ok
    assert [(finding.kind, finding.scope) for finding in report.violations] == [
        ("buffer_store", scope)
    ]


@T.prim_func
def _ordinary_call_argument_is_a_real_load(source: T.Buffer((1,), "float32")):
    T.evaluate(T.call_extern("consume", source[0], dtype="float32"))


def test_only_a_direct_address_operand_receives_the_exemption() -> None:
    report = inspect_low_level_ir(_ordinary_call_argument_is_a_real_load)

    assert not report.ok
    assert len(report.address_only_loads) == 0
    assert [finding.kind for finding in report.violations] == ["buffer_load"]


@T.prim_func
def _nested_index_load_remains_a_real_access(
    data: T.Buffer((2,), "float32"), index: T.Buffer((1,), "int32")
):
    T.evaluate(T.address_of(data[index[0]]))


def test_load_inside_an_address_index_is_not_exempt() -> None:
    report = inspect_low_level_ir(_nested_index_load_remains_a_real_access)

    assert not report.ok
    assert [finding.scope for finding in report.address_only_loads] == ["global"]
    assert [(finding.kind, finding.scope) for finding in report.violations] == [
        ("buffer_load", "global")
    ]


@T.prim_func
def _forbidden_load_in_buffer_load_predicate(guard: T.Buffer((1,), "bool")):
    local = T.alloc_local((1,), "float32")
    T.evaluate(local.vload([0], predicate=guard[0]))


def test_forbidden_load_in_buffer_load_predicate_is_rejected() -> None:
    report = inspect_low_level_ir(_forbidden_load_in_buffer_load_predicate)

    assert not report.ok
    assert [(finding.kind, finding.scope) for finding in report.violations] == [
        ("buffer_load", "global")
    ]


def test_one_invalid_function_rejects_the_whole_nested_result() -> None:
    result = {
        "legal": _local_accesses_are_legal,
        "nested": (_local_accesses_are_legal, [_real_load("shared.cluster")]),
    }

    with pytest.raises(LowLevelIRContractError) as caught:
        check_low_level_ir(result)

    finding = caught.value.report.violations[0]
    assert finding.scope == "shared.cluster"
    assert finding.node_type == "BufferLoad"
    assert "nested" in finding.function
    assert finding.span is not None
    assert caught.value.report.to_dict()["ok"] is False


@pytest.mark.parametrize("scope", ["globalish", "shared_memory", "local", "tmem"])
def test_unrelated_or_allowed_scope_names_are_not_rejected(scope: str) -> None:
    report = check_low_level_ir(_real_load(scope))

    assert report.ok


@pytest.mark.parametrize("empty", [[], (), {}, tvm.IRModule({})])
def test_empty_function_collections_are_rejected(empty) -> None:
    with pytest.raises(TypeError, match="at least one TIRx PrimFunc"):
        check_low_level_ir(empty)
