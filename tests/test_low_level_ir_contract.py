# Copyright (c) 2026 The TIRx Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

import tvm
from tirx_kernels.low_level_ir import inspect_registry_builder_contract, iter_prim_funcs
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T


@contextmanager
def _load_package(tmp_path: Path, files: dict[str, str]):
    package = tmp_path / "contract_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name, source in files.items():
        (package / f"{name}.py").write_text(source)
    sys.path.insert(0, str(tmp_path))
    try:
        modules = {}
        for name in files:
            module = ModuleType(f"contract_fixture.{name}")
            module.__file__ = str(package / f"{name}.py")
            sys.modules[module.__name__] = module
            modules[name] = module
        yield package, modules
    finally:
        sys.path.remove(str(tmp_path))
        for name in list(sys.modules):
            if name == "contract_fixture" or name.startswith("contract_fixture."):
                del sys.modules[name]


@pytest.mark.parametrize("parser_api", ["T.jit", "T.prim_func"])
def test_builder_contract_rejects_builder_wrapper_around_reachable_parser(tmp_path, parser_api):
    files = {
        "parser": f"""
from tvm.script import tirx as T

@{parser_api}
def parsed_kernel():
    return None
""",
        "kernel": """
from contract_fixture.parser import parsed_kernel

KERNEL_META = {"name": "fixture", "category": "fixture", "compute_capability": 10}

def get_kernel():
    from tvm.script.ir_builder import IRBuilder
    with IRBuilder() as builder:
        parsed_kernel()
    return builder.get()
""",
    }
    with _load_package(tmp_path, files) as (package, modules):
        report = inspect_registry_builder_contract(
            {"fixture": modules["kernel"]}, package_root=package
        )

    assert not report.ok
    assert any(finding.kind == "parser_decorator" for finding in report.violations)
    assert "contract_fixture.parser.parsed_kernel" in report.reachable_functions


def test_builder_contract_accepts_cross_module_explicit_builder_endpoint(tmp_path):
    files = {
        "builder": """
def build_kernel():
    from tvm.script.ir_builder import IRBuilder
    with IRBuilder() as builder:
        pass
    return builder.get()
""",
        "kernel": """
from contract_fixture.builder import build_kernel as make_kernel

KERNEL_META = {"name": "fixture", "category": "fixture", "compute_capability": 10}

def get_kernel():
    return make_kernel()
""",
    }
    with _load_package(tmp_path, files) as (package, modules):
        report = inspect_registry_builder_contract(
            {"fixture": modules["kernel"]}, package_root=package
        )

    assert report.ok, report.summary()
    assert report.builder_functions == ("contract_fixture.builder.build_kernel",)


def test_builder_contract_accepts_explicitly_reexported_builder_entrypoint(tmp_path):
    files = {
        "builder": """
def get_kernel():
    from tvm.script.ir_builder import IRBuilder
    with IRBuilder() as builder:
        pass
    return builder.get()
""",
        "kernel": """
from contract_fixture.builder import get_kernel as get_kernel

KERNEL_META = {"name": "fixture", "category": "fixture", "compute_capability": 10}
""",
    }
    with _load_package(tmp_path, files) as (package, modules):
        report = inspect_registry_builder_contract(
            {"fixture": modules["kernel"]}, package_root=package
        )

    assert report.ok, report.summary()
    assert report.builder_functions == ("contract_fixture.builder.get_kernel",)


def test_builder_contract_rejects_reexported_parser_entrypoint(tmp_path):
    files = {
        "parser": """
from tvm.script import tirx as T

@T.jit
def get_kernel():
    return None
""",
        "kernel": """
from contract_fixture.parser import get_kernel as get_kernel

KERNEL_META = {"name": "fixture", "category": "fixture", "compute_capability": 10}
""",
    }
    with _load_package(tmp_path, files) as (package, modules):
        report = inspect_registry_builder_contract(
            {"fixture": modules["kernel"]}, package_root=package
        )

    assert not report.ok
    assert any(finding.kind == "parser_decorator" for finding in report.violations)
    assert "contract_fixture.parser.get_kernel" in report.reachable_functions


def test_recursive_prim_func_traversal_preserves_paths_and_module_members():
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("first")
            T.evaluate(0)
    first = builder.get()
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("second")
            T.evaluate(1)
    second = builder.get()

    nested = {"entry": [first, tvm.IRModule({"secondary": second})]}

    assert [path for path, _ in iter_prim_funcs(nested)] == [
        "root['entry'][0]",
        "root['entry'][1].secondary",
    ]
