# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Exercise the original correctness entrypoint's source-oracle selection."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1] / "tirx_kernels/flashinfer/norm/fused_add_rmsnorm_quant.py"
)


class SourceUnavailable(RuntimeError):
    pass


class ExistingFallbackReached(RuntimeError):
    pass


def load_functions(*names):
    tree = ast.parse(SOURCE.read_text())
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(functions) == len(names)
    namespace = {"Any": object}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("h,rolled", [(131073, True), (1048576, True), (4096, False)])
@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7), (11, 0), (11, 1)])
def test_source_failure_cannot_fall_back_after_source_selection(monkeypatch, h, rolled, capability):
    namespace = load_functions("run_test")

    def unavailable(_device):
        raise SourceUnavailable("source failure must not fall back to the loose mathematical check")

    def synchronize():
        raise ExistingFallbackReached

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_capability=lambda _device: capability, synchronize=synchronize
            )
        ),
    )
    namespace.update(
        _DEFAULT_EPS=1e-6,
        _prepare_tensors=lambda _config: {
            "input": SimpleNamespace(device="cuda:0"),
            "y_row_stride": h,
        },
        _snapshot=lambda *_args: {},
        _prepare_output=lambda *_args, **_kwargs: {},
        _uses_compact_specialization=lambda *_args: True,
        _compiled_test_specialization=lambda *_args: None,
        _launch_tirx=lambda *_args: None,
        _uses_rolled_fragment_loops=lambda _h: rolled,
        _flashinfer_api=unavailable,
    )
    expected = SourceUnavailable if not rolled or capability == (11, 0) else ExistingFallbackReached
    with pytest.raises(expected):
        namespace["run_test"](
            M=3,
            H=h,
            input_dtype="bfloat16",
            output_dtype="float8_e5m2",
            input_layout="compact",
            residual_layout="compact",
            output_layout="compact",
            enable_pdl=True,
        )


def test_raw_source_check_rejects_zero_and_single_byte_corruption():
    import torch

    namespace = load_functions("_raw_bytes", "_assert_raw_equal")
    check = namespace["_assert_raw_equal"]
    expected = torch.tensor([1.0, -0.5, 2.0, 0.25]).to(torch.float8_e5m2)
    check(expected.clone(), expected, name="exact clone")
    for actual in (torch.zeros(4).to(expected.dtype), expected.clone()):
        if torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
            actual.view(torch.uint8)[0] ^= 1
        with pytest.raises(AssertionError, match="FP8 bytes differ"):
            check(actual, expected, name="corrupted source comparison")
