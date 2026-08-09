from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tirx_kernels.flashmla.sparse_decode_head64 import CONFIGS, get_kernel
from tvm import tirx


def _assert_static_parameter_protocol(kernel, *, total: int, buffers: int) -> None:
    buffer_parameters = [tirx.is_buffer_var(parameter) for parameter in kernel.params]

    assert len(kernel.params) == total
    assert buffer_parameters == [True] * buffers + [False] * (total - buffers)


@pytest.fixture(autouse=True)
def _sm100_compile_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148),
    )


@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config["label"])
def test_public_configs_expose_the_documented_static_argument_protocol(config) -> None:
    parameters = {key: value for key, value in config.items() if key != "label"}

    main, combine = get_kernel(**parameters)

    have_extra_cache = bool(config.get("extra_topk", 0))
    main_optional_buffers = sum(
        (
            bool(config.get("have_topk_length", False)),
            bool(config.get("have_attn_sink", False)),
            have_extra_cache,
            have_extra_cache,
            bool(config.get("have_extra_topk_length", False)),
        )
    )
    combine_optional_buffers = int(bool(config.get("have_attn_sink", False)))
    _assert_static_parameter_protocol(
        main, total=40 + main_optional_buffers, buffers=9 + main_optional_buffers
    )
    _assert_static_parameter_protocol(
        combine, total=20 + combine_optional_buffers, buffers=5 + combine_optional_buffers
    )
