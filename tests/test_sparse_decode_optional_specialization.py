from __future__ import annotations

import pytest
import torch

import tvm
from tirx_kernels.flashmla.sparse_decode_head64 import CONFIGS, get_kernel
from tvm import tirx


def _assert_static_parameter_protocol(kernel, *, total: int, buffers: int) -> None:
    buffer_parameters = [tirx.is_buffer_var(parameter) for parameter in kernel.params]

    assert len(kernel.params) == total
    assert buffer_parameters == [True] * buffers + [False] * (total - buffers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required by the public factory")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required by the public factory")
def test_public_model_selection_changes_only_the_model_specific_function() -> None:
    common = {
        "b": 4,
        "s_q": 2,
        "s_kv": 128,
        "topk": 64,
        "page_block_size": 64,
        "have_attn_sink": True,
    }

    main_model1, combine_model1 = get_kernel(model_type="MODEL1", **common)
    main_v32, combine_v32 = get_kernel(model_type="V32", **common)

    with pytest.raises(ValueError):
        tvm.ir.assert_structural_equal(main_model1, main_v32)
    tvm.ir.assert_structural_equal(combine_model1, combine_v32)
