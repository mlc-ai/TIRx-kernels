from __future__ import annotations

import pytest

from tirx_kernels.flashmla.sparse_decode_head64 import (
    COMBINE_OPTIONAL_BUFFER_PARAMS,
    MAIN_OPTIONAL_BUFFER_PARAMS,
    ModelType,
    _present_runtime_args,
    _specialized_combine_kernel,
    _specialized_decode_kernels,
    _specialized_main_kernel,
)

MAIN_PARAMETER_NAMES = (
    "q_h",
    "kv_h",
    "indices_h",
    "topk_length_h",
    "attn_sink_h",
    "lse_h",
    "out_h",
    "lse_accum_h",
    "o_accum_h",
    "tile_scheduler_metadata_h",
    "num_splits_h",
    "extra_kv_h",
    "extra_indices_h",
    "extra_topk_length_h",
    "sm_scale_div_log2",
    "stride_q_b",
    "stride_q_s_q",
    "stride_q_h_q",
    "stride_kv_block",
    "stride_kv_row",
    "stride_indices_b",
    "stride_indices_s_q",
    "stride_lse_b",
    "stride_lse_s_q",
    "stride_o_b",
    "stride_o_s_q",
    "stride_o_h_q",
    "stride_extra_kv_block",
    "stride_extra_kv_row",
    "stride_extra_indices_b",
    "stride_extra_indices_s_q",
    "stride_lse_accum_split",
    "stride_lse_accum_s_q",
    "stride_o_accum_split",
    "stride_o_accum_s_q",
    "stride_o_accum_h_q",
    "b",
    "s_q",
    "topk",
    "extra_topk",
    "num_blocks",
    "extra_num_blocks",
    "page_block_size",
    "extra_page_block_size",
    "num_sm_parts",
)
COMBINE_PARAMETER_NAMES = (
    "lse_h",
    "out_h",
    "lse_accum_h",
    "o_accum_h",
    "num_splits_h",
    "attn_sink_h",
    "stride_lse_b",
    "stride_lse_s_q",
    "stride_o_b",
    "stride_o_s_q",
    "stride_o_h_q",
    "stride_lse_accum_split",
    "stride_lse_accum_s_q",
    "stride_o_accum_split",
    "stride_o_accum_s_q",
    "stride_o_accum_h_q",
    "b",
    "s_q",
    "h_q",
    "d_v",
    "num_sm_parts",
)
MAIN_PRESENCE_MASKS = (
    (False, False, False, False, False),
    (True, False, False, False, False),
    (False, True, False, False, False),
    (True, True, False, False, False),
    (False, False, True, True, False),
    (False, False, True, True, True),
    (True, True, True, True, True),
)


def _expected_params(
    parameter_names: tuple[str, ...], optional_names: tuple[str, ...], presence: tuple[bool, ...]
) -> list[str]:
    optional_presence = dict(zip(optional_names, presence, strict=True))
    return [name for name in parameter_names if optional_presence.get(name, True)]


@pytest.mark.parametrize("model_type", (ModelType.MODEL1, ModelType.V32))
@pytest.mark.parametrize("presence", MAIN_PRESENCE_MASKS)
def test_main_optional_specializations_have_static_abi(model_type, presence) -> None:
    kernel = _specialized_main_kernel(model_type, presence)
    expected_params = _expected_params(MAIN_PARAMETER_NAMES, MAIN_OPTIONAL_BUFFER_PARAMS, presence)

    assert [param.name for param in kernel.params] == expected_params
    assert len(kernel.params) == 40 + sum(presence)
    optional_buffers = {
        param.name for param in kernel.buffer_map if param.name in MAIN_OPTIONAL_BUFFER_PARAMS
    }
    assert optional_buffers == {
        name
        for name, is_present in zip(MAIN_OPTIONAL_BUFFER_PARAMS, presence, strict=True)
        if is_present
    }


@pytest.mark.parametrize("have_attn_sink", (False, True))
def test_combine_optional_specialization_has_static_abi(have_attn_sink: bool) -> None:
    kernel = _specialized_combine_kernel(8, have_attn_sink)
    presence = (have_attn_sink,)

    assert [param.name for param in kernel.params] == _expected_params(
        COMBINE_PARAMETER_NAMES, COMBINE_OPTIONAL_BUFFER_PARAMS, presence
    )
    assert len(kernel.params) == 20 + have_attn_sink
    assert {param.name for param in kernel.buffer_map} & set(COMBINE_OPTIONAL_BUFFER_PARAMS) == (
        {"attn_sink_h"} if have_attn_sink else set()
    )


def test_main_and_combine_specialization_caches_are_independent() -> None:
    presence = (True, True, False, False, False)
    main_model1, combine_model1 = _specialized_decode_kernels(ModelType.MODEL1, 8, presence)
    main_v32, combine_v32 = _specialized_decode_kernels(ModelType.V32, 8, presence)

    assert main_model1 is _specialized_main_kernel(ModelType.MODEL1, presence)
    assert main_v32 is _specialized_main_kernel(ModelType.V32, presence)
    assert main_model1 is not main_v32
    assert combine_model1 is combine_v32
    assert combine_model1 is _specialized_combine_kernel(8, True)


def test_runtime_argument_filter_drops_absent_optionals() -> None:
    assert _present_runtime_args(("a", None, "c", None), (1, 3), (False, False)) == ("a", "c")
    assert _present_runtime_args(("a", "b", "c", "d"), (1, 3), (True, False)) == ("a", "b", "c")
