# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

import pytest
import torch

from tirx_kernels.flashinfer.topk import fast_topk_clusters, filtered_topk, radix_topk_single_cta
from tirx_kernels.flashinfer.utils import topk_harness

COMPARATORS = (
    radix_topk_single_cta._compare,
    topk_harness.compare_outputs,
    filtered_topk.compare_filtered_outputs,
)


def _config(**kwargs):
    return {
        "num_rows": 1,
        "length": 4,
        "k": 2,
        "mode": "basic",
        "deterministic": False,
        "tie_break": filtered_topk.TIE_NONE,
        **kwargs,
    }


def _output(indices, values=None):
    result = {"indices": torch.tensor(indices, dtype=torch.int32)}
    if values is not None:
        result["values"] = torch.tensor(values, dtype=torch.float32)
    return result


@pytest.mark.parametrize("compare", COMPARATORS)
@pytest.mark.parametrize("indices", ([[0, 0]], [[0, 9]], [[-2, 1]], [[-1, 1]]))
def test_rejects_invalid_nondeterministic_indices(compare, indices):
    data = {"scores": torch.tensor([[1.0, 1.0, 0.0, 1.0]])}
    reference = _output([[0, 1]], [[1.0, 1.0]])
    corrupt = _output(indices, [[1.0, 1.0]])
    with pytest.raises(AssertionError):
        compare(_config(), data, reference, corrupt)


@pytest.mark.parametrize("compare", COMPARATORS)
def test_allows_different_tied_indices_and_order(compare):
    data = {"scores": torch.tensor([[1.0, 1.0, 0.0, 1.0]])}
    compare(_config(), data, _output([[0, 1]], [[1.0, 1.0]]), _output([[3, 0]], [[1.0, 1.0]]))


@pytest.mark.parametrize("compare", COMPARATORS)
def test_rejects_values_assigned_to_the_wrong_indices(compare):
    data = {"scores": torch.tensor([[9.0, 4.0, 0.0, 1.0]])}
    with pytest.raises(AssertionError):
        compare(_config(), data, _output([[0, 1]], [[9.0, 4.0]]), _output([[0, 1]], [[4.0, 9.0]]))


@pytest.mark.parametrize("compare", COMPARATORS)
def test_checks_reference_indices_too(compare):
    data = {"scores": torch.tensor([[1.0, 1.0, 0.0, 1.0]])}
    corrupt = _output([[0, 0]], [[1.0, 1.0]])
    with pytest.raises(AssertionError):
        compare(_config(), data, corrupt, corrupt)


@pytest.mark.parametrize("deterministic,k", ((False, 2), (True, 1)))
def test_filtered_tie_break_checks_output_values(deterministic, k):
    config = _config(k=k, deterministic=deterministic, tie_break=filtered_topk.TIE_SMALL)
    data = {"scores": torch.tensor([[9.0, 4.0, 0.0, 1.0]])}
    indices = [list(range(k))]
    reference = _output(indices, [[9.0, 4.0][:k]])
    valid = _output(indices, [[9.0, 4.0][:k]])
    filtered_topk.compare_filtered_outputs(config, data, reference, valid)
    corrupt = _output(indices, [[float("nan")] * k])
    with pytest.raises(AssertionError):
        filtered_topk.compare_filtered_outputs(config, data, reference, corrupt)


@pytest.mark.parametrize("compare", COMPARATORS)
@pytest.mark.parametrize("mode", ("ragged", "page_table"))
def test_transformed_indices_with_independent_row_starts_and_padding(compare, mode):
    config = _config(num_rows=2, length=8, k=3, mode=mode)
    data = {
        "scores": torch.tensor(
            [[0.0, 9.0, 7.0, 7.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "row_starts": torch.tensor([1, 2]),
        "lengths": torch.tensor([3, 2]),
    }
    if mode == "ragged":
        data["offsets"] = torch.tensor([100, 200])
        reference = _output([[100, 101, 102], [200, 201, -1]])
        valid = _output([[102, 100, 101], [201, 200, -1]])
    else:
        data["row_to_batch"] = torch.tensor([1, 0])
        data["page_table_row_starts"] = torch.tensor([3, 1])
        reference = _output([[11, 12, 13], [1, 2, -1]])
        valid = _output([[13, 11, 12], [2, 1, -1]])
    compare(config, data, reference, valid)
    corrupt = {"indices": valid["indices"].clone()}
    corrupt["indices"][1, 0] += 1  # Row-local index 2 is outside the active length 2.
    with pytest.raises(AssertionError, match="outside the active row"):
        compare(config, data, reference, corrupt)


@pytest.mark.parametrize("compare", COMPARATORS)
def test_plain_padding_values_are_zero(compare):
    config = _config(length=2, k=3)
    data = {"scores": torch.tensor([[5.0, 4.0]])}
    reference = _output([[0, 1, -1]], [[5.0, 4.0, 0.0]])
    compare(config, data, reference, reference)
    corrupt = _output([[0, 1, -1]], [[5.0, 4.0, 3.0]])
    with pytest.raises(AssertionError):
        compare(config, data, reference, corrupt)


def test_fast_clusters_trivial_plain_checks_values():
    data = {"k": 3, "mode": "plain", "batch": 1, "seq_len": 2}
    reference = _output([[0, 1, -1]], [[5.0, 4.0, 0.0]])
    fast_topk_clusters.compare_outputs(data, reference, reference)
    corrupt = _output([[0, 1, -1]], [[5.0, float("nan"), 0.0]])
    with pytest.raises(AssertionError):
        fast_topk_clusters.compare_outputs(data, corrupt, reference)


@pytest.mark.parametrize("compare", COMPARATORS)
def test_zero_k_has_no_outputs_to_validate(compare):
    data = {"scores": torch.tensor([[5.0, 4.0, 3.0, 2.0]])}
    empty = _output([[]], [[]])
    compare(_config(k=0), data, empty, empty)
