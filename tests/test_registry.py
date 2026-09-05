# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from pathlib import Path
from types import SimpleNamespace

import pytest

from tirx_kernels import reference_requirements as refs
from tirx_kernels import registry


def _meta(**updates):
    meta = {"name": "kernel", "category": "basic", "runtime_cuda_archs": ["sm_100a"]}
    meta.update(updates)
    return meta


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ((), "must be list"),
        ([], "must not be empty"),
        (["sm_100a", 100], "entries must be strings"),
        (["SM_100a"], "canonical sm_* strings"),
        (["sm_100x"], "canonical sm_* strings"),
        (["sm_100a", "sm_100a"], "entries must be unique"),
    ],
)
def test_runtime_cuda_archs_validation(value, message):
    errors = registry._validate_meta(
        _meta(runtime_cuda_archs=value), category="basic", owner="test"
    )
    assert any(message in error for error in errors)


def test_runtime_cuda_archs_are_required():
    meta = _meta()
    meta.pop("runtime_cuda_archs")
    errors = registry._validate_meta(meta, category="basic", owner="test")
    assert "'runtime_cuda_archs' must be list" in errors


def test_exact_architectures_are_stored_in_source_index():
    index = registry.kernel_index(strict=True)
    assert index
    assert all(record.runtime_cuda_archs for record in index.values())
    assert index["bmm_fp8_rubin"].runtime_cuda_archs == ("sm_107a",)
    assert index["dense_blockscaled_gemm_sm107"].runtime_cuda_archs == ("sm_107a",)
    counts = {
        archs: sum(record.runtime_cuda_archs == archs for record in index.values())
        for archs in (("sm_100a",), ("sm_103a",), ("sm_107a",), ("sm_100a", "sm_103a", "sm_107a"))
    }
    assert counts == {
        ("sm_100a",): 11,
        ("sm_103a",): 6,
        ("sm_107a",): 4,
        ("sm_100a", "sm_103a", "sm_107a"): 90,
    }


def test_reference_requirements_are_stored_in_source_index():
    index = registry.kernel_index(strict=True)

    def packages(name):
        return tuple(item.package for item in index[name].reference_requirements)

    expected_by_category = {
        "cudnn": ("nvidia-cudnn-frontend", "nvidia-cutlass-dsl"),
        "deepep": ("deep-ep",),
        "deepgemm": ("deep-gemm",),
        "flashattention": ("flash-attn-4", "nvidia-cutlass-dsl"),
        "flashinfer": ("flashinfer-python", "nvidia-cutlass-dsl"),
        "msa": ("msa", "nvidia-cutlass-dsl", "quack-kernels"),
    }
    # Kernels whose upstream reference is not a CuTe-DSL program pin only the
    # source project (these routes load native CUDA through FlashInfer's own
    # tvm-ffi launcher).
    expected_by_kernel = {
        "blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103": ("flashinfer-python",),
        "blackwell_msa_decode_uniform_fp8_qkv_paged_sm100": ("flashinfer-python",),
        "blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100": ("flashinfer-python",),
        "blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103": ("flashinfer-python",),
        "blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103": ("flashinfer-python",),
        "cake_vsa_blk128_compact_sm100": ("flashinfer-python",),
        "cake_vsa_longseq_sm100": ("flashinfer-python",),
        "cake_vsa_longseq_sm103": ("flashinfer-python",),
        "cake_vsa_ultrasparse_bsr_sm100": ("flashinfer-python",),
        # The FP4 FA4 port quantizes its correctness inputs with FlashInfer.
        "flash_attention4_fp4": ("flash-attn-4", "nvidia-cutlass-dsl", "flashinfer-python"),
    }
    for name, record in index.items():
        if name in expected_by_kernel:
            assert packages(name) == expected_by_kernel[name]
        elif record.category in expected_by_category:
            assert packages(name) == expected_by_category[record.category]

    assert {
        name: packages(name) for name, record in index.items() if record.category == "basic"
    } == {
        "allgather_gemm": (),
        "fp16_bf16_gemm": (),
        "gemm_reduce_scatter": (),
        "nvfp4_gemm": ("flashinfer-python", "nvidia-cutlass-dsl"),
        "rmsnorm": ("flashinfer-python", "nvidia-cutlass-dsl"),
    }
    assert {
        name: packages(name) for name, record in index.items() if record.category == "flashmla"
    } == {
        "flash_mla_sparse_fwd": (),
        "sparse_flashmla_decode_head64": ("flash-mla",),
        "sparse_flashmla_prefill_head128_phase1": (),
        "sparse_flashmla_prefill_head128_small_topk_phase1": (),
        "sparse_flashmla_prefill_head64_phase1": (),
    }

    fla = index["agent_evolved_kda_forward_b1_t8192"].reference_requirements
    assert [(item.package, item.import_name) for item in fla] == [("flash-linear-attention", "fla")]
    assert fla[0].git.commit == "9c8e42e762fce087c27b673af4922795d9edb85e"

    msa = index["msa_sparse_atten_fwd_sm100"].reference_requirements
    assert [(item.package, item.specifier) for item in msa] == [
        ("msa", None),
        ("nvidia-cutlass-dsl", "==4.5.3"),
        ("quack-kernels", "==0.5.0"),
    ]
    assert index["cudnn_sm100_gdn_bprop_f16"].reference_requirements[1].specifier == (
        "==4.8.0.dev0"
    )
    assert index["flash_mla_sparse_fwd"].reference_requirements == ()


def test_runtime_cuda_archs_must_match_source_index(monkeypatch):
    record = registry.KernelRecord(
        name="kernel",
        category="basic",
        runtime_cuda_archs=("sm_100a",),
        reference_requirements=(),
        module_name="tirx_kernels.basic.kernel",
        source_path=Path(__file__),
    )
    module = SimpleNamespace(KERNEL_META=_meta(runtime_cuda_archs=["sm_103a"]))
    monkeypatch.setattr(registry.importlib, "import_module", lambda _name: module)

    with pytest.raises(ValueError, match="runtime runtime_cuda_archs"):
        registry._import_record(record, strict=True)


def test_reference_requirements_must_match_source_index(monkeypatch):
    requirement = refs.ReferenceRequirement(
        package="example", import_name="example", specifier="==1.0"
    )
    record = registry.KernelRecord(
        name="kernel",
        category="basic",
        runtime_cuda_archs=("sm_100a",),
        reference_requirements=(requirement,),
        module_name="tirx_kernels.basic.kernel",
        source_path=Path(__file__),
    )
    module = SimpleNamespace(KERNEL_META=_meta())
    monkeypatch.setattr(registry.importlib, "import_module", lambda _name: module)

    with pytest.raises(ValueError, match="runtime reference_requirements"):
        registry._import_record(record, strict=True)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "must be tuple"),
        ((), "must not be empty"),
        (({"package": "pkg", "import": "pkg"},), "at least one"),
        (({"package": "pkg", "specifier": "=1.0", "import": "pkg"},), "invalid"),
        (({"package": "pkg", "specifier": ">=1", "import": "bad-name"},), "canonical"),
        (
            (
                {
                    "package": "pkg",
                    "git": {"url": "https://example.test/repo.git", "commit": "abc"},
                    "import": "pkg",
                },
            ),
            "full lowercase Git SHA",
        ),
        (
            (
                {"package": "pkg", "specifier": ">=1", "import": "pkg"},
                {"package": "PKG", "specifier": "<2", "import": "pkg.other"},
            ),
            "duplicates",
        ),
    ],
)
def test_reference_requirements_validation(value, message):
    errors = registry._validate_meta(
        _meta(reference_requirements=value), category="basic", owner="test"
    )
    assert any(message in error for error in errors)


def test_discover_kernels_rejects_noncanonical_arch():
    with pytest.raises(ValueError, match="invalid exact CUDA architecture"):
        registry.discover_kernels(cuda_arch="sm100")


def test_probe_reference_requirement_reports_missing_import(monkeypatch):
    requirement = refs.ReferenceRequirement(package="missing", import_name="missing")
    refs.probe_reference_requirement.cache_clear()
    monkeypatch.setattr(refs.importlib.util, "find_spec", lambda _name: None)
    assert "is unavailable" in refs.probe_reference_requirement(requirement)


def test_probe_reference_requirement_checks_distribution_version(monkeypatch):
    requirement = refs.ReferenceRequirement(
        package="example", import_name="example", specifier=">=4.5.3,<4.6"
    )
    refs.probe_reference_requirement.cache_clear()
    monkeypatch.setattr(refs.importlib.util, "find_spec", lambda _name: SimpleNamespace())
    monkeypatch.setattr(refs.importlib.metadata, "version", lambda _name: "4.8.0.dev0")
    assert "installed version is 4.8.0.dev0" in refs.probe_reference_requirement(requirement)


def test_probe_reference_requirement_checks_git_identity(monkeypatch):
    requirement = refs.ReferenceRequirement(
        package="example",
        import_name="example",
        git=refs.GitRequirement(url="https://github.com/example/project.git", commit="1" * 40),
    )
    refs.probe_reference_requirement.cache_clear()
    monkeypatch.setattr(refs.importlib.util, "find_spec", lambda _name: SimpleNamespace())
    monkeypatch.setattr(
        refs,
        "_git_identity",
        lambda _requirement, _module_spec: ("git@github.com:example/project.git", "1" * 40),
    )
    assert refs.probe_reference_requirement(requirement) is None
