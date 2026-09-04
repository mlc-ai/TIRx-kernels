# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""CPU regressions for frozen FlashKDA source routing and numerical gates."""

import hashlib
import importlib
import sys
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace

import pytest
import torch

from tirx_kernels.flashinfer.kda import _source

DECODE_MODULES = (
    "flashkda_decode_t1_precomputed",
    "flashkda_decode_t2_precomputed",
    "flashkda_decode_t3_lower_bound",
    "flashkda_decode_t4_precomputed",
    "flashkda_decode_t5_gram",
    "flashkda_decode_t6_gram",
)
VARIANTS = tuple(
    name.removeprefix("flashkda_decode_").removesuffix(".cu")
    for name in _source._SOURCE_SHA256
    if name.startswith("flashkda_decode_d128_")
)


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("source adapter CPU test attempted to initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    cached = (_source._thor_source_spec, _source._thor_source_module, _source._thor_prefill_runner)
    for function in cached:
        function.cache_clear()
    yield
    for function in cached:
        function.cache_clear()


@pytest.fixture
def frozen_modules(tmp_path, monkeypatch):
    """Small source fixtures exercise staging without any optional dependency."""
    csrc = tmp_path / "csrc" / "kda"
    csrc.mkdir(parents=True)
    generated = tmp_path / "generated"
    modules = {}
    for name in (
        "flashinfer",
        "flashinfer.jit",
        "flashinfer.jit.env",
        "flashinfer.jit.core",
        "flashinfer.jit.utils",
        "flashinfer.jit.flash_kda",
        "flashinfer.jit.flash_kda_decode",
        "flashinfer.kda_prefill",
    ):
        module = ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    original = {}
    guards = {
        "flashkda_binding_common.cuh": "CheckFlashKDATarget",
        "flashkda_decode_binding_common.cuh": "CheckFlashKDADecodeTarget",
    }
    for name in _source._SOURCE_SHA256:
        text = "// frozen source\n"
        if name in guards:
            text += f"inline void {guards[name]}(int32_t device_id) {{\n"
            text += '  if (true) { CheckCuda(0, "unchanged input validation elsewhere"); }\n}\n'
        text += "__global__ void body() { /* instruction order must stay intact */ }\n"
        (csrc / name).write_text(text)
        original[name] = text
    monkeypatch.setattr(_source, "_SOURCE_SHA256", {k: _digest(v) for k, v in original.items()})
    modules["flashinfer.jit.env"].FLASHINFER_GEN_SRC_DIR = generated

    def write_if_different(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != text:
            path.write_text(text)

    modules["flashinfer.jit.utils"].write_if_different = write_if_different
    core = modules["flashinfer.jit.core"]
    core.sm110a_nvcc_flags = ["-gencode=arch=compute_110a,code=sm_110a", "-DSOURCE_FLAG"]
    calls = []

    def gen_jit_spec(name, sources, **kwargs):
        spec = SimpleNamespace(name=name, sources=sources, **kwargs)
        calls.append(spec)
        return spec

    core.gen_jit_spec = gen_jit_spec
    prefill_jit = modules["flashinfer.jit.flash_kda"]
    prefill_jit._get_flash_kda_csrc_dir = lambda: csrc
    prefill_jit._get_flash_kda_include_dir = lambda: tmp_path / "include"
    decode = modules["flashinfer.jit.flash_kda_decode"]
    decode.__file__ = str(tmp_path / "flash_kda_decode.py")
    Path(decode.__file__).write_text("frozen decode generator")
    monkeypatch.setattr(_source, "_DECODE_JIT_SHA256", _digest("frozen decode generator"))
    decode._get_csrc_dir = lambda: csrc
    decode._get_include_dir = lambda: tmp_path / "include"
    decode.FLASH_KDA_DECODE_VARIANT_METADATA = {variant: variant for variant in VARIANTS}
    decode._get_binding_cu = lambda variant, metadata: f"// {variant}, {metadata}\n"
    prefill = modules["flashinfer.kda_prefill"]
    prefill.__file__ = str(tmp_path / "kda_prefill.py")
    Path(prefill.__file__).write_text("frozen prefill wrapper")
    monkeypatch.setattr(_source, "_PREFILL_SHA256", _digest("frozen prefill wrapper"))
    return SimpleNamespace(
        csrc=csrc, generated=generated, original=original, modules=modules, calls=calls
    )


@pytest.mark.parametrize("variant", ("m128", *VARIANTS))
def test_source_spec_preserves_bodies_and_flags(frozen_modules, variant):
    fixtures = frozen_modules
    spec = _source._thor_source_spec(variant)
    generated = fixtures.generated / spec.name
    assert "sm110a" in spec.name
    assert spec.extra_cuda_cflags[:2] == [
        "-gencode=arch=compute_110a,code=sm_110a",
        "-DSOURCE_FLAG",
    ]
    assert spec.extra_include_paths == [
        generated,
        fixtures.csrc.parent,
        fixtures.csrc.parents[1] / "include",
    ]
    if variant == "m128":
        assert spec.extra_cuda_cflags[2:] == ["-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100"]
        assert spec.sources == [generated / "flashkda_bf16_fused_m128_binding.cu"]
    else:
        assert spec.extra_cuda_cflags[2:] == [
            "-DFLASHINFER_FLASH_KDA_DECODE_TARGET_KIND=100",
            "--maxrregcount=128",
        ]
        assert spec.sources == [generated / "flashkda_decode_binding.cu"]
    for name, original in fixtures.original.items():
        assert (fixtures.csrc / name).read_text() == original
        staged = generated / name
        if not staged.exists():
            continue
        if name.endswith("binding_common.cuh"):
            result = staged.read_text()
            assert "major == 11 && minor == 0" in result
            assert result.split("__global__", 1)[1] == original.split("__global__", 1)[1]
            assert result.startswith("// frozen source\n")
        else:
            assert staged.read_text() == original
    assert _source._thor_source_spec(variant) is spec
    assert len(fixtures.calls) == 1


@pytest.mark.parametrize(
    "name",
    (
        "flashkda_bf16_fused_m128.cu",
        "flashkda_binding_common.cuh",
        "flashkda_bf16_fused_m128_binding.cu",
    ),
)
def test_source_drift_fails_before_staging_or_build(frozen_modules, name):
    path = frozen_modules.csrc / name
    path.write_text(path.read_text() + "// drift")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        _source._thor_source_spec("m128")
    assert not frozen_modules.generated.exists()
    assert not frozen_modules.calls


def test_decode_generator_drift_fails_before_staging(frozen_modules):
    path = Path(frozen_modules.modules["flashinfer.jit.flash_kda_decode"].__file__)
    path.write_text("changed geometry")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        _source._thor_source_spec(VARIANTS[0])
    assert not frozen_modules.generated.exists()


@pytest.mark.parametrize("target", ("sm100a", "sm100f", "sm103a"))
def test_other_targets_delegate_unchanged(frozen_modules, target):
    calls = []
    loader = frozen_modules.modules["flashinfer.jit.flash_kda_decode"]
    loader.get_flash_kda_decode_module = lambda *args: calls.append(args) or "upstream"
    assert _source.get_decode_module("original_variant", target) == "upstream"
    assert calls == [("original_variant", target)]
    assert not frozen_modules.calls
    assert not frozen_modules.generated.exists()


def test_thor_loader_caches_module(monkeypatch):
    calls = []
    module = object()
    spec = SimpleNamespace(build_and_load=lambda: calls.append("build") or module)
    monkeypatch.setattr(_source, "_thor_source_spec", lambda variant: spec)
    assert _source.get_decode_module(VARIANTS[0], "sm110a") is module
    assert _source.get_decode_module(VARIANTS[0], "sm110a") is module
    assert calls == ["build"]


@pytest.mark.parametrize("variant", ("m64", "d128_t2_precomputed_split8", "../m128"))
def test_unproven_variants_are_rejected(frozen_modules, variant):
    with pytest.raises(ValueError, match="unsupported Thor"):
        _source._thor_source_spec(variant)
    assert not frozen_modules.calls


def test_prefill_clone_changes_only_two_globals(frozen_modules):
    prefill = frozen_modules.modules["flashinfer.kda_prefill"]

    def template(*, value=3):
        return (
            globals()["_select_flash_kda_prefill_target"],
            globals()["_get_flash_kda_prefill_module"],
            value,
        )

    namespace = {
        "_select_flash_kda_prefill_target": object(),
        "_get_flash_kda_prefill_module": object(),
        "shared_workspace": object(),
    }
    original = FunctionType(template.__code__, namespace)
    original.__kwdefaults__ = template.__kwdefaults__
    prefill._run_flash_kda_prefill = original
    adapted = _source._thor_prefill_runner()
    assert adapted.__code__ is original.__code__
    assert adapted.__kwdefaults__ == original.__kwdefaults__
    assert adapted.__globals__ is not namespace
    assert prefill._run_flash_kda_prefill is original
    assert adapted() == (_source._thor_prefill_target, _source._thor_prefill_module, 3)
    assert original() == (
        namespace["_select_flash_kda_prefill_target"],
        namespace["_get_flash_kda_prefill_module"],
        3,
    )
    assert {k for k in namespace if adapted.__globals__[k] is not namespace[k]} == {
        "_select_flash_kda_prefill_target",
        "_get_flash_kda_prefill_module",
    }


def test_prefill_wrapper_drift_is_rejected(frozen_modules):
    prefill = frozen_modules.modules["flashinfer.kda_prefill"]
    Path(prefill.__file__).write_text("changed wrapper")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        _source._thor_prefill_runner()


@pytest.mark.parametrize("capability", ((10, 0), (10, 3), (11, 1), (12, 0)))
def test_prefill_thor_route_rejects_other_devices(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    with pytest.raises(RuntimeError, match=r"requires compute capability 11\.0"):
        _source._thor_prefill_target("cpu")


def test_prefill_thor_module_requires_exact_variant_and_target(monkeypatch):
    calls = []
    monkeypatch.setattr(_source, "_thor_source_module", lambda v: calls.append(v) or "m128")
    assert _source._thor_prefill_module("m128", "sm110a") == "m128"
    for args in [("m64", "sm110a"), ("m128", "sm100f")]:
        with pytest.raises(ValueError, match="only supports m128/sm110a"):
            _source._thor_prefill_module(*args)
    assert calls == ["m128"]


@pytest.mark.parametrize(
    "bad_flag", (None, "use_qk_l2norm_in_kernel", "use_gate_in_kernel", "beta_is_logit")
)
def test_prefill_forwards_identical_arguments(monkeypatch, bad_flag):
    kwargs = {
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "beta_is_logit": True,
        "q": object(),
        "initial_state": object(),
        "cu_seqlens": object(),
        "lower_bound": -5.0,
    }
    forwarded = []
    monkeypatch.setattr(_source, "_thor_prefill_runner", lambda: lambda **kw: forwarded.append(kw))
    if bad_flag:
        kwargs[bad_flag] = False
        with pytest.raises(ValueError, match=f"{bad_flag}=True"):
            _source.recurrent_kda_m128(**kwargs)
        assert not forwarded
    else:
        _source.recurrent_kda_m128(**kwargs)
        assert forwarded == [
            {
                "q": kwargs["q"],
                "initial_state": kwargs["initial_state"],
                "cu_seqlens": kwargs["cu_seqlens"],
                "lower_bound": -5.0,
                "output": None,
                "prefill_workspace": None,
            }
        ]
    assert len(kwargs) == 7


def _decode_case():
    return {
        "device": "cpu",
        "spec": {"VALUE_SPLIT": 4},
        **{
            key: torch.zeros(1)
            for key in (
                "q",
                "k",
                "v",
                "g",
                "beta",
                "a_log",
                "dt_bias",
                "reference_state",
                "tirx_out",
                "tirx_state_raw",
                "reference_state_raw",
                "cu_seqlens",
                "ssm_state_indices",
                "num_accepted_tokens",
            )
        },
        "scale": 0.125,
        "lower_bound": -5.0,
    }


@pytest.mark.parametrize("module_name", DECODE_MODULES)
@pytest.mark.parametrize(
    ("capability", "cuda_version", "target"),
    (
        ((11, 0), "13.1", "sm110a"),
        ((10, 0), "12.8", "sm100a"),
        ((10, 0), "12.9", "sm100f"),
        ((10, 3), "13.1", "sm100f"),
    ),
)
def test_decode_reference_dispatch_preserves_abi(
    monkeypatch, module_name, capability, cuda_version, target
):
    module = importlib.import_module(f"tirx_kernels.flashinfer.kda.{module_name}")
    case = _decode_case()
    if module_name == DECODE_MODULES[0]:
        case["spec"]["VALUE_SPLIT"] = 16
        if capability == (10, 3):
            target = "sm103a"
    calls = []
    arguments = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=123)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(
        _source,
        "get_decode_module",
        lambda *args: (
            calls.append(args) or SimpleNamespace(run=lambda *args: arguments.append(args))
        ),
    )
    out = module._flashinfer_reference(case)
    assert calls[0][1] == target
    args = arguments[0]
    assert args[0:5] == tuple(case[key] for key in ("q", "k", "v", "g", "beta"))
    assert args[7] is case["reference_state"]
    assert args[8] is out
    assert args[9:12] == tuple(
        case[key] for key in ("cu_seqlens", "ssm_state_indices", "num_accepted_tokens")
    )
    assert args[12] == 0.125
    assert args[14] == 123
    if "lower_bound" in module_name:
        assert args[5] is case["a_log"] and args[6] is case["dt_bias"]
        assert args[13] == -5.0
    else:
        assert torch.equal(args[5], torch.ones(1)) and args[5] is args[6]
        assert args[13] == 0.0


@pytest.mark.parametrize("module_name", DECODE_MODULES)
@pytest.mark.parametrize("bad_buffer", ("output", "state"))
def test_decode_correctness_keeps_original_source_tolerances(monkeypatch, module_name, bad_buffer):
    from tirx_kernels import runner

    module = importlib.import_module(f"tirx_kernels.flashinfer.kda.{module_name}")
    case = _decode_case()
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    monkeypatch.setattr(module, "prepare_data", lambda **kwargs: case)
    monkeypatch.setattr(module, "get_kernel", lambda **kwargs: None)
    monkeypatch.setattr(module, "_tirx_args", lambda case: ())
    monkeypatch.setattr(runner, "compile_kernel", lambda kernel: lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    source_calls = []

    def reference(case):
        source_calls.append("source")
        if bad_buffer == "state":
            case["reference_state_raw"].fill_(0.004)
            return case["tirx_out"].clone()
        return torch.full_like(case["tirx_out"], 0.004)

    monkeypatch.setattr(module, "_flashinfer_reference", reference)
    # 0.004 would pass the old Thor mathematical floor (atol=2**-7), but
    # violates every original source tolerance near zero, including full state.
    with pytest.raises(AssertionError, match="vs flashinfer cake export"):
        module.run_test()
    assert source_calls == ["source"]


@pytest.mark.parametrize("module_name", DECODE_MODULES)
def test_decode_benchmark_retains_primary_source_on_thor(monkeypatch, module_name):
    from tirx_kernels import runner

    module = importlib.import_module(f"tirx_kernels.flashinfer.kda.{module_name}")
    case = _decode_case()
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    monkeypatch.setattr(module, "prepare_data", lambda **kwargs: case)
    monkeypatch.setattr(module, "_tirx_args", lambda case: ())
    monkeypatch.setattr(runner, "bench", lambda funcs, **kwargs: kwargs)
    result = module.run_gpu({"config": {}, "executable": lambda: None})
    assert tuple(result["references"]) == ("flashinfer_cake",)


def test_m128_benchmark_restores_both_original_sources(monkeypatch):
    from tirx_kernels import runner
    from tirx_kernels.flashinfer.kda import bf16_fused_m128 as module

    case = {"config": SimpleNamespace(use_initial_state=False), "dispatch_reason": "m128: domain"}
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    monkeypatch.setattr(module, "prepare_data", lambda **kwargs: case)
    monkeypatch.setattr(module, "_tirx_args", lambda case: ())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runner, "bench", lambda funcs, **kwargs: kwargs)
    result = module.run_gpu({"config": {}, "executable": lambda: None})
    assert tuple(result["references"]) == ("flashinfer_m128", "flashkda_raw")


@pytest.mark.parametrize("bad_buffer", ("output", "state", "missing_state"))
def test_m128_source_comparison_remains_required_with_mathematical_oracle(monkeypatch, bad_buffer):
    from tirx_kernels import runner
    from tirx_kernels.flashinfer.kda import bf16_fused_m128 as module

    case = {
        "config": SimpleNamespace(store_final_state=True, validate=lambda: None),
        "dispatch_reason": "m128: domain",
        "out": torch.zeros(1),
        "final_state": torch.zeros(1),
    }
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    monkeypatch.setattr(module, "prepare_data", lambda **kwargs: case)
    monkeypatch.setattr(module, "bf16_fused_m128", lambda **kwargs: None)
    monkeypatch.setattr(module, "_tirx_args", lambda case: ())
    monkeypatch.setattr(runner, "compile_kernel", lambda kernel: lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    calls = []
    monkeypatch.setattr(
        module,
        "_mathematical_reference",
        lambda case: calls.append("math") or (torch.zeros(1), torch.zeros(1)),
    )
    bad_out = torch.ones(1) if bad_buffer == "output" else torch.zeros(1)
    bad_state = (
        None if bad_buffer == "missing_state" else torch.full((1,), float(bad_buffer == "state"))
    )
    monkeypatch.setattr(
        module,
        "_flashinfer_cuda_reference",
        lambda case: calls.append("source") or (bad_out, bad_state),
    )
    with pytest.raises(AssertionError):
        module.run_test()
    assert calls == ["math", "source"]
