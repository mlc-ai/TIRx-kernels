# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f),
# Copyright (c) 2026 by FlashInfer team and NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Thor host adapters for the frozen FlashKDA source baselines.

Sources: ``flashinfer/jit/flash_kda{,_decode}.py``, ``flashinfer/kda_prefill.py``
and ``csrc/kda/flashkda_{bf16_fused_m128*,decode*,binding_common.cuh}`` at the
commit cited above. Only the two C++ host architecture checks and the Python
prefill target/module dispatch are retargeted. Device bodies, launch geometry,
compile options other than the architecture, and upstream packages stay intact.
Generated sources use a separate FlashInfer JIT key; no private library path or
global monkeypatch is used. Missing or changed frozen sources fail explicitly.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from types import FunctionType

_SOURCE_SHA256 = {
    "flashkda_binding_common.cuh": "cb6321b876de76af467752d3a14162a9ac636ec08c2156d2f81be3a230747440",
    "flashkda_bf16_fused_m128_binding.cu": "8e7a383131d5a7ca94982efee625a1f2d5d8170c7305b8bd74770270a2eb8ea4",
    "flashkda_bf16_fused_m128.cu": "9358199b5eb1d7ff9d8c5090deb0d7e90a2a5e2d11f470824daa2aa10fa3d682",
    "flashkda_decode_binding_common.cuh": "9cb9b5dc9fbf25a080b881bcc578848623a7e32b07c801a96e8545b38337c1d2",
    "flashkda_decode_binding.cuh": "90ecc9e5d2a9d01da992a1ed7e983f99a3f73ecf61af29be494caa50d3c51c0b",
    "flashkda_decode_binding_direct_impl.cuh": "c6e3adfb0e4310d180666feb2dd2995781e0acf8007ef0305ee087c9b149eff7",
    "flashkda_decode_binding_impl.cuh": "c0f6ebb0fd67e08d441d56730882a0189749614828f1915779f4224656d47d88",
    "flashkda_decode_d128_t1_precomputed_direct_split16.cu": "85f78721ecaef7132ee7b91b1e13ec29ef44fb861b0c1d502959029b5c80be81",
    "flashkda_decode_d128_t1_precomputed_direct_split8.cu": "31593b441c7ab3d330cc94493e8b87d34278e4f3e637486370591711a0b9de7e",
    "flashkda_decode_d128_t2_precomputed_split4.cu": "122b4071a5f32c291aaeb075a8e6093e93826689c5306d98a8b0b8140f6a8f11",
    "flashkda_decode_d128_t3_lower_bound_split4.cu": "dc5d675d9f3d8e459ea7bffe93e2b29ea39f29414d745c8616ad6b1acbe651f3",
    "flashkda_decode_d128_t4_precomputed_split2.cu": "7a0c481d397aea2f6cb60cff14b9d674fcbabfda7f962d60e6bcb22fc71e33f8",
    "flashkda_decode_d128_t5_precomputed_gram_split1.cu": "bc7172f4e28617b6fcee3518bc3fe0ca6f68a411c1b13a07bf773eca2e00fc79",
    "flashkda_decode_d128_t5_precomputed_gram_split2.cu": "662dc79b3230fddedbb6af323d24e834b249634770dfe60f5d3e899a72eaec59",
    "flashkda_decode_d128_t5_precomputed_gram_split4.cu": "adaa583edc1839c37a588e50a3c61440b485b2e72edc6b6072f367ca4df6c4df",
    "flashkda_decode_d128_t5_precomputed_gram_split8.cu": "75471f472a0b1322f0fc19a3d3e9ac26d5fdf49f7fd3a79b78488f7cc14b6a98",
    "flashkda_decode_d128_t6_precomputed_gram_split1.cu": "7039efdb20d745bfd64d1a59564b4ad1c27f052cf1e498fafb505e659f013bb1",
    "flashkda_decode_d128_t6_precomputed_gram_split2.cu": "af1aab95191cfa2a0b5124e4313351bbebbe3fdcc0df056f8d7b3c8b58a9998f",
    "flashkda_decode_d128_t6_precomputed_gram_split4.cu": "9789755bd2939201f82e9767a47eccc606524547eed2578bd2b1499d1b1ce1b6",
    "flashkda_decode_d128_t6_precomputed_gram_split8.cu": "5a0ff3fba48933ed92861f730a51ff9bfe93697cb2998c3d203059dcc858e846",
}

_DECODE_JIT_SHA256 = "db769e310b46e0a3434c6b8f83bef7d5adac22fd5d4b550836fd0ba0266f6a21"
_PREFILL_SHA256 = "f998258cef67c297e6ee209bdbbcdd210b771fe058c98968de7c5803d6ac4945"


def _checked_source(path: Path, expected: str) -> str:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"frozen FlashKDA source hash mismatch: {path}: {actual} != {expected}")
    return data.decode("utf-8")


def _retarget_host_guard(source: str, function_name: str) -> str:
    """Replace only a checked host function, preserving all other source bytes."""
    signature = f"inline void {function_name}(int32_t device_id) {{"
    if source.count(signature) != 1:
        raise RuntimeError(f"expected exactly one frozen host guard: {function_name}")
    start = source.index(signature)
    opening = start + len(signature) - 1
    depth = 1
    end = opening + 1
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if depth:
        raise RuntimeError(f"unclosed frozen host guard: {function_name}")
    replacement = (
        signature
        + """
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 11 && minor == 0)
      << "this frozen FlashKDA source module requires compute capability 11.0, got "
      << major << "." << minor;
}"""
    )
    return source[:start] + replacement + source[end:]


def _stage_sources(csrc: Path, generated: Path, names: tuple[str, ...]) -> None:
    from flashinfer.jit.utils import write_if_different

    guards = {
        "flashkda_binding_common.cuh": "CheckFlashKDATarget",
        "flashkda_decode_binding_common.cuh": "CheckFlashKDADecodeTarget",
    }
    # Verify every input before writing, including unchanged device bodies.
    sources = {name: _checked_source(csrc / name, _SOURCE_SHA256[name]) for name in names}
    for name, source in sources.items():
        if name in guards:
            source = _retarget_host_guard(source, guards[name])
        write_if_different(generated / name, source)


@functools.cache
def _thor_source_spec(variant: str):
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec, sm110a_nvcc_flags

    if variant == "m128":
        from flashinfer.jit.flash_kda import _get_flash_kda_csrc_dir, _get_flash_kda_include_dir

        csrc = _get_flash_kda_csrc_dir()
        include = _get_flash_kda_include_dir()
        name = "tirx_source_flashkda_m128_sm110a_v1"
        generated = jit_env.FLASHINFER_GEN_SRC_DIR / name
        binding_name = "flashkda_bf16_fused_m128_binding.cu"
        _stage_sources(
            csrc,
            generated,
            (binding_name, "flashkda_bf16_fused_m128.cu", "flashkda_binding_common.cuh"),
        )
        flags = [*sm110a_nvcc_flags, "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100"]
    else:
        from flashinfer.jit import flash_kda_decode as decode
        from flashinfer.jit.utils import write_if_different

        body_name = f"flashkda_decode_{variant}.cu"
        if body_name not in _SOURCE_SHA256:
            raise ValueError(f"unsupported Thor FlashKDA source variant: {variant}")
        _checked_source(Path(decode.__file__), _DECODE_JIT_SHA256)
        csrc = decode._get_csrc_dir()
        include = decode._get_include_dir()
        name = f"tirx_source_flashkda_decode_{variant}_sm110a_v1"
        generated = jit_env.FLASHINFER_GEN_SRC_DIR / name
        _stage_sources(
            csrc,
            generated,
            (
                body_name,
                "flashkda_decode_binding.cuh",
                "flashkda_decode_binding_common.cuh",
                "flashkda_decode_binding_direct_impl.cuh",
                "flashkda_decode_binding_impl.cuh",
            ),
        )
        binding_name = "flashkda_decode_binding.cu"
        metadata = decode.FLASH_KDA_DECODE_VARIANT_METADATA[variant]
        write_if_different(generated / binding_name, decode._get_binding_cu(variant, metadata))
        flags = [
            *sm110a_nvcc_flags,
            "-DFLASHINFER_FLASH_KDA_DECODE_TARGET_KIND=100",
            "--maxrregcount=128",
        ]
    return gen_jit_spec(
        name,
        [generated / binding_name],
        extra_cuda_cflags=flags,
        extra_include_paths=[generated, csrc.parent, include],
    )


@functools.cache
def _thor_source_module(variant: str):
    return _thor_source_spec(variant).build_and_load()


def get_decode_module(variant: str, target: str):
    """Use the unchanged upstream loader for every existing target."""
    if target == "sm110a":
        return _thor_source_module(variant)
    from flashinfer.jit.flash_kda_decode import get_flash_kda_decode_module

    return get_flash_kda_decode_module(variant, target)


def _thor_prefill_target(device) -> str:
    import torch

    capability = torch.cuda.get_device_capability(device)
    if capability != (11, 0):
        raise RuntimeError(
            f"Thor FlashKDA source adapter requires compute capability 11.0: {capability}"
        )
    return "sm110a"


def _thor_prefill_module(variant: str, target: str):
    if (variant, target) != ("m128", "sm110a"):
        raise ValueError(f"Thor prefill adapter only supports m128/sm110a: {variant}/{target}")
    return _thor_source_module("m128")


@functools.cache
def _thor_prefill_runner():
    from flashinfer import kda_prefill

    _checked_source(Path(kda_prefill.__file__), _PREFILL_SHA256)
    original = kda_prefill._run_flash_kda_prefill
    # A private globals mapping changes only dispatch for this function. Other
    # users/threads keep upstream's original function and module globals.
    namespace = dict(original.__globals__)
    namespace.update(
        _select_flash_kda_prefill_target=_thor_prefill_target,
        _get_flash_kda_prefill_module=_thor_prefill_module,
    )
    adapted = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    adapted.__kwdefaults__ = original.__kwdefaults__
    return adapted


def recurrent_kda_m128(**kwargs):
    """Run the frozen prefill wrapper for the port's exact M128 dispatch domain."""
    # These public-dispatch flags are intrinsic to this frozen fused body. Fail
    # rather than silently accepting a call meant for another recurrent kernel.
    for flag in ("use_qk_l2norm_in_kernel", "use_gate_in_kernel", "beta_is_logit"):
        if kwargs.pop(flag) is not True:
            raise ValueError(f"Thor FlashKDA M128 source requires {flag}=True")
    return _thor_prefill_runner()(**kwargs, output=None, prefill_workspace=None)
