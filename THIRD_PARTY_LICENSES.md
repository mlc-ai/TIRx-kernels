# Third-Party Licenses

Except where otherwise noted, this repository is licensed under the Apache
License, Version 2.0. It also contains TIRx ports of kernels from the upstream
projects listed below. Each ported file identifies the upstream source and
applicable license; the corresponding copyright notices and license texts are
collected here.

## DeepGEMM — MIT License

- Upstream: <https://github.com/deepseek-ai/DeepGEMM>
- Applies to:
  - `tirx_kernels/deepgemm/mqa_logits_fp4.py`
  - `tirx_kernels/deepgemm/mqa_logits_fp8.py`
  - `tirx_kernels/deepgemm/paged_mqa_logits_fp4.py`
  - `tirx_kernels/deepgemm/paged_mqa_logits_fp8.py`
  - `tirx_kernels/deepgemm/tf32_hc_prenorm_gemm.py`
  - `tirx_kernels/deepgemm/mega_moe.py`

  `tirx_kernels/deepgemm/fp8_blockwise_gemm.py` and
  `tirx_kernels/deepgemm/grouped_fp8_gemm_contiguous.py` are native TIRx
  implementations that only use DeepGEMM as a benchmark reference; they
  contain no DeepGEMM code.

```text
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## FlashMLA — MIT License

- Upstream: <https://github.com/deepseek-ai/FlashMLA>
- Applies to: `tirx_kernels/flashmla/` (kernel files and the shared
  `_gemm.py` / `_mask.py` / `_tma.py` helpers; the `_flashmla_bench.py` and
  `_trtllm_gen_bench.py` benchmarking harnesses are native code)

```text
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## flash-attention — BSD 3-Clause License

- Upstream: <https://github.com/Dao-AILab/flash-attention>
- Applies to:
  - `tirx_kernels/flashattention/flash_attention4.py`, ported from
    `flash_attn/cute/flash_fwd_sm100.py`
  - `tirx_kernels/flashattention/flash_attention_backward.py`, which retains
    its file-specific copyright and BSD 3-Clause notice in full
  - `tirx_kernels/flashattention/flash_attention_backward_sm100_sketch.md`,
    which documents the backward port's schedule

  `tirx_kernels/flashattention/__init__.py` is native repository code.

```text
BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The FlashAttention backward port additionally retains this upstream
file-specific notice:

```text
Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
All rights reserved.
```

## FlashInfer — Apache License 2.0 and BSD 3-Clause License

- Upstream: <https://github.com/flashinfer-ai/flashinfer>
- Applies to:
  - `tirx_kernels/flashinfer/tinygemm2_sm100.py`, ported from
    `csrc/tinygemm2_sm100.cu`; that upstream generated source records its Loom
    schedules as exact ports of NVIDIA TensorRT-LLM's TinyGEMM2 kernel
  - `tirx_kernels/flashinfer/bf16_fused_m128.py` and
    `tirx_kernels/flashinfer/bf16_fused_m128_tx_tile.py`, ported from
    `csrc/kda/flashkda_bf16_fused_m128.cu` in
    flashinfer-ai/flashinfer#4262
  - `tirx_kernels/flashinfer/gdn_prefill_sm100.py`, ported from
    `flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py`; the
    Apache-licensed `gdn_prefill.py` adapter is used only as an external frozen
    correctness and benchmark oracle
  - `.agents/sketch/flashkda_bf16_m128.md`,
    `.agents/sketch/gdn_prefill_sm100.md`, and
    `.agents/sketch/tinygemm2_sm100.md`, which document the corresponding
    ports under the same applicable terms

  `tirx_kernels/flashinfer/__init__.py` and
  `tirx_kernels/flashinfer/utils/_flashkda_bench.py` are native repository code.

### Apache License, Version 2.0 portions

The TinyGEMM2 and FlashKDA source kernels carry this notice:

```text
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
```

These upstream portions are licensed under the Apache License, Version 2.0.
The license text is reproduced in this repository's [LICENSE](LICENSE). The
ported files are modified versions of the upstream originals, and the relevant
attributions from FlashInfer's NOTICE are reproduced in [NOTICE](NOTICE).

### BSD 3-Clause portion

The GDN kernel body is ported from FlashInfer's
`gated_delta_net_chunked.py`, which carries the following license:

```text
BSD 3-Clause License

Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
