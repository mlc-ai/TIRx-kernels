#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Check that every tracked Python file carries an acceptable license header.

Headers are SPDX tags, in the style of vLLM's. Two shapes are accepted:

* the TIRx header alone, for code with no upstream lineage::

      # SPDX-License-Identifier: Apache-2.0
      # SPDX-FileCopyrightText: Copyright TIRx authors

* a port header, for files ported from an upstream project: a citation naming
  the project, its URL and the exact upstream commit, the upstream copyright
  notice, and an SPDX expression covering the upstream license as well as ours::

      # This file is a TIRx port of code from DeepGEMM
      # (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
      # SPDX-License-Identifier: Apache-2.0 AND MIT
      # SPDX-FileCopyrightText: Copyright TIRx authors

Ports must use the second shape, and the expected project name, URL and SPDX
expression are bound to the path: a DeepGEMM port tagged plain ``Apache-2.0``
(or ``AND BSD-3-Clause``) is an error, not a stylistic choice — the expression
states which terms apply to the file, and understating or misstating them is
exactly the drift this check exists to prevent. Unlike ASF-style header checks,
upstream ``Copyright`` lines are required rather than forbidden.

Full license texts live in ``licenses/``; ``LICENSE`` maps each bucket of the
tree to its license. Where an upstream license requires the conditions text
itself to travel with the source (BSD-3), that text must stay in the file and
its absence is an error.

Run ``--fix`` to insert the TIRx header into files that have no header at all;
port headers are never synthesized, since only a human knows the upstream terms.
Run ``--self-test`` to prove the checker is not vacuous: it feeds itself a set
of known violations and fails unless every one is caught.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SELF = "tests/lint/check_license_headers.py"

SPDX_ID = "# SPDX-License-Identifier: "
SPDX_COPYRIGHT = "# SPDX-FileCopyrightText: Copyright TIRx authors"
TIRX_HEADER = f"{SPDX_ID}Apache-2.0\n{SPDX_COPYRIGHT}"

PORT_LINE = "# This file is a TIRx port of code from "
CITATION_RE = re.compile(r"\((https://\S+) @ [0-9a-f]{7,40}\)")

# Path-bound expectations: project name, upstream URL, and the SPDX expression
# a port under this bucket must carry.
PORT_BUCKETS = {
    "tirx_kernels/cudnn/": (
        "cuDNN Frontend",
        "https://github.com/NVIDIA/cudnn-frontend",
        "Apache-2.0",
    ),
    "tirx_kernels/deepgemm/": (
        "DeepGEMM",
        "https://github.com/deepseek-ai/DeepGEMM",
        "Apache-2.0 AND MIT",
    ),
    "tirx_kernels/deepep/": (
        "DeepEP",
        "https://github.com/deepseek-ai/DeepEP",
        "Apache-2.0 AND MIT",
    ),
    "tirx_kernels/flashmla/": (
        "FlashMLA",
        "https://github.com/deepseek-ai/FlashMLA",
        "Apache-2.0 AND MIT",
    ),
    "tirx_kernels/flashattention/": (
        "flash-attention",
        "https://github.com/Dao-AILab/flash-attention",
        "Apache-2.0 AND BSD-3-Clause",
    ),
    "tirx_kernels/flashinfer/": (
        "FlashInfer",
        "https://github.com/flashinfer-ai/flashinfer",
        "Apache-2.0",
    ),
    "tirx_kernels/fastcu/": (
        "fast.cu",
        "https://github.com/pranjalssh/fast.cu",
        "Apache-2.0 AND MIT",
    ),
    "tirx_kernels/msa/": ("MSA", "https://github.com/MiniMax-AI/MSA", "Apache-2.0 AND MIT"),
}

# Files whose terms differ from their bucket, with header text the upstream
# license requires to stay in the file verbatim (BSD-3 clause 1: the copyright
# notice, the conditions list and the disclaimer travel with the source).
FILE_OVERRIDES = {
    # Port of hao-ai-lab/flash-attention-fp4, a fork of Dao-AILab/flash-attention whose
    # LICENSE and flash_attn/cute/AUTHORS are byte-identical to the upstream texts under
    # licenses/; only the canonical URL and commit history differ from the bucket.
    "tirx_kernels/flashattention/flash_attention4_fp4.py": {
        "project": "flash-attention-fp4",
        "url": "https://github.com/hao-ai-lab/flash-attention-fp4",
    },
    "tirx_kernels/cudnn/bsa/block_sparse_attention_forward_combine_sm100_blk64.py": {
        "spdx": "Apache-2.0 AND MIT AND BSD-3-Clause",
        "required_text": (
            "Redistribution and use in source and binary forms",
            'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        ),
    },
    "tirx_kernels/flashinfer/gdn_prefill/gdn_cp_prefill_sm100.py": {
        "spdx": "Apache-2.0 AND BSD-3-Clause",
        "required_text": (
            "Redistribution and use in source and binary forms",
            'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        ),
    },
    "tirx_kernels/flashinfer/gdn_prefill/gdn_prefill_sm100.py": {
        "spdx": "Apache-2.0 AND BSD-3-Clause",
        "required_text": (
            "Redistribution and use in source and binary forms",
            'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        ),
    },
    "tirx_kernels/flashinfer/gemm/bmm_fp8_rubin.py": {
        "spdx": "Apache-2.0 AND BSD-3-Clause",
        "required_text": (
            "Redistribution and use in source and binary forms",
            'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        ),
    },
    "tirx_kernels/flashinfer/gemm/grouped_gemm_masked_rubin.py": {
        "spdx": "Apache-2.0 AND BSD-3-Clause",
        "required_text": (
            "Redistribution and use in source and binary forms",
            'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        ),
    },
    "tirx_kernels/flashinfer/fused_moe/"
    "blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin.py": {
        "spdx": "Apache-2.0 AND BSD-3-Clause",
        "required_text": (
            "Redistribution and use in source and binary forms",
            'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        ),
    },
    "tirx_kernels/flashinfer/gemm/dense_blockscaled_gemm_sm107.py": {
        "spdx": "Apache-2.0 AND BSD-3-Clause",
        "required_text": (
            "Redistribution and use in source and binary forms",
            'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        ),
    },
    # Transcribes cub::BlockRadixSort, which FlashInfer's topk kernels instantiate.
    "tirx_kernels/flashinfer/utils/block_radix_sort.py": {
        "spdx": "Apache-2.0 AND BSD-3-Clause",
        "required_text": (
            "Redistribution and use in source and binary forms",
            'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        ),
    },
}

# Native modules inside port buckets — package markers and our own harnesses.
PORT_DIR_EXCEPTIONS = {
    "tirx_kernels/cudnn/_reference.py",
    "tirx_kernels/deepep/utils/_buffer.py",
    "tirx_kernels/deepep/utils/_runtime.py",
    "tirx_kernels/flashinfer/utils/_flashkda_bench.py",
    "tirx_kernels/flashmla/utils/_flashmla_bench.py",
    "tirx_kernels/flashmla/utils/_trtllm_gen_bench.py",
    "tirx_kernels/msa/utils/_msa_bench.py",
}

# Retired: the old per-file "Modifications" block and the pointer paragraph that
# went with it; attributions live in the SPDX tags, LICENSE and licenses/ now.
BANNED = (
    "THIRD_PARTY_LICENSES",
    "TIRX Authors",
    "Modifications Copyright",
    "Modifications are licensed",
    "See LICENSE, NOTICE",
)


def header_of(text: str) -> str:
    """Return the leading comment region (blank lines inside it kept)."""
    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    out: list[str] = []
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            out.append(ln)
        else:
            break
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def bucket_of(rel: str):
    if rel in PORT_DIR_EXCEPTIONS or rel.endswith("__init__.py"):
        return None
    for prefix, expect in PORT_BUCKETS.items():
        if rel.startswith(prefix):
            return expect
    return None


def check(rel: str, text: str) -> list[str]:
    errors = []
    header = header_of(text)

    # The checker itself spells out the banned strings; skip only that scan.
    if rel != SELF:
        for banned in BANNED:
            if banned in text:
                errors.append(f"{rel}: contains retired reference {banned!r}")

    if not header:
        errors.append(f"{rel}: missing license header (run --fix to add the TIRx one)")
        return errors

    lines = header.splitlines()
    ids = [ln[len(SPDX_ID) :] for ln in lines if ln.startswith(SPDX_ID)]
    if len(ids) != 1:
        errors.append(f"{rel}: expected exactly one {SPDX_ID.strip()} line, found {len(ids)}")
        return errors
    if SPDX_COPYRIGHT not in lines:
        errors.append(f"{rel}: missing {SPDX_COPYRIGHT!r}")

    expect = bucket_of(rel)
    if expect is None:
        if header != TIRX_HEADER:
            errors.append(f"{rel}: native file must carry exactly the two-line TIRx SPDX header")
        return errors

    project, url, spdx = expect
    override = FILE_OVERRIDES.get(rel, {})
    project = override.get("project", project)
    url = override.get("url", url)
    spdx = override.get("spdx", spdx)

    if ids[0] != spdx:
        errors.append(f"{rel}: SPDX expression must be {spdx!r} for this path, found {ids[0]!r}")
    if not any(ln.startswith(PORT_LINE + project) for ln in lines):
        errors.append(f"{rel}: ported file must cite '{PORT_LINE.strip()} {project}'")
    cited = CITATION_RE.search(header)
    if not cited:
        errors.append(f"{rel}: ported file is missing an upstream '(url @ commit)' citation")
    elif cited.group(1) != url:
        errors.append(f"{rel}: upstream URL must be {url!r}, found {cited.group(1)!r}")
    if "Copyright" not in header.split(SPDX_COPYRIGHT)[0]:
        errors.append(f"{rel}: ported file is missing its upstream copyright notice")
    for required in override.get("required_text", ()):
        if required not in header:
            errors.append(f"{rel}: upstream license requires this text in the header: {required!r}")

    return errors


def self_test() -> int:
    """Feed known violations to check(); fail unless every one is caught."""
    pair = f"{SPDX_ID}{{spdx}}\n{SPDX_COPYRIGHT}"
    port = (
        "# This file is a TIRx port of code from {project}\n"
        "# ({url} @ 559d79fb), Copyright (c) 2025 Upstream\n" + pair + "\n\nx = 1\n"
    )
    dg = dict(project="DeepGEMM", url="https://github.com/deepseek-ai/DeepGEMM")
    cudnn = dict(project="cuDNN Frontend", url="https://github.com/NVIDIA/cudnn-frontend")
    fi = dict(project="FlashInfer", url="https://github.com/flashinfer-ai/flashinfer")
    msa = dict(project="MSA", url="https://github.com/MiniMax-AI/MSA")
    fastcu = dict(project="fast.cu", url="https://github.com/pranjalssh/fast.cu")
    bsa_combine = "tirx_kernels/cudnn/bsa/block_sparse_attention_forward_combine_sm100_blk64.py"
    gdn = "tirx_kernels/flashinfer/gdn_prefill/gdn_prefill_sm100.py"
    gdn_cp = "tirx_kernels/flashinfer/gdn_prefill/gdn_cp_prefill_sm100.py"
    bmm_fp8_rubin = "tirx_kernels/flashinfer/gemm/bmm_fp8_rubin.py"
    grouped_gemm_masked_rubin = "tirx_kernels/flashinfer/gemm/grouped_gemm_masked_rubin.py"
    fused_moe_rubin = (
        "tirx_kernels/flashinfer/fused_moe/"
        "blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin.py"
    )
    fa4 = dict(project="flash-attention", url="https://github.com/Dao-AILab/flash-attention")
    fa4_fp4 = dict(
        project="flash-attention-fp4", url="https://github.com/hao-ai-lab/flash-attention-fp4"
    )
    fp4_port = "tirx_kernels/flashattention/flash_attention4_fp4.py"
    bsd_port = (
        "# Copyright (c) 2025 Upstream\n"
        "# Redistribution and use in source and binary forms\n"
        '# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"\n' + port
    )
    cases = [
        # (name, rel, text, must_error)
        (
            "valid flash-attention-fp4 fork port",
            fp4_port,
            port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fa4_fp4),
            False,
        ),
        (
            "flash-attention-fp4 port citing the Dao-AILab URL",
            fp4_port,
            port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fa4),
            True,
        ),
        (
            "flash-attention-fp4 port naming the parent project",
            fp4_port,
            port.format(
                spdx="Apache-2.0 AND BSD-3-Clause", project="flash-attention", url=fa4_fp4["url"]
            ),
            True,
        ),
        (
            "flash-attention-fp4 port tagged plain Apache",
            fp4_port,
            port.format(spdx="Apache-2.0", **fa4_fp4),
            True,
        ),
        (
            "flash-attention-fp4 port missing the upstream copyright",
            fp4_port,
            port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fa4_fp4).replace(
                "Copyright (c) 2025 Upstream", "2025 Upstream"
            ),
            True,
        ),
        (
            "sibling flashattention port citing the fork URL",
            "tirx_kernels/flashattention/x.py",
            port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fa4_fp4),
            True,
        ),
        (
            "valid flash-attention port",
            "tirx_kernels/flashattention/x.py",
            port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fa4),
            False,
        ),
        (
            "valid deepgemm port",
            "tirx_kernels/deepgemm/x.py",
            port.format(spdx="Apache-2.0 AND MIT", **dg),
            False,
        ),
        (
            "valid cudnn-frontend port",
            "tirx_kernels/cudnn/amax/x.py",
            port.format(spdx="Apache-2.0", **cudnn),
            False,
        ),
        (
            "valid fast.cu port",
            "tirx_kernels/fastcu/x.py",
            port.format(spdx="Apache-2.0 AND MIT", **fastcu),
            False,
        ),
        (
            "fast.cu tagged plain Apache-2.0",
            "tirx_kernels/fastcu/x.py",
            port.format(spdx="Apache-2.0", **fastcu),
            True,
        ),
        (
            "fast.cu port citing another upstream URL",
            "tirx_kernels/fastcu/x.py",
            port.format(
                spdx="Apache-2.0 AND MIT",
                project="fast.cu",
                url="https://github.com/deepseek-ai/DeepGEMM",
            ),
            True,
        ),
        (
            "fast.cu port with no upstream copyright",
            "tirx_kernels/fastcu/x.py",
            port.format(spdx="Apache-2.0 AND MIT", **fastcu).replace(
                ", Copyright (c) 2025 Upstream", ""
            ),
            True,
        ),
        (
            "valid cudnn-frontend BSD/MIT combine port",
            bsa_combine,
            bsd_port.format(spdx="Apache-2.0 AND MIT AND BSD-3-Clause", **cudnn),
            False,
        ),
        (
            "cudnn-frontend combine port missing MIT terms",
            bsa_combine,
            bsd_port.format(spdx="Apache-2.0 AND BSD-3-Clause", **cudnn),
            True,
        ),
        (
            "cudnn-frontend combine port missing BSD disclaimer",
            bsa_combine,
            bsd_port.format(spdx="Apache-2.0 AND MIT AND BSD-3-Clause", **cudnn).replace(
                'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
                "missing disclaimer",
            ),
            True,
        ),
        (
            "cudnn-frontend port tagged MIT",
            "tirx_kernels/cudnn/amax/x.py",
            port.format(spdx="Apache-2.0 AND MIT", **cudnn),
            True,
        ),
        (
            "cudnn-frontend port citing another URL",
            "tirx_kernels/cudnn/amax/x.py",
            port.format(
                spdx="Apache-2.0",
                project="cuDNN Frontend",
                url="https://github.com/deepseek-ai/DeepGEMM",
            ),
            True,
        ),
        (
            "cudnn-frontend port with no upstream copyright",
            "tirx_kernels/cudnn/amax/x.py",
            port.format(spdx="Apache-2.0", **cudnn).replace(", Copyright (c) 2025 Upstream", ""),
            True,
        ),
        ("valid native", "tirx_kernels/x.py", f"{TIRX_HEADER}\n\nx = 1\n", False),
        (
            "valid gdn_cp_prefill BSD port",
            gdn_cp,
            bsd_port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fi),
            False,
        ),
        (
            "valid bmm_fp8_rubin BSD port",
            bmm_fp8_rubin,
            bsd_port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fi),
            False,
        ),
        (
            "bmm_fp8_rubin tagged plain Apache-2.0",
            bmm_fp8_rubin,
            bsd_port.format(spdx="Apache-2.0", **fi),
            True,
        ),
        (
            "bmm_fp8_rubin missing BSD disclaimer",
            bmm_fp8_rubin,
            bsd_port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fi).replace(
                'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
                "missing disclaimer",
            ),
            True,
        ),
        (
            "valid grouped_gemm_masked_rubin BSD port",
            grouped_gemm_masked_rubin,
            bsd_port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fi),
            False,
        ),
        (
            "grouped_gemm_masked_rubin tagged plain Apache-2.0",
            grouped_gemm_masked_rubin,
            bsd_port.format(spdx="Apache-2.0", **fi),
            True,
        ),
        (
            "grouped_gemm_masked_rubin missing BSD disclaimer",
            grouped_gemm_masked_rubin,
            bsd_port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fi).replace(
                'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
                "missing disclaimer",
            ),
            True,
        ),
        (
            "valid fused_moe_rubin BSD port",
            fused_moe_rubin,
            bsd_port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fi),
            False,
        ),
        (
            "fused_moe_rubin tagged plain Apache-2.0",
            fused_moe_rubin,
            bsd_port.format(spdx="Apache-2.0", **fi),
            True,
        ),
        (
            "fused_moe_rubin missing BSD disclaimer",
            fused_moe_rubin,
            bsd_port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fi).replace(
                'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
                "missing disclaimer",
            ),
            True,
        ),
        (
            "deepgemm tagged plain Apache-2.0",
            "tirx_kernels/deepgemm/x.py",
            port.format(spdx="Apache-2.0", **dg),
            True,
        ),
        (
            "deepgemm tagged AND BSD-3-Clause",
            "tirx_kernels/deepgemm/x.py",
            port.format(spdx="Apache-2.0 AND BSD-3-Clause", **dg),
            True,
        ),
        (
            "flashinfer tagged AND MIT",
            "tirx_kernels/flashinfer/x.py",
            port.format(spdx="Apache-2.0 AND MIT", **fi),
            True,
        ),
        (
            "valid msa port",
            "tirx_kernels/msa/x.py",
            port.format(spdx="Apache-2.0 AND MIT", **msa),
            False,
        ),
        (
            "msa tagged plain Apache-2.0",
            "tirx_kernels/msa/x.py",
            port.format(spdx="Apache-2.0", **msa),
            True,
        ),
        (
            "msa port citing another upstream URL",
            "tirx_kernels/msa/x.py",
            port.format(
                spdx="Apache-2.0 AND MIT",
                project="MSA",
                url="https://github.com/deepseek-ai/DeepGEMM",
            ),
            True,
        ),
        (
            "msa port naming another project",
            "tirx_kernels/msa/x.py",
            port.format(
                spdx="Apache-2.0 AND MIT",
                project="DeepGEMM",
                url="https://github.com/MiniMax-AI/MSA",
            ),
            True,
        ),
        (
            "msa port with no upstream copyright",
            "tirx_kernels/msa/x.py",
            port.format(spdx="Apache-2.0 AND MIT", **msa).replace(
                ", Copyright (c) 2025 Upstream", ""
            ),
            True,
        ),
        (
            "gdn_prefill without the BSD conditions text",
            gdn,
            port.format(spdx="Apache-2.0 AND BSD-3-Clause", **fi),
            True,
        ),
        (
            "port citing the wrong upstream URL",
            "tirx_kernels/deepgemm/x.py",
            port.format(
                spdx="Apache-2.0 AND MIT",
                project="DeepGEMM",
                url="https://github.com/deepseek-ai/FlashMLA",
            ),
            True,
        ),
        (
            "port with no upstream copyright",
            "tirx_kernels/deepgemm/x.py",
            port.format(spdx="Apache-2.0 AND MIT", **dg).replace(
                ", Copyright (c) 2025 Upstream", ""
            ),
            True,
        ),
        (
            "port missing the citation line",
            "tirx_kernels/deepgemm/x.py",
            "\n".join(port.format(spdx="Apache-2.0 AND MIT", **dg).splitlines()[2:]),
            True,
        ),
        (
            "native with an extra header line",
            "tirx_kernels/x.py",
            f"# Copyright (c) 2026 extra\n{TIRX_HEADER}\n\nx = 1\n",
            True,
        ),
        (
            "banned Modifications block",
            "tirx_kernels/x.py",
            f"{TIRX_HEADER}\n# " + "Modifications" + " Copyright (c) 2026\n\nx = 1\n",
            True,
        ),
        ("two SPDX id lines", "tirx_kernels/x.py", f"{TIRX_HEADER}\n{SPDX_ID}MIT\n\nx = 1\n", True),
    ]
    failures = 0
    for name, rel, text, must_error in cases:
        errors = check(rel, text)
        ok = bool(errors) == must_error
        print(
            f"{'ok' if ok else 'SELF-TEST FAIL'}: {name}"
            + (f" -> {errors[0]}" if errors and ok else "")
        )
        failures += not ok
    if failures:
        print(f"\n{failures} self-test case(s) NOT caught; the checker is vacuous somewhere.")
        return 1
    print(f"\nself-test: all {len(cases)} cases behave as expected.")
    return 0


def tracked_python_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout.split()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix", action="store_true", help="insert the TIRx header into files that have none"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="verify the checker catches known violations"
    )
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    errors: list[str] = []
    for rel in tracked_python_files():
        path = REPO / rel
        text = path.read_text()
        if args.fix and not header_of(text) and bucket_of(rel) is None:
            lines = text.splitlines()
            shebang = [lines.pop(0)] if lines and lines[0].startswith("#!") else []
            body = "\n".join(lines).lstrip("\n")
            path.write_text(
                "\n".join(shebang) + ("\n" if shebang else "") + TIRX_HEADER + "\n\n" + body
            )
            text = path.read_text()
        errors += check(rel, text)

    for e in errors:
        print(f"ERROR: {e}")
    if errors:
        print(f"\n{len(errors)} license header problem(s).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
