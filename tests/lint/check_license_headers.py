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

Ports must use the second shape: dropping the upstream notice from a ported
file is exactly the drift this check exists to prevent. Unlike ASF-style header
checks, upstream ``Copyright`` lines are required rather than forbidden.

Full license texts live in ``licenses/``; ``LICENSE`` maps each bucket of the
tree to its license. Where an upstream license requires the conditions text
itself to travel with the source (BSD-3), that text stays in the file verbatim
and is not replaced by a tag.

Run ``--fix`` to insert the TIRx header into files that have no header at all;
port headers are never synthesized, since only a human knows the upstream terms.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

SPDX_ID = "# SPDX-License-Identifier: "
SPDX_COPYRIGHT = "# SPDX-FileCopyrightText: Copyright TIRx authors"
TIRX_HEADER = f"{SPDX_ID}Apache-2.0\n{SPDX_COPYRIGHT}"

# Our own contribution is always Apache-2.0; a port additionally carries the
# upstream license, spelled as an SPDX `AND` expression.
ID_RE = re.compile(r"^Apache-2\.0(?: AND (?:MIT|BSD-3-Clause))?$")

# A port must cite the upstream project, its URL and the exact commit ported.
PORT_LINE = "# This file is a TIRx port of code from "
CITATION_RE = re.compile(r"\(https://\S+ @ [0-9a-f]{7,40}\)")

# Buckets holding ports of third-party kernels: every module under them must
# carry a port header, not the plain TIRx one.
PORT_DIRS = (
    "tirx_kernels/deepgemm/",
    "tirx_kernels/flashattention/",
    "tirx_kernels/flashinfer/",
    "tirx_kernels/flashmla/",
)

# Native modules inside those buckets — package markers and our own harnesses.
PORT_DIR_EXCEPTIONS = {
    "tirx_kernels/flashinfer/utils/_flashkda_bench.py",
    "tirx_kernels/flashmla/utils/_flashmla_bench.py",
    "tirx_kernels/flashmla/utils/_trtllm_gen_bench.py",
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


def is_port(rel: str) -> bool:
    if rel in PORT_DIR_EXCEPTIONS or rel.endswith("__init__.py"):
        return False
    return rel.startswith(PORT_DIRS)


def check(rel: str, text: str) -> list[str]:
    errors = []
    header = header_of(text)

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
    elif not ID_RE.match(ids[0]):
        errors.append(f"{rel}: unexpected SPDX license expression {ids[0]!r}")
    if SPDX_COPYRIGHT not in lines:
        errors.append(f"{rel}: missing {SPDX_COPYRIGHT!r}")

    if is_port(rel):
        if not any(ln.startswith(PORT_LINE) for ln in lines):
            errors.append(f"{rel}: ported file is missing its {PORT_LINE.strip()!r} citation")
        if not CITATION_RE.search(header):
            errors.append(f"{rel}: ported file is missing an upstream '(url @ commit)' citation")
        if "Copyright" not in header.split(SPDX_COPYRIGHT)[0]:
            errors.append(f"{rel}: ported file is missing its upstream copyright notice")
    elif header != TIRX_HEADER:
        errors.append(f"{rel}: native file must carry exactly the two-line TIRx SPDX header")

    return errors


def tracked_python_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout.split()
    # This checker is skipped: it spells out the retired strings it bans.
    return [f for f in out if not f.startswith("tests/lint/")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix", action="store_true", help="insert the TIRx header into files that have none"
    )
    args = parser.parse_args()

    errors: list[str] = []
    for rel in tracked_python_files():
        path = REPO / rel
        text = path.read_text()
        if args.fix and not header_of(text) and not is_port(rel):
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
