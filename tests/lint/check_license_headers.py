#!/usr/bin/env python3
# Copyright (c) 2026 The TIRx Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Check that every tracked Python file carries an acceptable license header.

Two shapes are accepted:

* the TIRx Apache header, for code with no upstream lineage; and
* a port header, for files ported from an upstream project: the upstream
  copyright notice (reproduced verbatim from the upstream file, or an explicit
  statement of the upstream terms where upstream ships no per-file header),
  followed by the TIRx modification block.

Ports must use the second shape: dropping the upstream notice from a ported
file is exactly the drift this check exists to prevent. Unlike ASF-style header
checks, upstream ``Copyright`` lines are required rather than forbidden.

Run ``--fix`` to insert the TIRx Apache header into files that have no header
at all; port headers are never synthesized, since only a human knows what the
upstream terms are.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

TIRX_APACHE = """\
# Copyright (c) 2026 The TIRx Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License."""

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

MODS_LINE = "# Modifications Copyright (c) 2026 The TIRx Authors."
MODS_LICENSE_LINE = "# Modifications are licensed under the Apache License, Version 2.0."

# Retired file: attributions live in LICENSE + licenses/ now.
BANNED = ("THIRD_PARTY_LICENSES", "TIRX Authors")


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

    has_tirx_apache = header.startswith(TIRX_APACHE)
    has_mods = MODS_LINE in header and MODS_LICENSE_LINE in header

    if is_port(rel):
        if not has_mods:
            errors.append(f"{rel}: ported file is missing the TIRx modification block")
        upstream = header.split(MODS_LINE)[0]
        if "Copyright" not in upstream and "licensed under" not in upstream:
            errors.append(f"{rel}: ported file is missing its upstream copyright / license notice")
        if has_tirx_apache:
            errors.append(
                f"{rel}: ported file uses the plain TIRx header; reproduce the upstream "
                "header and add the modification block instead"
            )
    elif not has_tirx_apache and not has_mods:
        errors.append(f"{rel}: header is neither the TIRx Apache header nor a port header")

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
                "\n".join(shebang) + ("\n" if shebang else "") + TIRX_APACHE + "\n\n" + body
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
