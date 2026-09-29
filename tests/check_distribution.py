# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Smoke-test a wheel installed in an isolated environment, without GPU runtimes.

Run with ``python -I tests/check_distribution.py`` to exclude the checkout from
Python's import path. Optionally pass ``--expected-version v0.1.0``.
"""

import argparse
import json
import subprocess
from importlib.metadata import distribution
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.version import Version

import tirx_kernels
from tirx_kernels.registry import kernel_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default="")
    args = parser.parse_args()

    dist = distribution("tirx-kernels")
    if args.expected_version:
        assert Version(dist.version) == Version(args.expected_version), dist.version
        assert not Version(dist.version).is_devrelease, dist.version
        assert Version(dist.version).local is None, dist.version
    assert all(Requirement(req).url is None for req in dist.requires or ())

    checkout = Path(__file__).resolve().parents[1]
    package = Path(tirx_kernels.__file__).resolve().parent
    assert not package.is_relative_to(checkout), f"Imported checkout instead of wheel: {package}"

    # Compare every tracked package source and resource with the installed copy.
    sources = (
        subprocess.check_output(["git", "ls-files", "-z", "tirx_kernels"], cwd=checkout)
        .decode()
        .split("\0")
    )
    for source in filter(None, sources):
        installed = package.parent / source
        assert installed.is_file(), f"Missing from wheel: {source}"
        assert installed.read_bytes() == (checkout / source).read_bytes(), source

    records = kernel_index(strict=True)
    assert records, "No kernels found in installed wheel"
    configs = list((package / "bench_suite/config").rglob("*.yaml"))
    assert configs, "Missing benchmark configurations"
    for config in configs:
        assert yaml.safe_load(config.read_text()), config
    assert json.loads((package / "bench_suite/baseline.json").read_text())
    assert (package / "ported/fastcu/nvfp4_gemm_gb300_reference.cu").is_file()

    licenses = dist.metadata.get_all("License-File")
    expected_licenses = {"LICENSE", "NOTICE"} | {
        path.relative_to(checkout).as_posix() for path in (checkout / "licenses").glob("*.txt")
    }
    assert set(licenses or ()) == expected_licenses, licenses
    for license_file in expected_licenses:
        matches = [
            path
            for path in dist.files or ()
            if str(path).endswith(f".dist-info/licenses/{license_file}")
        ]
        assert len(matches) == 1, license_file
        assert dist.locate_file(matches[0]).read_bytes() == (checkout / license_file).read_bytes()

    print(f"tirx-kernels {dist.version}: {len(records)} kernels, {len(configs)} configs; wheel OK")


if __name__ == "__main__":
    main()
