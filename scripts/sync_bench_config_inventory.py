#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Synchronize selected bench YAML inventories with module-owned config labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
CONFIG_DIR = REPO_ROOT / "tirx_kernels" / "bench_suite" / "config"


def sync_file(path: Path) -> bool:
    from tirx_kernels.bench_suite.run import _bench_configs, _config_num_gpus
    from tirx_kernels.registry import load_kernel

    document = yaml.safe_load(path.read_text()) or {}
    kernel = document["kernel"]
    module = load_kernel(kernel, strict=True)
    module_configs = _bench_configs(module)
    existing = {row["config"]: dict(row) for row in document.get("configs") or []}
    synced = []
    for config in module_configs:
        label = config.get("label", "default")
        row = existing.pop(label, {"config": label, "default": False})
        required_num_gpus = _config_num_gpus(config)
        if required_num_gpus == 1:
            row.pop("num_gpus", None)
        else:
            row["num_gpus"] = required_num_gpus
        synced.append(row)
    if existing:
        raise ValueError(f"{path}: YAML-only config labels: {sorted(existing)}")
    document["configs"] = synced
    rendered = yaml.safe_dump(document, sort_keys=False, width=1000)
    if rendered == path.read_text():
        return False
    path.write_text(rendered)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        resolved = path if path.is_absolute() else Path.cwd() / path
        if CONFIG_DIR.resolve() not in resolved.resolve().parents:
            raise ValueError(f"refusing to edit config outside {CONFIG_DIR}: {path}")
        if sync_file(resolved):
            print(path)


if __name__ == "__main__":
    main()
