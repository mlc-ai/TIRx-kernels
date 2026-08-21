# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""CLI entry point for registry-owned kernel correctness."""

import argparse
import json
import sys
import traceback
from unittest import SkipTest

from tirx_kernels.registry import discover_kernels
from tirx_kernels.runner import run_kernel_test


def _config_num_gpus(config: dict) -> int:
    topology = [config[key] for key in ("world_size", "num_processes") if key in config]
    if len(topology) == 2 and topology[0] != topology[1]:
        raise ValueError(
            "conflicting config GPU counts: "
            f"world_size={topology[0]!r}, num_processes={topology[1]!r}"
        )
    value = topology[0] if topology else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"invalid config GPU count: {value!r}")
    return value


def _selected_configs(kernels: dict, *, label: str | None = None, num_gpus: int | None = None):
    for name, mod in sorted(kernels.items()):
        for config in getattr(mod, "CONFIGS", []):
            if label is not None and config.get("label", "default") != label:
                continue
            if num_gpus is not None and _config_num_gpus(config) != num_gpus:
                continue
            yield name, config


def main():
    parser = argparse.ArgumentParser(description="Run kernel correctness tests")
    parser.add_argument("--kernel", type=str, default=None, help="Run only this kernel")
    parser.add_argument("--config", type=str, default=None, help="Run only this config label")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    parser.add_argument("--cc", type=int, default=None, help="Compute capability filter")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Run only configs requiring this many GPUs (world_size/num_processes, default 1)",
    )
    args = parser.parse_args()
    if args.num_gpus is not None and args.num_gpus < 1:
        parser.error("--num-gpus must be positive")

    all_kernels = discover_kernels(min_compute_capability=args.cc)

    if args.kernel:
        if args.kernel not in all_kernels:
            print(
                f"ERROR: kernel '{args.kernel}' not found. Available: {sorted(all_kernels.keys())}"
            )
            sys.exit(1)
        all_kernels = {args.kernel: all_kernels[args.kernel]}

    results = []
    passed = 0
    failed = 0
    skipped = 0

    for name, cfg in _selected_configs(all_kernels, label=args.config, num_gpus=args.num_gpus):
        label = cfg.get("label", "default")
        try:
            run_kernel_test(name, cfg, registry=all_kernels)
            results.append({"kernel": name, "config": label, "status": "PASS"})
            passed += 1
            if not args.json:
                print(f"PASS  {name} [{label}]")
        except SkipTest as exc:
            results.append({"kernel": name, "config": label, "status": "SKIP", "reason": str(exc)})
            skipped += 1
            if not args.json:
                print(f"SKIP  {name} [{label}]: {exc}")
        except Exception as e:
            results.append({"kernel": name, "config": label, "status": "FAIL", "error": str(e)})
            failed += 1
            if not args.json:
                print(f"FAIL  {name} [{label}]: {e}")
                traceback.print_exc()

    if not results:
        message = "no kernel configs matched the requested filters"
        if args.json:
            print(
                json.dumps(
                    {"passed": 0, "failed": 0, "skipped": 0, "results": [], "error": message}
                )
            )
        else:
            print(f"ERROR: {message}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(
            json.dumps(
                {"passed": passed, "failed": failed, "skipped": skipped, "results": results},
                indent=2,
            )
        )
    else:
        print(f"\n{'=' * 60}")
        print(
            f"Total: {passed + failed + skipped}  "
            f"Passed: {passed}  Failed: {failed}  Skipped: {skipped}"
        )

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
