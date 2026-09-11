# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""CLI entry point: python -m tirx_kernels.bench [--kernel <name>] [--config <label>]"""

import argparse
import json
import os
import sys
import traceback
from unittest import SkipTest


def _get_bench_configs(mod):
    return getattr(mod, "BENCH_CONFIGS", getattr(mod, "CONFIGS", []))


def _find_bench_config(mod, label: str) -> dict:
    matches = [config for config in _get_bench_configs(mod) if config.get("label") == label]
    if len(matches) != 1:
        raise KeyError(
            f"kernel {mod.KERNEL_META['name']!r} config {label!r}: "
            f"expected exactly one match, found {len(matches)}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Run kernel benchmarks")
    parser.add_argument("--kernel", type=str, default=None, help="Run only this kernel")
    parser.add_argument("--config", type=str, default=None, help="Run only this config label")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    parser.add_argument(
        "--json-file",
        type=str,
        default=None,
        help="Write JSON results to this file instead of stdout",
    )
    parser.add_argument("--arch", type=str, default=None, help="Exact CUDA architecture filter")
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override the event/proton warmup budget in ms (else bench() default)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override the event/proton rep budget in ms (else bench() default)",
    )
    parser.add_argument(
        "--timer",
        type=str,
        choices=("event", "proton", "cudagraph_proton", "kineto", "megamoe", "e2e"),
        default=None,
        help="Override the kernel module's benchmark timer: 'event' = do_bench, "
        "'proton' = do_bench_proton, 'cudagraph_proton' = "
        "do_bench_cudagraph_proton [NVIDIA], 'kineto' = distributed full GPU "
        "activity span, 'megamoe' = DeepGEMM bench_kineto protocol for MegaMoE, "
        "'e2e' = distributed rank-max end-to-end wall protocol",
    )
    parser.add_argument(
        "--with-references",
        action="store_true",
        help="Run and time external reference implementations (off by default)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Independent standard-timer calls inside one process (default: runner protocol)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=None,
        help=("Seconds before every implementation in every round (default: runner protocol)"),
    )
    args = parser.parse_args()

    if args.json or args.json_file:
        os.environ["TIRX_BENCH_JSON"] = "1"

    from tirx_kernels.runner import set_external_references_enabled

    set_external_references_enabled(args.with_references)

    from tirx_kernels.registry import discover_kernels, load_kernel
    from tirx_kernels.runner import DEFAULT_BENCH_COOLDOWN_S, DEFAULT_BENCH_ROUNDS, run_kernel_bench

    if args.rounds is None:
        args.rounds = DEFAULT_BENCH_ROUNDS
    if args.cooldown is None:
        args.cooldown = DEFAULT_BENCH_COOLDOWN_S
    if args.rounds < 1:
        print("ERROR: --rounds must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.cooldown < 0:
        print("ERROR: --cooldown must be >= 0", file=sys.stderr)
        sys.exit(2)

    if args.kernel:
        try:
            mod = load_kernel(args.kernel)
        except KeyError:
            print(f"ERROR: kernel '{args.kernel}' not found.", file=sys.stderr)
            sys.exit(1)
        runtime_cuda_archs = tuple(mod.KERNEL_META.get("runtime_cuda_archs", ()))
        if args.arch is not None and args.arch not in runtime_cuda_archs:
            print(
                f"ERROR: kernel '{args.kernel}' runtime_cuda_archs={runtime_cuda_archs!r} "
                f"does not include filter {args.arch!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        all_kernels = {args.kernel: mod}
    else:
        all_kernels = discover_kernels(cuda_arch=args.arch)

    # Each kernel's run_bench() manages its own Proton session via bench(timer=...).
    # No global proton session needed.
    results = []

    for name, mod in sorted(all_kernels.items()):
        configs = _get_bench_configs(mod)
        for cfg in configs:
            label = cfg.get("label", "default")
            if args.config and label != args.config:
                continue
            try:
                # GPU flock is inside tvm.tirx.bench (prepare + rounds).
                result = run_kernel_bench(
                    name,
                    cfg,
                    registry=all_kernels,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    timer=args.timer,
                    rounds=args.rounds,
                    cooldown=args.cooldown,
                )
                results.append(result)
            except SkipTest as exc:
                results.append(
                    {"kernel": name, "label": label, "status": "SKIP", "reason": str(exc)}
                )
                if not args.json and not args.json_file:
                    print(f"SKIP  {name} [{label}]: {exc}", file=sys.stderr)
            except Exception as e:
                results.append({"kernel": name, "label": label, "status": "FAIL", "error": str(e)})
                if not args.json and not args.json_file:
                    print(f"FAIL  {name} [{label}]: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

    if args.json_file:
        with open(args.json_file, "w") as f:
            json.dump({"results": results}, f, indent=2)
    elif args.json:
        print(json.dumps({"results": results}, indent=2))
    else:
        # Print summary to stdout for human consumption
        for r in results:
            status = r.get("status", "ok")
            kernel = r.get("kernel", "?")
            label = r.get("label", "?")
            if status == "FAIL":
                print(f"FAIL  {kernel} [{label}]: {r.get('error', '?')}")
            elif status == "SKIP":
                print(f"SKIP  {kernel} [{label}]: {r.get('reason', '?')}")
            else:
                impls = r.get("impls", {})
                impl_str = ", ".join(f"{k}={v:.3f}µs" for k, v in impls.items())
                print(f"OK    {kernel} [{label}]: {impl_str}")


if __name__ == "__main__":
    main()
