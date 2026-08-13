# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""CLI entry point: python -m tirx_kernels.bench [--kernel <name>] [--config <label>]"""

import argparse
import json
import os
import socket
import sys
import time
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


def _send_control(control: socket.socket, message: dict) -> None:
    control.sendall(json.dumps(message, separators=(",", ":")).encode() + b"\n")


def _validated_gpu_assignment(gpu_indices: object, required_num_gpus: int) -> list[str]:
    if not isinstance(gpu_indices, list) or len(gpu_indices) != required_num_gpus:
        raise ValueError(f"invalid GPU assignment: {gpu_indices!r}")
    normalized = [str(index) for index in gpu_indices]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"GPU assignment contains duplicates: {gpu_indices!r}")
    if any(not index.isdigit() for index in normalized):
        raise ValueError(f"GPU assignment contains an invalid physical index: {gpu_indices!r}")
    return normalized


def _validated_physical_gpu_uuids(
    expected_gpu_uuids: object,
    actual_gpu_uuids: tuple[str, ...],
    required_num_gpus: int,
) -> list[str]:
    if not isinstance(expected_gpu_uuids, list) or len(expected_gpu_uuids) != required_num_gpus:
        raise ValueError(f"invalid physical GPU UUID assignment: {expected_gpu_uuids!r}")
    actual = list(actual_gpu_uuids)
    if actual != expected_gpu_uuids:
        raise RuntimeError(
            "late GPU assignment identity mismatch: "
            f"requested {expected_gpu_uuids}, visible {actual}"
        )
    return actual


def _prepared_child_main(args, *, child_started: float) -> int:
    """CPU-prepare in this process, then wait for a late GPU assignment."""
    control = socket.socket(fileno=args.prepared_control_fd)
    reader = control.makefile("r", encoding="utf-8")
    prepare_started = child_started
    try:
        try:
            framework_import_started = time.time()
            from tirx_kernels.registry import load_kernel
            from tirx_kernels.runner import (
                DEFAULT_BENCH_COOLDOWN_S,
                DEFAULT_BENCH_ROUNDS,
                cuda_is_initialized,
                physical_cuda_uuids,
                prepare_kernel_bench,
                run_prepared_kernel_bench,
            )

            framework_loaded = time.time()
            if args.rounds is None:
                args.rounds = DEFAULT_BENCH_ROUNDS
            if args.cooldown is None:
                args.cooldown = DEFAULT_BENCH_COOLDOWN_S
            if args.rounds < 1:
                raise ValueError("--rounds must be >= 1")
            if args.cooldown < 0:
                raise ValueError("--cooldown must be >= 0")
            if not args.kernel or not args.config:
                raise ValueError("prepared child requires --kernel and --config")
            if cuda_is_initialized():
                raise RuntimeError("CUDA was initialized before CPU prepare")
            module = load_kernel(args.kernel, strict=True)
            module_loaded = time.time()
            config = _find_bench_config(module, args.config)
            config_resolved = time.time()
            prepared = prepare_kernel_bench(
                args.kernel,
                config,
                module=module,
                require_cuda_uninitialized=True,
            )
            if prepared.required_num_gpus != args.prepared_num_gpus:
                raise ValueError(
                    f"workload declares num_gpus={args.prepared_num_gpus}, but prepared "
                    f"benchmark requires {prepared.required_num_gpus}"
                )
            ready = time.time()
            _send_control(
                control,
                {
                    "type": "READY",
                    "child_started": child_started,
                    "prepare_started": prepare_started,
                    "framework_import_started": framework_import_started,
                    "framework_loaded": framework_loaded,
                    "module_loaded": module_loaded,
                    "config_resolved": config_resolved,
                    "ready": ready,
                    "required_num_gpus": prepared.required_num_gpus,
                },
            )
        except BaseException as error:
            _send_control(
                control,
                {
                    "type": "FAIL",
                    "phase": "prepare",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
            return 1

        line = reader.readline()
        if not line:
            return 1
        command = json.loads(line)
        if command.get("type") == "CANCEL":
            return 0
        if command.get("type") != "ASSIGN":
            raise ValueError(f"expected ASSIGN or CANCEL, got {command!r}")

        normalized_gpu_indices = _validated_gpu_assignment(
            command.get("gpu_indices"), args.prepared_num_gpus
        )
        if cuda_is_initialized():
            raise RuntimeError("CUDA was initialized before late GPU assignment")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(normalized_gpu_indices)
        if cuda_is_initialized():
            raise RuntimeError("setting CUDA_VISIBLE_DEVICES initialized CUDA")
        actual_gpu_uuids = _validated_physical_gpu_uuids(
            command.get("gpu_uuids"),
            physical_cuda_uuids(args.prepared_num_gpus),
            args.prepared_num_gpus,
        )
        if cuda_is_initialized():
            raise RuntimeError("physical GPU identity validation initialized a CUDA context")

        gpu_started = time.time()
        _send_control(
            control,
            {
                "type": "RUNNING_GPU",
                "gpu_started": gpu_started,
                "physical_gpu_uuids": actual_gpu_uuids,
            },
        )
        try:
            result = run_prepared_kernel_bench(
                prepared,
                warmup=args.warmup,
                repeat=args.repeat,
                timer=args.timer,
                rounds=args.rounds,
                cooldown=args.cooldown,
            )
            # RESULT is the GPU-ownership handoff. Complete all device work and
            # return cached allocations before telling the scheduler to reuse
            # the physical card; subsequent process teardown is host-only.
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except SkipTest as error:
            result = {
                "kernel": args.kernel,
                "label": args.config,
                "status": "SKIP",
                "reason": str(error),
            }
        except BaseException as error:
            _send_control(
                control,
                {
                    "type": "FAIL",
                    "phase": "gpu",
                    "gpu_started": gpu_started,
                    "gpu_finished": time.time(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
            return 1

        _send_control(
            control,
            {
                "type": "RESULT",
                "gpu_finished": time.time(),
                "result": result,
            },
        )
        return 0
    except BaseException as error:
        try:
            _send_control(
                control,
                {
                    "type": "FAIL",
                    "phase": "protocol",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
        except OSError:
            pass
        return 1
    finally:
        reader.close()
        control.close()


def main():
    child_started = time.time()
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
    parser.add_argument("--cc", type=int, default=None, help="Compute capability filter")
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
        choices=("event", "proton", "cudagraph_proton", "kineto", "megamoe"),
        default=None,
        help="Override the kernel module's benchmark timer: 'event' = do_bench, "
        "'proton' = do_bench_proton, 'cudagraph_proton' = "
        "do_bench_cudagraph_proton [NVIDIA], 'kineto' = distributed full GPU "
        "activity span, 'megamoe' = DeepGEMM bench_kineto protocol for MegaMoE",
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
        help=(
            "Seconds before every implementation in every round "
            "(default: runner protocol)"
        ),
    )
    parser.add_argument("--prepared-control-fd", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--prepared-num-gpus", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepared_num_gpus < 1:
        print("ERROR: --prepared-num-gpus must be >= 1", file=sys.stderr)
        sys.exit(2)

    if args.json or args.json_file:
        os.environ["TIRX_BENCH_JSON"] = "1"

    if args.prepared_control_fd is not None:
        sys.exit(_prepared_child_main(args, child_started=child_started))

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
        if args.cc is not None and mod.KERNEL_META.get("compute_capability") != args.cc:
            print(
                f"ERROR: kernel '{args.kernel}' compute_capability="
                f"{mod.KERNEL_META.get('compute_capability')} != filter {args.cc}",
                file=sys.stderr,
            )
            sys.exit(1)
        all_kernels = {args.kernel: mod}
    else:
        all_kernels = discover_kernels(min_compute_capability=args.cc)

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
