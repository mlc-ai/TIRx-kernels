# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Route the existing correctness, single-kernel and suite CLIs."""

import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description="TIRx kernel tests and benchmarks")
    parser.add_argument("command", choices=("test", "run", "suite", "list"))
    args = parser.parse_args(sys.argv[1:2])
    if args.command == "list":
        from tirx_kernels.bench.registry import kernel_index

        for name, record in sorted(kernel_index(strict=True).items()):
            print(f"{name}: {record.module_name}")
        return
    sys.argv = [f"{sys.argv[0]} {args.command}", *sys.argv[2:]]
    importlib.import_module(f"tirx_kernels.bench.{args.command}").main()


if __name__ == "__main__":
    main()
