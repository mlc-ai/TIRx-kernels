# Task and hardware layout

Kernel source lives at `tirx_kernels/<task_set>/<task>/<device>/`, for example
`tirx_kernels/basic/rmsnorm/b200/`. Task sets such as `basic`, `flashinfer`, and
`deepgemm` are sibling directories inside the root-level `tirx_kernels/` package.
A task has one home even when its implementations have different origins.
RMSNorm implementations share `tirx_kernels/basic/rmsnorm/`; the fast.cu NVFP4
GEMM shares `tirx_kernels/basic/nvfp4_gemm/` with the B200 implementation.
Former curated implementations live in their actual tasks.

The resulting layout is illustrated below; the README lists every kernel and
links to its implementation.

```text
tirx-kernels/
├── tirx_kernels/
│   ├── __init__.py
│   ├── basic/
│   │   ├── _shared/
│   │   ├── rmsnorm/b200/
│   │   │   ├── rmsnorm.py
│   │   │   └── flashinfer_rmsnorm.py
│   │   ├── nvfp4_gemm/
│   │   │   ├── b200/nvfp4_gemm.py
│   │   │   └── gb300/nvfp4_gemm_gb300.py
│   │   ├── kda_forward/b200/
│   │   ├── kda_backward_packed/b200/
│   │   └── ...
│   ├── flashinfer/
│   │   ├── _shared/
│   │   ├── kda_decode/b200/
│   │   │   ├── kda_decode_multishape.py
│   │   │   ├── recurrent_kda_decode_grouped.py
│   │   │   └── recurrent_kda_decode_one_warp.py
│   │   └── ...
│   ├── cudnn/<task>/<device>/
│   ├── deepep/<task>/<device>/
│   ├── deepgemm/<task>/<device>/
│   ├── flashattention/<task>/<device>/
│   ├── flashmla/<task>/<device>/
│   ├── msa/<task>/<device>/
│   ├── tirx_lite/
│   └── bench/
│       ├── __main__.py
│       ├── test.py
│       ├── run.py
│       ├── suite.py
│       ├── registry.py
│       ├── runner.py
│       ├── config/<task_set>/<task>/<kernel>.yaml
│       └── ...
├── tests/
└── docs/
```

Multiple kernel files may coexist in a hardware directory. There is no required
`kernel.py`, shared entry point, dispatcher, or implementation manifest. The
existing names in `KERNEL_META`, Python `CONFIGS` / `BENCH_CONFIGS`, correctness
functions, benchmark functions, and exact architecture allowlists remain intact.
The directory names `b200`, `gb300`, and `rubin` identify the primary target, not a
replacement for those allowlists. Shared implementation helpers live in
`tirx_kernels/<task_set>/_shared/` or in private directories alongside their owning kernel.

A task may contain `definition.json`, `workloads.json`, and `tirx_kernels.bench.py` when those
files already exist. They are not required for discovering the existing Python
kernels. This directory refactor does not synthesize metadata, convert the
existing configuration formats, or introduce a new FlashInfer-Bench adapter.

`tirx_kernels/bench/` contains the former correctness CLI, single-kernel benchmark CLI,
regression suite, registry, runner, and remote client. The commands delegate to
the original implementations and keep their existing flags:

```bash
python -m tirx_kernels.bench list
python -m tirx_kernels.bench test --kernel rmsnorm --config hs128_bs32
python -m tirx_kernels.bench run --kernel rmsnorm --config hs128_bs32
python -m tirx_kernels.bench suite --check-imports
python -m tirx_kernels.bench suite --server http://127.0.0.1:8901
```

Suite YAML remains under `tirx_kernels/bench/config/<task_set>/<task>/`. Historical baseline
JSON and Markdown retain their original measurements and provenance; moving them
does not promote a new baseline. Framework unit tests remain in `tests/`.

Python imports use `tirx_kernels.<task_set>.<task>.<device>.<module>` for kernels,
`tirx_kernels.tirx_lite` for the substrate, and `tirx_kernels.bench` for the
harness. Source checkouts and wheels use the same package layout. The task sets
live inside `tirx_kernels`, so they do not shadow installed reference libraries
such as `flashinfer`, `cudnn`, or `deepgemm`. Optional references remain lazy imports.

Remote archives include `tirx_kernels/`, excluding caches and baseline artifacts.
The source fingerprint covers the repository tree; its result field name is
retained for report compatibility. Paired A/B runs can still extract revisions
from before the migration. Their kernel source and substrate stay in the old
layout while import aliases share the current harness; stable public names
locate the current benchmark contract.

New task sets under `tirx_kernels/` are discovered automatically by packaging,
the registry, and remote archives.
