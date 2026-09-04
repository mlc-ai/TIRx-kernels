# Frozen source references on Thor

DeepGEMM and FlashMLA need separate host adaptations and builds for `sm_110a`.
The canonical reference installer and non-Thor imports remain unchanged. Thor
workers require an explicit variant manifest; missing or invalid selection is a
reference infrastructure failure, not permission to omit the primary baseline.

| Reference | Frozen source commit | Allowed adaptation |
| --- | --- | --- |
| DeepGEMM | `559d79fb6994a58b8a15b4b93bf13ccc16edf247` | Select the original SM100 host schedule for capability 11.0, preserving the actual device pair used by the JIT target. |
| FlashMLA | `9241ae3ef9bac614dd25e45e507e089f888280e0` | Admit capability 11.0 in the host SM100 predicate; compile the unchanged translation-unit list for `compute_110a/sm_110a`; use the already pinned CUTLASS checkout. |

Use the same Python, PyTorch and CUDA toolkit environment as the worker. The
script expects the canonical pinned checkouts, including submodules, to have
already been installed by `scripts/install_reference_dependencies.py`.
For example, create a new isolated build with:

```bash
python scripts/setup_thor_source_references.py \
  --name deep-gemm --source-root /path/to/canonical/.reference-deps \
  --variant-root /path/to/new/deep-gemm-thor --build

python scripts/setup_thor_source_references.py \
  --name flash-mla --source-root /path/to/canonical/.reference-deps \
  --variant-root /path/to/new/flash-mla-thor --build
```

Omitting `--build` prepares the source only and does not create a usable runtime
manifest. Use `--build-prepared` with that same variant root to compile it later.
The script refuses to replace an existing destination, keeps all
tracked source files and clones pinned submodules locally. Both source builds
and binary scans hold `/tmp/tirx-kernels-gpu.lock` because CPU compilation can
contend with GPU timing through Thor's unified memory. CUDA toolkit CCCL headers
are supplied explicitly to NVCC. FlashMLA's actual build flags and extension
cubin listing must contain only the selected `sm_110a` target.

A retained private build can be registered without rebuilding:

```bash
python scripts/setup_thor_source_references.py --name deep-gemm \
  --variant-root /path/to/retained/deep-gemm \
  --register-existing --build-log /path/to/retained/actual-build.log
```

Registration verifies the frozen Git origin and commit, every tracked parent
file, pinned submodule revisions and unmodified tracked submodule files, the
exact permitted host patch, unchanged device files, include links, extension
hash, Python/PyTorch ABI, and retained build evidence. DeepGEMM uses the exact host compile/link commands
from its successful setuptools log; FlashMLA also requires retained Ninja flags
and the actual cubin listing. Legacy private DeepGEMM
copies may have omitted three unused TileLang Python files. Registration records
these omissions explicitly; new clones retain them. Submodule symlinks are
recorded with their resolved locations. These are total-tree differences, even
when the CUDA device files are unchanged. A manifest is build provenance, not
proof of numerical correctness or benchmark acceptance.

Apply the generated `tirx-thor-environment.json` to a **new worker before Python
starts**, preserving any other required `PYTHONPATH` entries. The principal keys
are `TIRX_DEEP_GEMM_VARIANT_MANIFEST` or `TIRX_FLASH_MLA_VARIANT_MANIFEST`, the
variant root on `PYTHONPATH`, and `TIRX_PREPARE_CUDA_ARCH=sm_110a`. DeepGEMM also
uses a separate JIT cache. FlashMLA's `FLASH_MLA_PATH` points to that same root.
The loader checks the imported extension's exact path and rejects switching
variants within a worker. It does not change `sys.path` or report a false GPU
identity. Existing `KERNEL_META.reference_requirements` Git checks still apply
to the imported checkout; the canonical installer's path check is separate.
Benchmark results include `reference_variant` with the selected manifest and
extension hashes when a variant was loaded.

The numerical contract remains explicit:

- DeepGEMM GEMMs retain the Thor mathematical checks and restore the original
  direct source comparison threshold. MQA and paged MQA restore source checks at
  their existing tolerance. HC retains its original `1e-8` oracle threshold;
  its observed source/oracle failure remains a failure pending separate review.
- MegaMoE admits Thor only for `num_processes=1`, retaining exact numerical
  output and integer statistics comparisons. Multiple processes are rejected
  before GPU allocation. The required workload matrix is unchanged.
- FlashMLA decode restores the original source output/LSE and exact scheduler
  metadata comparisons while retaining the Thor Torch oracle. Prefill checks
  all three public source outputs against the original oracle and TIRx on Thor.
  All existing per-output tolerances remain unchanged.

Run the full required correctness and benchmark matrices on the integrated
commit. Successful source import, source identity checks, or a diagnostic shape
alone do not admit a kernel or workload as passing.
