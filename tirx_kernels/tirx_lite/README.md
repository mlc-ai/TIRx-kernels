# tirx-lite authoring guide

tirx-lite (txl) is the canonical source language for kernels in this package. It is a
traced, PTX-level DSL over TIRx:

```python
# Do not enable `from __future__ import annotations` in a tirx-lite module. tirx-lite
# reads live annotations while tracing the function at decoration time.
import tirx_kernels.tirx_lite as txl


@txl.kernel(launch=txl.cuda.LaunchConfig(grid=1, block=32), arch="sm_100a")
def zero(out: txl.gptr(txl.f32)):
    txl.ptx.st.global_.f32(out.ptr_to([0]), txl.float32(0))
```

`@txl.kernel` returns a `txl.Kernel`. Its principal views are:

- `zero.func`: the pre-lowering TIRx `Function` used by analysis tools and
  package runners;
- `zero.mod`: an `IRModule` containing that function;
- `zero.compile()`: a runnable module compiled through the TIRx pipeline.

## Authoring contract

- Kernel entries use `@txl.kernel`; parser entry points and raw TIRx builder
  forwarding are not part of the author-facing API.
- `txl.gptr`, `txl.TensorMap`, and scalar dtype annotations define the entry ABI.
- Ordinary tensors use `txl.alloc_tensor(...)` and `txl.decl_tensor(...)` with
  the default layout. `txl.alloc_local(...)` is the shorthand for
  `txl.alloc_tensor(..., scope="local")`; writable scalars use `txl.local_scalar`.
  Tensor memory uses raw columns and PTX operations rather than `scope="tmem"`.
- `txl.cta_id`, `txl.warp_id`, `txl.lane_id`, and `txl.thread_id` are the entry-owned
  coordinates. `txl.specialize()` defines named warp roles when the schedule is
  warp-specialized.
- Shared memory is owned by `txl.smem_pool()`. PTX and CUDA instructions are
  spelled through `txl.ptx` and `txl.cuda`; higher-level reusable instruction
  sequences live under `txl.idioms`.
- Every kernel is checked against the low-level IR contract when it is traced.
  Keep the default `check_ir=True`. `allowed_func_calls` is only for a task that
  explicitly owns a named runtime-call exception.
- Statements and instruction calls traced from file-backed Python code carry
  TIRx source spans pointing to the kernel or helper line that emitted them;
  the resulting `Function` points to the `@txl.kernel` declaration.

You can learn tirx-lite APIs and complete implementation patterns from the canonical
modules under `tirx_kernels/`. Do not copy API spellings from historical TIRx
parser kernels.

## Launch configuration

Use `launch=txl.cuda.LaunchConfig(grid=..., block=..., cluster=...)` and optional
`options=txl.cuda.KernelOptions(...)`. `grid` counts CTAs and `cluster` counts CTAs
per cluster; `block` is a static one-dimensional multiple of 32 in tirx-lite.
The raw TIRx API also supports three-dimensional blocks.

A factory can compute launch operands from the kernel's bound ABI parameters:

```python
@txl.kernel(
    launch=lambda p: txl.cuda.LaunchConfig(grid=(p["n"] + 127) // 128, block=128),
    options=txl.cuda.KernelOptions(min_blocks_per_sm=1),
)
def zero(out: txl.gptr(txl.f32), n: txl.i32):
    i = txl.cta_id() * 128 + txl.thread_id()
    with txl.If(i < n), txl.Then():
        txl.ptx.st.global_.f32(out.ptr_to([i]), txl.float32(0))
```

The no-argument coordinate helpers remain entry-owned. `cta_id()` returns a
scalar for a one-dimensional authored grid and an array for a two- or
three-dimensional grid. Explicit hardware coordinates are also available as
`txl.cuda.block_idx("x")`, `txl.cuda.cluster_cta_id("x")`, and the other finite
CUDA index calls. Coordinates do not supply launch extents.

## Selecting shared-memory descriptors

`SmemDescriptor` and `KDesc` implement TVM's `__tvm_ffi_object__` conversion
protocol, so an FFI-backed expression constructor such as `txl.Select` accepts
either wrapper directly. Inside a kernel, given compatible tile views and a
runtime condition `choose_a`:

```python
a_desc = a_tile.mma_desc(major="k", mma_k=16)
b_desc = b_tile.mma_desc(major="k", mma_k=16)
selected = txl.Select(choose_a, a_desc, b_desc)
```

The conversion uses the existing scalar accessor for each type:

| Wrapper | Construction | Explicit scalar accessor |
| --- | --- | --- |
| `SmemDescriptor` | `txl.SmemDescriptor()`, then `.init(...)` | `.desc` |
| `KDesc` | `tile.mma_desc(...)` or the first result of `tile.encode(...)` | `.value` |

Both accessors remain available, and either `Select` arm may use an explicit
scalar expression. Initialize both descriptors before evaluating the selection;
conversion reads their local storage and does not initialize or snapshot it.

`selected` is a `uint64` IR expression usable as a raw MMA descriptor operand.
It does not carry `KDesc.k_step` or `KDesc.major`. Both candidates must match the
consuming instruction's operand layout and format. To select a nonzero k-tile,
apply each descriptor's offset before selection, for a trace-time integer `kp`:

```python
a_desc, a_off = a_tile.encode(major="k", mma_k=16)
b_desc, b_off = b_tile.encode(major="k", mma_k=16)
selected = txl.Select(choose_a, a_desc + a_off(kp), b_desc + b_off(kp))
```
