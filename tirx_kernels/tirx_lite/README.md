# tirx-lite authoring guide

tirx-lite (txl) is the canonical source language for kernels in this package. It is a
traced, PTX-level DSL over TIRx:

```python
# Do not enable `from __future__ import annotations` in a tirx-lite module. tirx-lite
# reads live annotations while tracing the function at decoration time.
import tirx_kernels.tirx_lite as txl


@txl.kernel(warps=1, arch="sm_100a", grid=False)
def zero(out: txl.gptr(txl.f32)):
    txl.ptx.st.global_.f32(out.ptr_to([0]), txl.float32(0))
```

`@txl.kernel` returns a `txl.Kernel`. Its principal views are:

- `zero.func`: the pre-lowering TIRx `PrimFunc` used by analysis tools and
  package runners;
- `zero.mod`: an `IRModule` containing that function;
- `zero.compile()`: a runnable module compiled through the TIRx pipeline.

## Authoring contract

- Kernel entries use `@txl.kernel`; parser entry points and raw TIRx builder
  forwarding are not part of the author-facing API.
- `txl.gptr`, `txl.TensorMap`, and scalar dtype annotations define the entry ABI.
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
  the resulting `PrimFunc` points to the `@txl.kernel` declaration.

You can learn tirx-lite APIs and complete implementation patterns from the canonical
modules under the root-level task sets. Do not copy API spellings from historical TIRx
parser kernels.

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
