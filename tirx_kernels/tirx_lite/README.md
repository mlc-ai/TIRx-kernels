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
modules under `tirx_kernels/`. Do not copy API spellings from historical TIRx
parser kernels.
