# Kern authoring guide

Kern is the canonical source language for kernels in this package. It is a
traced, PTX-level DSL over TIRx:

```python
# Do not enable `from __future__ import annotations` in a Kern module. Kern
# reads live annotations while tracing the function at decoration time.
import tirx_kernels.kern as K


@K.kernel(warps=1, arch="sm_100a", grid=False)
def zero(out: K.gptr(K.f32)):
    K.ptx.st.global_.f32(out.ptr_to([0]), K.float32(0))
```

`@K.kernel` returns a `K.Kernel`. Its principal views are:

- `zero.func`: the pre-lowering TIRx `PrimFunc` used by analysis tools and
  package runners;
- `zero.mod`: an `IRModule` containing that function;
- `zero.compile()`: a runnable module compiled through the TIRx pipeline.

## Authoring contract

- Kernel entries use `@K.kernel`; parser entry points and raw TIRx builder
  forwarding are not part of the author-facing API.
- `K.gptr`, `K.TensorMap`, and scalar dtype annotations define the entry ABI.
- `K.cta_id`, `K.warp_id`, `K.lane_id`, and `K.thread_id` are the entry-owned
  coordinates. `K.specialize()` defines named warp roles when the schedule is
  warp-specialized.
- Shared memory is owned by `K.smem_pool()`. PTX and CUDA instructions are
  spelled through `K.ptx` and `K.cuda`; higher-level reusable instruction
  sequences live under `K.idioms`.
- Every kernel is checked against the low-level IR contract when it is traced.
  Keep the default `check_ir=True`. `allowed_func_calls` is only for a task that
  explicitly owns a named runtime-call exception.

You can learn Kern APIs and complete implementation patterns from the canonical
modules under `tirx_kernels/`. Do not copy API spellings from historical TIRx
parser kernels.
