# Allocate shared memory beyond the static ceiling explicitly

**Symptoms:** `illegal_memory_access`, `zero_dynamic_smem`, `smem_capacity_limit`

## Symptom

Runtime illegal-memory-access faults on arena accesses that read like a
transcription bug anywhere in the kernel; the launch reserves zero dynamic
shared bytes.

## What to change

A per-CTA arena above the 48 KB static limit cannot be a static buffer. Allocate
it from the shared-memory pool at the reference's alignment, keep every region
byte offset unchanged, and declare the dynamic-shared launch parameter on the
kernel.

```python
# before: fits under the 49152 B static ceiling.
arena = T.alloc_buffer((SMEM_TOTAL,), "uint8", scope="shared", align=1024)

# after: past the ceiling, so the arena comes from the pool at the same
# alignment, and every region byte offset is unchanged.
pool = T.SMEMPool()
arena = pool.alloc((SMEM_TOTAL,), "uint8", align=1024)
pool.commit()
```

Regions are carved as views onto that arena:

```python
s_q = T.decl_buffer(
    (64 * 128 * 2,), "float16", data=arena.data,
    scope="shared.dyn", byte_offset=SMEM_Q_OFF, align=1024,
)
```

The launch parameter is what actually backs the arena:

```python
LAUNCH_TAGS = ("blockIdx.x", "blockIdx.y", "threadIdx.x", "tirx.use_dyn_shared_memory")


def get_kernel(**kwargs):
    kernel = _kernel.specialize(**_specialization(kwargs))
    return kernel.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))
```

## Rationale

Omitting the dynamic-shared declaration is not a compile error. The launch
reserves zero dynamic bytes and every arena access faults at runtime.

Static versus dynamic is a declaration detail, not a configuration one. Measured
on both sides of one port, the L1/shared split was the same 65536 bytes either
way and the reference's carveout hint changed nothing. Scaffold-stage notes that
such a hint is "not needed" are assumptions until a counter is read; this one
sat unexamined while the port chased a latency deficit a wrong split could
plausibly have caused.

## Verification

Prove the path before writing the body: compile and run each specialization's
arena size through the real swizzled store and matrix-load shapes, and include
an oversized arena as a negative control so a passing result means something.
