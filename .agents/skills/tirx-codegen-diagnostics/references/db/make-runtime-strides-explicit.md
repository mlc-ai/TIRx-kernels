# Make runtime strides explicit when a kernel owns no layout

**Symptoms:** `runtime_stride_ignored`, `contiguous_alias_lowering`, `combine_only_mismatch`, `split_buffer_misaddress`

## Symptom

A multi-dimensional buffer alias declared with a runtime stride lowers to
contiguous addressing: generated CUDA multiplies by the extent rather than by
the stride variable, and only the stage that reads the strided buffer
mismatches.

## What to change

Where kernel policy forbids layouts, do not expect a `strides=` declaration on a
default-layout alias to preserve runtime-strided indexing through every
lowering. Flatten the alias and put the runtime stride in the pointer arithmetic
that feeds the native load or store.

```python
# before: a two-dimensional alias carrying a declared runtime stride; the
# generated address collapses to `split * 8`.
value = lse_2d[split, warp]

# after: a flat alias, with the runtime stride in the address expression.
value = lse_flat[split * stride_lse_accum_split + warp]
```

## Rationale

One split-combine kernel declared a two-dimensional LSE alias with runtime split
stride 128, but current CUDA addressed it as `split * 8`, while the
layout-bearing baseline emitted `split * stride_lse_accum_split`. Flattening the
buffer and addressing `split * stride_lse_accum_split + warp` explicitly
restored the minimal reproducer and all 15 sparse-decode configurations.

## Boundary

This is for buffers whose layout the kernel does not own. A buffer with
compile-time strides needs no such flattening.

## Verification

Read the generated CUDA address expression for the strided access and confirm
the stride variable appears in it; a literal extent in its place is the failure.
