# Match block-scale MMA spelling to the certified target

**Symptoms:** `ptxas_feature_not_supported`, `scale_vec_target_mismatch`, `block32_spelling`, `collector_modifier_mismatch`

## Symptom

A block-scale MMA that constructs and dispatches cleanly in the IR, then fails
in ptxas with a message naming the modifier and the target, such as `Feature
'.scale_vec::2X' not supported on .target 'sm_100f'`.

The inverse can fail before assembly: the typed PTX table cannot represent a
source opcode because it requires collector B whenever collector A is present,
even though the target no-block-size block-scale form permits collector A by
itself.

## What to change

Read the table entry's certified architecture and put both the entry attribute
and the compile target on it. The typed PTX table exposes
`kind::mxf4.block_scale.scale_vec::2X` and certifies that spelling for
`sm_100a`.

```python
@K.kernel(warps=..., arch="sm_100a")  # matches the table's cert_arch
```

Do not restore a source-string `.block32` wrapper to escape the table.

Match modifier optionality as well as the architecture. For the SM107
activation-stationary no-block-size form, collector A is required and collector
B is independently optional. Do not synthesize a collector-B action when the
source uses collector A alone.

```python
K.ptx[
    "tcgen05.mma.cta_group::1.kind::mxf8f6f4"
    ".block_scale.collector::a::discard"
](d, a, b, idesc, sfa, sfb, K.ptx.pred(enable_d))
```

## Rationale

For one paged FP4 MQA kernel, changing both targets together retained the native
typed instruction and made the previously failing correctness case pass.
Successful IR dispatch alone does not prove the target accepts the modifier.

In the no-block-size SM107 family, making collector B optional added 48 legal
typed variants (762,023 to 762,071). The collector-A-only helper assembled with
CUDA 13.4 ptxas for `sm_107a`, the complete PTX certification suite passed, and
the FP8 collector-A-only plus collector-B fill/last-use correctness
specializations passed full-allocation bitwise comparison. The production
performance matrix was unaffected because it selected a separate FP4 block16
form.

## Boundary

This applies where the instruction genuinely requires an architecture-specific
feature. Moving a whole kernel family onto the architecture-specific target to
satisfy one instruction is a separate decision with its own portability cost.

Do not transfer the optionality across grammar families. Explicit
scale-vector/block qualifiers have their own tables, and collector B without
collector A remains invalid. The collector-A-only evidence applies to the
SM107-certified no-block-size block-scale family.

## Verification

Check the table entry and the final ptxas target together, and confirm
the native typed instruction survives into the generated PTX. Assemble the
actual generated helper through nvcc and ptxas rather than validating only its
rendered text; cover collector-A-only and collector-A-plus-B variants, then run
the correctness specializations that select each form.
