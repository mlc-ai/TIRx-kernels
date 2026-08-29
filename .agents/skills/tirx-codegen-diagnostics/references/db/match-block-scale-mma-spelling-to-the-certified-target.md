# Match block-scale MMA spelling to the certified target

**Symptoms:** `ptxas_feature_not_supported`, `scale_vec_target_mismatch`, `block32_spelling`

## Symptom

A block-scale MMA that constructs and dispatches cleanly in the IR, then fails
in ptxas with a message naming the modifier and the target, such as `Feature
'.scale_vec::2X' not supported on .target 'sm_100f'`.

## What to change

Read the table entry's certified architecture and put both the entry attribute
and the compile target on it. The typed PTX table exposes
`kind::mxf4.block_scale.scale_vec::2X` and certifies that spelling for
`sm_100a`.

```python
@K.kernel(warps=..., arch="sm_100a")  # matches the table's cert_arch
```

Do not restore a source-string `.block32` wrapper to escape the table.

## Rationale

For one paged FP4 MQA kernel, changing both targets together retained the native
typed instruction and made the previously failing correctness case pass.
Successful IR dispatch alone does not prove the target accepts the modifier.

## Boundary

This applies where the instruction genuinely requires an architecture-specific
feature. Moving a whole kernel family onto the architecture-specific target to
satisfy one instruction is a separate decision with its own portability cost.

## Verification

Check the table's `cert_arch` and the final ptxas target together, and confirm
the native typed instruction survives into the generated PTX.
