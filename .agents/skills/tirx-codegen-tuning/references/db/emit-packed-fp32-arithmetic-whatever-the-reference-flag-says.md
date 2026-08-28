# Emit packed FP32 arithmetic whatever the reference flag says

**Symptoms:** `instruction_count_gap`, `local_memory_traffic`, `slow_epilogue`, `register_spill`

## Symptom

A reference exposes a `vectorized_f32`-style switch and the port mirrors it, so
half the specializations lower their epilogue arithmetic scalar. On those
specializations the reference retires roughly a quarter of the port's FP32
instruction count, which no scalar lowering can reach.

## What to change

Build the pairwise arithmetic helpers in packed form unconditionally. The flag
governs which intrinsics the reference's *author* writes, not what its machine
executes: its compiler pairs the scalar operations anyway.

```python
# before: the port mirrors the reference's source-level switch.
ops = _arithmetic(vectorized_f32, packed_pair)

# after: the packed form on every specialization.
ops = _arithmetic(True, packed_pair)


def _binary(mnemonic, out, left, right):
    K.ptx[f"{mnemonic}.rn.f32x2"](
        packed,
        K.cuda.make_float2(left[0], left[1]),
        K.cuda.make_float2(right[0], right[1]),
    )
    K.ptx["mov.b64"](out[0], out[1], packed)
```

Keep the approximations and comparisons per-element -- `ex2`, `rcp`, `tanh`,
`setp` have no packed form. For a difference, use the packed `sub` directly:
negating operands ahead of `make_float2` pays one scalar `FADD` per operand,
because the negation does not fold across the pack.

```python
# before: two scalar negations per pair survive into SASS.
K.ptx["add.rn.f32x2"](packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(-b0, -b1))
# after: the packed sub folds them.
K.ptx["sub.rn.f32x2"](packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1))
```

## Rationale

Each half of `mul.rn.f32x2` / `add.rn.f32x2` / `sub.rn.f32x2` rounds exactly
as its scalar sibling, and `sub` equals `neg` plus `add` bit for bit, so the
rewrite is numerically exact and needs no tolerance argument. Folding the
negation into the packed `sub` removed about 520K dynamic scalar `FADD` on one
latency-bound shape; it measured timing-neutral there only because that chain
sat in stall shadow, and it survived the complete correctness matrix and the
final performance-matrix winner. In one block-scaled MoE
grouped GEMM every activation family with real arithmetic depth moved above
parity at once -- 0.864 to 1.224, 0.987 to 1.120, 0.955 to 1.120 -- and the
worst row of the whole port stopped being the worst.

The second effect is larger than the instruction count suggests: the stack frame
went from 176 bytes to 0, which removed the local traffic a profile had already
flagged as 42.84% of L1TEX sectors against the reference's 27.20%. Halving the
number of value-carrying registers in a wide epilogue is what buys that, not the
issue slots.

## Boundary

Only for element-wise pairs that are genuinely independent. A packed operation
ties its two results to one 64-bit register pair, and where a later consumer
wants the halves apart that constraint can cost more moves than the pairing
saves.

## Verification

Compare packed and scalar FP opcode counts on both sides, and check the stack
frame and local-memory traffic as well as the instruction total -- the register
effect is usually the larger half.
