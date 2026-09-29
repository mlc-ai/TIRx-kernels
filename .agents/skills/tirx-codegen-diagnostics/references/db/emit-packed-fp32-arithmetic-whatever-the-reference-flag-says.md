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
    txl.ptx[f"{mnemonic}.rn.f32x2"](
        packed,
        txl.cuda.make_float2(left[0], left[1]),
        txl.cuda.make_float2(right[0], right[1]),
    )
    txl.ptx["mov.b64"](out[0], out[1], packed)
```

Keep the approximations and comparisons per-element -- `ex2`, `rcp`, `tanh`,
`setp` have no packed form. For a difference, use the packed `sub` directly:
negating operands ahead of `make_float2` pays one scalar `FADD` per operand,
because the negation does not fold across the pack.

```python
# before: two scalar negations per pair survive into SASS.
txl.ptx["add.rn.f32x2"](packed, txl.cuda.make_float2(a0, a1), txl.cuda.make_float2(-b0, -b1))
# after: the packed sub folds them.
txl.ptx["sub.rn.f32x2"](packed, txl.cuda.make_float2(a0, a1), txl.cuda.make_float2(b0, b1))
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

Pairing and register budgets can interact even without reducing the reported
stack frame. In a measured warp-specialized 64-row pipeline, paired independent
FP32 promotions alone changed pure GPU time from 53.380 to 53.239 us; also
raising the math role from 224 to 240 registers reduced it to 52.435 us, with
bitwise identical output. A different redistribution (96 producer, 240 math,
168 quantizer registers) reduced the stack to 248 bytes but ran at 53.052 us;
the faster combination reported 512 bytes. Use timing and the affected code
paths rather than the smallest stack figure to select the allocation.

An instruction reduction can also be correct but too small to address the
observed bottleneck. Pairing two independent accumulators that shared a
key-times-decay factor reduced scalar/packed FMA counts from 32/96 to 16/104
in one stable reduction and from 48/160 to 24/172 in another. Register, stack,
and local-load/store counts stayed unchanged. Six paired cases passed with all
outputs exactly equal. The strong cases improved only from 634 to 625 us and
6255 to 6219 us, while mixed distributions changed by less than 0.5%. This
limits the benefit of that particular pairing; it does not establish that
scalar arithmetic was the main stall source.

## Verification

Compare packed and scalar FP opcode counts on both sides, and check the stack
frame and local-memory traffic as well as the instruction total -- the register
effect is usually the larger half.
