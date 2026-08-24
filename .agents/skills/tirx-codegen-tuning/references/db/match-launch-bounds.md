# Match launch bounds

**Symptoms:** `register_spill`, `register_budget_mismatch`, `local_memory_traffic`, `low_occupancy`

## Symptom

STL/LDL traffic or global rescheduling in a kernel whose reference uses more
registers; realized allocation capped well below the reference's.

## What to change

Set `tirx.launch_bounds_min_blocks_per_sm` from the reference kernel's realized
occupancy target, not from theoretical occupancy. It is a statement attribute
placed right after `T.device_entry()`, not a function attribute.

```python
T.device_entry()
T.attr({"tirx.launch_bounds_min_blocks_per_sm": 8})
```

Do not copy one minimum-block value across block-size families; select it from
the shape when the families differ.

```python
T.device_entry()
if ILP_ROWS == 4 and SEQ_LEN == 8:
    if USE_SMEM_V and NUM_HEADS >= 8:
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 9})
    else:
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 8})
```

## Rationale

The value becomes the second CUDA `__launch_bounds__` argument and imposes a
hard ptxas register budget: roughly 65536 registers divided by (threads per CTA
times the bound), rounded down to the allocation granularity. One measured
512-thread quantization kernel was capped at 32 registers with a bound of 4
while its reference ran at about 50; a bound of 2 restored parity.

In one measured selector, a 160-thread family used nine minimum blocks and a
representative FP16-state specialization moved from 53 to 40 registers, while
its 288-thread family used one minimum block and stayed at 53 registers. Forcing
nine on the larger block cut its allocation to 32 registers before timing. The
shape-aware 9/1 selector cleared its five-workload boundary matrix at
1.003-1.028x.

In a measured 128-thread single-wave family, setting the minimum-block count to
one moved the allocation from 63 to 90 registers, reduced static SASS from 664
to 656 instructions, and introduced no spill. Two non-protocol paths reached
1.016x and 1.005x, while the dependency-protocol path improved from about
0.945x to 0.959x.

## Boundary

Treat a large allocation shift as a separate shape A/B even when neither variant
spills: ptxas can trade registers for recomputation and address instructions.

The inverse case matters too: a shorter instruction stream does not pay for a
lost resident CTA. One 128-thread decode rewrite cut static SASS from 1992/1984
to 1608 instructions yet raised allocation from 124 to 130 registers, reducing
the realizable CTA count from four to three. Pinning four blocks per SM produced
128 registers, zero spills, and 1568/1560 instructions; all 19 correctness
shapes passed, and a clean 45-round A/B measured 0.96957 and 0.96002
after/before on the two previously failing production rows. Use this only where
the reference allocation already sits below the exact occupancy boundary.

Lower allocation is not the objective. In the same family, a minimum-block
count of six later realized 80 registers with zero stack and local traffic, yet
regressed the gate from 0.981x to 0.976x. Match an occupancy mechanism, not the
reference's register number.

In a measured 384-thread SM100 block-scaled GEMM/SwiGLU family, bounds of one
and two left three representative ordinary, FP4, and FP32 specializations at
122-130 registers. A bound of three moved all three to 96 registers with zero
stack. The change passed 58 correctness configurations and a 23-row targeted
matrix at a 1.00109 minimum reference/TIRx ratio. Its supplemented 234-row
matrix passed at a 0.99768 minimum after replacing one frozen set of eight
host-interfered measurements. This result supports sweeping the resident-block
boundary across heterogeneous specializations; it does not establish that 96
registers or a bound of three is generally optimal.

Use `tirx.max_registers` when an exact per-specialization ceiling, rather than a
minimum resident-block target, is the demonstrated lever. The two contracts are
mutually exclusive.

## Verification

Compare resource usage, achieved occupancy, and dynamic local-memory traffic on
both sides; do not infer success from the declared launch bound alone.
