# Match the reference ptxas register-usage level

**Symptoms:** `register_spill`, `local_memory_traffic`, `register_budget_mismatch`, `sass_schedule_divergence`, `warp_retry_region`

## Symptom

Same key-instruction counts as the reference (`UTCHMMA`, `LDTM`, `STTM`,
`UTMALDG`, `MUFU`) but several times more `MOV.SPILL`/`R2UR.FILL` pairs,
`LDL.64`/`STL.64` of 64-bit descriptor packs in a 48-register issuer or
producer warp, and a larger stack frame, in a build that goes through the
nvcc path with its default `--register-usage-level=10`.

The same lever can matter without allocation traffic: paired SASS profiles may
show equal occupancy, register counts, and zero spill, yet different barrier or
long-scoreboard behavior and issue schedules across register-usage levels.

## What to change

Use the ptxas register-usage level the reference is built with (its native
default is 5) as the first candidate. When generated SASS or paired profiles
still show schedule divergence, sweep a compact level range and pin the measured
winner around the compile call. Restore the caller's environment exactly.

```python
_PTXAS_REG_LEVEL = "5"

previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _PTXAS_REG_LEVEL
try:
    executable = compile_kernel(func, cuda_compile_mode="nvcc")
finally:
    ...  # restore the previous value
```

## Rationale

Level 10 lets ptxas schedule aggressively; in narrow `setmaxnreg` roles that
stretches the lifetimes of `mov.b64` descriptor packs and uniform barrier and
coordinate values until they spill. One 16-warp persistent block-sparse
attention kernel (roles 192/80/48) went from 336 bytes of stack, 120 `LDL`,
99 `STL` and 268 uniform spill/fill pairs at level 10 to 88 bytes, 30 `LDL`,
27 `STL` and 23 pairs at level 5, the reference's own profile (96 bytes,
48/34/23). Bench-suite ratios moved from 1.0074x to 1.0667x on a 1.46 ms
streaming shape and from 1.0457x to 1.0717x on a 0.74 ms shape; the other
four required shapes gained 6-7% as well and outputs stayed bitwise identical.

In a second 16-warp block-sparse attention kernel, both implementations still
allocated 128 registers, but level 10 introduced a 56-byte stack and dynamic
local traffic that grew from 7,168 spill instructions on a short path to
1,970,176 on the deepest measured path; the reference had none. Level 5
removed the stack and all local spill traffic, and reduced static SASS from
2,248 to 2,072 instructions, exactly matching the reference count. On a
representative streaming path it also reduced synchronization try-waits from
14,170,782 to 11,061,280, close to the reference's 10,968,668. The complete
six-row bench-suite minimum/geometric-mean ratios moved from
0.9734x/0.9749x to 0.9926x/1.0055x, with all seven correctness configurations
remaining bitwise identical to the reference.

In a 12-warp persistent attention pipeline, levels 0 through 6 and 10 all used
151 registers with zero stack and local traffic, but produced different
schedules: levels 0-3 emitted 3,048 static instructions, levels 4-5 emitted
2,824, and levels 6/10 emitted 2,984. Level 2 beat the reference-default level 5
despite its larger static program, improving a four-shape targeted
minimum/geometric mean from 0.9393x/0.9681x to 0.9552x/0.9816x while preserving
bitwise correctness. The change helped but did not clear the gate; a separate
unsigned-index lowering fix was still required.

An independent 16-warp persistent paged-attention pipeline showed the zero-spill
case more strongly. Both builds used 128 registers, 148,480 dynamic shared bytes,
one CTA per SM, and zero stack or local traffic; the generated build was already
slightly shorter at 2,800 static SASS instructions against the reference's 2,808,
yet paired profiles showed extra uniform address/control instructions and higher
long-scoreboard exposure. A compact sweep was sharply non-monotonic: levels 2,
4, 5, and 6 produced targeted minimum/geometric-mean ratios of
0.9925x/0.9990x, 0.9695x/0.9810x, 0.9699x/0.9822x, and 0.9615x/0.9773x.
Pinning level 2 with exact environment restoration kept all four targeted rows
above 0.99x, and the complete ten-row suite cleared at a 0.9904x minimum and
0.9961x geometric mean; all seven correctness configurations remained bitwise
identical.

## Boundary

The sibling kernel of the same source family, which had zero spills at level 10,
measured level 5 as neutral (within 0.5%). Zero spills alone are therefore not a
reason to sweep. Without allocation traffic, require paired evidence of schedule
divergence such as barrier/scoreboard stalls or instruction-order differences.
Record the toolkit version and re-check this beta option after toolchain moves.

The register-usage-level response is non-monotonic. In a fixed three-shape
sweep around the reference level, levels 4, 5, and 6 produced targeted
minimum/geometric-mean ratios of 0.9904x/0.9997x, 0.9929x/1.0085x, and
0.9741x/0.9833x respectively; level 6 failed two rows. Match the reference as
the primary candidate, but measure adjacent levels instead of assuming that a
more aggressive setting is better.

## Verification

Compare stack bytes, static and dynamic `LDL`/`STL`/`MOV.SPILL`/`R2UR.FILL`
against the reference build at both levels, rerun the bitwise correctness
rows, and accept only on bench-suite timing.
