# Retarget SM100-only PTX with equivalent sequences

**Symptoms:** `ptxas_feature_not_supported`, `redux_f32_sm110`, `packed_cvt_sm110`

## Symptom

An SM100a kernel traces and lowers for another architecture, but ptxas rejects
an otherwise well-typed instruction.  CUDA 13.1 produced two concrete Thor
failures while compiling for `sm_110a`:

- `Instruction 'redux.f32' not supported on .target 'sm_110a'` for
  `redux.sync.max.NaN.f32`.
- `Unexpected instruction types specified for 'cvt'` for
  `cvt.rn.bf16x2.e4m3x2`.

## What to change

Keep the certified one-instruction path when the effective prepare target is
`sm_100a`.  For an explicitly retargeted architecture, use a sequence whose
individual instructions are accepted by that target:

- expand warp max reduction to five full-warp
  `shfl.sync.bfly.b32` + `max.NaN.f32` pairs;
- widen packed E4M3 through `cvt.rn.f16x2.e4m3x2`, unpack it, widen each F16
  to F32, then repack with `cvt.rn.bf16x2.f32`.

Put the selection in a shared traced idiom so all callers retain the native
SM100a spelling and a retarget cannot accidentally emit an uncertified opcode.

## Rationale

The warp sequence preserves full-mask reduction and NaN propagation.  Every
finite E4M3 value is exactly representable in F16, F32, and BF16, so the
conversion chain introduces no intermediate rounding beyond the native result.
On a 20-SM Thor, the substitutions removed the ptxas errors and the affected
kernels completed real launches.  Correctness remains a separate requirement
when an upstream B200-only reference refuses SM11.0.

## Boundary

Do not broaden an instruction table's certified architecture merely because a
related mnemonic exists on the new target.  Do not count successful assembly
or launch-only testing as correctness, and do not replace the native B200 path
without performance evidence.

## Verification

Check both traced branches: SM100a must contain the native instruction and
`sm_110a` must contain only the fallback sequence.  Compile the actual failing
kernel with ptxas for `sm_110a`, launch it on Thor, and run its registered
correctness case whenever the reference implementation supports that device.
