# Keep acquire polling control flow exact

**Symptoms:** `polling_loop_hang`, `acquire_load`, `device_function_boundary`, `unstable_benchmark_ratio`

## Symptom

A source helper containing an acquire load followed by a sleep-and-retry loop,
where replacing the arbitrary source call appears to require flattening or
otherwise restructuring the loop.

## What to change

Preserve the exact acquire scope, retry condition, sleep, and reload, whichever
boundary carries them. Spell the sequence directly in the caller, with no source
call and no extra memory operation:

```python
value: T.int32
T.ptx.ld.acquire.sys.global_.b32(value, flag.ptr_to([0]))
while value < 0:
    T.cuda.nano_sleep(40)
    T.ptx.ld.acquire.sys.global_.b32(value, flag.ptr_to([0]))
```

A private scalar-return PrimFunc carrying the same statements is the closest
compiler-visible replacement for the source helper, and it keeps the loop behind
a reusable function boundary:

```python
@T.prim_func(private=True)
def while_ld_global_acquire(addr_h: T.handle("int32", "global")) -> T.int32:
    addr = T.decl_buffer((1,), "int32", data=addr_h, scope="global")
    value: T.int32
    T.ptx.ld.acquire.sys.global_.b32(value, addr.ptr_to([0]))
    while value < 0:
        T.cuda.nano_sleep(40)
        T.ptx.ld.acquire.sys.global_.b32(value, addr.ptr_to([0]))
    return value
```

Use that form only when the active backend compiles private device calls, and
bind the helper and the entrypoint to the same CUDA target.

## Rationale

For one distributed reduce-scatter polling loop, the typed private helper and
the source helper produced byte-identical loadable SASS and kernel metadata. TP1
source/private timing was approximately 590.430/590.049 us, with a later final
run at 590.412 us; TP4 source/private timing was 277.541/278.789 us (ratio
0.9955). TP1 and TP4 correctness both completed without permanent spin. The
function boundary therefore costs nothing; what matters is that the acquire,
condition, sleep, and reload survive unchanged.

## Boundary

As of 2026-08-18 the sibling TVM pipeline cannot compile the private helper: a
scalar return is treated as an invalid device-kernel return, and a void variant
returning through a local pointer lacks a device id. Spell the sequence in the
caller until private device calls carry a supported target/device contract.

Multi-reference TP4 campaigns were order-sensitive and unstable, so do not
replace paired fixed-topology A/B or final-binary comparison with an unpaired
aggregate. Stop immediately on a hang, an acquire-scope change, different final
control flow, or a reproducible ratio at or below the gate.

## Verification

Compare final SASS and kernel metadata against the source-helper build, and run
correctness at every topology without a permanent spin.
