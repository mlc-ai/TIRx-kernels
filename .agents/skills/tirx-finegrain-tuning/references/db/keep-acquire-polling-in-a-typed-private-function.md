# Keep acquire polling in a typed private function boundary

**Symptoms:** `polling_loop_hang`, `acquire_load`, `device_function_boundary`, `unstable_benchmark_ratio`

## Symptom

A source helper containing an acquire load followed by a sleep-and-retry loop,
where replacing the arbitrary source call appears to require flattening the loop
into its caller.

## What to change

It does not. Express the loop as a private scalar-return PrimFunc containing the
typed acquire load, retry condition, sleep, and reload, and call that function
through the IR module.

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

Bind the helper and the entrypoint to the same CUDA target; the function
boundary preserves compiler-visible control flow without keeping an arbitrary
CUDA source string.

## Rationale

For one distributed reduce-scatter polling loop, the typed private helper and
the source helper produced byte-identical loadable SASS and kernel metadata. TP1
source/private timing was approximately 590.430/590.049 us, with a later final
run at 590.412 us; TP4 source/private timing was 277.541/278.789 us (ratio
0.9955). TP1 and TP4 correctness both completed without permanent spin.

## Boundary

Multi-reference TP4 campaigns were order-sensitive and unstable, so do not
replace paired fixed-topology A/B or final-binary comparison with an unpaired
aggregate. Stop immediately on a hang, an acquire-scope change, different final
control flow, or a reproducible ratio at or below the gate.

## Verification

Compare final SASS and kernel metadata against the source-helper build, and run
correctness at every topology without a permanent spin.
