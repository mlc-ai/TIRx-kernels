# Lower specialized global accesses explicitly

**Symptoms:** `low_level_ir_contract_failure`, `config_specific_buffer_load`, `config_specific_buffer_store`

## Symptom

Contract violations that appear in some configurations and not others:
specialization hides raw global `BufferLoad` and `BufferStore` nodes from the
commonly exercised shape while leaving them in a less frequent branch.

## What to change

Express the access with the kernel's typed `ld.global` / `st.global` helper, and
preserve signed values through bit reinterpretation rather than conversion.

```python
# before: reached only in a less frequent specialization branch.
if not FAST_PATH:
    dst[index] = src[index]

# after: the same typed helpers the fast path already uses.
if not FAST_PATH:
    bits = _load_u32(src, index)
    _store_u32(dst, index, bits)
```

## Rationale

Configuration-wide inspection exposed this in 82 radix-top-k configurations and
all four combine configurations. Replacing only the raw accesses with existing
typed PTX helpers left zero violations across the complete 234-config radix and
four-config matrices; all 86 previously rejected configurations then passed
their upstream oracles under a 16-worker run.

## Verification

Inspect every public configuration after specialization; validating only one
default `get_kernel()` result is insufficient.
