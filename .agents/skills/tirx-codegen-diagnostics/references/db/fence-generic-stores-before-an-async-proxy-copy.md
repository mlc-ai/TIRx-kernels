# Fence generic stores before an async-proxy copy reads them

**Symptoms:** `bitwise_mismatch`, `nondeterministic_output`, `stale_shared_data`, `multi_tile_only_failure`

## Symptom

Outputs that are bitwise-exact on one-tile-per-CTA grids differ on a few rows of
a few tiles once CTAs process several tiles, and two identical launches of the
same binary disagree with each other. The affected rows are those whose per-row
scale words are written to shared memory with ordinary stores and then consumed
through the tensor-memory copy path.

## What to change

Put `fence.proxy.async.shared::cta` between the generic-proxy `st.shared` of
data that a `tcgen05.cp` (or any async-proxy consumer) will read and the
barrier arrive that publishes it.

```python
# before: the scale word is published to the matrix warp without a proxy fence.
K.ptx.st.shared.b32(sSFP.ptr_to([offset]), sf_word)
K.ptx[TMEM_ST](tmem_col, *p_words)
K.ptx.tcgen05.wait__st.sync.aligned()
P_full.arrive(stage)            # matrix warp then issues tcgen05.cp from sSFP

# after: order the generic stores before the async-proxy read.
K.ptx.st.shared.b32(sSFP.ptr_to([offset]), sf_word)
K.ptx.fence.proxy.async_.shared__cta()
K.ptx[TMEM_ST](tmem_col, *p_words)
K.ptx.tcgen05.wait__st.sync.aligned()
P_full.arrive(stage)
```

## Rationale

The mbarrier orders generic-proxy work; `tcgen05.cp` reads shared memory
through the async proxy and can observe the previous step's scale factors. The
race only surfaces when timing shifts, which here meant persistent grids with
several tiles per CTA. The reference kernel itself omitted the fence: on a
4096-key, 32-head NVFP4-P shape two back-to-back launches of the unpatched
reference differed on 100,050 and 133,944 of 16.8M elements, and its error
against an fp64 oracle on the disputed rows was the larger one. Adding the one
fence to the reference made it deterministic and bitwise-equal to the port,
which issues the same fence: 45 configurations then matched bit for bit. One
fence per softmax step was not measurable in the 74-row benchmark matrix.

## Boundary

Needed only when a generic store feeds an async-proxy reader (`tcgen05.cp`,
TMA store, bulk copy). Stores consumed by `tcgen05.st`/`tcgen05.ld` register
paths or by ordinary loads after an mbarrier do not need it. Do not treat a
bitwise-clean single-tile matrix as proof: the failing regime is multi-tile.

## Verification

Launch the reference twice on a multi-tile persistent shape and `torch.equal`
the two outputs before using it as a bitwise oracle; then compare the port.
Include at least one configuration with several tiles per CTA for every
quantized-P mode in the correctness matrix.
