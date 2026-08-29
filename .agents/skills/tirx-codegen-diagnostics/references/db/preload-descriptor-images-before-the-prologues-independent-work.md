# Preload descriptor images before the prologue's independent work

**Symptoms:** `serialized_prologue`, `fixed_overhead`, `slow_small_shape`, `long_scoreboard`

## Symptom

A descriptor-publishing prologue runs measurably longer than the reference's
(a fixed per-launch cost the short shapes cannot amortize), and its publish
phase reads 128-byte host-encoded TensorMap images through global pointers at
the point of use. The reference reads the same images from kernel-parameter
space, which costs nothing.

## What to change

K's PTX surface has no parameter-space load, so a port passes the reference's
grid-constant descriptors as small global buffers. Do not read them inside the
publish loop: issue the image loads into registers at kernel entry, on the
elected lane of the warp that owns each array, run the prologue's unrelated
phase, then publish and patch from the registers. One load per array serves
every per-sequence slot.

```python
# before: each slot's publish re-reads the image through cold global memory.
with K.If(warp == array_index), K.Then():
    with K.If(_elected()), K.Then():
        with K.serial(n_batch) as batch:
            _copy_image_global_to_slot(base_map, slot(batch))   # 4x ld.global.v4.b64
            _patch_slot(slot(batch), batch)

# after: the image is register-resident before the ordering phase begins.
payload = K.alloc_local((16,), "uint64")
with K.If(warp == array_index), K.Then():
    with K.If(_elected()), K.Then():
        for group in range(4):
            K.ptx.ld.global_.v4.b64(
                payload[group * 4],
                payload[group * 4 + 1],
                payload[group * 4 + 2],
                payload[group * 4 + 3],
                _image_ptr(base_map, group * 32),
            )
...  # the work-ordering phase, independent of the images
with K.If(warp == array_index), K.Then():
    with K.If(_elected()), K.Then():
        with K.serial(n_batch) as batch:
            _store_image_from_registers(slot(batch), payload)
            _patch_slot(slot(batch), batch)
```

## Rationale

The image buffers are written once by the host and first touched here, so the
point-of-use reads are cold DRAM chains serialized into the publish phase, and
a per-slot copy multiplies them by the batch count. Preloading moved the
measured prologue from 5.4us to 4.6us, into the reference's 4.4-4.9us band, and
the shortest shape from 0.985x to 0.988x -- margin the complete matrix needed.
The registers cost thirty-two 32-bit slots on one lane, paid only until the
publish.

## Boundary

The payload lives in the loading lane's registers, so the same elected lane
must load and publish. The covering phase must not write the images -- given
they are host-written inputs, any prologue phase qualifies.

## Verification

Measure the prologue kernel's duration in isolation before and after; the gain
then appears only on the shortest shapes of the timed two-launch closure.
