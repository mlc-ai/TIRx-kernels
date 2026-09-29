# Choose the L2 eviction policy by reuse distance

**Symptoms:** `excess_dram_read`, `low_dram_throughput`, `cold_cache_regression`, `unsaturated_bandwidth`

## Symptom

DRAM read traffic well above the reference's while every request-side counter
matches: same global load instructions, same requests, same sectors, same
stores, same DRAM writes. L2 read sectors only slightly higher, but nearly all
of the extra ones miss.

## What to change

Bind the TMA cache hint to each tensor's reuse distance instead of using one
policy for every descriptor. The question is how many CTAs read the same bytes:

- read once by the CTA that owns it and never again -- evict-first, so it does
  not sit in L2 displacing something that will be read again;
- read by every CTA that carries the row -- evict-last, so the later CTAs hit.

```python
# before: one hint for every descriptor, so the streamed tensor and the
# reused one compete on equal terms.
_HINT = T.uint64(0x14F0000000000000)

# after: name the policy and bind it to the reuse.
_EVICT_FIRST = T.uint64(0x12F0000000000000)  # this CTA reads it once
_EVICT_LAST = T.uint64(0x14F0000000000000)  # every CTA carrying this row reads it

T.ptx[_TMA_G2S_CACHE](..., _EVICT_FIRST)  # the streamed block
T.ptx[_TMA_GATHER4_CACHE](..., _EVICT_LAST)  # the shared rows
```

Getting the pairing backwards is easy and silent: both constants differ in one
nibble, both compile, and every output value is unchanged.

## Rationale

Reversed, the streamed tensor stays resident and pushes the reused one out, so
each of the later CTAs re-fetches from DRAM what should have been an L2 hit.
Measured with stores already coalesced and DRAM writes equal at 464.6 against
463.4 MB, the port read 140.3 MB where the reference read 60.9 MB. The 2.41M
extra L2 read sectors times 32 B is 77 MB, essentially all of the 79 MB gap:
the extra L2 reads all missed. Correcting the pairing took the two worst shapes
from 0.9305 to 1.0227 and 0.9327 to 1.0584.

Cache hit rate alone points the wrong way here. The port's L1 and L2 hit rates
were *higher* than the reference's while its DRAM traffic was worse, because a
higher hit rate over far more requests still leaves more misses.

## Boundary

Worth distinguishing only when one tensor genuinely has cross-CTA reuse. With
nothing reused, uniform evict-first is right; where the whole working set fits,
the policies stop mattering. Reuse distance is a property of the schedule, not
of the tensor: the same tensor read by one CTA per block in one dispatch and by
K CTAs in another wants different policies in the two.

## Verification

Diff `dram__bytes_read`, `dram__bytes_write` and `lts__t_sectors_op_read`
against the reference. Establish first that the store side matches -- equal
store sectors and equal DRAM writes localize the whole difference to the read
side before any code changes.
