# Remove idempotent tail replays after proving vector coverage

**Symptoms:** `duplicate_global_traffic`, `instruction_count_bloat`, `vector_tail_overlap`, `bandwidth_bound`

## Symptom

A vector loop and a scalar remainder loop execute matching work over an
overlapping suffix. Dynamic global instruction counts exceed the unique tensor
extent even though every vector transaction is aligned and coalesced.

## What to change

Prove the vector loop's covered interval from its actual bound and stride, then
remove the scalar replay only when the operation is stateless and the dispatch
contract makes vector coverage complete.

```python
n_vec = d // vec_size
source_rem = d % (block_size * vec_size)

# idx < n_vec already covers [0, d) because d % vec_size == 0.
rem = 0 if prepare_cuda_arch() == "sm_110a" else source_rem
```

## Rationale

One activation port used `d=11008`, vector width eight, and 1024 threads. Its
vector loop covered all 1376 chunks, while a second loop replayed the final 2816
elements. Removing that replay on Thor reduced dynamic warp instructions from
61,538,304 to 41,091,072 and global load/store instructions from
2,146,304/1,073,152 to 704,512/352,256. All 90 million representative outputs
remained bitwise equal.

The timing gain was much smaller than the instruction reduction: the removed
input loads mostly hit cache, so a final 30-round standard suite moved only to
`1.0010x` versus the source. A compiler register-level sweep supplied additional
schedule stability, and a 30-pair counterbalanced diagnostic measured `1.0057x`.
Use traffic removal as mechanism evidence, not as a predicted speedup.

## Boundary

Do not remove a remainder merely because its count is nonzero. First prove that
the main loop reaches every scalar element; many kernels intentionally vectorize
only the largest block-aligned prefix. Replays are removable only for
idempotent, non-atomic, stateless stores with no externally visible intermediate
state. Preserve target paths whose source-compatibility contract requires the
original control structure.

Do not combine this rewrite with a smaller CTA or persistent grid without
independent evidence. On the measured Thor activation, 512 threads regressed to
`0.9006x`, and persistent grids from 20 through 640 CTAs measured only
`0.8637x-0.9160x`; CTA turnover was helping hide streaming latency.

## Verification

Check output equality on every activation and dtype sharing the loop, including
the exact vector boundary and replay-producing extents. Profile before and after
global instructions, L2 sectors, local traffic, registers, and total dynamic
instructions. Use the complete bench-suite gate for acceptance; when the margin
is sub-percent, add counterbalanced-order and cold-state diagnostics and record
their disagreement instead of hiding it.
