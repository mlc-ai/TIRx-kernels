# Buy overlap once the instruction count already matches

**Symptoms:** `instruction_parity_with_deficit`, `short_scoreboard`, `exposed_load_latency`, `unroll_no_effect`

## Symptom

The port issues as many instructions as the reference or fewer, and still takes
more cycles. Removing more instructions changes nothing, or loses.

## What to change

Stop shortening the instruction stream and start shortening the dependency
chain. Where a region drains one batch of long-latency results before issuing
the next, issue the whole batch first and wait once — accepting more
instructions, more live registers, and a larger static footprint to get the
transfers overlapping.

```python
# before: each pass drains before the next is issued.
for pass_idx in range(N_PASS):
    frag_a = _load(pass_idx, half=0)
    frag_b = _load(pass_idx, half=1)
    _wait()
    _consume(frag_a); _consume(frag_b)

# after: every load issued, one wait, then all consumers. The wait drains every
# outstanding load this thread has, so one after the last issue still orders
# all of them ahead of the first consumer.
frags = [_load(p, h) for p in range(N_PASS) for h in (0, 1)]
_wait()
for f in frags:
    _consume(f)
```

## Rationale

On a kernel at roughly a third of memory throughput and two fifths of compute,
five consecutive attempts to remove instructions bought nothing: four produced
byte-identical machine code, and the fifth removed two thirds of an address
chain for no change in the timed path. A re-profile then showed the port
executing 4.2% *fewer* instructions than the reference while taking 4.3% more
cycles — surplus arithmetic was hiding in stall shadow, so the metric that had
driven every expansion was pointing the wrong way.

Inverting the bet worked immediately. Issuing four tile reads before a single
wait, instead of draining one column pass at a time, raised static instructions
from 3552 to 3872 and stores from 18 to 34, and moved four measured shapes by
+0.0345, +0.0134, +0.0046 and +0.0030.

## Boundary

The trade is bounded by register live range, and the ceiling is
dispatch-specific rather than global: the same rewrite that gained on eight
shapes cost 13% on one and 3% on another, both of them the specializations whose
operand dtype left least room for the extra live fragments. Select the form from
the same compile-time predicate that decides the operand dtype or the load
program, and measure both arms.

Deeper pipelines follow the same split. Raising a load pipeline from two stages
to three cost nothing in occupancy on a register-limited kernel and still helped
one dtype while regressing another by 0.023.

## Verification

Compare executed instructions against the reference before choosing a
direction. Parity or a deficit there, with a cycle surplus, means the remaining
gap is stalls: read the stall breakdown and treat instruction-count reductions
as unlikely to pay.
