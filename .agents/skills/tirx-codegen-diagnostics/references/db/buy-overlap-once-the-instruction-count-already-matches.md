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

The same scheduling rule applies to independent special functions. Do not
serialize every element through the whole dependency chain when the reference
issues one operation class for the complete fragment before beginning the next
class.

```python
# before: every element waits on its own complete chain.
for i in K.unroll(FRAGMENT):
    exp[i] = _exp2(gate[i])
    denom[i] = 1.0 + exp[i]
    reciprocal[i] = _reciprocal(_round_if_required(denom[i]))
    out[i] = value[i] * gate[i] * reciprocal[i]

# after: expose independent operations within each dependency level.
for i in K.unroll(FRAGMENT):
    reciprocal[i] = _exp2(gate[i])
for i in K.unroll(FRAGMENT):
    reciprocal[i] = 1.0 + reciprocal[i]
for i in K.unroll(FRAGMENT):
    reciprocal[i] = _round_if_required(reciprocal[i])
for i in K.unroll(FRAGMENT):
    reciprocal[i] = _reciprocal(reciprocal[i])
for i in K.unroll(FRAGMENT):
    out[i] = value[i] * gate[i] * reciprocal[i]
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

A 32-value activation epilogue showed the same effect without memory loads.
Grouping all `ex2`, add, optional round-trip conversion, reciprocal, and final
multiply operations by dependency level matched the reference PTX order and
reduced one realized allocation from 69 to 64 registers without spill. On the
five directly comparable guard rows, the minimum ratio moved from 0.9909x to
0.9967x; two affected rows improved by 0.0084x and 0.0101x. The staged form
then survived the complete correctness matrix and the complete performance
matrix.

## Boundary

The staged width is itself the lever, and it can want the COMPLETE fragment.
On a scalar pass reading three shared tiles per token, staging eight tokens at a
time beat one token at a time on every required shape. Narrowing to four then
regressed every one of them (-0.0025, -0.0019, -0.0008, -0.0004, -0.0004) even
though it lowered the MIO pressure the staging had added, and widening to the
whole sixteen-token fragment -- 48 loads in flight -- was the best of the three,
taking four shapes stuck at 0.988x to 0.9907x, 0.9917x, 0.9921x and 0.9924x and
the matrix from five failing shapes to one. Sweep the width in both directions
before settling; the small-trip guidance that suits a global-load staging loop
is not general, and here the extra live range was worth paying for.

The transform is site-selective, and the site can be judged in advance. Of five
scalar shared-memory ladders staged the same way in one kernel, the three that
paid all fed a long arithmetic chain in a register-rich consumer warpgroup and
ran every iteration of the hot loop. The two that did not pay failed for
identifiable reasons: one fed tensor-memory stores rather than arithmetic and
sat behind a conditional so it did not run every iteration (measured -0.0008 and
-0.0007 on the two tightest shapes), and the other was already being scheduled
ahead of its consumers. Prefer ladders whose staged values feed a long dependent
chain, and check whether the region is even entered every iteration.

Removing genuinely redundant work in these loops loses. Unpacking each packed
pair once into a per-token array, in place of unpacking every word twice and
selecting a half, removes real converts and real selects that the generated code
was keeping (`SEL` 39 against the reference's zero). It was measured twice, once
before and once after an unrelated front-end fix, and lost both times -- on the
clean measurement it took the matrix from two failing shapes to six and cost the
worst shape 0.0125. The redundant arithmetic was sitting in stall shadow, and
removing it shortened the very chain the staging had been lengthening on purpose.

The trade is bounded by register live range, and the ceiling is
dispatch-specific rather than global: the same rewrite that gained on eight
shapes cost 13% on one and 3% on another, both of them the specializations whose
operand dtype left least room for the extra live fragments. Select the form from
the same compile-time predicate that decides the operand dtype or the load
program, and measure both arms.

Dependency-level staging also lengthens the lifetime of every intermediate in
the fragment. Keep it only when the generated schedule exposes independent
special-function work without introducing stack or local-memory traffic; a
larger source-level array is not evidence of more overlap by itself.

Deeper pipelines follow the same split. Raising a load pipeline from two stages
to three cost nothing in occupancy on a register-limited kernel and still helped
one dtype while regressing another by 0.023.

In a warp-specialized kernel, exclude the warp that issues the collectives from
this trade. That warp is typically on a small register budget, and three
independent attempts to buy overlap inside it all lost: fully unrolling its
matrix chains ran 4x slower, keeping one running descriptor per issue slot cost
2.4-3.0%, and straight-lining only its eight shortest chains -- two static
issues becoming four -- cost 6.9-10.7% on the wider of two compiled programs
while leaving the narrower one flat. The same kernel gained from the opposite
direction: removing a runtime multiply from each issue was worth 0.4-1.7%.
Spend the extra instructions in the consumer warps, and measure per compiled
program, since the cost of code growth there did not transfer between them.

## Verification

Compare executed instructions against the reference before choosing a
direction. Parity or a deficit there, with a cycle surplus, means the remaining
gap is stalls: read the stall breakdown and treat instruction-count reductions
as unlikely to pay.
