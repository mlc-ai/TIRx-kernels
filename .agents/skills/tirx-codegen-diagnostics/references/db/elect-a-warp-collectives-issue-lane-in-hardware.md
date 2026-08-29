# Elect a warp collective's issue lane in hardware

**Symptoms:** `branch_reconvergence`, `excess_control_instructions`, `warp_retry_region`, `instruction_parity_with_deficit`

## Symptom

A single-issue warp collective -- a matrix MMA, a commit -- already carries its
guard as `pred=` rather than as a branch around the block, and SASS still shows
a fresh election and a reconvergence around *every* issue:

```text
@P   ELECT P1, URZ, PT
     @!UP UTCHMMA  ...
@P   PLOP3.LUT P2, PT, P1, ...
     PLOP3.LUT P1, PT, PT, ...
@P   BRA.U.ANY
```

The reference issues the same instruction under a uniform predicate with none of
that machinery. Static predicate operations run one to two orders of magnitude
above the reference while the instruction total is at or below it.

## What to change

Take the issue predicate from the hardware election rather than from a lane
comparison, and let one election own every issue in the role.

```python
# before: a per-lane comparison materialized as the guard.
leader = T.alloc_local((1,), "uint32")
T.assign(leader[0], T.cast(lane == 0, "uint32"))
for chain in chains:
    T.evaluate(T.ptx[mma](acc, a_desc, b_desc, idesc, pred=leader[0]))

# after: the hardware election, which lowers to a uniform predicate.
leader = T.local_scalar("uint32", init=T.cuda.elect_sync())
for chain in chains:
    T.evaluate(T.ptx[mma](acc, a_desc, b_desc, idesc, pred=leader))
```

## Rationale

These instructions are warp collectives: something has to establish a single
issuing lane. A per-lane comparison does not let ptxas prove that one lane is
selected, so it rebuilds the election and its reconvergence around each issue.
The hardware election yields the uniform predicate the instruction already
wants, and the surrounding machinery disappears.

Measured on a warp-specialized backward attention kernel whose issuing warp runs
ten chains and ten commits per tile: static `PLOP3` fell from 213 to 48 against
the reference's 3, `ELECT` from 58 to 18 against 13, `VOTEU` from 52 to 1, and
total static SASS from 3679 to 3359. Four benchmark shapes moved by -9.5%,
-10.2%, -10.5% and -11.4%, taking the required matrix from 0 of 16 shapes
passing to 10 of 16 in one change.

The predicate *form* is the lever, not the absence of a branch. Moving the same
guards off branches and onto `pred=` first -- the standard predication rewrite,
with a lane comparison as the predicate -- was worth only 1-4% on the same
shapes. The remaining ten percent was entirely in what the predicate was.

## Boundary

This pays where the guarded instruction is a warp collective and the guard sits
on *every* issue. Making the same substitution for a transfer group already
behind a single guard measured neutral, -0.5% to +0.3%: there was one region's
machinery to remove, not one per instruction.

`elect_sync` is itself a warp collective, so materialize it where the warp is
converged. Binding it inside a guard leaves the excluded lanes never reaching
it.

Check that the two spellings differ before rewriting one into the other. Taking
the predicate from the hardware-election helper, in place of comparing a raw
elect against one, can lower to exactly the same machine code: two builds of one
kernel differing only in that substitution disassembled to identical opcode
histograms -- `ELECT` 80/80, `WARPSYNC.COLLECTIVE` 8/8, `BSSY` 20/20, `PLOP3`
96/96 -- and measured indistinguishable. The lever is whether the guard rides on
the instruction or wraps a branch, not which helper produced the predicate; a
substitution that leaves a multi-statement guarded region still a branch changes
nothing. A contaminated benchmark run initially credited that no-op with a
0.0174 gain, so confirm a codegen difference exists before believing a timing
one.

An election establishes *a* lane, not lane 0. Do not swap it in where the
guarded code also depends on being the lowest lane for addressing or ordering.

## Verification

Count static `ELECT`, `PLOP3` and `VOTEU` against the reference and confirm the
collective is issued under a uniform predicate (`@!UP`) rather than inside an
elect-and-branch region, then re-measure. Counters alone do not show this --
paired stall breakdowns attributed the same gap to `long_scoreboard` and
`barrier`, and only the disassembly named the cause.
