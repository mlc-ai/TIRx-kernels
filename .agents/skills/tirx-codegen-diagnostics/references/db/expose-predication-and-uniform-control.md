# Expose predication and uniform control

**Symptoms:** `branch_reconvergence`, `excess_control_instructions`, `branch_in_hot_loop`, `excess_guard_math`, `serialized_stores`

## Symptom

Excess BRA/BSYNC and reconvergence against the reference, a branch in a hot
loop, or stores issued too sparsely to saturate DRAM at identical occupancy with
no spill.

## What to change

- For one isolated load or store, express the predicate on the PTX instruction
  when an outer branch blocks if-conversion. The `pred=` keyword is the `@p`
  guard on the instruction; `T.ptx.pred(x)` is a different thing, a
  predicate-typed *operand* such as an accumulate or select flag.

  ```python
  # before: a real branch plus a BSYNC reconvergence per CTA.
  if cond:
      T.evaluate(T.ptx.st.global_.b16(buffer.ptr_to([index]), bits))

  # after: one genuinely predicated store.
  T.evaluate(T.ptx.st.global_.b16(buffer.ptr_to([index]), bits, pred=cond))
  ```

- For a loop-invariant uniform condition, hoist it and duplicate the hot loop
  only when that exposes a dense path without changing recurrence state.
- When compile-time launch and tiling facts prove complete rows and vectors,
  expose a separate guard-free specialization instead of carrying the generic
  row, column, and zero-byte-copy predicates into every unrolled issue. Keep the
  guarded path for partial rows and columns.

  ```python
  # before: the complete-vector specialization still materializes both guards.
  source_bytes = T.if_then_else(row_valid and col_valid, COPY_BYTES, 0)
  _copy_async(src, dst, source_bytes)

  # after: launch and tiling proofs make the complete path literal.
  if ROWS_PER_CTA == 1 and FULL_COLUMNS:
      _copy_async(src, dst, T.uint32(COPY_BYTES))
  else:
      source_bytes = T.if_then_else(row_valid and col_valid, COPY_BYTES, 0)
      _copy_async(src, dst, source_bytes)
  ```
- When the condition is runtime at kernel entry but constant throughout the CTA,
  an inline TIRx helper with a `T.constexpr` mode can force the two branch
  bodies to specialize independently: dispatch once on the runtime condition,
  then call the helper with `True` or `False`. This is a control-flow lowering
  tool, not permission to change dispatch semantics.

  ```python
  @T.inline
  def run_body(IS_PAD: T.constexpr):
      # Inside, IS_PAD is a literal, so each copy folds away the other path.
      if not IS_PAD and member_col < DSTATE:
          ...

  if is_pad != 0:
      run_body(True)
  else:
      run_body(False)
  ```

- For a whole elected-lane region in a warp-specialized mainloop, flatten the
  region and predicate each single-issue matrix and copy instruction
  individually instead of guarding the block.

  ```python
  # before: branching on the elected lane costs a BSSY/BSYNC pair per K block.
  if elected:
      for c in T.unroll(0, num_chunks):
          T.evaluate(T.ptx[utccp_chain](tmem_addr(c), desc_sf))

  # after: the warp never diverges; each issue carries the guard.
  for c in T.unroll(0, num_chunks):
      T.evaluate(
          T.ptx[utccp_chain](tmem_addr(c), desc_sf, pred=elected == T.uint32(1))
      )
  ```

- When such a region also contains barrier *waits*, split it by what actually
  needs one lane instead of predicating everything in it. A wait is safe
  warp-wide -- every lane spins on the same barrier and phase -- so it leaves
  the guard entirely, while the arrivals and transfer issues keep it as `@p`.
  Electing once above the region and dropping the branch is what removes the
  reconvergence; predicating the issues is what preserves single-lane
  semantics.

  ```python
  # before: one elected region holds the waits, the arrivals and the issues.
  with K.If(_elected()), K.Then():
      _wait_barrier(...)
      _expect_tx(...)
      _issue_transfer(...)

  # after: the waits run warp-wide; only what must be single-lane is predicated.
  leader = K.cuda.elect_sync()
  _wait_barrier(...)
  _expect_tx(..., pred=leader)
  _issue_transfer(..., pred=leader)
  ```

- When complementary lane groups choose between two pure arithmetic results,
  and both results are defined for every lane, match a reference that uses
  exact zero/one weights instead of duplicating a long unrolled body under a
  divergent branch.

  ```python
  # before: both unrolled arithmetic bodies survive behind half-warp control.
  with K.If(lane < HALF), K.Then():
      result = _left_path(inputs)
  with K.Else():
      result = _right_path(inputs)

  # after: one straight-line body selects with exact arithmetic identities.
  right_weight = K.cast(lane // HALF, "float32")
  left_weight = K.float32(1.0) - right_weight
  left = _left_path(inputs)
  right = _right_path(inputs)
  scaled_left = _packed_mul(left, left_weight)
  result = _packed_fma(right, right_weight, scaled_left)
  ```

## Rationale

- Replacing repeated pad checks with the `T.constexpr` specialization removed
  262,144 dynamic `CS2R` instructions in the profiled specialization by letting
  each copy dead-code-eliminate the opposite path.
- One store-heavy recurrence carried the predicate per row and issued stores too
  sparsely to saturate DRAM -- 61.1% against the reference's 62.5%, at identical
  occupancy with no spill. Hoisting the predicate and duplicating the loop
  matched the reference's static counts exactly and moved the largest shapes
  from 0.988x, 1.000x and 1.003x to 1.013x, 1.023x and 1.026x.
- One output store written as a guarded branch cost a real branch plus a
  reconvergence per CTA; reissuing it as a predicated store matched the
  reference exactly at BSYNC 9216 to 6144 and BRA 7680 to 6144, and moved two
  shapes from 0.982x and 0.988x to 1.007-1.019x and 1.012x.
- In a measured 128-thread dependency-protocol path, moving the same tail guard
  from an outer C++ branch onto the vector stores removed the remaining BSSY and
  BSYNC, reduced registers from 105 to 96 with no spill, and moved the gate from
  0.959x to 0.976x. The change was useful even though another register-budget
  change was still needed to clear the final threshold.
- Guarding a block of single-issue matrix and copy instructions on the elected
  lane costs a reconvergence pair per iteration, measured at 4.53 instructions
  per K block against the reference's roughly zero, because the reference's
  compiler predicates those instructions individually instead. Flattening the
  region and predicating each issue -- in the mainloop and in both epilogue
  store paths -- took reconvergence-marked BSSY and BSYNC from 494,814 and
  989,628 to 222 and 444, the kernel from 58.81M to 55.64M instructions, and the
  tensor pipe from 67.3% to 74.2% of cycles against the reference's 73.5%. On
  the gate the mainloop half is what moved the family, from 0.939-0.983x to
  0.995-1.035x.
- A transfer warp's per-chunk block -- ten barrier waits, eight expect-tx
  arrivals and eight tensor-copy issues inside one elected region entered every
  chunk -- was the single largest divergent region left in a warp-specialized
  backward kernel, which carried `ELECT` 80 against the reference's 52, `PLOP3`
  96 against 28, and `WARPSYNC.COLLECTIVE`/`ENDCOLLECTIVE`/`BSSY`/`BSYNC` at
  8/8/20/20 against zero, worth 3928 branch-resolving samples. Letting the waits
  run warp-wide and predicating the sixteen single-lane issues moved the
  tightest shape by +0.0116 to 1.0016 and five more by +0.0057 to +0.0099,
  taking the required matrix from one failing shape to none with the worst shape
  at 0.993x. It was the last change the gate needed, after twelve expansions
  that had each moved a required shape by 0.002 or less.

- In a rows-one, full-vector specialization, making the launch proof explicit
  removed eight asynchronous-copy row/source-byte guards, eight weight-column
  guards, and eight output-row guards. Static SASS fell from 693 to 669
  instructions, `BRA` from 9 to 1, and `ISETP` from 17 to 1 while the eight
  copies, eight loads, and eight stores were unchanged. Two affected ratios
  moved from 0.9794 to 0.9895 and from 0.9869 to 0.9955; a neighboring multi-row
  shape stayed flat at 0.9836 to 0.9834.

- A half-warp arithmetic branch around two 32-step unrolled bodies measured
  94.53% branch efficiency against the reference's 99.96%. Replacing it with
  the reference's exact zero/one packed blend kept 102 registers and zero stack
  use, while static SASS fell from 2775 to 2352 instructions. Two production
  workloads moved from 97.32/191.57 us to 92.72/186.00 us, and a one-work guard
  moved from 21.62 to 16.77 us; all output orientations retained full numerical
  agreement.

The reference's source text does not reveal whether it wants this. nvcc
routinely duplicates a loop around a store predicate the reference wrote per
iteration, so transcribing the text faithfully keeps a per-iteration branch the
reference never compiles to. Detect it by counting static arithmetic against the
reference: a recurrence block appearing an odd multiple of its logical count is
duplicated, and matching that multiple is the target.

## Boundary

Do not predicate substantial computation or duplicate a body that causes
instruction-cache pressure, spills, or lower occupancy.

A complete-row proof must follow from the specialization's launch identity, not
only from a favorite runtime sample. Preserve the generic guards for tail rows,
partial vectors, strided layouts, and any ABI whose runtime extent can differ
from the compile-time tile. Removing guard instructions is mechanism evidence,
not a timing result: a separate divisible-row proof removed control and address
instructions but still failed its affected performance shape.

Not every predicate is worth rewriting. Rebuilding an integer-materialized
condition as a boolean conjunction, aimed at an excess of logic ops and
reconvergence, changed nothing: both forms lowered identically, down to equal
totals. Check the lowering before assuming the written form survived.

In the elected-lane case, extending the rewrite to the epilogue produced no
separation from run-to-run drift; that is where to stop.

Hoisting a uniform guard can be execution-path-specific. Replacing eight
repeated asynchronous-copy branches with one outer branch improved the
non-protocol path but moved the dependency-protocol path from 3.386 to 3.425
microseconds, so the shared rewrite was reverted. Match the source topology,
then retain the hoist only on the paths where the gate confirms it.

The grouping width itself is a scheduling parameter. In one asynchronous-copy
path, merging guards across the complete copy group reduced control and address
instructions but regressed every guarded shape by roughly 4.6-8.1%. Grouping
only two adjacent copies still removed dynamic instructions and preserved
logical memory traffic, but regressed the affected shapes by roughly 5.3-10.3%
because more addresses and predicates stayed live together. Compare the
smallest legal group, intermediate groups, and the original per-issue form;
fewer reconvergence instructions do not prove that a broader branch is useful.

Encoding a row predicate as an asynchronous copy's zero-fill source size is
also path-sensitive. One measured rewrite removed more than two million dynamic
warp instructions, removed more than sixty million predicated-on thread
instructions, and lowered the allocation without spilling, yet both
dependency-protocol shapes fell to about 0.987x while their non-protocol guards
held. Source-size predication changes issue and dependency behavior, not just
branch syntax, so preserve protocol-on and protocol-off shapes as separate
performance guards.

The payoff scales with what the region holds and how often it runs, not with
how many guards exist. In the same kernel whose per-chunk transfer block was
worth +0.0116, predicating seven elected regions that each wrapped exactly ONE
barrier arrival was worth nothing twice: on the first protocol the target shape
fell 0.0016, and on a clean re-measurement it gained 0.0002 while a passing
shape lost 0.0007, merely moving which shape failed. Those arrivals ran a couple
of times per CTA against the transfer block's once per chunk. Count the dynamic
executions of the region and the instructions inside it before rewriting a guard.

Predication pays where a branch DIVERGES, not wherever a branch exists. In a
gather whose validity flag is row-uniform -- every lane of the warp takes the
same arm -- replacing the if/else with a predicated copy and a predicated
zero-fill measured neutral at +0.4%, +0.0%, -0.1%, -0.4%, because those branches
never diverged in the first place. Check that the guard actually varies across
the warp before rewriting it; divergent-branch counters separate the two cases
where branch counts do not.

An identity-weighted arithmetic blend is legal only when both arms are pure and
defined for every lane. It cannot replace guarded memory accesses, barriers, or
other side effects, and inactive-arm NaNs or infinities can defeat the zero
weight. Preserve the reference's rounding sequence and test both lane groups.

## Verification

Confirm predicate polarity and inactive-lane memory behavior, then compare BRA,
BSYNC, reconvergence, code size, registers, and both control-flow outcomes. For
the `T.constexpr` specialization, test both copies and compare code size,
registers, and every affected workload. For identity-weighted arithmetic,
confirm the long body is no longer duplicated in SASS and validate both lane
groups with asymmetric finite inputs.
