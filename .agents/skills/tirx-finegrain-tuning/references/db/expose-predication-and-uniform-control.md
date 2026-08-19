# Expose predication and uniform control

**Symptoms:** `branch_reconvergence`, `warp_divergence`, `excess_control_instructions`, `branch_in_hot_loop`, `serialized_stores`

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

The reference's source text does not reveal whether it wants this. nvcc
routinely duplicates a loop around a store predicate the reference wrote per
iteration, so transcribing the text faithfully keeps a per-iteration branch the
reference never compiles to. Detect it by counting static arithmetic against the
reference: a recurrence block appearing an odd multiple of its logical count is
duplicated, and matching that multiple is the target.

## Boundary

Do not predicate substantial computation or duplicate a body that causes
instruction-cache pressure, spills, or lower occupancy.

Not every predicate is worth rewriting. Rebuilding an integer-materialized
condition as a boolean conjunction, aimed at an excess of logic ops and
reconvergence, changed nothing: both forms lowered identically, down to equal
totals. Check the lowering before assuming the written form survived.

In the elected-lane case, extending the rewrite to the epilogue produced no
separation from run-to-run drift; that is where to stop.

## Verification

Confirm predicate polarity and inactive-lane memory behavior, then compare BRA,
BSYNC, reconvergence, code size, registers, and both control-flow outcomes. For
the `T.constexpr` specialization, test both copies and compare code size,
registers, and every affected workload.
