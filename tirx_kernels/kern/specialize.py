# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""Warp-role partition of the CTA exposed as ``K.specialize``.

The primitive owns the one kernel-level structural fact neither PTX nor the
in-tree ``lang`` helpers carry: which warps run which role. It emits the
dispatch guard plus ``setmaxnreg``, and checks the invariants the hardware
requires but nothing else verifies.
"""

from __future__ import annotations

import tvm
from tvm.script import tirx as T

from . import entry as _entry

# setmaxnreg operand constraints (PTX ISA 9.7.14.6).
_REGS_MIN = 24
_REGS_MAX = 256
_REGS_GRANULARITY = 8


def _validate_register_target(session, kind, name, regs):
    if regs is None:
        return
    if regs % _REGS_GRANULARITY:
        raise ValueError(f"{kind} {name!r} regs={regs} is not a multiple of {_REGS_GRANULARITY}")
    if not _REGS_MIN <= regs <= _REGS_MAX:
        raise ValueError(
            f"{kind} {name!r} regs={regs} is outside the setmaxnreg range "
            f"[{_REGS_MIN}, {_REGS_MAX}]"
        )
    if session.min_blocks_per_sm is None:
        raise ValueError(
            f"{kind} {name!r} asks for regs={regs}, but setmaxnreg requires "
            "K.kernel(..., min_blocks_per_sm=...) to pin the entry allocation"
        )


class RegisterScope:
    """One warpgroup-aligned register transition with caller-owned timing."""

    def __init__(self, owner, name, warps, regs):
        self.owner = owner
        self.name = name
        self.warps = list(warps)
        self.regs = regs
        self.emissions = 0

    def emit(self):
        """Emit the transition at the caller's current predicate and position."""
        self.emissions += 1
        direction = "inc" if self.regs > self.owner.session.entry_regs else "dec"
        T.evaluate(T.ptx[f"setmaxnreg.{direction}.sync.aligned.u32"](self.regs))

    def __repr__(self):
        return f"<K.register_scope {self.name} warps={self.warps} regs={self.regs}>"


class WarpGroup:
    """One four-warp convergence/register scope containing narrower roles."""

    def __init__(self, owner, name, warps, regs):
        self.owner = owner
        self.name = name
        self.warps = list(warps)
        self.regs = regs
        self.first_warp = self.warps[0]
        self.n_warps = len(self.warps)
        self._frames = None

    def __enter__(self):
        if self.owner.active is not None or self.owner.active_group is not None:
            raise RuntimeError(f"warpgroup {self.name!r} cannot nest inside another role scope")
        warp_id = self.owner.session.warp_id()
        cond = tvm.tirx.all(self.first_warp <= warp_id, warp_id <= self.warps[-1])
        self._frames = [T.If(cond), T.Then()]
        self.owner.dispatch.append((self._frames[0].condition, self))
        for frame in self._frames:
            frame.__enter__()
        if self.regs is not None:
            direction = "inc" if self.regs > self.owner.session.entry_regs else "dec"
            T.evaluate(T.ptx[f"setmaxnreg.{direction}.sync.aligned.u32"](self.regs))
        self.owner.active_group = self
        return self

    def __exit__(self, *exc):
        self.owner.active_group = None
        for frame in reversed(self._frames):
            frame.__exit__(*exc)
        self._frames = None
        return False

    def __repr__(self):
        return f"<K.warpgroup {self.name} warps={self.warps} regs={self.regs}>"


class Role:
    """One warp role. Use as a context manager: ``with role:``."""

    def __init__(self, owner, name, warps, regs, when, group, register_scope):
        self.owner = owner
        self.name = name
        self.warps = list(warps)
        self.regs = regs
        self.when = when
        self.group = group
        self.register_scope = register_scope
        self.first_warp = self.warps[0]
        self.n_warps = len(self.warps)
        self._frames = None

    def __enter__(self):
        if self.owner.active is not None:
            raise RuntimeError(
                f"role {self.name!r} opened inside role {self.owner.active.name!r}; "
                "roles partition the CTA and cannot nest"
            )
        if self.group is not None and self.owner.active_group is not self.group:
            raise RuntimeError(
                f"role {self.name!r} must be emitted inside warpgroup {self.group.name!r}"
            )
        if self.group is None and self.owner.active_group is not None:
            raise RuntimeError(
                f"role {self.name!r} does not belong to active warpgroup "
                f"{self.owner.active_group.name!r}"
            )
        warp_id = self.owner.session.warp_id()
        if self.group is None:
            role_warp_id = warp_id
            lo, hi = self.first_warp, self.warps[-1]
        else:
            role_warp_id = warp_id % self.group.n_warps
            lo = self.first_warp - self.group.first_warp
            hi = self.warps[-1] - self.group.first_warp
        cond = (
            role_warp_id == lo if lo == hi else tvm.tirx.all(lo <= role_warp_id, role_warp_id <= hi)
        )
        if self.when is not None:
            cond = tvm.tirx.all(cond, self.when)
        self._frames = [T.If(cond), T.Then()]
        # Remember the exact condition *node* the frame will build the
        # IfThenElse from, so the finished body can be rewritten into an
        # else-if chain (Specialize.chain_dispatch). Read back off the frame
        # rather than kept from above because `warp_id == lo` hands back an
        # EqualOp proxy, and the node the frame holds is the converted one.
        # Identity, not structure: a user-written `K.If` spelling the same
        # predicate is a different node and must not be chained.
        self.owner.dispatch.append((self._frames[0].condition, self))
        for frame in self._frames:
            frame.__enter__()
        if self.regs is not None and self.register_scope is None:
            direction = "inc" if self.regs > self.owner.session.entry_regs else "dec"
            T.evaluate(T.ptx[f"setmaxnreg.{direction}.sync.aligned.u32"](self.regs))
        self.owner.active = self
        return self

    def __exit__(self, *exc):
        self.owner.active = None
        for frame in reversed(self._frames):
            frame.__exit__(*exc)
        self._frames = None
        return False

    def __repr__(self):
        return f"<K.role {self.name} warps={self.warps} regs={self.regs}>"


class Specialize:
    """The set of roles a kernel declares; validated when the kernel finalizes."""

    def __init__(self, session, chain_dispatch=True):
        self.session = session
        self.roles = []
        self.groups = []
        self.register_scopes = []
        self.active = None
        self.active_group = None
        #: whether adjacent role guards are folded into an else-if chain; see
        #: :func:`specialize` for why this defaults on and when to turn it off.
        self.chaining = chain_dispatch
        # (condition node, role) for every `with role:` entered, in emission
        # order. Consumed by chain_dispatch once the body is built.
        self.dispatch = []

    def role(self, name, warps, regs=None, when=None, group=None, register_scope=None):
        """Declare a role and, optionally, its absolute register target.

        A ``regs=`` target requires the kernel to set ``min_blocks_per_sm``;
        that pinned entry allocation is the single source of truth for whether
        the transition increases or decreases registers. ``when=`` adds a
        CTA-uniform participation condition to the role's dispatch guard
        instead of nesting a second predicate inside it. ``register_scope=``
        inherits that scope's target for budget validation without re-emitting
        its transition inside the functional role.
        """
        warps = sorted(set(warps)) if not isinstance(warps, int) else [warps]
        if not warps:
            raise ValueError(f"role {name!r} owns no warps")
        if register_scope is not None:
            if regs is not None:
                raise ValueError(
                    f"role {name!r} inherits register scope {register_scope.name!r}; "
                    "do not repeat regs="
                )
            if register_scope not in self.register_scopes:
                raise ValueError(f"role {name!r} references a foreign register scope")
            outside = sorted(set(warps) - set(register_scope.warps))
            if outside:
                raise ValueError(
                    f"role {name!r} warp(s) {outside} lie outside register scope "
                    f"{register_scope.name!r}"
                )
            regs = register_scope.regs
        else:
            _validate_register_target(self.session, "role", name, regs)
        if warps != list(range(warps[0], warps[-1] + 1)):
            raise ValueError(
                f"role {name!r} warps={warps} are not contiguous; a role is one "
                "warp-id range so the dispatch is a single compare"
            )
        if group is not None:
            if group not in self.groups:
                raise ValueError(f"role {name!r} references a foreign warpgroup")
            outside = sorted(set(warps) - set(group.warps))
            if outside:
                raise ValueError(
                    f"role {name!r} warp(s) {outside} lie outside warpgroup {group.name!r}"
                )
        for other in self.roles:
            overlap = sorted(set(warps) & set(other.warps))
            if overlap:
                raise ValueError(
                    f"role {name!r} and role {other.name!r} both claim warp(s) {overlap}"
                )
        role = Role(self, name, warps, regs, when, group, register_scope)
        self.roles.append(role)
        return role

    def warpgroup(self, name, warps, regs=None):
        """Declare one aligned four-warp scope shared by narrower roles."""
        warps = sorted(set(warps)) if not isinstance(warps, int) else [warps]
        if len(warps) != 4 or warps != list(range(warps[0], warps[0] + 4)):
            raise ValueError(f"warpgroup {name!r} must own four contiguous warps")
        if warps[0] % 4:
            raise ValueError(f"warpgroup {name!r} must start at a multiple of four")
        _validate_register_target(self.session, "warpgroup", name, regs)
        for other in self.groups:
            overlap = sorted(set(warps) & set(other.warps))
            if overlap:
                raise ValueError(
                    f"warpgroup {name!r} and {other.name!r} both claim warp(s) {overlap}"
                )
        group = WarpGroup(self, name, warps, regs)
        self.groups.append(group)
        return group

    def register_scope(self, name, warps, regs):
        """Declare a register transition without owning functional dispatch.

        The caller emits the scope at the original temporal point and under
        the original warp-uniform predicate. K emits only ``setmaxnreg`` and
        owns its participant metadata and CTA register-budget validation.
        """
        warps = sorted(set(warps)) if not isinstance(warps, int) else [warps]
        if not warps:
            raise ValueError(f"register scope {name!r} owns no warps")
        if warps != list(range(warps[0], warps[-1] + 1)):
            raise ValueError(f"register scope {name!r} warps={warps} are not contiguous")
        if warps[0] % 4 or len(warps) % 4:
            raise ValueError(
                f"register scope {name!r} must own a warpgroup-aligned multiple of four warps"
            )
        _validate_register_target(self.session, "register scope", name, regs)
        if regs is None:
            raise ValueError(f"register scope {name!r} requires regs")
        scope = RegisterScope(self, name, warps, regs)
        self.register_scopes.append(scope)
        return scope

    def chain_dispatch(self, func):
        """Rewrite each run of **adjacent** role guards into an else-if chain.

        Sibling ``if``s and an if/else-if chain are semantically identical here
        — :meth:`role` proves the warp ranges disjoint, so at most one guard is
        ever true — but they are not identical to ``ptxas``. Sibling guards let
        it treat each role's live values as simultaneously live; chained ones
        tell it the bodies are mutually exclusive. At the 255-register cliff
        that is the difference between assembling and ``ptxas fatal C7600``:
        hand-editing a generated kernel's sibling ``if``s into if/else (same
        code otherwise) dropped its consumer role to 255 and let it assemble.

        The C7600 is the *loud* form. The same pressure also resolves as silent
        spills, which is why this is on by default rather than reached for when
        a build breaks — see :func:`specialize` for the trade and its cost.

        Adjacency is the rule, and it is defined structurally: guards chain only
        when they are **consecutive statements of the same block**. Anything the
        body emits at CTA scope between two ``with role:`` blocks — a barrier,
        a store, a pipeline init — breaks the run, and the roles either side of
        it stay siblings. That is what keeps the rewrite safe: it reassembles
        whole statements that were already built, so no role body can end up
        nested inside another's scope, and code between roles cannot be
        swallowed into a branch.

        A run also never chains the same role twice. Re-entering one role in
        two adjacent blocks (``with cg0: ...`` twice) means "do both on those
        warps"; chaining that would make the second body unreachable.

        Off entirely under ``K.specialize(chain_dispatch=False)``, which is a
        measured trade rather than a preference — see :func:`specialize`.
        """
        if not self.chaining or len(self.dispatch) < 2:
            return func

        def role_of(stmt):
            """The role whose guard *stmt* is, or None if it is not one.

            The ``else_case is not None`` arm is **defensive and currently
            unreachable**: a role guard is built by :meth:`Role.__enter__` as
            ``If`` + ``Then`` only, and ``K.Else`` cannot be attached to it
            (the frames close on ``__exit__``, and the role is not exposed
            during the window where an else could be opened). It is written
            anyway because ``chain_run`` below *discards* ``else_case`` when it
            rebuilds a guard — so if a role guard ever does grow an else, the
            rewrite must skip it rather than silently drop that branch. Do not
            read this arm as evidence that the case is handled; it is only
            evidence that it is refused.
            """
            if not isinstance(stmt, tvm.tirx.IfThenElse) or stmt.else_case is not None:
                return None
            for cond, role in self.dispatch:
                if stmt.condition.same_as(cond):
                    return role
            return None

        def chain_run(run):
            """[if a, if b, if c] -> if a else { if b else { if c } }.

            The whole construction lives here. Each level currently re-uses its
            guard's own condition node, which the codegen re-materialises at
            every else-level; the pending experiment is to hoist the comparison
            into one uniform local ahead of the chain and have the levels test
            that. That variant is a change to this function alone — the run
            detection, adjacency rule and rewrite plumbing outside it do not
            move — which is why it is kept as a separate step rather than
            inlined into ``rewrite``.
            """
            chained = None
            for stmt in reversed(run):
                chained = tvm.tirx.IfThenElse(stmt.condition, stmt.then_case, chained, stmt.span)
            return chained

        def rewrite(seq):
            out, i, changed = [], 0, False
            while i < len(seq):
                run, seen = [], set()
                j = i
                while j < len(seq):
                    role = role_of(seq[j])
                    if role is None or id(role) in seen:
                        break
                    seen.add(id(role))
                    run.append(seq[j])
                    j += 1
                if len(run) > 1:
                    out.append(chain_run(run))
                    changed = True
                    i = j
                else:
                    out.append(seq[i])
                    i += 1
            return out if changed else None

        def postorder(stmt):
            if not isinstance(stmt, tvm.tirx.SeqStmt):
                return None
            new = rewrite(list(stmt.seq))
            if new is None:
                return None
            return new[0] if len(new) == 1 else tvm.tirx.SeqStmt(new, stmt.span)

        body = tvm.tirx.stmt_functor.ir_transform(func.body, None, postorder)
        return func.with_body(body)

    def finalize(self):
        """Check the partition invariants. Called when the kernel body ends."""
        total = self.session.warps
        owned = []
        for role in self.roles:
            owned.extend(role.warps)
        missing = sorted(set(range(total)) - set(owned))
        extra = sorted(w for w in owned if w >= total or w < 0)
        if extra:
            raise ValueError(f"roles claim warp(s) {extra} but the kernel is {total} warps wide")
        if missing:
            raise ValueError(
                f"warp(s) {missing} belong to no role; K.specialize must partition "
                f"0..{total - 1} exactly"
            )

        for scope in self.register_scopes:
            if scope.emissions != 1:
                raise ValueError(
                    f"register scope {scope.name!r} must be emitted exactly once, "
                    f"got {scope.emissions}"
                )

        def validate_register_state(scopes, state_name):
            register_targets = [self.session.entry_regs] * total
            register_owners = [None] * total
            for scope in scopes:
                if scope.regs is None:
                    continue
                for warp in scope.warps:
                    if not 0 <= warp < total:
                        raise ValueError(
                            f"register scope {scope.name!r} claims warp {warp}, "
                            f"but the kernel is {total} warps wide"
                        )
                    previous = register_owners[warp]
                    if previous is not None and register_targets[warp] != scope.regs:
                        raise ValueError(
                            f"warp {warp} has competing {state_name} targets "
                            f"{previous.name}={register_targets[warp]} and "
                            f"{scope.name}={scope.regs}"
                        )
                    register_targets[warp] = scope.regs
                    register_owners[warp] = scope

            if any(scope.regs is not None for scope in scopes):
                budget = sum(register_targets) * 32
                # The CTA pool follows the resident-warpgroup share rounded to
                # setmaxnreg's 8-register granularity. Do not apply the
                # separate 255-register entry-usage cap: setmaxnreg may claim
                # the legal 256-register ownership target from this pool.
                ceiling = _entry.cta_register_pool(total, self.session.min_blocks_per_sm)
                available = (
                    f"{ceiling}-register CTA pool "
                    f"at min_blocks_per_sm={self.session.min_blocks_per_sm}"
                )
                if budget > ceiling:
                    raise ValueError(
                        f"{state_name} budget {budget} exceeds the {available}; "
                        "regs*32*warps must fit"
                    )

            # setmaxnreg is warpgroup-collective: every warp of a warpgroup
            # reaches the same instruction with the same operand.
            for group_begin in range(0, total, 4):
                targets = set(register_targets[group_begin : group_begin + 4])
                if len(targets) > 1:
                    names = ", ".join(
                        f"warp{warp}={register_targets[warp]}"
                        for warp in range(group_begin, min(group_begin + 4, total))
                    )
                    raise ValueError(
                        f"warpgroup {group_begin // 4} has non-uniform {state_name} "
                        f"targets ({names})"
                    )

        validate_register_state([*self.groups, *self.roles], "register")
        if self.register_scopes:
            validate_register_state(self.register_scopes, "temporal register")


def specialize(chain_dispatch: bool = True):
    """Open the kernel's role partition. One per kernel.

    Parameters
    ----------
    chain_dispatch : bool
        Fold adjacent role guards into an if/else-if chain
        (:meth:`Specialize.chain_dispatch`). Per **kernel**, not per role:
        chaining is a property of a *run* of guards, so there is nothing for an
        individual role to opt out of.

        **Default True**, and the reason is asymmetry of failure, not size of
        win. Sibling guards let ptxas treat every role's live values as
        simultaneously live, and the cost of that is not reliably loud: it
        surfaces as ``ptxas fatal C7600`` sometimes and as **silent spills**
        other times. Both modes are now measured, not asserted: a 2-role GEMM
        at N=2048 fails C7600 outright with chaining off, and a 5-role paged
        kernel with ``setmaxnreg``-pinned 168 registers went from 4/32 to
        24/68 spill store/load bytes with chaining off while still compiling
        and still producing bit-identical output — the quiet case no
        correctness test can see. The quiet case has since been TIMED on a
        block-scaled fp4 kernel: chaining off took one config from 0 to 104
        spill bytes and cost **+57%** wall-clock (three runs 1.563-1.571,
        reference stable to 0.08%), monotone in spill across five configs
        (0 B → ≤1.3%, 76 B → +8.2%, 104 B → +57%), registers pinned at 168
        in every build, output bit-identical either way. So the trade is a
        chaining cost measured free-to-mildly-beneficial everywhere it has
        been TIMED, against a not-chaining cliff measured at up to +57% and
        invisible to every correctness check — and a kernel cannot tell in
        advance which side it is on. Do not paraphrase this as "unchaining is
        never better": one kernel measured chaining WORSE on its reachable
        configs - unchained ran 0.25-0.5% faster, six of six in-window runs
        agreeing to 1e-4, coherently with its static picture (the unchained
        build carries +40-72 MORE instructions yet runs faster, and on its
        config-unreachable shapes the CHAINED build is the one that spills,
        8-52 B vs zero, matching its original's spill exactly since that
        original itself writes an if/elif chain). Its port kept chaining
        anyway: with the default ON it sits at parity with its original, and
        turning it off would have made the port FASTER than the original
        while no longer reproducing the original's control flow — a faster
        restructuring is a finding, never a shipped deviation. The only
        inheritable statement is the rule itself: per-kernel, per-shape
        measurement, and the measured range now spans -0.5% to +57%.

        Chaining is **not pure upside**, and this is the counter-measurement of
        record. It fixes a hard cliff, and on FA-bwd — a kernel nowhere near
        that cliff — it measured FREE within noise: same-window on/off over
        two replicates x seven configs gave mean +0.06%, median -0.10%, and
        the SIGN VARIES across shapes of one kernel (range 0.9981-1.0047).
        An earlier ~1% figure was a cross-session comparison artifact,
        retained here as methodology provenance: same-window A/B is the only
        sound instrument for this delta. The +24 SASS (R2UR +16, EXIT +4)
        is real and its time cost is below measurement — and it is a property
        of THAT kernel, not of chaining: a second 5-role kernel measured
        chained-vs-flat at identical instruction and R2UR counts (897/897,
        34/34), differing only in control-flow shape, and timed chaining
        free-to-mildly-beneficial. No kernel should inherit ANY figure from
        this docstring — the campaign protocol is a per-kernel on/off
        measurement. Pass ``False``
        when a kernel has measured that it pays this and does not need the
        relief.

        The instruction delta is REAL and construction-independent — the
        artifact hypothesis was tested and falsified twice over. Hand-hoisting
        the guard comparisons into uniform locals ahead of the chain produces
        an instruction-identical binary (the predicate is already uniform at
        its source: the dispatch id IS the shfl broadcast), and forcing a
        further broadcast makes it worse (+24 SASS). Positionally, the extra
        R2UR sits in the compute body (deciles 2/4 of the stream), not at the
        dispatch points — it is ptxas's uniform-register allocation responding
        to live ranges that now cross else boundaries, plus the chain's tail
        turning one fall-through into several EXITs. In short: the +24 SASS
        is what telling ptxas the bodies are mutually exclusive costs in
        instructions, its time cost is below measurement, and this flag is a
        stable part of the API, not a temporary escape hatch.

        Campaign protocol: a multi-role acceptance package reports the on/off
        delta, so the trade is re-measured per kernel rather than inherited
        from this docstring.
    """
    session = _entry.current()
    if session.specialize is not None:
        raise RuntimeError("K.specialize() was already called for this kernel")
    session.specialize = Specialize(session, chain_dispatch=chain_dispatch)
    return session.specialize


def _active_role():
    session = _entry.current()
    if session.specialize is None or session.specialize.active is None:
        raise RuntimeError(
            "K.tid_in_role() / K.warp_id_in_role() are only defined inside a `with role:` block"
        )
    return session, session.specialize.active


def tid_in_role():
    """Flattened thread index within the active role."""
    session, role = _active_role()
    if role.group is not None:
        return warp_id_in_role() * 32 + session.lane_id
    if role.first_warp == 0:
        return session.thread_id
    return session.thread_id - role.first_warp * 32


def warp_id_in_role():
    """Warp index within the active role."""
    session, role = _active_role()
    if role.group is not None:
        return session.warp_id() % role.group.n_warps - (role.first_warp - role.group.first_warp)
    if role.first_warp == 0:
        return session.warp_id()
    return session.warp_id() - role.first_warp
