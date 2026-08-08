<!--
Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
All rights reserved.
Modifications Copyright (c) 2026 The TIRX Authors.
SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

This design sketch documents the modified TIRx port in
flash_attention_backward.py. See LICENSE and THIRD_PARTY_LICENSES.md for the
applicable terms.
-->

# SM100 FlashAttention-4 backward: schedule sketch

This is a review-oriented sketch of the fp16, head-dimension-128, equal-length
specialization in `flash_attention_backward.py`. It records the schedule and
dependency structure; TensorMap construction, layouts, swizzles, PTX forms,
phase arithmetic, masking details, and bounds checks stay in the implementation.

## Fixed geometry

```text
cluster          = 2 CTAs cooperating with tcgen05 cta_group=2
CTA              = 16 warps = 4 warpgroups
query tile M     = 128 rows
cluster KV tile N= 256 rows = 128 rows per CTA
head dimension   = 128
scheduler        = one physical cluster per (KV tile, batch-head)
causal scheduler = skip query tiles strictly above the KV tile
```

## Warp specialization

```text
WG3  infrastructure
  warp 0: leader-CTA MMA issue; both CTAs allocate/deallocate TMEM
  warp 1: TMA producer for K, V, Q, Q-col, dO, LSE, and dPsum
  warp 2: relay peer dS-exchange completion to the leader CTA
  warp 3: idle
  register budget: setmaxnreg dec 104
WG1 + WG2  compute and dKV epilogue
  8 warps per CTA: form P and dS, publish/exchange dS, then store dV/dK
  each WG owns one 64-column half of the compute and epilogue tiles
  register budget: setmaxnreg inc 136
WG0  dQ reduction
  4 warps per CTA: TMEM -> registers -> four-stage SMEM ring -> TMA add
  register budget: setmaxnreg inc 136
```

The active 12 compute/reduce warps and WG3 warp 0 rendezvous around TMEM
allocation. TMA and relay warps remain outside the compute-side rendezvous.

## Storage plan

```text
TMEM columns (512 total)
    0..127 : S, then P
   64..127 : dQ aliases the upper half of S/P after S has drained
  128..255 : accumulated dV
  256..383 : dP, then dS
  384..511 : accumulated dK
SMEM physical order
  Q-row, K, V, dO-row, Q-col, dO-col, dS-send, K-col, dS-exchange,
  LSE, dPsum, dQ reduction ring

SMEM lifetime reuse
  V storage -> two-stage dV epilogue
  K storage -> two-stage dK epilogue
```

The implementation uses about 217.5 KiB of dynamic SMEM. The TMA input
streams are single-stage; dQ output reduction alone uses a four-stage ring.

## Per-query-tile operations

```text
A[i] = S[i]  = K @ Q[i]^T
B[i] = dP[i] = V @ dO[i]^T
P[i] = exp(S[i] - LSE[i]), including the causal mask when enabled
dS[i] = P[i] * (dP[i] - dPsum[i])
C[i] = dV += P[i]^T @ dO[i]
D[i] = dK += dS[i]^T @ Q[i]
E[i] = dQ[i] = dS[i] @ K
```

## Dependency DAG

```text
TMA(K, Q-row) --------------------> A[i] ----------------> S-ready
                                                              |
TMA(LSE) ------------------------------------------------------+
                                                              v
                                                         form P[i]
                                                        /         \
TMA(dO-col) ------------------------------------------> C[i]      |
                                                                   |
TMA(V, dO-row) ------------------> B[i] ----------------> dP-ready |
                                                              \    |
TMA(dPsum) ----------------------------------------------------> dS[i]
                                                               /      \
                                      dS-in-TMEM ready --------       \
                                             |                         \
TMA(Q-col) --------------------------------> D[i]                       \
                                                                        v
                                          local store + peer DSMEM exchange
                                                                        |
TMA(K-col) ----------------------------------------------------------> E[i]
                                                                        |
                                                      dQ TMEM -> staged TMA add
```

`D` consumes the TMEM representation of dS and is released independently of
the DSMEM path. `E` waits for the local-plus-peer dS image. This edge cutting
keeps dK issue off the DSMEM exchange critical path.

## MMA software pipeline

```text
prologue:
  A[0] -> B[0] -> C[0]

steady state for i = 1 .. last:
  A[i] -> D[i-1] -> B[i] -> E[i-1] -> C[i]

tail:
  publish dV-done
  D[last]
  publish dK-done
  E[last]
  wait until the reducer drains the final dQ alias
```

The dV epilogue starts after `C[last]` and overlaps the remaining dK/dQ tail.
The dK epilogue starts after `D[last]` and overlaps `E[last]`.

## Buffer-release edges

```text
A commit  -> Q-row may be refilled
C commit  -> dO-row and dO-col may be refilled
D commit  -> Q-col may be refilled; dP/dS TMEM may be reused
E commit  -> the single dS-exchange buffer may be reused
dQ drain  -> the aliased upper S/P TMEM columns may be overwritten by A
```

Proxy fences bridge generic SMEM accesses and async/TMA accesses at each reuse
boundary. Barrier phase advances exactly once per logical production or
consumption. TMEM deallocation occurs only after the final reducer release.

## Control-flow sketch

```text
initialize barriers and TMEM; cluster rendezvous
parallel by warp role:
  TMA warp:     prefetch descriptors; stream fixed KV and per-i Q/dO/stat tiles
  MMA warp:     execute the prologue, steady-state order, and tail above
  WG1 + WG2:    for each i, wait A/B; form P/dS; signal D; exchange dS; signal E
                then run the split dV epilogue followed by the split dK epilogue
  relay warp:   convert per-CTA dS-exchange completion into leader-local E-ready
  WG0:          for each i, wait E; drain dQ; pipeline TMA-add into global dQ
join active warps; relinquish allocation permit; deallocate TMEM
```
