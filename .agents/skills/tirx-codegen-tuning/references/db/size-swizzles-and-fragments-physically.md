# Size swizzles and fragments physically

**Symptoms:** `smem_bank_conflict`, `register_spill`, `slow_epilogue`

## Symptom

Shared-memory bank conflicts, register spill from wide live fragments, or a slow
epilogue. Shared-pipe conflicts appear even where a kernel allocates no shared
memory, because shuffles use that pipe and divergence serializes them.

## What to change

Derive shared-memory swizzles from the lane-to-bank map and apply the same
transform at every access to a region.

```python
def _swz(byte_off):
    """XOR swizzle applied to every byte offset into this region."""
    return T.bitwise_xor(byte_off, T.shift_left(T.bitwise_and(T.shift_right(byte_off, 7), 7), 4))


def _st_shared_f32(arena, byte_off, value):
    T.evaluate(T.ptx.st.shared.b32(arena.ptr_to([_swz(byte_off)]), T.reinterpret("uint32", value)))
```

Where a matrix load consumes the region, the swizzle folds into the column
index instead:

```python
col_sw: T.uint32 = row % T.uint32(8) ^ col
T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
    frag[0], frag[1], frag[2], frag[3],
    base + row * T.uint32(128) + col_sw * T.uint32(16),
)
```

Keep a live register fragment no wider than the next consumed tile, especially
across barriers and epilogue casts: allocate it inside the tile loop.

## Rationale

A measured swizzle removed a 96x store-conflict gap; tiling a 128-register
epilogue fragment down to at most 16 live registers removed dynamic local
traffic. One kernel with zero shared allocation ran at nearly double the
reference's conflict rate; removing the branch around a guarded store took its
conflicts to zero against the reference's 612.

## Boundary

Smaller is not automatically better when it adds synchronization or breaks
vector alignment.

Twist on the byte offset, never on the row index. A `Swizzle<B,4,3>` XORs bits
`[4, 4+B)` of the byte offset with bits `[7, 7+B)`, and those source bits sit at
128 bytes whatever the row is: a 128-byte row twists on the row index only by
coincidence, a 64-byte row twists on `row >> 1` and a 32-byte row on `row >> 2`.
A formula written from the wide case passes its own correctness matrix and reads
the right bytes from the wrong places on every narrower element type.

## Verification

Measure bank conflicts, static registers, dynamic LDL/STL, and writeback depth
together across all affected shapes.
