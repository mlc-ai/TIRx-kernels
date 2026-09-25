# Exporting the source's line-info PTX

Both the kernel-sketch stage and the sketch reviewer need a line-info PTX export
of the exact source specialization. The writer's export grounds the sketch's
`instruction_selection` annotations; the reviewer's is independent and does not
reuse the writer's. This file exists so the mechanics are looked up once instead
of rediscovered every port.

Preserve the export under `${PORT_DIR}` and cite its path from the sketch, so the
reviewer can diff its own export against it.

## What a usable export looks like

- `.file` directives naming the source file, and `.loc` directives through the
  body. An export with zero `.loc` is not usable for check 5 -- line info was not
  enabled.
- For a code generator (CuTeDSL, Gluon, Triton), the generated intermediate
  source preserved alongside, since `.loc` may point into it rather than into the
  code you read.

## CuTeDSL

```bash
mkdir -p "$DUMP"          # the directory must already exist; the DSL will not create it
MM_SPARSE_ATTN_AOT_DISABLE=1 \
CUTE_DSL_NO_CACHE=1 \
CUTE_DSL_KEEP=ptx \
CUTE_DSL_LINEINFO=1 \
CUTE_DSL_DUMP_DIR="$DUMP" \
python <driver>
```

- `CUTE_DSL_LINEINFO=1` is what adds `.loc`; `CUTE_DSL_KEEP=ptx` alone gives an
  export with none.
- Disable both caches or a previously compiled binary is returned and nothing is
  dumped: the DSL's own JIT cache (`CUTE_DSL_NO_CACHE=1`) and any project-level
  AOT cache (above, MSA's is `MM_SPARSE_ATTN_AOT_DISABLE=1`).
- Keep `CUTE_DSL_KEEP=ptx`. Adding `ir-debug` or `cubin` has been observed to
  segfault the compiler.
- `CUTE_DSL_KEEP=ir` also dumps MLIR, but the "clean" MLIR is post-canonicalize
  and carries no `loc`, so it does not substitute for the PTX.

**The driver matters.** A kernel that consumes another kernel's output, or that
needs real payload data, cannot be exported by calling it with empty tensors --
it will not run, or will run a degenerate path. Drive the source's own host
wrapper with data shaped like production, and assert something about the result
so a silently wrong run is caught. Doing so doubles as a semantic check: one such
run confirmed a per-group degree bound that the correctness oracle then depended
on. If the wrapper runs an upstream kernel first, the dump directory will hold
more than one `.ptx`; take the one whose name carries the target symbol.

## Triton

```bash
TRITON_CACHE_DIR="$DUMP" python <driver>
```

The cache holds `.ttir`, `.ttgir`, `.ptx` and `.cubin` per compiled kernel.
Preserve the `.ttgir` as the generated intermediate source. Line info requires
the kernel to be compiled with debug info enabled (`TRITON_DEBUG=1`, or
`@triton.jit(debug=True)` depending on version).

## CUDA C++

```bash
nvcc -arch=<sm_xx> -ptx -lineinfo <source> -o "$DUMP/kernel.ptx"
```

When the source is JIT-built by the project (torch `cpp_extension`, a project
JIT), add `-lineinfo` to that build's flag list rather than compiling by hand, so
the export matches the specialization the project actually runs.

## Reading the export

Two commands cover most of what the sketch needs:

```bash
# key operations mapped to source lines
awk '/\.loc/{loc=$0} /ld\.global|st\.global|ld\.shared|st\.shared|atom\.|bar\.|shfl\.|mbarrier|cp\.async/ \
     {gsub(/^[ \t]+/,"",$0); gsub(/^[ \t]+/,"",loc); print loc" ||| "$0}' "$PTX" \
  | sed 's/\[[^]]*\]/[addr]/' | sort | uniq -c | sort -rn

# static opcode histogram, for the sketch's evidence table
grep -oE "^\s+[a-z][a-z0-9_.]+" "$PTX" | sed 's/^[ \t]*//' \
  | grep -v "^ld.param\|^\." | sort | uniq -c | sort -rn
```

Read the loop bodies, not only the totals. A port can match the reference's
static opcode counts exactly and still carry a fatter loop body, which is where
the time goes.

## Counting convention

Sketches in this corpus report instruction counts as instruction lines minus
predicated lines. State the convention if you report a figure, so a reviewer
recounting it does not read a mismatch as an error.
