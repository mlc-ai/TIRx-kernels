"""Native TIRx IRBuilder conveniences shared by quantization kernels."""

from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T


def name(value, name_hint):
    """Give a builder-created IR value its source-level name."""
    IRBuilder.current().name(name_hint, value)
    return value


def scalar(name_hint, dtype, value=None):
    """Create a named mutable local scalar and optionally initialize it."""
    result = T.alloc_local([1], dtype)
    name(result, name_hint)
    if value is not None:
        T.buffer_store(result, value, [0])
    return result[0]


def store(target, value):
    """Assign a mutable scalar returned by :func:`scalar`."""
    T.buffer_store(target.buffer, value, target.indices)


def bind(name_hint, value):
    """Emit a named immutable binding."""
    return name(T.Bind(value), name_hint)


def flat_attr(attrs):
    """Open parser-style flat attribute frames until the PrimFunc exits."""
    builder = IRBuilder.current()
    prim_func_frame = next(
        frame for frame in reversed(builder.frames) if type(frame).__name__ == "PrimFuncFrame"
    )
    scope = T.attr(attrs)
    frames = scope.frames if hasattr(scope, "frames") else [scope]
    for frame in frames:
        frame_ref = frame
        prim_func_frame.add_callback(
            lambda frame_ref=frame_ref: frame_ref.__exit__(None, None, None)
        )
        frame.__enter__()
