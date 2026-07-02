# Stack-based ABI: TORCH_BOX, StableIValue, and the dispatcher

Read this when:

- A `TORCH_BOX` macro isn't applicable (function signature uses unsupported types).
- You need to call the PyTorch dispatcher from C++ inside a stable kernel.
- You need to manually box/unbox values across the ABI boundary.

For routine migration, `TORCH_BOX(&fn)` handles everything; you don't need this doc.

## How TORCH_BOX works

`STABLE_TORCH_LIBRARY_IMPL` requires kernels to be **boxed**: they accept a uniform `(StableIValue* stack, uint64_t num_args, uint64_t num_outputs)` signature, regardless of the underlying typed signature. `TORCH_BOX` generates the boxing wrapper for you:

```cpp
// User writes:
torch::stable::Tensor mymuladd_cuda(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    double c);

STABLE_TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("mymuladd", TORCH_BOX(&mymuladd_cuda));
}

// TORCH_BOX expands to roughly:
void boxed_fn(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
    STD_TORCH_CHECK(num_args == 3, "arity mismatch");
    STD_TORCH_CHECK(num_outputs == 1, "return count mismatch");

    auto args = std::make_tuple(
        to<torch::stable::Tensor>(stack[0]),
        to<torch::stable::Tensor>(stack[1]),
        to<double>(stack[2]));

    auto res = std::apply(&mymuladd_cuda, args);

    stack[0] = from<torch::stable::Tensor>(res);
}
```

`to<T>` and `from<T>` are in `torch/csrc/stable/stableivalue_conversions.h` and convert between user-extension types and `StableIValue`, the ABI-stable 64-bit container that crosses the libtorch boundary.

## When TORCH_BOX doesn't fit

`TORCH_BOX` handles types in the supported-conversions table (see below). If your kernel takes a type that isn't supported (e.g., `c10::Stream`, `at::Generator`), you must hand-roll the boxing — typically by accepting the argument as a raw `StableIValue` and passing it through to another dispatcher call rather than introspecting it.

## Supported StableIValue conversions

These types have stable, defined conversions via `to<T>` / `from<T>`:

| Type in your extension | StableIValue rep | Schema type |
|---|---|---|
| `torch::stable::Tensor` | Bitwise copy of `AtenTensorHandle` into low bytes of `uint64_t` | `Tensor` |
| `torch::headeronly::ScalarType` | Bitwise copy of translated enum | `ScalarType` |
| `torch::headeronly::Layout` | Bitwise copy of translated enum | `Layout` |
| `torch::headeronly::MemoryFormat` | Bitwise copy of translated enum | `MemoryFormat` |
| `bool` | Bitwise copy into low byte | `bool` |
| `int64_t` | Bitwise copy | `int` |
| `double` | Bitwise copy | `float` |
| `torch::stable::Device` | Bitwise copy of index + type | `Device` |
| `std::string` / `std::string_view` | Bitwise copy of `StringHandle` | `str` |
| `std::optional<S>` | Pointer to a heap-allocated `StableIValue` for `S`, or `nullptr` | `Type?` |
| `std::vector<T>` / `torch::headeronly::HeaderOnlyArrayRef<T>` | Pointer to heap-allocated list of recursively boxed elements | `Type[]` |

Types not in this table (`c10::Stream`, `c10::SymInt`, `at::Scalar`, `at::Storage`, `at::Generator`, tuples, etc.) are **not** ABI-stable for direct conversion. For some literal-representable types (like `c10::Device`), a default `reinterpret_cast` may work but is best-effort and may change in future versions — avoid for production.

## Stack conventions

Two invariants:

1. **Populated left to right.** Argument `arg0` at `stack[0]`, `arg1` at `stack[1]`, etc. Returns follow the same pattern: `ret0` at `stack[0]` after the call.
2. **Stack always owns the objects it holds.**
   - When **calling** a stack-based API: pass owning references in, steal references out of the returned stack.
   - When **registering** a function to receive a stack: steal references from the argument stack, push new references onto it for returns.

Mishandling ownership is a tensor leak at best and a use-after-free at worst.

## Calling the dispatcher from C++

`torch_call_dispatcher` lets a stable kernel invoke another op through the PyTorch dispatcher:

```cpp
AOTITorchError torch_call_dispatcher(
    const char* opName,
    const char* overloadName,
    StableIValue* stack,
    uint64_t extension_build_version);
```

The `extension_build_version` is your `TORCH_ABI_VERSION` (defined in `torch/headeronly/version.h`), used by libtorch to apply any compatibility shims for the op's schema.

**Warning:** This is the escape hatch for calling ops that aren't yet exposed in `torch/csrc/stable/ops.h`. It does **not** provide strong forward/backward compatibility guarantees on the op's schema — if the op's signature changes upstream, your call will break. Prefer the wrapped form in `torch::stable::ops` whenever it exists.

Don't use `torch_call_dispatcher` to call ops registered by **other** extensions unless you can guarantee the schemas match. The dispatcher won't catch a mismatch — it'll silently corrupt the stack.

## TORCH_ABI_VERSION layout

```text
[ MAJ ][ MIN ][ PATCH ][      ABI TAG (reserved)       ]
```

Defined in `torch/headeronly/version.h`. The `TORCH_TARGET_VERSION` macro you set as a compile flag is encoded in the same format (e.g., `0x020b000000000000` for 2.11). C-shim APIs introduced after the target version are unavailable; using one produces a compile-time error.

C-shim APIs carry a **2-year backward-compatibility window** from the release that introduced them. `torch/csrc/stable/*` and `torch/headeronly/*` follow PyTorch's standard FC/BC policy.
