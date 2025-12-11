# 07_exceptions

This module explains structured error handling in C++. You’ll learn when to use exceptions vs return codes, how to write exception‑safe code, and how exception specifications (`noexcept`) affect performance and guarantees.

## What to cover

- Throwing and catching: `try`/`catch`, exception object lifetimes, rethrow with `throw;`
- Standard exceptions: `std::runtime_error`, `std::logic_error`, `std::bad_alloc`, etc.
- Exception safety levels: basic, strong, and nothrow guarantees
- RAII and unwinding: destructors run during stack unwinding
- Writing exception‑safe constructors and move/copy operations
- `noexcept` on functions and move ops; its impact on containers and optimizers
- Alternatives to exceptions: error codes, `expected` pattern, `std::optional`
- Assertions and contracts style checks for invariants (runtime vs debug)

## Suggested examples (future .cpps)

- divide_safe.cpp – safe division API returning optional/expected vs throwing
- vector_strong_guarantee.cpp – commit/rollback pattern to provide strong guarantee
- rethrow_demo.cpp – catch to log, then rethrow
- noexcept_move_vector.cpp – how `noexcept` changes behavior on reallocation

## Notes and pitfalls

- Don’t throw exceptions from destructors; if unavoidable, swallow or terminate intentionally
- Ensure invariants hold after exceptions; use RAII to keep code exception‑safe by construction
- Prefer precise exception types; avoid catching by value, catch by const reference
