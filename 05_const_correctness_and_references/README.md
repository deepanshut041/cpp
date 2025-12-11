# 05_const_correctness_and_references

This module focuses on writing correct and intention‑revealing interfaces using const‑correctness and references. It bridges your core language knowledge (01–04) with modern performance‑friendly practices used in 06_move_semantics and in competitive programming.

## What to cover

- const on data and functions
  - Top‑level vs low‑level const (const object vs pointer/reference to const)
  - const member functions and logical constness
  - Mutable members for caches (when and why)
- References and ref qualifiers on member functions
  - `&` and `&&` qualifiers to differentiate lvalue/rvalue object receivers
  - Overload patterns like `operator[] &` vs `operator[] &&`
- Passing parameters: by value, by const reference, by forwarding reference
  - Small types by value; large/expensive types by `const&`
  - Return by value and copy elision
- Non‑owning views for safer/faster APIs
  - `std::string_view` for read‑only string params
  - `std::span<T>` for contiguous ranges
- Interop with algorithms
  - Prefer ranges of views; avoid needless copies
  - Iterator invalidation and lifetime of views

## Suggested examples (future .cpps)

- const_api_basics.cpp — add const overloads and const data accessors
- ref_qualified_methods.cpp — demonstrate `&`/`&&` qualified member function overloads
- string_view_params.cpp — take `std::string_view` without copying
- span_slice.cpp — use `std::span` to pass subranges into functions

## Notes and pitfalls

- Do not return references to local variables (dangling!)
- Beware of `string_view` lifetime: it does not own data; keep referenced storage alive
- Prefer `const` everywhere it communicates intent and enables compiler optimizations
- Use ref qualifiers to prevent accidental moves from lvalues or to enable efficient rvalue operations
