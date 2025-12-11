# 06_move_semantics

This module covers rvalues, move semantics, and perfect forwarding. Mastering these topics lets you write fast, zero‑overhead abstractions and avoid unnecessary copies in both systems and competitive programming contexts.

## What to cover

- Value categories: lvalue, xvalue, prvalue (what they mean and how to spot them)
- Move semantics: move constructors and move assignment operators
- `std::move` vs `std::forward` and when to use each
- Forwarding (aka universal) references and perfect forwarding
- Rule of 0/3/5 and special member functions
- Copy elision and return value optimization (RVO/NRVO)
- noexcept and its effect on move operations and container optimizations
- Small optimization patterns: move‑aware APIs, sink parameters

## Suggested examples (future .cpps)

- movable_buffer.cpp – implement copy and move operations, track moves vs copies
- forwarding_factory.cpp – template function that perfectly forwards args to construct T
- vector_emplace_vs_push.cpp – demonstrate why `emplace_back` can be faster
- noexcept_move.cpp – show how `noexcept` affects `std::vector` reallocation

## Notes and pitfalls

- Never use an object after moving from it except to destroy/assign; only valid but unspecified state
- `std::move` does not move by itself; it only casts to an rvalue reference enabling move overloads
- Mark move ctor/assign `noexcept` when you can; standard containers prefer moving only if it’s nothrow
