# 02_intermediate

This directory collects intermediate‑level C++ examples that bridge core language features with the Standard Library (STL). It covers enums, `typedef`/`using`, aggregates/`struct`, type deduction (`auto`), raw pointers and dynamic memory basics, iteration patterns, and a tour of STL containers: fixed (`std::array`), dynamic (`std::vector`), node‑based (`std::list`), double‑ended (`std::deque`), adapters (`std::stack`, `std::queue`), and associative containers (`std::set`, `std::map`). Each `.cpp` under `src` is a standalone program with its own CMake target.

Use this as a quick revision sheet: skim “What’s included” to jump to an example, then review the cheat sheet for complexities, invalidation rules, and best practices.

## What’s included

- 01_enum.cpp – Traditional enums; implicit integral values; scope and conversion caveats.
- 02_enum.cpp – Scoped enums (`enum class`); strong typing and explicit casts.
- 03_enum.cpp – Underlying types and usage patterns for enums.
- 04_typdef.cpp – `typedef` and modern `using` aliases; readability and template aliasing.
- 05_struct.cpp – `struct` basics, aggregate initialization, public members by default.
- 06_struct.cpp – Member functions in `struct`, encapsulation patterns.
- 07_struct.cpp – Nested structs, initialization, and simple composition.
- 08_auto.cpp – Type deduction with `auto`, `decltype`, initializer impact.
- 09_dynamic_memory.cpp – `new`/`delete`, arrays, basic ownership pitfalls; prefer RAII.
- 10_pointers.cpp – Raw pointer fundamentals, `nullptr`, pointer arithmetic cautions.
- 11_for_each.cpp – Iteration patterns, `std::for_each`, lambda usage.
- 12_vector.cpp – `std::vector`: growth, capacity, `push_back`/`emplace_back`, invalidation.
- 13_array.cpp – `std::array<N,T>`: fixed size, stack allocation, `.at()` vs `operator[]`.
- 14_returning.cpp – Returning values/references; lifetime and copy elision notes.
- 15_tuple.cpp – `std::tuple`, `std::tie`/structured bindings, returning multiple values.
- 16_list.cpp – `std::list` (doubly‑linked list): stable iterators, splice, no random access.
- 17_stack.cpp – `std::stack` adapter: LIFO interface over an underlying container.
- 18_queue.cpp – `std::queue` adapter: FIFO interface, operations and complexity.
- 19_deque.cpp – `std::deque`: double‑ended growth, iterator validity rules.
- 20_sets.cpp – `std::set`: ordered unique keys, log‑time ops, custom comparators.
- 21_maps.cpp – `std::map`: ordered key→value, insertion/lookup, `operator[]` vs `.at()`.

## Revision Cheat Sheet

### 1. Language & Type Foundations
| Topic | Key Insight |
| :--- | :--- |
| `enum` vs `enum class` | `enum class` is scoped and strongly typed; requires explicit casts, avoids name collisions/implicit int conversions. |
| `typedef`/`using` | Prefer `using` for readability and template aliasing (e.g., `template<class T> using Vec = std::vector<T>;`). |
| `struct` | Public by default; supports methods/constructors. Aggregates can use brace init; once you add private members/ctors, aggregate rules change. |
| `auto`/`decltype` | `auto` deduces by rules of template argument deduction; `decltype(expr)` preserves references/cv as in expression. |

### 2. Pointers, Dynamic Memory, and Returns
| Topic | Key Insight |
| :--- | :--- |
| Raw pointers | Prefer non‑owning semantics; initialize to `nullptr`. Avoid pointer arithmetic unless necessary. |
| Dynamic memory | Pair every `new` with `delete` (and `new[]` with `delete[]`). In modern C++, prefer `std::unique_ptr`/`std::make_unique`. |
| Returning values | NRVO/copy elision makes returning by value efficient. Do not return references to local variables. |

### 3. Iteration, Algorithms, and Iterators
| Topic | Key Insight |
| :--- | :--- |
| `std::for_each` & lambdas | Great for side‑effects; prefer algorithm + lambda over manual loops when possible. |
| Range‑for | `for (auto &x : v)` is clear and safe; choose `const auto&` to avoid copies. |
| Iterator categories | `array`/`vector`/`deque` provide random‑access; `list` provides bidirectional. Algorithms may require certain categories. |
| Erase‑remove idiom | `v.erase(std::remove(v.begin(), v.end(), val), v.end());` to erase by value in sequence containers. |

### 4. Sequence Containers (Big Picture)
| Container | Strengths | Weaknesses | Iterator invalidation (common ops) |
| :--- | :--- | :--- | :--- |
| `std::array` | Fixed size, trivial, stack allocation, contiguous | Size known at compile time only | Never invalidates (size fixed) |
| `std::vector` | Contiguous, best cache locality, amortized O(1) push_back | Insertion/erase in middle is O(n); reallocation moves elements | Reallocation invalidates all; `erase` invalidates from point to end |
| `std::deque` | Fast push/pop front/back, stable references more than vector | Not contiguous; random access ok but weaker locality | Insertion at ends may invalidate iterators/pointers; middle ops can invalidate many |
| `std::list` | Splice/insert/erase O(1) given iterator; stable iterators | No random access; extra memory per node; cache‑unfriendly | Iterators remain valid except to erased elements |

### 5. Container Adapters
| Adapter | Underlying | Operations | Notes |
| :--- | :--- | :--- | :--- |
| `std::stack` | default `deque<T>` | `push`, `pop`, `top` | LIFO. Can choose `vector`/`list` as underlying if needed. |
| `std::queue` | default `deque<T>` | `push`, `pop`, `front`, `back` | FIFO. No iterators; use underlying if traversal needed. |

### 6. Associative Containers (Ordered)
| Container | Keys | Ordering | Complexity | Invalidation |
| :--- | :--- | :--- | :--- | :--- |
| `std::set` | unique keys | `Compare` (default `<`) | insert/erase/find: O(log n) | Iterators/pointers stable except erased elements |
| `std::map` | unique keys -> values | `Compare` (default `<`) | `operator[]` inserts default value if not present; `.at()` throws | Stable iterators except erased elements |
| Custom compare | struct/functor/lambda | Strict weak ordering required | For `map/set`, comparator forms the sort order | Ensure comparator imposes strict weak ordering to avoid UB |

### 7. Tuple and Multiple Returns
| Topic | Key Insight |
| :--- | :--- |
| `std::tuple` | Heterogeneous values; access via `std::get<N>` or structured bindings `auto [a,b]=...;`. |
| Tying/ignoring | Use `std::tie(x, std::ignore, y) = tuple;` to assign selectively. |
| Returning multi‑values | Prefer returning a `struct` with named fields when semantics matter; `tuple` is fine for ad‑hoc results. |

## Quick Notes and Pitfalls

- Bounds‑checked access
  - Prefer `.at(i)` for bounds checking (throws) during learning/debugging; `operator[]` is unchecked.

- Reserve and capacity
  - For `std::vector`, use `reserve(n)` when you know size to avoid reallocations; `shrink_to_fit()` is non‑binding.

- Invalidation rules matter
  - After `vector` reallocation, all pointers/refs/iterators to elements are invalid. Re‑acquire them.
  - `list` iterators are stable across insert/erase (except erased element).
  - `set/map` iterators remain valid unless element erased; insertion does not invalidate others.

- `emplace` vs `insert`/`push`
  - `emplace` constructs in place and can avoid extra moves/copies; pass constructor args directly.

- Map access semantics
  - `m[key]` inserts default‑constructed value if `key` missing; use `.at(key)` for read‑only access that throws on miss.

- Erase while iterating
  - Use iterator‑returning `erase` or post‑increment pattern: `it = c.erase(it);` to continue safely.

- Prefer RAII
  - Replace raw `new/delete` with smart pointers where ownership is needed; keep raw pointers for non‑owning observers.

## Build and Run

- Use the provided CMake targets to build a single example. For instance, to build and run the vector example:
  - Target name: `02_intermediate_src_12_vector`
  - Executable is generated in your active build directory under the same name.

Tip: In CLion, select the desired target from the run configurations; from the terminal, build the specific target to avoid rebuilding everything.
