# 03_classes

This directory contains focused examples around C++ classes: defining types, constructors/destructors, const‑correctness, static members, pointers to objects, class templates, and copy semantics. Each `.cpp` in `src` is a standalone program with a corresponding CMake target you can build and run.

Use this as a quick revision sheet: skim “What’s included” to locate an example, then check the cheat sheet for gotchas and next steps.

## What’s included

- 01_classes.cpp – Basic class definition, access specifiers, member functions, and object usage.
- 02_classes.cpp – Constructors (default/parameterized), `this` pointer basics, member initialization.
- 03_class_ptr.cpp – Pointers to objects, `->` vs `.`, dynamic allocation, and object lifetime notes.
- 04_classes.cpp – Encapsulation patterns (getters/setters), simple invariants; possibly additional constructor forms.
- 05_classes_static.cpp – `static` data members and `static` member functions; shared state across instances.
- 06_classess_const.cpp – Const‑correctness: `const` objects, `const` member functions, `mutable` fields.
- 07_classess_template.cpp – Class templates and templated members; basics of CTAD if applicable.
- 08_classess_copy.cpp – Copy constructor, copy assignment, destructor; the Rule of 3/5.

## Revision Cheat Sheet

### 1. Object Model & Syntax
| Topic | Key Insight |
| :--- | :--- |
| Access control | `public` API, `private` invariants, optional `protected` for inheritance. Keep data private; expose behavior. |
| Member functions | Use qualifiers properly: `const`, `&`, `&&`, and `noexcept` where appropriate. |
| Initialization | Prefer member‑initializer lists; they run before the body and avoid default‑then‑assign. |

### 2. Constructors & Destructors
| Topic | Key Insight |
| :--- | :--- |
| Special members | If you declare any of dtor/copy/move, consider the Rule of 3/5/0. |
| Default vs parameterized | Provide clear defaults; consider `explicit` single‑arg ctors to prevent implicit conversions. |
| Order of init | Members initialize in the order of declaration in the class, not the order in the initializer list. |

### 3. Copy, Move, and Ownership
| Topic | Key Insight |
| :--- | :--- |
| Rule of 3/5 | Manage resources with RAII. If you own resources, define copy/move special members or delete them. |
| Copy vs move | Copy duplicates state; move transfers ownership, leaving the source in a valid but unspecified state. Mark moved‑from objects safe to destroy. |
| Self‑assignment | Handle `if (this == &other) return *this;` in copy/move assignment operators. |

### 4. Const‑Correctness
| Topic | Key Insight |
| :--- | :--- |
| `const` objects | Can only call `const` member functions. Mark read‑only operations as `const`. |
| `mutable` | Allows modification in `const` functions for non‑logical state (e.g., caches, mutex). Use sparingly. |
| Overloads | Provide both `T& get();` and `const T& get() const;` to preserve constness. |

### 5. Static Members
| Topic | Key Insight |
| :--- | :--- |
| Static data | Shared across all instances. Define out‑of‑class storage where needed (pre‑C++17) or use inline variables (C++17+). |
| Static functions | No `this`; call via `Type::func()`; great for helpers tied to the type. |

### 6. Class Templates & Generic Code
| Topic | Key Insight |
| :--- | :--- |
| Class templates | Parameterize types/values; you can also template constructors or members. |
| CTAD | Class Template Argument Deduction can infer template args from constructors if guides exist/are implicit. |
| Separate definition | For templates, keep definitions in headers (or the same TU) so the compiler can instantiate. |

### 7. Pointers, References, and Lifetime
| Topic | Key Insight |
| :--- | :--- |
| Object pointers | Use `obj.ptr->member` vs `obj.member`. Prefer smart pointers (`unique_ptr`, `shared_ptr`) over raw owning pointers. |
| Lifetime | Pair `new` with `delete` (or better: avoid raw `new/delete`). Ensure a single clear owner (RAII). |
| References | Use references where lifetime is guaranteed; avoid storing references unless you are certain about owners. |

## Quick Notes and Pitfalls

- Prefer RAII: acquire resources in constructors, release in destructors; avoid manual `new/delete` in user code.
- Mark single‑argument constructors `explicit` to avoid surprising implicit conversions.
- Use member‑initializer lists; initialize members in the order they are declared in the class.
- If your class owns a resource, implement or delete copy/move operations intentionally (Rule of 3/5/0).
- Provide both `const` and non‑`const` accessors to preserve const correctness.
- For templates, keep definitions visible at the point of instantiation (header or same TU).

## Build and Run

- Use the provided CMake targets to build a single example. For instance, to build and run the static members example:
  - Target name: `03_classes_src_05_classes_static`
  - Executable is generated in your active build directory under the same name.

Tip: In CLion, select the desired target from run configurations; from the terminal, build the specific target to avoid rebuilding everything.
