# 04_templates

This directory contains focused examples of C++ templates: function and class templates, non-type template parameters (NTTP), deduction, parameter packs, fold expressions, and C++20 concepts/constraints. Each `.cpp` in `src` is a standalone program with a corresponding CMake target you can build and run.

Use this as a quick revision sheet: skim “What’s included” to locate an example, then check the notes for gotchas and next steps.

## What’s included

- 01_function_templates.cpp – Function templates, basic usage and instantiation.
- 02_class_templates.cpp – Class templates, templated members, CTAD basics if present.
- 03_nttp_templates.cpp – Non‑type template parameters (array sizes, integral constants).
- 04_auto_template_parameters.cpp – `auto` as NTTP, C++17/20 constant parameters.
- 05_explicit_template_args.cpp – Calling templates with explicit template arguments.
- 06_template_overloading.cpp – Overloading templates vs non‑templates and specialization order.
- 07_partial_ordering.cpp – Partial ordering of function templates and selection rules.
- 08_deduction_overloading.cpp – Template argument deduction with competing overloads/deduction guides.
- 09_requires.cpp – Intro to C++20 `requires` clauses to constrain templates.
- 10_requires_with_return_type.cpp – Constraining return types and trailing `requires`.
- 11_compound_requirements.cpp – Requires‑expressions with compound requirements.
- 12_requires_expression.cpp – Using `requires (...) { ... }` to check expressions.
- 13_requires_exprss_multiple_parameter.cpp – Requires‑expressions with multiple parameters.
- 14_abbreviated_function_templates.cpp – Abbreviated templates using `auto` parameters.
- 15_overloading_templates_by_concepts.cpp – Overloading selected by concepts.
- 16_combining_concepts.cpp – Combining constraints with logical operators.
- 17_constraining_class_templates.cpp – Applying concepts to class templates.
- 18_fallback_overload_pattern.cpp – Fallback/unconstrained overloads for graceful degradation.
- 19_constraint_strength_overload_pattern.cpp – Prefer stronger constraints over weaker ones.
- 20_template_parameter_packs.cpp – Template parameter packs fundamentals.
- 21_variadic_templates.cpp – Variadic function/class templates, pack expansion.
- 22_fold_expressions.cpp – C++17 fold expressions over parameter packs.
- 23_variadic_templates_c20.cpp – C++20 improvements with packs (and constraints where relevant).
- 24_tag_dispatch_via_concepts.cpp – Tag dispatch and overload selection guided by concepts.
- 25_compile_time_selection.cpp – `if constexpr` / concept‑based compile‑time branching.
- 26_type_traits_concepts.cpp – Using `<type_traits>` with concepts for clean constraints.
- 27_expression_checking_concepts.cpp – Checking operation validity with requires‑expressions.

## Revision Cheat Sheet

### 1. Template Fundamentals
| Topic | Key Insight |
| :--- | :--- |
| Function templates | Instantiated per unique set of template arguments. Prefer `auto` return only if it doesn’t hinder deduction; otherwise use trailing return or `decltype(auto)`. |
| Class templates | Constructors can be templated too. CTAD (Class Template Argument Deduction) works when guides exist or are implicit. |
| NTTP (non‑type params) | Must be structural types. Common: integral constants, pointers/references, `auto` (since C++17/20) when value is compile‑time constant. |

### 2. Deduction, Overloads, and Specialization
| Topic | Key Insight |
| :--- | :--- |
| Overload resolution | Non‑template overloads are preferred over templates; more specialized templates beat more general ones. |
| Forwarding references | Use `T&&` with deduction and `std::forward<T>(t)` to preserve value category; avoid unintended copies. |
| Decay pitfalls | Arrays/functions decay to pointers unless taken by reference (e.g., `T(&)[N]` to keep size). |
| Partial specialization | Not allowed for function templates; use overloads or helper structs with partial specialization. |

### 3. Concepts and `requires`
| Topic | Key Insight |
| :--- | :--- |
| Constrained templates | Participate in overload resolution only if constraints are satisfied. Prefer precise, minimal constraints. |
| Requires‑clause vs expression | Clause sits on template/func; expression checks validity of operations, return types, and `noexcept`. |
| Constraint strength | Provide stronger and weaker constrained overloads; the stronger one should be preferred when both are satisfied. |

### 4. Variadics, Packs, and Folds
| Topic | Key Insight |
| :--- | :--- |
| Parameter packs | Expand with `...`. Ensure there’s a base case when not using folds. |
| Fold expressions | Choose left vs right fold thoughtfully; pick an identity element appropriate for the operator. |
| Pack constraints | You can constrain packs with concepts (e.g., `std::integral auto... xs`). |

### 5. Compile‑time Selection
| Topic | Key Insight |
| :--- | :--- |
| `if constexpr` | Discards ill‑formed branches at compile time; great for tag dispatch alternatives. |
| Tag dispatch | Combine lightweight tags with concepts to route to optimal overloads without SFINAE maze. |

## Quick Notes and Pitfalls

- Template argument deduction
  - Prefer forwarding references and `std::forward` when propagating value category.
  - Beware of decay: arrays/functions decay to pointers in deduction unless taken by reference.

- Specialization and overloading
  - Overload resolution considers non‑template functions before templates; more specialized templates beat more general ones.
  - Avoid partial specialization of function templates; use overloads or helper structs.

- NTTP and `auto` parameters
  - NTTPs must be structural types; with `auto` NTTPs ensure values are compile‑time constants.

- Concepts and `requires`
  - Constrained templates participate in overload resolution only when constraints are satisfied.
  - Prefer precise constraints; use `&&`/`||` to compose. Stronger constraints should be preferred when both match.
  - Use requires‑expressions to check member presence, valid operations, complexity (`noexcept`), and return types.

- Packs and folds
  - Remember base cases for recursion if not using folds.
  - For folds, choose the correct fold form: unary/binary, left vs right, and identity element.

## Build and Run

- Use the provided CMake targets to build a single example. For instance, to build and run the fold expression example:
  - Target name: `04_templates_src_22_fold_expressions`
  - Executable is generated in your active build directory under the same name.

Tip: In CLion, select the desired target from the run configurations; from the terminal, build the specific target to avoid rebuilding everything.
