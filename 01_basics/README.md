# 01_basics

This directory collects small C++ programs that cover introductory concepts and basic I/O. Each source file in `src` is a standalone example that can be built as an executable through the provided CMake targets.

Use this README as a quick revision sheet: skim the “What’s included” list to recall which file demonstrates which idea, then jump to the notes section to refresh the concept, pitfalls, and a next step.

## What’s included

- **01_hello_world.cpp** – Print a simple greeting using `std::cout`; shows the minimal `main` shape and newline handling.
- **02_user_input.cpp** – Read an integer from stdin and echo it back; demonstrates formatted extraction with `std::cin`.
- **03_input_sum.cpp** – Prompt for two integers, read them safely, then output their sum; illustrates sequential input and operator precedence in expressions.
- **04_divison.cpp** – Divide two integers and display quotient and remainder; highlights integer division vs modulo and the need for zero checks.
- **05_find_size.cpp** – Show `sizeof` results for fundamental types (int, float, long, double, char); contrasts type widths and platform dependence.
- **06_swap_number.cpp** – Swap two integers without a temporary variable using arithmetic; introduces value swapping patterns and overflow caveats.
- **07_ascii.cpp** – Display the ASCII code of a character literal; shows implicit promotion of `char` to `int` when printing.
- **08_multiply_float.cpp** – Multiply two floating-point values from user input; introduces float parsing and precision expectations.
- **09_even_odd.cpp** – Determine if an integer is even or odd using `%`; reinforces boolean conditions and branching.
- **10_vowel.cpp** – Check whether a character is a vowel using `switch`; demonstrates grouped cases and `default` handling.
- **11_largest_of_3.cpp** – Find the greatest of three integers via nested conditionals; shows stepwise comparisons.
- **12_leap_year.cpp** – Test if a given year is a leap year with the standard rule (divisible by 4 and not by 100 unless by 400); demonstrates compound logical operators.
- **13_sum_of_n_natural.cpp** – Compute the sum of the first *n* natural numbers using the closed-form formula; shows functions and return values.
- **14_factorial.cpp** – Calculate factorial iteratively; demonstrates accumulation loops and integer growth.
- **15_fibonacci.cpp** – Generate the Fibonacci sequence up to *n* terms and print the list; shows vector growth and sequence generation.

## Revision Cheat Sheet

### 1. Input/Output & Types
| File | Concept | Key Insight |
| :--- | :--- | :--- |
| `01_hello_world.cpp` | **Basic Output** | `std::cout` is the stream, `<<` is the insertion operator. `\n` is generally faster than `std::endl` (which forces a flush). |
| `02_user_input.cpp` | **Basic Input** | `std::cin >> variable;` reads formatted input. It skips leading whitespace (spaces, tabs, newlines) automatically. |
| `03_input_sum.cpp` | **Chaining I/O** | You can chain operators: `cin >> a >> b;`. Input is type-safe; entering text for an `int` puts the stream in a fail state. |
| `05_find_size.cpp` | **Memory Size** | `sizeof(type)` returns size in bytes (`size_t`). `int` is usually 4 bytes, but it's platform-dependent. |
| `07_ascii.cpp` | **Char as Int** | `char` is just a small integer (usually 1 byte). Printing it displays the symbol; casting to `int` displays the ASCII code. |
| `08_multiply_float.cpp` | **Floating Point** | `float` (single precision) vs `double` (double precision). Watch out for precision errors in comparisons. |

### 2. Math & Logic Operations
| File | Concept | Key Insight |
| :--- | :--- | :--- |
| `04_divison.cpp` | **Integer Math** | `int / int` results in an `int` (truncates towards zero). `5/2` is `2`, not `2.5`. Use `%` for remainder. |
| `06_swap_number.cpp` | **Swapping** | Swapping without temp (`a=a+b...`) is a cool trick but risky (overflow). **Best practice:** `std::swap(a, b)`. |
| `09_even_odd.cpp` | **Modulo** | `n % 2 == 0` checks for even numbers. Modulo operator `%` only works with integers. |
| `11_largest_of_3.cpp` | **Logic Gates** | Nested `if`s work, but `&&` (AND) and `||` (OR) make conditions cleaner. `if (a > b && a > c)`. |
| `12_leap_year.cpp` | **Complex Logic** | Precedence matters. Rule: `(div by 4) AND (NOT div by 100 OR div by 400)`. |

### 3. Control Flow (Branching & Loops)
| File | Concept | Key Insight |
| :--- | :--- | :--- |
| `10_vowel.cpp` | **Switch Case** | Great for single variable equality checks. Don't forget `break;` or execution "falls through" to the next case. |
| `13_sum_of_n...` | **Formula vs Loop** | Math formulas (O(1)) are faster than loops (O(n)). Sum = `n*(n+1)/2`. |
| `14_factorial.cpp` | **Accumulation** | Initialize accumulators correctly (sum starts at 0, product starts at 1). Factorials grow insanely fast; `int` overflows after 12!. |
| `15_fibonacci.cpp` | **Sequence Logic** | Requires tracking state: `prev` and `current`. Loop updates state: `next = a + b; a = b; b = next;`. |

---

## Deep Dive Notes (For Interview/Exam)

### The "Stream" Concept
- **`std::cin`** is "smart." If you ask for an `int`, it tries to parse characters as digits. If it hits a letter, it stops reading and sets a "failbit."
- **Pitfall**: If `cin` fails, the variable stays unchanged (or 0 in newer standards), and future I/O operations might be ignored until you `cin.clear()` and ignore bad input.

### Type Promotion & Casting
- In `07_ascii.cpp`: When you do `char c = 'A'; cout << c;`, it prints 'A'.
- To print '65', you must cast: `cout << (int)c;` or `cout << +c;` (unary plus promotes to int).

### Integer Division Trap
- `float x = 5 / 2;` -> `x` will be `2.0`, not `2.5`.
- **Why?** `5` and `2` are ints, so integer division happens *first*.
- **Fix**: Cast one operand: `float x = 5.0 / 2;` or `(float)5 / 2`.

### The `switch` Statement
- Only works with integral types (`int`, `char`, `long`, `enum`). You cannot switch on `std::string`.

### Overflow Risks
- In `factorial.cpp`: `13!` is 6,227,020,800. A standard signed 32-bit `int` maxes out at ~2 billion (2,147,483,647).
- **Lesson**: Use `long long` for big numbers, or `unsigned` if negative values are impossible (gives you 2x range).
