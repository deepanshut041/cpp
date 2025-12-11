# 08_io_and_utilities

This module focuses on practical I/O, formatting, time, and filesystem utilities you’ll use in scripts, tools, and competitive programming. It rounds out the “core C++” track before diving into problem‑solving with arrays, strings, graphs, etc.

## What to cover

- iostream basics refresher: `cin`, `cout`, sync with stdio, fast I/O tips
- String formatting:
  - iostream manipulators: `std::fixed`, `std::setprecision`, `std::setw`
  - `std::format` (C++20) overview and fallbacks if unavailable
- File I/O: `std::ifstream`, `std::ofstream`, `std::fstream` (text/binary modes)
- Reading lines efficiently, tokenizing input (`std::getline`, `std::istringstream`)
- `std::filesystem`: paths, iterate directories, file size/time, create/remove
- Time utilities: `std::chrono` clocks, durations, timing a block of code
- Random numbers: `std::mt19937`, distributions, seeding
- Parsing performance tips for competitive programming (fast scan/print patterns)

## Suggested examples (future .cpps)

- fast_io.cpp – untie/sync tricks, buffered output
- format_vs_iomanip.cpp – comparing iostream manipulators with `std::format`
- file_copy.cpp – copy a file with buffers and with `std::filesystem::copy`
- chrono_timer.cpp – measure runtime of an algorithmic snippet
- list_directory.cpp – enumerate files, filter by extension

## Notes and pitfalls

- `std::getline` followed by `operator>>` needs newline handling; consume leftover `\n`
- Always check stream state; prefer RAII on streams so files close on scope exit
- Beware locale effects on parsing/formatting; set locale if needed
