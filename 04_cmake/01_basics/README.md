# Build with GCC on Windows (CMake)

This project uses **GCC (MinGW-w64)** instead of MSVC (`cl.exe`).
Below are clean, repeatable commands for building with **MinGW Makefiles** (no Ninja).

---

## Prerequisites

* **CMake** (Kitware build)
* **GCC/G++ (MinGW-w64)**

  * MSYS2: `pacman -S mingw-w64-x86_64-gcc`
  * Or standalone MinGW-w64 (ensure `gcc`/`g++` are in your `PATH`)

Verify:

```bash
gcc --version
g++ --version
cmake --version
```

---

## CMakeLists.txt (minimum)

> Note: CMake **4.x** does not exist (latest is 3.x). Use a valid 3.x version.

```cmake
cmake_minimum_required(VERSION 3.10)
project(HelloCMake VERSION 1.0)

add_executable(hello main.cpp)
```

---

## Build (MSYS2 MinGW64 shell)

```bash
# from project root
rm -rf build
mkdir build && cd build

cmake -G "MinGW Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
cmake --build . -v
```

Run:

```bash
./hello
```

---

## Build (Windows CMD/PowerShell with MinGW-w64 in PATH)

```bat
:: from project root
rmdir /s /q build
mkdir build && cd build

cmake -G "MinGW Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
cmake --build . -v
```

Run:

```bat
hello.exe
```

---

## Troubleshooting

* **Mixing toolchains**: If you previously configured with MSVC, delete the entire `build/` dir before switching to GCC.
* **Wrong CMake**: Prefer Kitware CMake (avoid MSYS2’s `/mingw64/bin/cmake.exe` if it causes confusion).
* **`gcc` not found**: Ensure MinGW-w64 `bin` folder is in `PATH`.
