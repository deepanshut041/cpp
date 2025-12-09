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
./hello_cli
```

Test:

```bash
./hello_tests
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
./hello_cli.exe
```

Test:

```bat
./hello_tests.exe
```