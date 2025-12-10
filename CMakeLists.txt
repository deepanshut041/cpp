cmake_minimum_required(VERSION 3.20)

# Root CMake for the whole repo (excluding the standalone examples in 09_cmake)
project(cpp_sandbox LANGUAGES CXX)

# Use C++20 across the project
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for IDE IntelliSense/name resolution (VS Code, clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Collect all .cpp files recursively and create an executable per file.
# Exclusions:
#   - 09_cmake/**           (standalone CMake examples)
#   - **/build/**           (any build directories)
#   - 10_array/tests/**     (gtest-based tests managed separately)

file(GLOB_RECURSE ALL_CPP RELATIVE ${CMAKE_SOURCE_DIR} "*.cpp")

set(SOURCES "")
foreach(src IN LISTS ALL_CPP)
    # Normalize to forward slashes for string matching
    file(TO_CMAKE_PATH "${src}" src_norm)

    # Skip build folders
    string(FIND "${src_norm}" "/build/" pos_build)
    if(NOT pos_build EQUAL -1)
        continue()
    endif()

    # Skip the dedicated CMake example area
    string(FIND "${src_norm}" "09_cmake/" pos_cmake)
    if(NOT pos_cmake EQUAL -1)
        continue()
    endif()

    # Skip tests; they may require gtest dependencies
    string(FIND "${src_norm}" "10_array/tests/" pos_tests)
    if(NOT pos_tests EQUAL -1)
        continue()
    endif()

    list(APPEND SOURCES "${src}")
endforeach()

if(SOURCES STREQUAL "")
    message(WARNING "No source files discovered for targets. Check exclusions or folder structure.")
endif()

foreach(src IN LISTS SOURCES)
    get_filename_component(src_path "${src}" DIRECTORY)
    get_filename_component(src_name_we "${src}" NAME_WE)
    set(target_name "${src_path}/${src_name_we}")
    string(REPLACE "/" "_" target_name "${target_name}")

    add_executable(${target_name} "${src}")

    # target_compile_options(${target_name} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive-> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)
endforeach()
