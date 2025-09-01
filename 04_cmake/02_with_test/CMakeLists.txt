cmake_minimum_required(VERSION 3.16)
project(HelloCli VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Enable building tests" ON)

include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.0.2
)
FetchContent_MakeAvailable(fmt)

# Library with business logic (so tests can link to it)
add_library(hello_lib src/lib.cpp)
add_library(hello::lib ALIAS hello_lib)

target_include_directories(hello_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(hello_lib PUBLIC fmt::fmt)

add_executable(hello_cli src/main.cpp)
target_link_libraries(hello_cli PRIVATE hello::lib)

if(BUILD_TESTING)
    include(CTest)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )

    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)


    add_executable(hello_tests tests/hello_tests.cpp)
    target_link_libraries(hello_tests PRIVATE hello::lib GTest::gtest_main)


    include(GoogleTest)
    gtest_discover_tests(hello_tests)
endif()