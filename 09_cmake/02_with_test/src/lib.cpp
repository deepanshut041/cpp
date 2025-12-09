#include "hello/lib.hpp"
#include <fmt/core.h>
#include <algorithm>


namespace hello {
    int add(int a, int b) { return a + b; };

    std::string greet(std::string name) {
        if (name.empty()) name = "world";
        name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char c){return c=='\n' || c=='\r';}), name.end());
        return fmt::format("Hello, {}!", name);
    };

};