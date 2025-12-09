#include "hello/lib.hpp"
#include <fmt/core.h>
#include <string>
#include <iostream>


int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print("Usage:\n {} greet <name>\n {} add <a> <b>\n", argv[0], argv[0]);
        return 1;
    }

    std::string cmd = argv[1];
    if (cmd == "greet") {
        std::string name = (argc >= 3) ? argv[2] : "";
        fmt::print("{}\n", hello::greet(name));
        return 0;
    }
    if (cmd == "add") {
        if (argc < 4) {
            fmt::print(stderr, "error: add requires two integers\n");
            return 2;
        }
        int a = std::stoi(argv[2]);
        int b = std::stoi(argv[3]);
        fmt::print("{}\n", hello::add(a, b));
        return 0;
    }


    fmt::print(stderr, "error: unknown command '{}'\n", cmd);
    return 3;
}