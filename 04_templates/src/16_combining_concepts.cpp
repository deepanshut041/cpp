#include <iostream>
#include <concepts>
#include <string>

using namespace std;

template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;

template <typename T>
concept Integer = std::integral<T>;

template <typename T>
concept Real = std::floating_point<T>;

void number_type(Integer auto) {
    cout << "integer\n";
}

void number_type(Real auto) {
    cout << "real\n";
}

void number_type(auto) {
    cout << "not a number\n";
}

int main() {
    int i = 42;
    double d = 3.14;
    float f = 1.5f;
    string s = "hello";
    bool b = true;

    cout << "int: ";
    number_type(i);

    cout << "double: ";
    number_type(d);

    cout << "float: ";
    number_type(f);

    cout << "string: ";
    number_type(s);

    cout << "bool: ";
    number_type(b);

    return 0;
}
