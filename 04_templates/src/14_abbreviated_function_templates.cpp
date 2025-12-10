#include <iostream>
#include <concepts>
#include <type_traits>
#include <string>

using namespace std;

template<typename T>
concept Floating = is_floating_point_v<T>;

auto divide(Floating auto a, Floating auto b) {
    return a / b;
}

template<typename T>
concept Printable = requires(T x) {
    cout << x;
};

auto print_value(Printable auto s) {
    cout << s;
}

struct NotPrintable {};

int main() {
    static_assert(Floating<float>);
    static_assert(Floating<double>);
    static_assert(!Floating<int>);

    static_assert(Printable<int>);
    static_assert(Printable<double>);
    static_assert(Printable<string>);
    static_assert(!Printable<NotPrintable>);

    auto r1 = divide(4.5, 1.5);
    auto r2 = divide(3.0f, 2.0f);

    cout << "divide(4.5, 1.5) = " << r1 << '\n';
    cout << "divide(3.0f, 2.0f) = " << r2 << '\n';

    cout << "print_value(42): ";
    print_value(42);
    cout << '\n';

    cout << "print_value(string(\"hello\")): ";
    print_value(string("hello"));
    cout << '\n';

    cout << "print_value(3.14): ";
    print_value(3.14);
    cout << '\n';

    return 0;
}
