#include <iostream>
#include <concepts>
#include <string>

using namespace std;

template<typename T>
concept SafeAdd = requires(T a, T b) {
    { a + b } noexcept -> std::same_as<T>;
};

template<typename T>
concept NoThrowIncrement = requires(T x) {
    { x + 1 } noexcept -> std::same_as<T>;
};

int main() {
    static_assert(SafeAdd<int>);
    static_assert(SafeAdd<double>);
    static_assert(!SafeAdd<std::string>);

    static_assert(NoThrowIncrement<int>);
    static_assert(NoThrowIncrement<double>);
    static_assert(!NoThrowIncrement<std::string>);

    cout << boolalpha;
    cout << "SafeAdd<int>: " << SafeAdd<int> << '\n';
    cout << "SafeAdd<double>: " << SafeAdd<double> << '\n';
    cout << "SafeAdd<string>: " << SafeAdd<std::string> << '\n';

    cout << "NoThrowIncrement<int>: " << NoThrowIncrement<int> << '\n';
    cout << "NoThrowIncrement<double>: " << NoThrowIncrement<double> << '\n';
    cout << "NoThrowIncrement<string>: " << NoThrowIncrement<std::string> << '\n';

    return 0;
}
