#include <iostream>
#include <concepts>
#include <vector>
#include <string>

using namespace std;

template <typename T>
concept Iterable = requires(T x) {
    { x.begin() };
    { x.end() };
};

template <typename T>
concept CallableWithDouble = requires(T f) {
    { f(0.0) } -> convertible_to<double>;
};

int main() {
    auto f = [](double x) { return x * 2; };

    static_assert(CallableWithDouble<decltype(f)>);
    static_assert(!CallableWithDouble<int>);

    vector<int> v{1, 2, 3};
    string s = "hello";

    static_assert(Iterable<vector<int>>);
    static_assert(Iterable<string>);
    static_assert(!Iterable<int>);

    cout << boolalpha;
    cout << "CallableWithDouble<lambda>: " << CallableWithDouble<decltype(f)> << endl;
    cout << "CallableWithDouble<int>: " << CallableWithDouble<int> << endl;

    cout << "Iterable<vector<int>>: " << Iterable<vector<int>> << endl;
    cout << "Iterable<string>: " << Iterable<string> << endl;
    cout << "Iterable<int>: " << Iterable<int> << endl;

    return 0;
}
