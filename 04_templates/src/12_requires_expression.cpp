#include <iostream>
#include <concepts>
#include <string>

using namespace std;

template <typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> same_as<T>;
};

template <Addable T>
T add_values(const T& a, const T& b) {
    return a + b;
}

int main() {
    cout << add_values(2, 3) << endl;
    cout << add_values(string("hi"), string("yo")) << endl;

    // add_values(2, string("hello"));   // must fail to compile

    return 0;
}
