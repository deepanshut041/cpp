#include <iostream>
using namespace std;

template<typename T>
concept HasPlus = requires(T a, T b) {
    a + b;
};

struct NoPlus {};

template<typename T>
concept CanMultiply = requires(T a, T b) {
    a * b;
};



int main() {
    static_assert(HasPlus<int>);
    static_assert(!HasPlus<NoPlus>);

    cout << boolalpha;
    cout << "HasPlus<int>: " << HasPlus<int> << endl;
    cout << "HasPlus<NoPlus>: " << HasPlus<NoPlus> << endl;

    static_assert(CanMultiply<int>);
    static_assert(CanMultiply<double>);
    static_assert(!CanMultiply<string>);

    cout << "CanMultiply<int>: " << CanMultiply<int> << endl;
    cout << "CanMultiply<double>: " << CanMultiply<double> << endl;
    cout << "CanMultiply<std::string>: " << CanMultiply<string> << endl;
    return 0;
}
