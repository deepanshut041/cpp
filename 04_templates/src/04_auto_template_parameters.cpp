#include <iostream>
using namespace std;

template<auto Value>
struct Constant {
    static constexpr auto v = Value;
};

template<auto Value>
struct Multiplier {
    static constexpr auto v = Value;

    auto multiply(auto m) {
        return m * v;
    }
};

int main() {
    Constant<5> a;
    cout << "Constant value: " << a.v << endl;

    Multiplier<3> mul;
    int x = 10;
    cout << "Multiplier value: " << mul.v << endl;
    cout << "multiply(" << x << ") = " << mul.multiply(x) << endl;

    double y = 2.5;
    cout << "multiply(" << y << ") = " << mul.multiply(y) << endl;

    return 0;
}
