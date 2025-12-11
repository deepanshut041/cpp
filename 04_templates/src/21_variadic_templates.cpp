#include <iostream>
#include <concepts>

using namespace std;

template<integral... Args>
auto add_integrals(Args... args) {
    return (... + args);
}

template<typename... Args>
requires (floating_point<Args> && ...)
auto add_floats(Args... args) {
    return (... + args);
}

int main() {
    auto i1 = add_integrals(1, 2, 3, 4);
    auto i2 = add_integrals(10, 20, 30);

    auto f1 = add_floats(1.5, 2.5, 3.0);
    auto f2 = add_floats(0.1f, 0.2f, 0.3f);

    cout << "add_integrals(1, 2, 3, 4) = " << i1 << endl;
    cout << "add_integrals(10, 20, 30) = " << i2 << endl;
    cout << "add_floats(1.5, 2.5, 3.0) = " << f1 << endl;
    cout << "add_floats(0.1f, 0.2f, 0.3f) = " << f2 << endl;

    return 0;
}
