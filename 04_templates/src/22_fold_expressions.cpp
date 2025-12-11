#include <iostream>
using namespace std;

template <typename... Args>
auto sum(Args... args) {
    return (... + args);
}

template <typename... Args>
auto multiply_all(Args... args) {
    return (... * args);
}

int main() {
    auto s1 = sum(1, 2, 3, 4);
    auto s2 = sum(10.5, 2.5);

    auto m1 = multiply_all(2, 3, 4);
    auto m2 = multiply_all(1.5, 2.0, 3.0);

    cout << "sum(1,2,3,4) = " << s1 << endl;
    cout << "sum(10.5,2.5) = " << s2 << endl;
    cout << "multiply_all(2,3,4) = " << m1 << endl;
    cout << "multiply_all(1.5,2.0,3.0) = " << m2 << endl;

    return 0;
}
