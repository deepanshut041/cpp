#include <iostream>
#include <concepts>

using namespace std;

template<integral... Ts>
auto add_integral(Ts... xs) {
    return (... + xs);
}

template<typename... Ts>
requires (floating_point<Ts> && ...)
auto add_floats(Ts... xs) {
    return (... + xs);
}

int main() {
    auto a = add_integral(1, 2, 3, 4);
    auto b = add_integral(10, 20);

    auto c = add_floats(1.5, 2.5, 3.0);
    auto d = add_floats(0.1f, 0.2f, 0.3f);

    cout << a << endl;
    cout << b << endl;
    cout << c << endl;
    cout << d << endl;

    return 0;
}
