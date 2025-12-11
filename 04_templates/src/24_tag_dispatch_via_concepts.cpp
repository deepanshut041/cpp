#include <iostream>
#include <type_traits>
#include <string>

using namespace std;

template <typename T>
concept Integral = std::is_integral_v<T>;

template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

void process(Integral auto x) {
    cout << x << " Integral!" << endl;
}

void process(FloatingPoint auto x) {
    cout << x << " Floating Point!" << endl;
}

void process(auto x) {
    cout << x << " Unknown!" << endl;
}

int main() {
    process(42);
    process(3.14);
    process(1.0f);
    process('A');
    process(true);
    process(string("Hello"));

    return 0;
}
