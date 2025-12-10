#include <iostream>
#include <concepts>

using namespace std;

template<typename T>
concept Integral = is_integral_v<T>;

template<typename T>
concept Floating = is_floating_point_v<T>;

void process(Integral auto x) {
    cout << x << " is a Integral" << endl;
}

void process(Floating auto x) {
    cout << x << " is a Floating" << endl;
}

int main() {
    process(10);
    process(3.14);
    return 0;
}