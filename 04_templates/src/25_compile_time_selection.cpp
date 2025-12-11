#include <iostream>

using namespace std;

template <typename T>
void handle(T x) {
    if constexpr (integral<T>) {
        cout << x << " Integral!" << endl;
    } else if constexpr (floating_point<T>) {
        cout << x << " Floating Point!" << endl;
    } else {
        cout << x << " Unknown!" <<  endl;
    }
}

int main() {
    handle(42);
    handle(3.14);
    handle(1.0f);
    handle('A');
    handle(true);
    handle(string("Hello"));

    return 0;
}