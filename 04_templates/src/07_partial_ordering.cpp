#include <iostream>
using namespace std;

template<typename T>
void debug(T x) { cout << "general: " << x << endl; }

template<typename T>
void debug(T* p) { cout << "pointer: " << p << endl; }

int main() {
    int a = 10;
    int* pa = &a;

    double d = 3.14;
    double* pd = &d;

    debug(a);
    debug(pa);
    debug(d);
    debug(pd);
    debug("hi");

    return 0;
}
