#include <iostream>
using namespace std;

template<typename T>
T square(T x){
    return x * x;
}

template<typename T>
T sum(T a, T b){
    return a + b;
}

template<typename T>
T max_value(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    int a = sum(3, 5);
    auto b = square(2.5);
    auto m = max_value(5, 10);

    cout << "sum(3, 5) = " << a << endl;
    cout << "square(2.5) = " << b << endl;
    cout << "max_value(5, 10) = " << m << endl;

    return 0;
}
