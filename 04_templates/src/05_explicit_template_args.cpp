#include <iostream>
using namespace std;

template<typename T>
T twice(T x) {
    return x + x;
}

template<typename T>
T convert_and_add(T a, T b) {
    return a + b;
}

int main() {
    auto v = twice<int>(2.5);
    cout << "twice<int>(2.5) = " << v << endl;

    auto s = convert_and_add<double>(3, 4.5);
    cout << "convert_and_add<double>(3, 4.5) = " << s << endl;

    auto i = convert_and_add<int>(5.9, 3.2);
    cout << "convert_and_add<int>(5.9, 3.2) = " << i << endl;

    return 0;
}
