#include <iostream>
using namespace std;

template<typename T>
struct Pair {
    T first;
    T second;
    Pair(T _first, T _second){
        first = _first;
        second = _second;
    }
};

template<typename T>
struct Holder {
    T value;
    Holder(T _value) {
        value = _value;
    }
    T get() {
        return value;
    }
};

int main() {
    Pair<int> a(2, 4);
    cout << "Pair first: " << a.first << ", second: " << a.second << endl;

    Holder<double> h(3.14);
    cout << "Holder value: " << h.get() << endl;

    return 0;
}
