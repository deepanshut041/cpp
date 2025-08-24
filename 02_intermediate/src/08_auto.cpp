#include <iostream>
#include <typeinfo>
using namespace std;

auto add(int a, int b) {
    return a + b;
}

int main() {
    auto v = 50.0;
    auto t = "Name";
    auto sum = add(5, 7);

    cout << "Name: v"
         << ", Value: " << v
         << ", Type: " << typeid(v).name() << "\n";

    cout << "Name: t"
         << ", Value: " << t
         << ", Type: " << typeid(t).name() << "\n";

    cout << "Name: sum"
         << ", Value: " << sum
         << ", Type: " << typeid(sum).name() << "\n";
}