#include <iostream>
#include <concepts>
#include <type_traits>
#include <string>

using namespace std;

template<typename T>
concept Addable = requires(T x) { x + x; };

template<typename T>
concept Arithmetic = Addable<T> && std::is_arithmetic_v<T>;

void f(Addable auto){ cout << "addable\n"; }
void f(Arithmetic auto){ cout << "arithmetic\n"; }

int main() {
    int i = 10;
    double d = 3.14;
    string s = "hi";

    cout << "f(i): ";
    f(i);

    cout << "f(d): ";
    f(d);

    cout << "f(s): ";
    f(s);

    // struct NotAddable {};
    // NotAddable na;
    // f(na); // must fail to compile if uncommented

    return 0;
}
