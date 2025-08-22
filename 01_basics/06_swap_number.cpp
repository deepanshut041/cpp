#include <iostream>

using namespace std;

int main() {
    int a = 5, b = 10;

    cout << "A: " << a << endl;
    cout << "B: " << b << endl;

    a = a + b;
    b = a - b;
    a = a - b;

    cout << "A: " << a << endl;
    cout << "B: " << b << endl;

}