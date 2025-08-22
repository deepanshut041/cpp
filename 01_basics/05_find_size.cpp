#include <iostream>

using namespace std;

int main() {
    int i = 5;
    float f = 6.0;
    long l = 7;
    double d = 9;
    char c = 'a';

    cout << "Size of int: " << i << " is -> " << sizeof(i) << endl;
    cout << "Size of float: " << f << " is -> " << sizeof(f) << endl;
    cout << "Size of long: " << l << " is -> " << sizeof(l) << endl;
    cout << "Size of double: " << d << " is -> " << sizeof(d) << endl;
    cout << "Size of char: " << c << " is -> " << sizeof(c) << endl;

}