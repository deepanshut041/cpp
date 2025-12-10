#include <iostream>
using namespace std;

void print_value(int x) {
    cout << "Non-template print_value(int): " << x << endl;
}

template<typename T>
void print_value(T x) {
    cout << "Template print_value(T): " << x << endl;
}

int main() {
    print_value(10);
    print_value(3.14);
    print_value("hi");
    return 0;
}
