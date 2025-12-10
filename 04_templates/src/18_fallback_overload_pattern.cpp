#include <iostream>
#include <string>

using namespace std;

template<typename T>
concept Floating = std::floating_point<T>;

void analyze(Floating auto x) {
    cout << "'" << x << "' is Floating" << endl;
}

void analyze(auto x) {
    cout << "'" << x << "' is anything else" << endl;
}


int main() {
    analyze(3.5);
    analyze(5);
    analyze("hi");
    return 0;
}
