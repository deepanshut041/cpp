#include <iostream>
using namespace std;

template<typename... Args>
void print_all(Args... args) {
    (cout << ... << args);
}

template<typename... Args>
int count_args(Args... args) {
    return sizeof...(args);
}

int main() {
    print_all("Hello", " ", "world", "!");
    cout << '\n';

    print_all("The sum is: ", 2, " + ", 3, " = ", 2 + 3);
    cout << '\n';

    cout << "Number of args in first call: "
         << count_args("Hello", " ", "world", "!") << '\n';

    cout << "Number of args in second call: "
         << count_args("The sum is: ", 2, " + ", 3, " = ", 2 + 3) << '\n';

    cout << "Number of args (ints only): "
         << count_args(1, 2, 3, 4, 5) << '\n';

    return 0;
}
