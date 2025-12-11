#include <iostream>
#include <vector>

using namespace std;

template <typename T>
concept IsVector = requires(T t) {
    typename T::value_type;
    t.size();
};

void print_size(const IsVector auto& v) {
    cout << "Container size: " << v.size() << endl;
}

void print_elements(const IsVector auto& v) {
    for (const auto& x : v) cout << x << " ";
    cout << endl;
}

int main() {
    vector<int> vi = {1, 2, 3, 4, 5};
    vector<string> vs = {"hello", "world"};

    print_size(vi);
    print_elements(vi);

    print_size(vs);
    print_elements(vs);

    // int x = 10;
    // print_size(x);   // error: does not satisfy IsVector
    return 0;
}
